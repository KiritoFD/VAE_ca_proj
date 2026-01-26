import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import json
import random
from pathlib import Path
from tqdm import tqdm
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from structure_net import LearnableStructureExtractor

# ==========================================
# 1. Dataset Wrapper
# ==========================================
class ProxyDataset(Dataset):
    def __init__(self, cfg):
        self.files = []
        data_root = Path(cfg['data']['data_root'])
        for subdir in cfg['data']['style_subdirs']:
            self.files.extend(list((data_root / subdir).glob("*.pt")))
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location='cpu')

# ==========================================
# 2. Training Loop
# ==========================================
def train_proxy():
    device = 'cuda'
    print(f"🚀 Training Structural Proxy on {device}...")
    
    # Load Config
    with open('config.json', 'r') as f:
        cfg = json.load(f)
        
    # Load VAE (Teacher)
    print("Loading Teacher (VAE)...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    # Init Student
    student = LearnableStructureExtractor().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    
    # Dataloader
    dataset = ProxyDataset(cfg)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    
    # Training
    epochs = 5 # 5 Epochs is enough for convergence
    loss_fn = nn.BCELoss()
    
    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        
        for latents in loop:
            latents = latents.to(device)
            if latents.ndim == 3: latents = latents.unsqueeze(1)
            
            # --- Teacher Step (Ground Truth Generation) ---
            with torch.no_grad():
                # 1. Decode
                pixels = vae.decode(latents / 0.18215).sample
                pixels = (pixels / 2 + 0.5).clamp(0, 1) # [0, 1]
                
                # 2. Canny Edge Detection (Batch Processing on GPU via Sobel approximation for speed)
                # Or simply loop to CPU OpenCV (Slower but accurate Canny)
                # Let's use CPU OpenCV for high quality ground truth
                
                gt_edges = []
                pixels_np = pixels.permute(0, 2, 3, 1).cpu().numpy() # [B, H, W, 3]
                
                for i in range(pixels_np.shape[0]):
                    img_u8 = (pixels_np[i] * 255).astype(np.uint8)
                    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
                    # Use wider thresholds to capture main structure
                    edges = cv2.Canny(gray, 50, 150)
                    gt_edges.append(edges)
                
                gt_tensor = torch.from_numpy(np.stack(gt_edges)).to(device).float() / 255.0
                gt_tensor = gt_tensor.unsqueeze(1) # [B, 1, 512, 512]
                
                # Downsample GT back to Latent Size for Student Target
                # Use max pooling to preserve thin lines
                gt_latent = F.max_pool2d(gt_tensor, kernel_size=8, stride=8)
            
            # --- Student Step ---
            pred_edges = student(latents) # [B, 1, H, W]
            
            # Resize pred if dimensions mismatch (just in case)
            if pred_edges.shape[-1] != gt_latent.shape[-1]:
                gt_latent = F.interpolate(gt_latent, size=pred_edges.shape[-2:], mode='nearest')
            
            loss = loss_fn(pred_edges, gt_latent)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
    # Save the trained proxy
    torch.save(student.state_dict(), "structure_proxy.pt")
    print("\n✅ Proxy trained and saved to 'structure_proxy.pt'")

if __name__ == "__main__":
    train_proxy()