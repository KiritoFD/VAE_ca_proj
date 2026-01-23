import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Import project modules
from model import LGTUNet
from losses import PatchSlicedWassersteinLoss, CosineSSMLoss
from train import InMemoryLatentDataset

def calibrate_ode(config_path='config.json', num_batches=10):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 2. Initialize Model (Untrained, Random Weights)
    # This simulates the gradient landscape at the very beginning of training
    print("Initializing LGT Model...")
    model = LGTUNet(
        latent_channels=config['model']['latent_channels'],
        base_channels=config['model']['base_channels'],
        style_dim=config['model']['style_dim'],
        time_dim=config['model']['time_dim'],
        num_styles=config['model']['num_styles'],
        num_encoder_blocks=config['model']['num_encoder_blocks'],
        num_decoder_blocks=config['model']['num_decoder_blocks']
    ).to(device)
    model.eval() # We calculate gradients w.r.t input/latent, not weights for now
    
    # 3. Setup Dataset
    print("Loading dataset...")
    dataset = InMemoryLatentDataset(
        data_root=config['data']['data_root'],
        num_styles=config['model']['num_styles'],
        style_subdirs=config['data']['style_subdirs']
    )
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, drop_last=True)
    
    # 4. Setup Losses
    patch_size = config['loss'].get('patch_size', 5)
    print(f"Calibration Config: Batch={config['training']['batch_size']}, Patch={patch_size}, ODE Steps=10")
    
    loss_swd = PatchSlicedWassersteinLoss(
        patch_size=patch_size, 
        num_projections=128, 
        max_samples=8192,
        use_fp32=True
    ).to(device)
    
    loss_ssm = CosineSSMLoss(use_fp32=True).to(device)
    
    # 5. Dynamic Calibration Loop
    ratios = []
    
    print(f"\n🚀 Running ODE-aware Calibration on {num_batches} batches...")
    print("-" * 75)
    print(f"{'Batch':<6} | {'Content Grad':<12} | {'Style Grad':<12} | {'Ratio (C/S)':<10}")
    print("-" * 75)
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches: break
        
        # Prepare Inputs
        x_src = batch['latent'].to(device)
        B = x_src.shape[0]
        style_id = batch['style_id'].to(device)
        
        # Target Style Reference
        rand_idx = torch.randperm(len(dataset))[:B]
        x_style = dataset.latents_tensor[rand_idx].to(device)
        
        # --- ODE Integration Simulation ---
        # We must simulate the forward pass to get x_1
        # Initial condition
        x0 = torch.randn_like(x_src)
        t_val = torch.rand(B, device=device) # Random t ~ [0,1]
        
        # Construct path x_t (Linear interpolation anchored to content)
        # This is what the model sees as input
        t_expand = t_val.view(-1, 1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x_src
        
        # Integrate ODE (Simplified Euler for 10 steps to get x_1)
        # Note: We don't need gradients for the integration steps themselves for this calibration,
        # we only need gradients AT x_1. So we can use torch.no_grad() for the loop to save memory,
        # then enable grad for the final calculation.
        
        # Wait! To measure the force "at the end", we just need to know WHERE the end is.
        # With an untrained model, the velocity v is random.
        # Let's compute a "hypothetical" x_1 based on current random model.
        
        curr_x = x_t.clone()
        dt = 1.0 / 10.0
        with torch.no_grad():
            for step in range(10):
                # Simple Euler
                curr_t = t_val + step * dt
                # Clamp t to [0,1]
                curr_t = torch.clamp(curr_t, 0.0, 1.0)
                v = model(curr_x, curr_t, style_id)
                curr_x = curr_x + v * dt
        
        # Now curr_x is our x_1 (Terminal State).
        # We attach gradient requirement HERE.
        x_1 = curr_x.detach().requires_grad_(True)
        
        # --- Measure Forces at Terminal State ---
        
        # Force 1: Style Pull
        l_style = loss_swd(x_1, x_style)
        l_style.backward()
        g_style = x_1.grad.norm().item()
        
        # Force 2: Content Pull
        x_1.grad = None
        l_content = loss_ssm(x_1, x_src)
        l_content.backward()
        g_content = x_1.grad.norm().item()
        
        # Ratio
        if g_style > 1e-9:
            ratio = g_content / g_style
            ratios.append(ratio)
            print(f"{i:<6} | {g_content:.6f}      | {g_style:.6f}      | {ratio:.2f}")
            
    # 6. Conclusion
    avg_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    
    print("-" * 75)
    print(f"✅ Final Ratio (Content Force / Style Force): {avg_ratio:.2f} (±{std_ratio:.2f})")
    print(f"💡 This means SSM gradient is {avg_ratio:.1f}x stronger than SWD gradient.")
    print(f"👉 Recommended w_style: {avg_ratio:.1f}")
    print("-" * 75)

if __name__ == "__main__":
    calibrate_ode()