import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import random
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from diffusers import AutoencoderKL

# ==========================================
# 1. 待测组件: Robust Latent Filter
# ==========================================
class LatentEdgeFilter(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.k3 = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
        self.k5 = torch.tensor([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ], dtype=torch.float32, device=device).view(1, 1, 5, 5).repeat(4, 1, 1, 1)

    def forward(self, x):
        with torch.no_grad():
            e3 = F.conv2d(x, self.k3, groups=4, padding=1)
            e5 = F.conv2d(x, self.k5, groups=4, padding=2)
            response = torch.max(torch.abs(e3), torch.abs(e5))
            edge_map, _ = torch.max(response, dim=1, keepdim=True)
            
            B = edge_map.shape[0]
            flat = edge_map.view(B, -1)
            v_min = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            v_max = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            return (edge_map - v_min) / (v_max - v_min + 1e-6)

# ==========================================
# 2. 评估工具函数
# ==========================================
def compute_metrics(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    pred_area = pred_mask.sum()
    gt_area = gt_mask.sum()
    
    iou = intersection / (union + 1e-6)
    precision = intersection / (pred_area + 1e-6)
    recall = intersection / (gt_area + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return iou, precision, recall, f1

def save_comparison(idx, latent_map, gt_map, score, title, output_dir):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Latent Filter (IoU={score:.2f})")
    plt.imshow(latent_map, cmap='inferno')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Ground Truth (Canny)")
    plt.imshow(gt_map, cmap='gray')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_dir / f"bench_{idx}_{title.lower().replace(' ', '_')}.png")
    plt.close()

# ==========================================
# 3. 主流程
# ==========================================
def main():
    NUM_SAMPLES = 50 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Starting Benchmark on {device}...")
    
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    
    try:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        vae.eval()
    except:
        print("❌ VAE load failed.")
        return

    data_root = Path(cfg['data']['data_root'])
    all_files = []
    for subdir in cfg['data']['style_subdirs']:
        all_files.extend(list((data_root / subdir).glob("*.pt")))
    
    if len(all_files) == 0:
        print("❌ No data found.")
        return
        
    samples = random.sample(all_files, min(NUM_SAMPLES, len(all_files)))
    filter_mod = LatentEdgeFilter(device)
    
    metrics = {'iou': [], 'prec': [], 'rec': [], 'f1': []}
    best_iou = -1.0
    worst_iou = 1.0
    
    for i, file_path in enumerate(tqdm(samples, desc="Benchmarking")):
        try:
            # 1. Load Latent
            latent = torch.load(file_path, map_location=device)
            if latent.ndim == 3: latent = latent.unsqueeze(0)
            
            # 2. Decode First (To get Ground Truth Size)
            with torch.no_grad():
                pixels = vae.decode(latent / 0.18215).sample
                pixels = (pixels / 2 + 0.5).clamp(0, 1) * 255
            
            # 获取真实的 H, W (例如 256, 256)
            B, C, H, W = pixels.shape
            
            # 3. Path B: Ground Truth (Canny)
            img_np = pixels[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            canny = cv2.Canny(img_gray, 50, 150)
            gt_mask = (canny > 0)
            
            # 4. Path A: Latent Filter (Interpolate to MATCH GT size)
            latent_edge = filter_mod(latent) # [1, 1, h, w]
            
            # 🔥 修复核心：动态使用 H, W，而不是写死 512
            latent_up = F.interpolate(latent_edge, size=(H, W), mode='bilinear')[0, 0].cpu().numpy()
            
            thresh = latent_up.mean() + 1.5 * latent_up.std()
            pred_mask = (latent_up > thresh)
            
            # 5. Compute Metrics
            iou, prec, rec, f1 = compute_metrics(pred_mask, gt_mask)
            
            metrics['iou'].append(iou)
            metrics['prec'].append(prec)
            metrics['rec'].append(rec)
            metrics['f1'].append(f1)
            
            if iou > best_iou:
                best_iou = iou
                save_comparison(i, latent_up, gt_mask, iou, "Best Case", out_dir)
            if iou < worst_iou:
                worst_iou = iou
                save_comparison(i, latent_up, gt_mask, iou, "Worst Case", out_dir)
                
        except Exception as e:
            print(f"Error on {file_path}: {e}")
            continue

    if len(metrics['iou']) == 0:
        print("❌ No samples processed successfully.")
        return

    print(f"\n📊 Benchmark Report (N={len(metrics['iou'])})")
    print(f"{'-'*40}")
    print(f"IoU       : {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}")
    print(f"Precision : {np.mean(metrics['prec']):.4f} ± {np.std(metrics['prec']):.4f}")
    print(f"Recall    : {np.mean(metrics['rec']):.4f} ± {np.std(metrics['rec']):.4f}")
    print(f"F1 Score  : {np.mean(metrics['f1']):.4f}  ± {np.std(metrics['f1']):.4f}")
    print(f"{'-'*40}")
    
    avg_iou = np.mean(metrics['iou'])
    if avg_iou > 0.35:
        print("✅ PASSED: Strong correlation found. Latent filter is robust.")
        print("   Action: Enable 'RobustLatentStructureLoss' in training.")
    elif avg_iou > 0.20:
        print("⚠️ MARGINAL: Weak correlation but usable as regularization.")
        print("   Action: Consider increasing 'edge_boost' weight.")
    else:
        print("❌ FAILED: No meaningful correlation.")
        print("   Action: Revert to Plan C (Learnable CNN Structure Extractor).")
        
    print(f"\nVisualizations saved to: {out_dir}")

if __name__ == "__main__":
    main()