import torch
import torch.nn as nn
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
# 1. 重新定义小网络结构 (确保与训练时一致)
# ==========================================
class LearnableStructureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 4 channels (Latent) -> Output: 1 channel (Edge Probability)
        self.net = nn.Sequential(
            # Layer 1: Expand features
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            
            # Layer 2: Dilated Conv (Receptive Field ↑)
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # Layer 3: Feature consolidation
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            # Layer 4: Projection to Edge Map
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 评估指标计算
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
    plt.title(f"Proxy Prediction (IoU={score:.2f})")
    plt.imshow(latent_map, cmap='inferno')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Ground Truth (Canny)")
    plt.imshow(gt_map, cmap='gray')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_dir / f"proxy_eval_{idx}_{title.replace(' ', '_')}.png")
    plt.close()

# ==========================================
# 3. 主流程
# ==========================================
def main():
    NUM_SAMPLES = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    proxy_path = "structure_proxy.pt"
    
    out_dir = Path("proxy_benchmark_results")
    out_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Benchmarking Learned Proxy on {device}...")
    
    # 1. 加载 Config 和 VAE
    with open('config.json', 'r') as f:
        cfg = json.load(f)
        
    try:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        vae.eval()
    except:
        print("❌ VAE load failed.")
        return

    # 2. 加载训练好的 Proxy
    if not Path(proxy_path).exists():
        print(f"❌ Weight file '{proxy_path}' not found! Run train_proxy.py first.")
        return
        
    proxy = LearnableStructureExtractor().to(device)
    proxy.load_state_dict(torch.load(proxy_path, map_location=device))
    proxy.eval()
    print("✓ Loaded Proxy Weights")

    # 3. 准备数据
    data_root = Path(cfg['data']['data_root'])
    all_files = []
    for subdir in cfg['data']['style_subdirs']:
        all_files.extend(list((data_root / subdir).glob("*.pt")))
    
    if len(all_files) == 0:
        print("❌ No data found.")
        return
        
    samples = random.sample(all_files, min(NUM_SAMPLES, len(all_files)))
    
    metrics = {'iou': [], 'prec': [], 'rec': [], 'f1': []}
    best_iou = -1.0
    worst_iou = 1.0
    
    print("running evaluation...")
    for i, file_path in enumerate(tqdm(samples)):
        try:
            # Load Latent
            latent = torch.load(file_path, map_location=device)
            if latent.ndim == 3: latent = latent.unsqueeze(0)
            
            # --- Path A: Proxy Prediction (Pure Latent) ---
            with torch.no_grad():
                pred_raw = proxy(latent) # [1, 1, 32, 32] or [1, 1, 64, 64]
                
            # --- Path B: Ground Truth (Decode -> Canny) ---
            with torch.no_grad():
                pixels = vae.decode(latent / 0.18215).sample
                pixels = (pixels / 2 + 0.5).clamp(0, 1) * 255
            
            # 获取 GT 尺寸
            B, C, H, W = pixels.shape
            
            # 生成 Canny GT
            img_np = pixels[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            # 宽阈值以捕捉主要轮廓
            canny = cv2.Canny(img_gray, 50, 150)
            gt_mask = (canny > 0)
            
            # 对齐预测图到 GT 尺寸
            pred_up = F.interpolate(pred_raw, size=(H, W), mode='bilinear')[0, 0].cpu().numpy()
            
            # 二值化 (Proxy 输出是 0-1 概率，0.5 是自然阈值)
            pred_mask = (pred_up > 0.5)
            
            # 计算指标
            iou, prec, rec, f1 = compute_metrics(pred_mask, gt_mask)
            
            metrics['iou'].append(iou)
            metrics['prec'].append(prec)
            metrics['rec'].append(rec)
            metrics['f1'].append(f1)
            
            if iou > best_iou:
                best_iou = iou
                save_comparison(i, pred_up, gt_mask, iou, "Best Case", out_dir)
            if iou < worst_iou:
                worst_iou = iou
                save_comparison(i, pred_up, gt_mask, iou, "Worst Case", out_dir)
                
        except Exception as e:
            print(f"Error: {e}")
            continue

    # 4. 最终报告
    print(f"\n📊 Proxy Benchmark Report (N={len(metrics['iou'])})")
    print(f"{'-'*40}")
    print(f"IoU       : {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}")
    print(f"Precision : {np.mean(metrics['prec']):.4f} ± {np.std(metrics['prec']):.4f}")
    print(f"Recall    : {np.mean(metrics['rec']):.4f} ± {np.std(metrics['rec']):.4f}")
    print(f"{'-'*40}")
    
    avg_iou = np.mean(metrics['iou'])
    if avg_iou > 0.4:
        print("✅ SUCCESS: Proxy effectively captures structure.")
        print("   Action: Safe to integrate into main training.")
    elif avg_iou > 0.25:
        print("⚠️ MODERATE: Better than static (0.08), but not perfect.")
        print("   Action: Train longer or increase model capacity.")
    else:
        print("❌ FAILURE: Proxy failed to learn.")
        print("   Action: Check training data or VAE scaling.")

if __name__ == "__main__":
    main()