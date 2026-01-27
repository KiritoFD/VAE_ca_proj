import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
from diffusers import AutoencoderKL

class LatentEdgeFilter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Laplacian kernel for 4 channels
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
        self.register_buffer('kernel', k)

    def forward(self, x):
        # 4070 SIMD 优化：depthwise conv
        edges = F.conv2d(x, self.kernel, groups=4, padding=1)
        return torch.max(torch.abs(edges), dim=1, keepdim=True)[0]

@torch.no_grad()
def run():
    device = 'cuda'
    # 加载真实 VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    
    data_root = Path(cfg['data']['data_root'])
    # 找一张真实的图片 latent
    pt_file = next(data_root.rglob("*.pt"))
    latent = torch.load(pt_file, map_location=device)
    if latent.ndim == 3: latent = latent.unsqueeze(0)

    # 1. 物理层验证：VAE 解码真值图像 (必须除以缩放因子)
    # SD VAE Scaling Factor = 0.18215
    rec_img = vae.decode(latent / 0.18215).sample
    img_np = (rec_img[0].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    canny_gt = cv2.Canny(gray, 100, 200)

    # 2. 算子层验证：直接在 Latent 空间提边缘
    ef = LatentEdgeFilter().to(device)
    l_edge = ef(latent)
    
    # 3. 关键：将 Canny GT 下采样到 32x32 进行“公平对齐”
    # 只有在相同分辨率下，IoU 才能反映语义正确性
    gt_32 = F.interpolate(torch.from_numpy(canny_gt).unsqueeze(0).unsqueeze(0).float().to(device), 
                          size=(32, 32), mode='area')[0, 0]
    gt_bin = (gt_32 > 0).float()
    
    # Latent 边缘二值化 (自适应阈值)
    l_bin = (l_edge[0, 0] > l_edge.mean() + l_edge.std()).float()
    
    # 计算公平 IoU
    inter = (l_bin * gt_bin).sum()
    union = (l_bin + gt_bin).clamp(0, 1).sum()
    iou = inter / (union + 1e-6)

    print(f"Fair Resolution IoU (32x32): {iou.item():.4f}")
    
    # 保存结果看一眼，别在那瞎猜
    cv2.imwrite("debug_canny_gt.png", canny_gt)
    cv2.imwrite("debug_latent_edge.png", (l_bin.cpu().numpy() * 255).astype(np.uint8))

if __name__ == "__main__":
    run()