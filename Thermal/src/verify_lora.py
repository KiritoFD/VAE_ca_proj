import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from diffusers import AutoencoderKL

class StyleHyperLinear(nn.Module):
    def __init__(self, in_features, out_features, style_dim=256, rank=8, alpha=1.0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.rank, self.alpha = rank, alpha
        
        # Base Linear 必须初始化为 Identity (如果是方阵) 或保持固定
        self.base_linear = nn.Linear(in_features, out_features)
        
        self.num_params = (in_features + out_features) * rank
        self.lora_gen = nn.Sequential(
            nn.Linear(style_dim, style_dim // 2),
            nn.SiLU(),
            nn.Linear(style_dim // 2, self.num_params)
        )
        # 严格零初始化
        nn.init.zeros_(self.lora_gen[-1].weight)
        nn.init.zeros_(self.lora_gen[-1].bias)

    def forward(self, x, style_emb):
        base_out = self.base_linear(x)
        
        # 处理非连续内存
        lora_params = self.lora_gen(style_emb)
        B_size = x.shape[0]
        split = self.in_features * self.rank
        
        A = lora_params[:, :split].reshape(B_size, self.rank, self.in_features).contiguous()
        B = lora_params[:, split:].reshape(B_size, self.out_features, self.rank).contiguous()
        
        # x @ A.T @ B.T
        delta = torch.bmm(torch.bmm(x, A.transpose(1, 2)), B.transpose(1, 2))
        return base_out + self.alpha * delta

class IdentityHyperBench(nn.Module):
    """
    使用残差结构确保结构不丢失。
    y = x + StyleLoRA(x)
    """
    def __init__(self, dim=4, style_dim=256):
        super().__init__()
        # 直接在 Latent 通道上做 LoRA，rank=4
        self.lora_q = StyleHyperLinear(dim, dim, style_dim, rank=4)
        self.lora_v = StyleHyperLinear(dim, dim, style_dim, rank=4)

    def forward(self, x, style_emb):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, -1, C).contiguous()
        
        # 模拟残差注意力
        q = self.lora_q(x_flat, style_emb)
        v = self.lora_v(x_flat, style_emb)
        attn = F.softmax(q @ q.transpose(1, 2) / (C**0.5), dim=-1)
        out = attn @ v
        
        # 残差连接是保结构的核心
        out = x_flat + out 
        return out.transpose(1, 2).reshape(B, C, H, W).contiguous()

def main():
    device = 'cuda'
    # 加载 VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    
    # 获取测试样本
    data_root = Path(cfg['data']['data_root'])
    sample_pt = next(data_root.rglob("*.pt"))
    latent = torch.load(sample_pt, map_location=device)
    if latent.ndim == 3: latent = latent.unsqueeze(0)

    # 实例化模型
    bench = IdentityHyperBench().to(device).eval()
    
    # 模拟训练后的权重扰动
    with torch.no_grad():
        for m in bench.modules():
            if isinstance(m, nn.Linear) and m.out_features > 10:
                m.weight.normal_(0, 0.05)

    style_A = torch.randn(1, 256).to(device)
    style_B = torch.randn(1, 256).to(device)

    with torch.no_grad():
        # 这里必须除以 0.18215，否则 VAE 解码出来全是噪点
        def decode(l):
            return vae.decode(l / 0.18215).sample

        out_A = bench(latent, style_A)
        out_B = bench(latent, style_B)
        
        img_A = decode(out_A)
        img_B = decode(out_B)

        # 转为灰度计算 IoU
        def get_edges(img):
            im = (img[0] / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            gray = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            return cv2.Canny(gray, 50, 150)

        edge_A = get_edges(img_A)
        edge_B = get_edges(img_B)

        iou = np.sum((edge_A > 0) & (edge_B > 0)) / (np.sum((edge_A > 0) | (edge_B > 0)) + 1e-6)
        
    print(f"Verified Edge IoU: {iou:.4f}")
    if iou > 0.5:
        print("✅ Correct: Identity path with LoRA perturbation maintains structure.")
    else:
        print("❌ Still failing: Check latent scaling or VAE normalization.")

if __name__ == "__main__":
    main()