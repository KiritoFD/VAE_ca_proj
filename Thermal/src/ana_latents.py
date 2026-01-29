import torch
import torch.nn.functional as F
import numpy as np
from diffusers import AutoencoderKL
from pathlib import Path
import random
import sys

# ================= 配置 =================
# 自动寻找 Latent 路径
POTENTIAL_PATHS = [
    Path("../../data/latents"),
    Path("../data/latents"),
    Path("./data/latents"),
    Path("/mnt/c/Users/xy/data/latents")
]
VAE_ID = "stabilityai/sd-vae-ft-mse"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE = 500  # 采样数量，越多越准
# =======================================

def load_vae():
    print(f"⚡ Loading VAE ({VAE_ID})...")
    try:
        vae = AutoencoderKL.from_pretrained(VAE_ID).to(DEVICE, dtype=DTYPE)
    except:
        print("⚠️  Download failed, trying local cache...")
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(DEVICE, dtype=DTYPE)
    vae.encoder = None
    return vae

def decode(vae, latents, is_raw=True):
    """解码 Latent -> Pixel"""
    z = latents.to(dtype=DTYPE)
    if is_raw: z = 1.0 / 0.18215 * z
    with torch.no_grad():
        imgs = vae.decode(z).sample
    return (imgs / 2 + 0.5).clamp(0, 1)

def calc_metrics(img_orig, img_mod):
    """计算核心统计指标"""
    # 1. 颜色漂移 (Color Shift): 全图均值的 L1 误差
    # 如果潜空间滤波破坏了语义，解码后的色调会整体偏移
    mu_orig = img_orig.mean(dim=(2, 3)) # [B, 3]
    mu_mod = img_mod.mean(dim=(2, 3))
    color_shift = torch.abs(mu_orig - mu_mod).mean().item() * 255.0 # 转为像素级误差

    # 2. 结构一致性 (Structure L1): 像素级差异
    # 我们期望它有一定差异（因为去掉了高频），但不能太大（否则结构崩了）
    pixel_diff = torch.abs(img_orig - img_mod).mean().item() * 255.0

    # 3. 边缘能量残留 (High-Freq Residual):
    # 滤波后的图，边缘应该变糊。如果方差依然很高，说明没滤干净（或者变成了噪点）
    # 使用简单的拉普拉斯算子计算图像梯度的方差
    def get_gradient_energy(x):
        # Sobel-like simple gradient
        dx = torch.abs(x[..., :, 1:] - x[..., :, :-1])
        dy = torch.abs(x[..., 1:, :] - x[..., :-1, :])
        return dx.mean() + dy.mean()
    
    energy_orig = get_gradient_energy(img_orig).item()
    energy_mod = get_gradient_energy(img_mod).item()
    energy_ratio = energy_mod / (energy_orig + 1e-6) # 越低说明模糊效果越好

    return color_shift, pixel_diff, energy_ratio

def main():
    # 1. 寻找数据
    data_root = None
    for p in POTENTIAL_PATHS:
        if p.exists() and any(p.glob("**/*.pt")):
            data_root = p
            break
    
    if not data_root:
        print(f"❌ 未找到数据目录！请检查路径: {[str(p) for p in POTENTIAL_PATHS]}")
        return

    print(f"📂 Data Root: {data_root}")
    files = list(data_root.rglob("*.pt"))
    target_files = random.sample(files, min(BATCH_SIZE, len(files)))
    print(f"📊 Sampling {len(target_files)} latents for statistical verification...")

    vae = load_vae()
    
    # 统计容器
    stats = {
        'AvgPool': {'color': [], 'pixel': [], 'energy': []},
        'Pyramid': {'color': [], 'pixel': [], 'energy': []},
        'Laplacian': {'color': [], 'pixel': [], 'energy': []}
    }

    for f in target_files:
        raw = torch.load(f, map_location=DEVICE).float()
        if raw.dim() == 3: raw = raw.unsqueeze(0)
        is_raw = raw.std() < 0.5 # 自动检测是否需要缩放

        # 原始解码
        img_orig = decode(vae, raw, is_raw)

        # === 方案 A: AvgPool (频域分离法) ===
        # 对应 FrequencyDecoupledLoss
        lat_avg = F.avg_pool2d(raw, kernel_size=5, stride=1, padding=2)
        img_avg = decode(vae, lat_avg, is_raw)
        c, p, e = calc_metrics(img_orig, img_avg)
        stats['AvgPool']['color'].append(c)
        stats['AvgPool']['pixel'].append(p)
        stats['AvgPool']['energy'].append(e)

        # === 方案 B: Pyramid (尺度金字塔法) ===
        # 对应 PyramidStructuralLoss
        lat_down = F.interpolate(raw, scale_factor=0.25, mode='area')
        lat_up = F.interpolate(lat_down, size=raw.shape[-2:], mode='bilinear')
        img_pyr = decode(vae, lat_up, is_raw)
        c, p, e = calc_metrics(img_orig, img_pyr)
        stats['Pyramid']['color'].append(c)
        stats['Pyramid']['pixel'].append(p)
        stats['Pyramid']['energy'].append(e)

        # === 方案 C: Laplacian (当前的边缘锁) ===
        # 只是为了看它解码出来是不是一团糟
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=raw.dtype, device=DEVICE)
        k = k.view(1, 1, 3, 3).repeat(4, 1, 1, 1)
        lat_lap = F.conv2d(raw, k, padding=1, groups=4)
        img_lap = decode(vae, lat_lap * 2.0 + raw, is_raw) # 叠加回原图看是不是锐化了，或者直接看残差
        # 这里比较特殊，我们直接看它和原图的差异是否巨大
        c, p, e = calc_metrics(img_orig, img_lap) 
        stats['Laplacian']['color'].append(c)
        stats['Laplacian']['pixel'].append(p)
        stats['Laplacian']['energy'].append(e)

    # 打印最终报告
    print("\n" + "="*60)
    print(f"{'Method':<12} | {'Color Shift':<12} | {'Struct Diff':<12} | {'Blur Ratio'}")
    print(f"{'':<12} | {'(Lower Better)':<12} | {'(Stability)':<12} | {'(Freq Filter)':<12}")
    print("-" * 60)

    for method, metrics in stats.items():
        c_mean = np.mean(metrics['color'])
        p_mean = np.mean(metrics['pixel'])
        e_mean = np.mean(metrics['energy'])
        
        # 简单的自动诊断标记
        flag = ""
        if method == 'AvgPool' and c_mean > 5.0: flag = "⚠️ Color Drift!"
        if method == 'Pyramid' and c_mean < 3.0: flag = "✅ Robust"
        
        print(f"{method:<12} | {c_mean:.2f} px      | {p_mean:.2f} px      | {e_mean:.2f} {flag}")
    print("="*60)
    print("【判读指南】")
    print("1. Color Shift > 5.0 说明该滤波破坏了 Latent 的语义通道平衡 -> 不能用该方案。")
    print("2. Blur Ratio 越低，说明去高频效果越好。Pyramid 通常在 0.5-0.7 之间。")
    print("3. 如果 AvgPool 的 Color Shift 显著高于 Pyramid，请坚决选用 PyramidLoss。")

if __name__ == "__main__":
    main()