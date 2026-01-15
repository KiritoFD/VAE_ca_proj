import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# ================= 配置 =================
DATA_ROOT = r"G:\GitHub\VAE_ca_proj\wikiart_latents"
DEVICE = "cuda"
# =======================================

def decode_latents(vae, latents):
    # 核心：强制转 FP32，防止噪点
    latents = latents.to(DEVICE).float()
    # 猜测缩放因子：SDXL通常是0.13025，SD1.5是0.18215
    # 我们先不除缩放因子，或者尝试标准缩放
    # 这里直接让 VAE decode，观察大概轮廓即可
    with torch.no_grad():
        image = vae.decode(latents / 0.13025).sample # 假设是 SDXL 因子
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    return (image * 255).astype(np.uint8)

def check():
    # 找一个文件
    files = glob.glob(os.path.join(DATA_ROOT, "*", "*.pt"))
    if not files:
        print("❌ 没找到数据文件！")
        return
    test_file = files[0]
    print(f"🧐 正在检查文件: {test_file}")
    
    # 加载潜码 (130KB -> 128x128 FP16)
    latents = torch.load(test_file, map_location=DEVICE)
    print(f"📊 潜码形状: {latents.shape} (验证: 应该也是 [4, 128, 128])")
    
    # 1. 尝试用 SDXL VAE 解码
    print("正在尝试 SDXL VAE 解码...")
    vae_xl = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(DEVICE)
    img_xl = decode_latents(vae_xl, latents.unsqueeze(0))
    del vae_xl
    
    # 2. 尝试用 SD1.5 VAE 解码
    print("正在尝试 SD1.5 VAE 解码...")
    vae_15 = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    # SD1.5 VAE 也能处理 128x128 (对应 1024图)，只是如果不匹配会古怪
    img_15 = decode_latents(vae_15, latents.unsqueeze(0)) 
    del vae_15
    
    # 保存对比图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Decode with SDXL VAE")
    plt.imshow(img_xl)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Decode with SD1.5 VAE")
    plt.imshow(img_15)
    plt.axis("off")
    
    plt.savefig("data_check.png")
    print("✅ 诊断完成！请打开 data_check.png 查看结果。")
    print("👉 如果左边清晰，说明你的数据是 SDXL 的。")
    print("👉 如果右边清晰，说明你的数据是 SD1.5 的。")

if __name__ == "__main__":
    check()