import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from inference import load_vae

# 配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def verify_spectrum():
    print("🧪 Running 5-Minute Validation: Latent Spectrum Analysis...")
    vae = load_vae(device)
    
    # 1. 生成自然图像的 Latent (模拟训练数据)
    # 我们用随机噪声模拟 Latent，因为 VAE 的 KL 正则化导致它统计上接近高斯
    # 但为了更真实，我们最好有一张真实图片。这里我们合成一个简单的“方块”结构
    # 来看看 Latent 能否保留这个低频结构。
    
    # 创建一个 512x512 的图像，中间有个白块 (明显的低频结构)
    img = torch.zeros(1, 3, 512, 512, device=device)
    img[:, :, 128:384, 128:384] = 1.0 
    img = img * 2.0 - 1.0 # [-1, 1]
    
    # Encode
    with torch.no_grad():
        latent = vae.encode(img.half()).latent_dist.sample() * 0.18215
    
    # 2. Pixel Space FFT
    img_gray = img.mean(dim=1, keepdim=True).float()
    fft_pixel = torch.fft.rfft2(img_gray, norm='ortho')
    # 计算径向平均能量 (Radial Profile)
    # 简单的做法：取 X 轴和 Y 轴的平均
    spec_pixel = torch.log(torch.abs(fft_pixel).mean(dim=0) + 1e-8)
    
    # 3. Latent Space FFT
    # 将 4 通道平铺或取平均
    latent_gray = latent.mean(dim=1, keepdim=True).float()
    fft_latent = torch.fft.rfft2(latent_gray, norm='ortho')
    spec_latent = torch.log(torch.abs(fft_latent).mean(dim=0) + 1e-8)
    
    # 4. 可视化对比
    # 4. 可视化对比 (修正维度错误)
    plt.figure(figsize=(12, 5))
    
    # 强制将 (1, 512, 257) 压缩为 (512, 257)
    spec_pixel_2d = spec_pixel.squeeze().cpu().numpy()
    spec_latent_2d = spec_latent.squeeze().cpu().numpy()

    plt.subplot(1, 2, 1)
    plt.title("Pixel Space Log-Spectrum")
    plt.imshow(spec_pixel_2d, cmap='inferno') # 现在 shape 正确了
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Latent Space Log-Spectrum")
    plt.imshow(spec_latent_2d, cmap='inferno')
    plt.colorbar()
    
    plt.savefig("validation_spectrum.png")
    print("✅ Spectrum saved to validation_spectrum.png")
    print("   Observation: If Latent Spectrum looks 'flat' or 'noisy' compared to Pixel,")
    print("                then FFT-based filtering is invalid.")

if __name__ == "__main__":
    verify_spectrum()