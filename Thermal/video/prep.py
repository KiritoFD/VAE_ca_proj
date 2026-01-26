import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from diffusers import AutoencoderKL
from tqdm import tqdm

def process_video(src, dst, device='cuda'):
    # Ampere 优化：Tensor Core 加速
    torch.backends.cuda.matmul.allow_tf32 = True
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(device)

    # 获取图像列表
    src_path = Path(src)
    img_files = sorted([f for f in src_path.iterdir() if f.suffix.lower() in {'.jpg', '.png', '.jpeg'}])
    if not img_files: raise FileNotFoundError(f"Empty: {src}")

    latents_list = []
    batch_size = 16 # 针对 8G 显存的稳健选择

    print(f"Streaming {len(img_files)} frames from {src}...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            batch_files = img_files[i : i + batch_size]
            batch_data = []
            
            for f in batch_files:
                img = cv2.imread(str(f))
                # 动态计算 VAE 友好尺寸 (8的倍数)
                H, W = img.shape[:2]
                scale = 512.0 / min(H, W)
                new_H, new_W = (int(H * scale) // 8) * 8, (int(W * scale) // 8) * 8
                
                img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_data.append(img)
            
            # 关键优化：仅在 GPU 上进行数据类型转换和归一化
            # 这样 CPU 内存只保留 uint8，GPU 只保留一小块 float16
            t = torch.from_numpy(np.stack(batch_data)).permute(0, 3, 1, 2).to(device)
            t = (t.to(torch.float16) / 127.5) - 1.0
            
            # VAE 编码
            latents = vae.encode(t).latent_dist.mean * 0.18215
            latents_list.append(latents.cpu())
            
            # 显式清理显存防止碎片化
            del t
            # torch.cuda.empty_cache() # 只有在真的 OOM 时才取消注释，平时会拖慢速度

    final_latents = torch.cat(latents_list, dim=0).numpy()
    np.savez(dst, latents=final_latents, fps=24.0, shape=final_latents.shape[-2:])
    print(f"Success. Latents: {final_latents.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    args = parser.parse_args()
    process_video(args.src, args.dst)