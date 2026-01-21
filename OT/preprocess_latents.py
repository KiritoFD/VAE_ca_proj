"""
数据预处理工具：将图片预编码为 VAE Latent
严禁在训练循环中运行 VAE Encoder，必须预计算所有数据
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def load_vae():
    """加载 VAE 模型"""
    try:
        from diffusers import AutoencoderKL
        print("Loading VAE from stabilityai/sd-vae-ft-mse...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        return vae
    except ImportError:
        print("Error: diffusers not installed. Please run: pip install diffusers")
        return None
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return None


def preprocess_image(image_path, target_size=256):
    """
    预处理单张图片
    
    Args:
        image_path: 图片路径
        target_size: 目标尺寸
    Returns:
        image_tensor: [3, H, W], 范围 [-1, 1]
    """
    try:
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        
        # Resize (保持宽高比)
        w, h = image.size
        if w < h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Center crop
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        image = image.crop((left, top, left + target_size, top + target_size))
        
        # To tensor: [0, 255] -> [0, 1] -> [-1, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        image = image * 2.0 - 1.0
        
        return image
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def encode_to_latent(vae, image_tensor, device):
    """
    将图片编码为 VAE Latent
    
    Args:
        vae: VAE 模型
        image_tensor: [3, H, W], 范围 [-1, 1]
        device: 设备
    Returns:
        latent: [4, H//8, W//8]
    """
    vae = vae.to(device)
    vae.eval()
    
    with torch.no_grad():
        # 添加 batch 维度
        image_batch = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
        
        # 编码
        latent_dist = vae.encode(image_batch).latent_dist
        latent = latent_dist.sample()  # [1, 4, H//8, W//8]
        
        # 缩放因子 (Stable Diffusion 标准)
        latent = latent * vae.config.scaling_factor
        
        # 移除 batch 维度
        latent = latent.squeeze(0).cpu()  # [4, H//8, W//8]
    
    return latent


def process_style_folder(vae, style_dir, output_dir, style_id, device, target_size=512):
    """
    处理一个风格文件夹
    
    Args:
        vae: VAE 模型
        style_dir: 风格文件夹路径
        output_dir: 输出文件夹路径
        style_id: 风格ID
        device: 设备
        target_size: 目标图片尺寸
    """
    style_dir = Path(style_dir)
    output_dir = Path(output_dir) / f"style_{style_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(style_dir.glob(f"*{ext}"))
        image_files.extend(style_dir.glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"⚠ No images found in {style_dir}")
        return
    
    print(f"\nProcessing style {style_id}: {style_dir}")
    print(f"Found {len(image_files)} images")
    
    # 处理每张图片
    success_count = 0
    for img_path in tqdm(image_files, desc=f"Style {style_id}"):
        try:
            # 预处理
            image_tensor = preprocess_image(img_path, target_size)
            if image_tensor is None:
                continue
            
            # 编码为 latent
            latent = encode_to_latent(vae, image_tensor, device)
            
            # 保存
            output_path = output_dir / f"{img_path.stem}.pt"
            torch.save(latent, output_path)
            
            success_count += 1
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"✓ Style {style_id}: {success_count}/{len(image_files)} images encoded successfully")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess images to VAE latents')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--target_size', type=int, default=256, help='Target image size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    data_cfg = config['data']
    raw_data_root = Path(data_cfg['raw_data_root'])
    output_root = Path(data_cfg['data_root'])
    num_styles = data_cfg['num_classes']
    
    print("="*80)
    print("VAE Latent Preprocessing")
    print("="*80)
    print(f"Raw data root: {raw_data_root}")
    print(f"Output root: {output_root}")
    print(f"Number of styles: {num_styles}")
    print(f"Target size: {args.target_size}")
    print("="*80)
    
    # 设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载 VAE
    vae = load_vae()
    if vae is None:
        print("Failed to load VAE. Exiting.")
        return
    
    print("✓ VAE loaded successfully")
    
    # 创建输出目录
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 处理每个风格文件夹
    # 假设数据结构为:
    # raw_data_root/
    #   style_0/
    #     image1.jpg
    #     image2.jpg
    #   style_1/
    #     image3.jpg
    #     image4.jpg
    
    for style_id in range(num_styles):
        style_dir = raw_data_root / f"style_{style_id}"
        
        if not style_dir.exists():
            # 尝试其他可能的命名
            alternatives = [
                raw_data_root / f"class_{style_id}",
                raw_data_root / f"{style_id}",
                raw_data_root / f"train_{style_id}"
            ]
            
            for alt_dir in alternatives:
                if alt_dir.exists():
                    style_dir = alt_dir
                    break
            else:
                print(f"⚠ Warning: Style directory not found for style_id={style_id}")
                print(f"   Tried: {style_dir}")
                continue
        
        process_style_folder(
            vae=vae,
            style_dir=style_dir,
            output_dir=output_root,
            style_id=style_id,
            device=device,
            target_size=args.target_size
        )
    
    print("\n" + "="*80)
    print("✓ Preprocessing completed!")
    print(f"Latents saved to: {output_root}")
    print("="*80)
    
    # 显示统计信息
    print("\nDataset statistics:")
    for style_id in range(num_styles):
        style_dir = output_root / f"style_{style_id}"
        if style_dir.exists():
            num_files = len(list(style_dir.glob("*.pt")))
            print(f"  Style {style_id}: {num_files} samples")


if __name__ == "__main__":
    main()
