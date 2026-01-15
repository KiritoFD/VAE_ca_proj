import torch
import os
import shutil
import numpy as np
from tqdm import tqdm
from diffusers import AutoencoderKL
from datasets import load_dataset
from PIL import Image
import json
from collections import defaultdict
from glob import glob

# ================= SD1.5 标准配置 =================
VAE_ID = "stabilityai/sd-vae-ft-mse"
IMG_SIZE = 512             # SD1.5 标准尺寸
SCALING_FACTOR = 0.18215   # SD1.5 标准缩放
SAVE_ROOT = "./wikiart_latents"
DEVICE = "cuda"
PROGRESS_FILE = os.path.join(SAVE_ROOT, "encode_progress.json")

# ❌ 移除了 TARGET_STYLES (跑所有风格)
# ❌ 移除了 MAX_IMAGES (跑所有图片)
# =================================================

def scan_saved_counts():
    counts = defaultdict(int)
    if not os.path.isdir(SAVE_ROOT):
        return counts
    for entry in os.listdir(SAVE_ROOT):
        path = os.path.join(SAVE_ROOT, entry)
        if os.path.isdir(path):
            counts[entry] = sum(1 for name in os.listdir(path) if name.endswith(".pt"))
    return counts

def scan_saved_counts_and_max_idx():
    counts = defaultdict(int)
    max_idx = -1
    if not os.path.isdir(SAVE_ROOT):
        return counts, max_idx
    for entry in os.listdir(SAVE_ROOT):
        path = os.path.join(SAVE_ROOT, entry)
        if os.path.isdir(path):
            pt_files = [f for f in os.listdir(path) if f.endswith(".pt")]
            counts[entry] = len(pt_files)
            # 提取所有编号
            indices = []
            for fname in pt_files:
                try:
                    idx = int(os.path.splitext(fname)[0])
                    indices.append(idx)
                except Exception:
                    continue
            if indices:
                max_idx = max(max_idx, max(indices))
    return counts, max_idx

def load_progress():
    # 不再从文件读取进度，只用磁盘扫描
    return scan_saved_counts_and_max_idx()

def save_progress(last_idx, counters):
    # 仅用于显示，不影响断点续传逻辑
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as fp:
        json.dump({"last_idx": last_idx, "counters": counters}, fp, indent=2)

def scan_existing_indices():
    """Return a set of all image indices already encoded (from all style dirs)."""
    existing = set()
    if not os.path.isdir(SAVE_ROOT):
        return existing
    for entry in os.listdir(SAVE_ROOT):
        path = os.path.join(SAVE_ROOT, entry)
        if os.path.isdir(path):
            for fname in os.listdir(path):
                if fname.endswith(".pt"):
                    try:
                        idx = int(os.path.splitext(fname)[0])
                        existing.add(idx)
                    except Exception:
                        continue
    return existing

def run_encode_all():
    print(f"🚀 初始化 VAE: {VAE_ID} (FP32)...")
    # 强制 FP32，确保精度
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(DEVICE).float()
    vae.eval()

    print(f"📥 加载 WikiArt 全量数据集...")
    # 不使用 streaming，方便通过索引跳过
    dataset = load_dataset("huggan/wikiart", split="train")
    
    # 处理风格标签映射
    if 'style' in dataset.features:
        int2str = dataset.features['style'].int2str
    else:
        int2str = lambda x: str(x)

    # ⚠️ 修改：建议不要每次都强制删除目录，防止误删跑了几个小时的成果
    # if os.path.exists(SAVE_ROOT): shutil.rmtree(SAVE_ROOT) 
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # 断点续传：找到已处理的最大索引
    existing_indices = scan_existing_indices()
    start_idx = max(existing_indices) + 1 if existing_indices else 0
    print(f"⏱ 已存在 {len(existing_indices)} 张图片，从索引 {start_idx} 开始处理。")
    print(f"⚡ 开始全量编码 (512x512), 共 {len(dataset)} 张图片...")

    counters = defaultdict(int)
    processed_idx = -1
    with torch.no_grad():
        # 直接从 start_idx 开始遍历
        for i in tqdm(range(start_idx, len(dataset)), desc="Encoding"):
            try:
                item = dataset[i]
                # 1. 获取风格名称
                style_idx = item['style']
                raw_style = int2str(style_idx)
                style = raw_style.replace(" ", "_").replace("/", "_")
                
                if style not in counters:
                    counters[style] = 0

                # 2. 保存路径
                save_dir = os.path.join(SAVE_ROOT, style)
                save_path = os.path.join(save_dir, f"{i}.pt")
                
                # 3. 图片预处理
                img = item['image'].convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                arr = np.array(img).astype(np.float32) / 127.5 - 1.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                
                # 4. 编码 + 缩放
                latents = vae.encode(tensor).latent_dist.mode() * SCALING_FACTOR
                
                # 5. 保存
                os.makedirs(save_dir, exist_ok=True)
                torch.save(latents.cpu(), save_path)
                
                counters[style] += 1
                processed_idx = i
                if counters[style] % 500 == 0:
                    save_progress(processed_idx, counters)
            except Exception as e:
                print(f"\n⚠️ 图片 {i} 处理失败: {e}")
                continue
                    
    save_progress(processed_idx, counters)
    print("\n✅ 全量编码完成！")
    print("📊 最新风格统计:")
    for s, c in sorted(counters.items()):
        print(f"  - {s}: {c} 张")

def encode_images_in_dir(img_dir, vae_id=VAE_ID, device=DEVICE, scaling_factor=SCALING_FACTOR, img_size=IMG_SIZE):
    """
    编码指定目录下所有图片，保存为 {img_dir}/latents/{相对子目录结构}/{原文件名}.pt
    """
    vae = AutoencoderKL.from_pretrained(vae_id).to(device).float()
    vae.eval()
    img_dir = os.path.abspath(img_dir)
    save_root = os.path.join(img_dir, "latents")
    os.makedirs(save_root, exist_ok=True)

    # 支持常见图片格式，递归查找所有子目录
    img_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        img_files.extend(glob(os.path.join(img_dir, "**", ext), recursive=True))
    img_files = sorted(img_files)
    if not img_files:
        print(f"❌ No images found in {img_dir}")
        return

    print(f"🚀 Encoding {len(img_files)} images from {img_dir} (with subdirs) ...")
    with torch.no_grad():
        for img_path in tqdm(img_files, desc="Encoding images"):
            try:
                img = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.LANCZOS)
                arr = np.array(img).astype(np.float32) / 127.5 - 1.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                latents = vae.encode(tensor).latent_dist.mode() * scaling_factor
                # 保留相对子目录结构
                rel_path = os.path.relpath(img_path, img_dir)
                rel_dir = os.path.dirname(rel_path)
                base = os.path.splitext(os.path.basename(img_path))[0]
                save_dir = os.path.join(save_root, rel_dir)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{base}.pt")
                torch.save(latents.cpu(), save_path)
            except Exception as e:
                print(f"⚠️ Failed to encode {img_path}: {e}")
    print(f"✅ All images encoded and saved to {save_root}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default=None, help="要编码的图片目录（可选）")
    args = parser.parse_args()

    if args.img_dir:
        encode_images_in_dir(args.img_dir)
    else:
        run_encode_all()