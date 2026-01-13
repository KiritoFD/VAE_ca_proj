import torch
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL
from datasets import load_dataset
import argparse
import gc
import time
import psutil
import json

# ================= 配置区 =================
DEVICE = "cuda" 
VAE_MODEL = "stabilityai/sd-vae-ft-mse"  # SD 1.5/2.1 兼容 VAE
IMG_SIZE = 512
SAVE_ROOT = "./wikiart_latents"
DATASET_CACHE = "./wikiart_dataset"  # 数据集缓存路径
MIN_STYLE_IMAGES = 50  # 过滤小样本风格
MAX_IMAGES_PER_STYLE = 1000  # 防止数据倾斜
MAX_BATCH_SIZE = 4  # 8G显存安全上限
MEMORY_CLEANUP_INTERVAL = 10  # 每N个batch清理一次内存
# ==========================================

def download_and_cache_dataset():
    """第一步：下载数据集到本地硬盘"""
    os.makedirs(DATASET_CACHE, exist_ok=True)
    cache_marker = os.path.join(DATASET_CACHE, "downloaded.marker")
    
    if os.path.exists(cache_marker):
        print(f"✅ Dataset already cached at {DATASET_CACHE}")
        return
    
    print(f"📥 Downloading WikiArt dataset to {DATASET_CACHE}...")
    print("⏳ This is a one-time operation, may take 10-30 minutes...")
    
    start_time = time.time()
    # 数据集会自动缓存到 HF_HOME 或自定义位置
    os.environ['HF_DATASETS_CACHE'] = DATASET_CACHE
    
    dataset = load_dataset("huggan/wikiart", split="train", cache_dir=DATASET_CACHE)
    
    download_time = time.time() - start_time
    print(f"✅ Downloaded {len(dataset)} samples in {download_time:.2f}s")
    
    # 创建标记文件
    with open(cache_marker, 'w') as f:
        f.write(f"Downloaded at {time.ctime()}\nTotal samples: {len(dataset)}\n")

def analyze_and_filter_dataset():
    """第二步：分析和筛选数据集，保存元数据"""
    print("\n📊 Analyzing and filtering dataset...")
    os.environ['HF_DATASETS_CACHE'] = DATASET_CACHE
    
    metadata_path = os.path.join(DATASET_CACHE, "metadata.json")
    
    # 获取数据集
    dataset = load_dataset("huggan/wikiart", split="train", cache_dir=DATASET_CACHE)
    
    # 建立风格名称映射函数
    style_feature = dataset.features["style"]
    def get_style_name(s_code):
        if s_code == -1 or s_code is None: return "na"
        if isinstance(s_code, int) and hasattr(style_feature, "int2str"):
            return str(style_feature.int2str(s_code))
        return str(s_code)

    # 如果已有元数据，直接读取并返回
    if os.path.exists(metadata_path):
        print(f"✅ Loading existing metadata from {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata

    process = psutil.Process()
    print(f"Total samples: {len(dataset)}")
    
    # 第一次遍历：统计风格分布
    print("\n📈 Counting styles...")
    style_counts = {}
    start_time = time.time()
    
    for i in tqdm(range(len(dataset)), desc="Analyzing"):
        item = dataset[i]
        style = get_style_name(item.get("style", -1))
        
        if style != "na":
            style_counts[style] = style_counts.get(style, 0) + 1
        
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"  [{i+1:6d}] styles: {len(style_counts):4d} | "
                  f"Speed: {rate:.0f} samples/s | Memory: {mem_mb:.0f}MB")
    
    count_time = time.time() - start_time
    print(f"✅ Analysis completed in {count_time:.2f}s")
    print(f"   Found {len(style_counts)} unique styles")
    
    # 过滤和排序
    valid_styles = sorted([str(s) for s, count in style_counts.items() if count >= MIN_STYLE_IMAGES])
    style_to_id = {s: i for i, s in enumerate(valid_styles)}
    
    print(f"\n📊 Filtering criteria:")
    print(f"   Total styles: {len(style_counts)}")
    print(f"   Valid styles (>= {MIN_STYLE_IMAGES}): {len(valid_styles)}")
    print(f"   Filtered out: {len(style_counts) - len(valid_styles)}")
    print(f"   First 5 valid styles: {valid_styles[:5]}")
    
    # 第二次遍历：确认符合条件的样本
    print("\n🔍 Building valid sample list...")
    valid_samples = []
    style_image_count = {s: 0 for s in valid_styles}
    
    start_time = time.time()
    for idx in tqdm(range(len(dataset)), desc="Filtering"):
        item = dataset[idx]
        style = get_style_name(item.get("style", -1))
        
        if style in valid_styles and style_image_count[style] < MAX_IMAGES_PER_STYLE:
            valid_samples.append({
                "idx": idx,
                "style": style,
                "style_id": style_to_id[style]
            })
            style_image_count[style] += 1
        
        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            print(f"  [{idx+1:6d}] valid: {len(valid_samples):6d} | Speed: {rate:.0f} samples/s")
    
    filter_time = time.time() - start_time
    print(f"✅ Filtering completed in {filter_time:.2f}s")
    print(f"   Selected {len(valid_samples)} samples")
    print(f"   Samples per style: min={min(style_image_count.values())}, "
          f"max={max(style_image_count.values())}, "
          f"avg={np.mean(list(style_image_count.values())):.1f}")
    
    # 保存元数据到硬盘
    metadata = {
        "valid_styles": valid_styles,
        "style_to_id": style_to_id,
        "valid_samples": valid_samples,
        "total_samples": len(valid_samples),
        "num_styles": len(valid_styles),
        "created_at": time.ctime()
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    return metadata

def setup_vae():
    """加载并配置 VAE"""
    print(f"Loading VAE: {VAE_MODEL} on {DEVICE}...")
    
    # 清空缓存
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    vae.eval()
    
    # 禁用梯度计算节省内存
    for param in vae.parameters():
        param.requires_grad = False
    
    return vae

def get_optimal_batch_size(vae, img_size=512):
    """自动检测最大 batch size（保守策略）"""
    if DEVICE == "cpu":
        return 2
    
    torch.cuda.empty_cache()
    gc.collect()
    vae.eval()
    
    # 8G显存保守策略：从小尺寸开始测试
    for bs in [4, 3, 2, 1]:
        try:
            dummy_input = torch.randn(bs, 3, img_size, img_size, device=DEVICE, dtype=torch.float16)
            with torch.no_grad():
                latent = vae.encode(dummy_input).latent_dist.sample()
                # 测试decode确保双向安全
                _ = vae.decode(latent).sample
            
            # 清理测试数据
            del dummy_input, latent
            torch.cuda.empty_cache()
            gc.collect()
            
            print(f"✅ Optimal batch size: {bs}")
            return min(bs, MAX_BATCH_SIZE)  # 不超过安全上限
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e
    return 1

def print_memory_stats():
    """打印显存使用情况"""
    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU Memory: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")

def run_encoding():
    """第三步：使用预处理的元数据，从硬盘流式读取并编码"""
    print("\n🚀 Starting VAE encoding...")
    
    # 加载元数据
    metadata_path = os.path.join(DATASET_CACHE, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found at {metadata_path}")
        print("   Please run: python encode.py --analyze")
        return
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    valid_styles = [str(s) for s in metadata["valid_styles"]]
    valid_samples = metadata["valid_samples"]
    
    # 提前加载数据集以修复整数索引问题
    os.environ['HF_DATASETS_CACHE'] = DATASET_CACHE
    dataset = load_dataset("huggan/wikiart", split="train", cache_dir=DATASET_CACHE)
    
    # --- 修复逻辑：如果元数据存的是整数，将其转回字符串名称 ---
    # ---------------------------------------------------

    print(f"Loaded metadata: {len(valid_samples)} samples across {len(valid_styles)} styles")
    print(f"First style: {valid_styles[0]} (type: {type(valid_styles[0])})")
    
    # 创建保存目录
    os.makedirs(SAVE_ROOT, exist_ok=True)
    for style_name in valid_styles:
        style_dir = os.path.join(SAVE_ROOT, style_name)
        os.makedirs(style_dir, exist_ok=True)
    
    print(f"✅ Created directories for {len(valid_styles)} styles")
    
    # 加载 VAE
    vae = setup_vae()
    VAE_DTYPE = vae.dtype # 获取模型实际精度 (float16 或 float32)
    BATCH_SIZE = get_optimal_batch_size(vae, IMG_SIZE)
    print(f"Using batch size: {BATCH_SIZE}")
    
    # 编码循环
    total_processed = 0
    skipped = 0
    batch_count = 0
    
    print(f"\n⏳ Encoding {len(valid_samples)} samples...")
    with torch.no_grad():
        pbar = tqdm(range(0, len(valid_samples), BATCH_SIZE), desc="Encoding")
        for batch_start in pbar:
            batch_end = min(batch_start + BATCH_SIZE, len(valid_samples))
            batch_samples = valid_samples[batch_start:batch_end]
            
            # 从硬盘读取这一批样本的图片
            imgs = []
            labels = []
            indices = []
            
            for sample in batch_samples:
                try:
                    item = dataset[sample["idx"]]
                    img = item['image'].convert("RGB").resize(
                        (IMG_SIZE, IMG_SIZE),
                        Image.LANCZOS
                    )
                    img_np = np.array(img, dtype=np.float32)
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
                    img_tensor = (img_tensor / 127.5) - 1.0
                    
                    imgs.append(img_tensor)
                    labels.append(sample["style_id"])
                    indices.append(sample["idx"])
                    
                    del img, img_np
                except Exception as e:
                    print(f"⚠️ Skipping sample {sample['idx']}: {e}")
                    skipped += 1
            
            if not imgs:
                continue
            
            # 批量转换并移动到设备，强制转换 dtype 以匹配 VAE 偏置类型
            imgs_tensor = torch.stack(imgs).to(DEVICE, dtype=VAE_DTYPE, non_blocking=True)
            latents = vae.encode(imgs_tensor).latent_dist.sample()
            latents_cpu = latents.cpu()
            
            # 保存
            for i, (latent, label, idx) in enumerate(zip(latents_cpu, labels, indices)):
                style_name = valid_styles[label]
                save_path = os.path.join(SAVE_ROOT, style_name, f"img_{idx:06d}.pt")
                
                if os.path.exists(save_path):
                    skipped += 1
                    continue
                
                torch.save(latent, save_path)
                total_processed += 1
            
            pbar.set_postfix({'Saved': total_processed, 'Skipped': skipped})
            
            # 内存清理
            del imgs, imgs_tensor, latents, latents_cpu
            batch_count += 1
            
            if batch_count % MEMORY_CLEANUP_INTERVAL == 0:
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
    
    print(f"\n✅ Encoding completed!")
    print(f"   Total saved: {total_processed}")
    print(f"   Skipped: {skipped}")
    print(f"   Output: {SAVE_ROOT}")
    print_memory_stats()

def verify_reconstruction():
    """验证潜码质量（可选）"""
    print("\n🔍 Verifying latent quality with reconstruction...")
    
    # 清理内存
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    vae.eval()
    
    # 找一个样本测试
    for style_dir in os.listdir(SAVE_ROOT):
        style_path = os.path.join(SAVE_ROOT, style_dir)
        if os.path.isdir(style_path):
            for file in os.listdir(style_path):
                if file.endswith('.pt'):
                    latent_path = os.path.join(style_path, file)
                    latent = torch.load(latent_path, map_location='cpu')
                    
                    with torch.no_grad():
                        latent = latent.unsqueeze(0).to(DEVICE)
                        recon = vae.decode(latent).sample.cpu()
                    
                    # 保存重构图
                    recon_img = (recon[0].permute(1,2,0).numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(recon_img).save("reconstruction_test.jpg")
                    print(f"✅ Reconstruction test saved to: reconstruction_test.jpg")
                    
                    # 清理
                    del latent, recon, recon_img
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download dataset to cache")
    parser.add_argument("--analyze", action="store_true", help="Analyze and filter dataset")
    parser.add_argument("--encode", action="store_true", help="Encode with VAE")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--verify", action="store_true", help="Verify reconstruction")
    args = parser.parse_args()
    
    if args.all:
        download_and_cache_dataset()
        metadata = analyze_and_filter_dataset()
        run_encoding()
    else:
        if args.download:
            download_and_cache_dataset()
        if args.analyze:
            analyze_and_filter_dataset()
        if args.encode:
            run_encoding()
        if args.verify:
            verify_reconstruction()
        if not any([args.download, args.analyze, args.encode, args.verify]):
            print("Usage: python encode.py [--download] [--analyze] [--encode] [--verify] [--all]")