import torch
import os
import json
import numpy as np
import gc
from tqdm import tqdm
from diffusers import AutoencoderKL
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ================= 🚀 加速配置区 =================
# 在 FP32 + Tiling 模式下：
# 12G 显存试 1-2
# 16G 显存试 2-4
# 24G 显存试 4-8
BATCH_SIZE = 1 

NUM_WORKERS = 32
VAE_ID = "stabilityai/sdxl-vae"
IMG_SIZE = 1024
SCALING_FACTOR = 0.13025
SAVE_ROOT = "./wikiart_latents"
PROGRESS_FILE = os.path.join(SAVE_ROOT, "progress.json")
DATASET_ID = "huggan/wikiart"
DEVICE = "cuda"
# 【新增】支持从本地缓存加载模型（若存在）
CACHE_DIR = os.getenv("HF_CACHE_DIR") or os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
# ===============================================

class WikiArtDataset(Dataset):
    def __init__(self, hf_dataset, img_size):
        self.data = hf_dataset
        self.img_size = img_size
        if 'style' in hf_dataset.features:
            self.int2str = hf_dataset.features['style'].int2str
        else:
            self.int2str = lambda x: str(x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((self.img_size, self.img_size), resample=Image.LANCZOS)
        img_arr = np.array(image).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
        style_idx = item['style']
        style_name = self.int2str(style_idx)
        safe_style_name = style_name.replace(" ", "_").replace("/", "_")
        filename = f"img_{idx:06d}.pt"
        return {
            "pixel_values": img_tensor,
            "style_dir": safe_style_name,
            "filename": filename,
            "idx": idx
        }

def load_progress():
    """加载进度文件，返回已处理的 (style_dir, filename) 集合"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
            processed = set()
            for item in data.get('processed', []):
                processed.add((item['style_dir'], item['filename']))
            return processed, data.get('last_idx', -1)
    return set(), -1

def save_progress(processed_set, last_idx):
    """保存进度到文件"""
    data = {
        'processed': [
            {'style_dir': sd, 'filename': fn} 
            for sd, fn in sorted(processed_set)
        ],
        'last_idx': last_idx
    }
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def _load_vae_from_cache_or_online(model_id, cache_dir=None, torch_dtype=torch.float32):
    """Try loading from local cache first (local_files_only=True). On failure, fall back to online."""
    if cache_dir and os.path.exists(cache_dir):
        try:
            print(f"🔁 尝试从缓存加载 {model_id} (cache_dir={cache_dir}) ...")
            return AutoencoderKL.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True, torch_dtype=torch_dtype)
        except Exception as e:
            print(f"⚠️ 本地加载失败: {e}，将回退到在线下载。")
    print(f"🌐 从在线下载 {model_id} ...")
    return AutoencoderKL.from_pretrained(model_id, torch_dtype=torch_dtype)

def run_fast_encoding():
    # 1. 初始化 VAE (严格保持 FP32)
    print(f"🚀 加载 VAE: {VAE_ID} (FP32)...")
    vae = _load_vae_from_cache_or_online(VAE_ID, cache_dir=CACHE_DIR, torch_dtype=torch.float32).to(DEVICE)
    
    # 【关键修改点 1】开启 Tiling (切块)
    # 这一步不动数据精度，而是将大图切成小块分别进显卡计算，最后拼合。
    # 它是以"计算时间"换"显存空间"的唯一 FP32 救命稻草。
    vae.enable_tiling()
    
    # 【关键修改点 2】开启 xFormers (如果环境支持)
    # 这会优化 Attention 的显存占用，且在 FP32 下精度无损。
    # 如果报错，请注释掉这一行。
    try:
        vae.enable_xformers_memory_efficient_attention()
        print("🧠 xFormers memory-efficient attention 已启用")
    except Exception as exc:
        print(f"⚠️ 无法启用 xFormers: {exc}")

    vae.eval()
    vae.requires_grad_(False)

    # 【新增】加载进度
    processed_set, last_idx = load_progress()
    print(f"📊 已处理 {len(processed_set)} 张图片，上次中断在 idx={last_idx}")

    print(f"📥 加载数据集...")
    hf_dataset = load_dataset(DATASET_ID, split="train")
    torch_dataset = WikiArtDataset(hf_dataset, IMG_SIZE)
    
    loader = DataLoader(
        torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,     
        drop_last=False
    )

    print(f"⚡ 开始 FP32 编码 (Tiling 开启，断点续跑已启用)")
    
    os.makedirs(SAVE_ROOT, exist_ok=True)
    
    current_last_idx = last_idx
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding"):
            # 保持 FP32
            imgs = batch["pixel_values"].to(DEVICE, dtype=torch.float32)
            
            # 编码
            latents = vae.encode(imgs).latent_dist.mode()
            latents = latents * SCALING_FACTOR
            
            # 搬回 CPU
            latents = latents.cpu()
            
            # 【关键修改点 3】手动清理显存引用
            # 在 Python 中，虽然变量出了作用域会销毁，但显存释放有时有滞后。
            # 手动删除 GPU 上的变量引用，有助于缓解显存碎片化。
            del imgs

            for i in range(latents.shape[0]):
                style_dir = batch["style_dir"][i]
                fname = batch["filename"][i]
                current_idx = batch["idx"][i].item()
                
                # 【新增】检查是否已处理过
                if (style_dir, fname) in processed_set:
                    continue
                
                latent_tensor = latents[i]
                
                full_dir = os.path.join(SAVE_ROOT, style_dir)
                os.makedirs(full_dir, exist_ok=True)
                
                save_path = os.path.join(full_dir, fname)
                if not os.path.exists(save_path):
                    torch.save(latent_tensor, save_path)
                
                # 【新增】更新进度
                processed_set.add((style_dir, fname))
                current_last_idx = current_idx
                
                # 每处理 100 张图就保存一次进度（防止全部丢失）
                if len(processed_set) % 100 == 0:
                    save_progress(processed_set, current_last_idx)

    # 【新增】最后保存一次进度
    save_progress(processed_set, current_last_idx)
    print(f"✅ 全部完成！共处理 {len(processed_set)} 张图片")
    print(f"📝 进度已保存至 {PROGRESS_FILE}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    run_fast_encoding()