import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import os
import numpy as np
from collections import defaultdict
import gc
import sys

# 设置标准输出编码
sys.stdout.reconfigure(encoding='utf-8')

from diffusers import AutoencoderKL
from SAFlow import SAFModel

# ==========================================
# 1. VGG Network & Fixed Gram Matrix
# ==========================================
class VGGStyleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4): self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9): self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16): self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23): self.slice4.add_module(str(x), vgg[x])
        for param in self.parameters(): param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

def gram_matrix(input):
    """
    修正后的 Gram 矩阵计算
    输入: (B, C, H, W)
    输出: (B, C, C)
    """
    B, C, H, W = input.size()
    # [B, C, N] where N = H*W
    features = input.view(B, C, H * W)
    # Batch Matrix Multiplication: (B, C, N) @ (B, N, C) -> (B, C, C)
    G = torch.bmm(features, features.transpose(1, 2))
    # 归一化: 除以 C*H*W
    return G.div(C * H * W)

# ==========================================
# 2. Datasets
# ==========================================
class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.files = [f for f in Path(root_dir).rglob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']]
        self.transform = transforms.Compose([
            transforms.Resize(512), transforms.CenterCrop(512),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): 
        try: return self.transform(Image.open(self.files[idx]).convert('RGB'))
        except: return torch.zeros(3, 512, 512)

class MatrixImageDataset(Dataset):
    def __init__(self, root_dir, size=512):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")
        
        self.files = []
        for f in self.root_dir.rglob('*'):
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                sub_dir = str(f.parent.relative_to(self.root_dir))
                fname = f.stem
                self.files.append((f, sub_dir, fname))
        
        self.transform = transforms.Compose([
            transforms.Resize(size), transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path, sub_dir, fname = self.files[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), sub_dir, fname
        except:
            return torch.zeros(3, 512, 512), "error", "error"

# ==========================================
# 3. Evaluator
# ==========================================
class Evaluator:
    def __init__(self, ckpt_path, ref_root=None, config_path="config.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Init] Device: {self.device}")
        
        gc.collect()
        torch.cuda.empty_cache()

        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)

        # 1. Find Refs
        self.style_refs = self._detect_reference_dirs(ref_root)
        if not self.style_refs:
            print("\n[ERROR] Could not find reference datasets (trainA/trainB).")
            print("Please use --ref_dir to specify the folder containing trainA/trainB images.")
            raise ValueError("Reference dataset not found.")

        # 2. Load VGG
        print("[Load] VGG16 (Frozen)...")
        self.vgg = VGGStyleEncoder().to(self.device).eval()
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
        
        # 3. Pre-compute Grams
        self.ref_grams = {}
        with torch.no_grad():
            for s_id, path in self.style_refs.items():
                print(f"[Stats] Computing Gram Matrix for Style {s_id} from: {path}")
                self.ref_grams[s_id] = self.compute_dataset_gram(path)

        # 4. Load Model
        print(f"[Load] SAFModel from {ckpt_path}...")
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(self.device).eval()
        self.model = SAFModel(**self.cfg['model']).to(self.device).eval()
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)
        
        # 5. Cache
        self.cache_root = Path("eval_cache")
        self.cache_root.mkdir(exist_ok=True)
        print(f"[Cache] Directory: {self.cache_root.resolve()}")

    def _detect_reference_dirs(self, manual_root):
        roots_to_try = []
        if manual_root: roots_to_try.append(Path(manual_root))
        
        cfg_root = self.cfg.get("dataset", {}).get("root_dir", "")
        if cfg_root: roots_to_try.append(Path(cfg_root.strip('"').strip("'")))
            
        inf_path = self.cfg.get("inference", {}).get("image_path", "")
        if inf_path: roots_to_try.append(Path(inf_path.strip('"').strip("'")).parent)

        print(f"[Search] Looking for trainA/trainB in: {[str(p) for p in roots_to_try]}")

        refs = {}
        candidates_0 = ['trainA', 'testA', 'class0', 'A', 'photo', 'monet']
        candidates_1 = ['trainB', 'testB', 'class1', 'B', 'art']
        
        for root in roots_to_try:
            if not root.exists(): continue
            if 0 not in refs:
                for c in candidates_0:
                    if (root/c).exists(): refs[0] = root/c; break
            if 1 not in refs:
                for c in candidates_1:
                    if (root/c).exists(): refs[1] = root/c; break
            if 0 in refs and 1 in refs: break
        
        return refs

    @torch.no_grad()
    def compute_dataset_gram(self, data_dir):
        dataset = SimpleDataset(data_dir)
        if len(dataset) == 0: return None
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        
        # Initialize accumulators
        grams_acc = [0, 0, 0, 0]
        count = 0
        
        for imgs in tqdm(loader, desc=f"Scanning {data_dir.name}"):
            imgs = imgs.to(self.device)
            B = imgs.size(0)
            feats = self.vgg(imgs)
            
            for i, f in enumerate(feats):
                # f shape: [B, C, H, W]
                # gram_matrix(f) shape: [B, C, C]
                # .sum(0) shape: [C, C] -> 正确的累加尺寸
                grams_acc[i] += gram_matrix(f).sum(0)
            
            count += B
            del imgs, feats
            
        return [g / count for g in grams_acc]

    @torch.no_grad()
    def evaluate(self, bs=1, save_dir=None):
        content_path_str = self.cfg.get("inference", {}).get("image_path", "").strip('"').strip("'")
        content_path = Path(content_path_str)
        if not content_path.exists():
             print(f"[ERROR] Inference path not found: {content_path}")
             return

        dataset = MatrixImageDataset(content_path)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
        
        num_styles = self.cfg['model'].get('num_styles', 2)
        matrix_results = defaultdict(lambda: defaultdict(list))
        
        print(f"\n[Start] Evaluation on: {content_path}")
        
        for imgs, sub_dirs, fnames in tqdm(loader, desc="Processing"):
            imgs = imgs.to(self.device)
            B = imgs.size(0)
            
            # Check Cache
            all_cached = True
            cached_paths_map = {} 
            for tgt_id in range(num_styles):
                cached_paths_map[tgt_id] = []
                for idx in range(B):
                    subdir = sub_dirs[idx]
                    fname = fnames[idx]
                    save_path = self.cache_root / subdir / f"{fname}_style{tgt_id}.png"
                    if not save_path.exists(): all_cached = False
                    cached_paths_map[tgt_id].append(save_path)

            if all_cached:
                # Load Cache
                for tgt_id in range(num_styles):
                    loaded_imgs = []
                    for p in cached_paths_map[tgt_id]:
                        pil_img = Image.open(p).convert("RGB").resize((512,512))
                        loaded_imgs.append(transforms.ToTensor()(pil_img))
                    
                    batch_tensor = torch.stack(loaded_imgs).to(self.device)
                    batch_vgg = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(batch_tensor)
                    
                    feats = self.vgg(batch_vgg)
                    ref_grams = self.ref_grams.get(tgt_id)
                    if ref_grams:
                        for b_idx in range(B):
                            current_subdir = sub_dirs[b_idx]
                            sample_loss = 0.0
                            for f_idx, f_layer in enumerate(feats):
                                # 这里也要改成 [b_idx:b_idx+1] 切片
                                g = gram_matrix(f_layer[b_idx:b_idx+1]).squeeze(0) # [C, C]
                                ref = ref_grams[f_idx]
                                sample_loss += torch.nn.functional.mse_loss(g, ref).item()
                            matrix_results[current_subdir][tgt_id].append(sample_loss)
                    del batch_tensor, feats

            else:
                # Inference
                x_c = self.vae.encode(imgs).latent_dist.sample() * 0.18215
                
                for tgt_id in range(num_styles):
                    s_ids = torch.full((B,), tgt_id, device=self.device, dtype=torch.long)
                    x_t = x_c.clone()
                    dt = 1.0/10
                    for i in range(10):
                        t_vec = torch.ones(B, device=self.device) * (i * dt)
                        v_pred = self.model(x_t, x_c, t_vec, s_ids)
                        x_t = x_t + v_pred * dt
                    
                    gen_imgs = self.vae.decode(x_t / 0.18215).sample.clamp(-1, 1)
                    
                    # Save Cache
                    for b_idx in range(B):
                        subdir = sub_dirs[b_idx]
                        fname = fnames[b_idx]
                        cache_save_dir = self.cache_root / subdir
                        cache_save_dir.mkdir(parents=True, exist_ok=True)
                        ndarr = gen_imgs[b_idx].cpu().float().numpy()
                        ndarr = (ndarr / 2 + 0.5).clip(0, 1)
                        ndarr = ndarr.transpose(1, 2, 0) * 255
                        Image.fromarray(ndarr.astype(np.uint8)).save(cache_save_dir / f"{fname}_style{tgt_id}.png")

                    # Calc Loss
                    gen_imgs_vgg = (gen_imgs + 1) / 2
                    gen_imgs_vgg = (gen_imgs_vgg - self.vgg_mean) / self.vgg_std
                    
                    feats = self.vgg(gen_imgs_vgg)
                    ref_grams = self.ref_grams.get(tgt_id)
                    if ref_grams:
                        for b_idx in range(B):
                            current_subdir = sub_dirs[b_idx]
                            sample_loss = 0.0
                            for f_idx, f_layer in enumerate(feats):
                                g = gram_matrix(f_layer[b_idx:b_idx+1]).squeeze(0)
                                ref = ref_grams[f_idx]
                                sample_loss += torch.nn.functional.mse_loss(g, ref).item()
                            matrix_results[current_subdir][tgt_id].append(sample_loss)
                    
                    del gen_imgs, gen_imgs_vgg, feats, x_t
                del x_c
            del imgs

        # 计算汇总统计
        summary = {
            "metric": "VGG_Style",
            "description": "VGG Gram style distance (lower is better)",
            "per_subdirectory": {},
            "overall_average": {}
        }
        
        all_subdirs = sorted(matrix_results.keys())
        overall_by_style = defaultdict(list)
        
        for sd in all_subdirs:
            summary["per_subdirectory"][sd] = {}
            for t_id in range(num_styles):
                scores = matrix_results[sd][t_id]
                avg = float(np.mean(scores)) if scores else 0.0
                summary["per_subdirectory"][sd][f"style_{t_id}"] = {
                    "mean": avg,
                    "mean_scaled": avg * 1e5,  # 保存缩放版本便于阅读
                    "std": float(np.std(scores)) if scores else 0.0,
                    "count": len(scores)
                }
                overall_by_style[t_id].extend(scores)
        
        # 全局平均
        for t_id in range(num_styles):
            all_scores = overall_by_style[t_id]
            avg_val = float(np.mean(all_scores)) if all_scores else 0.0
            summary["overall_average"][f"style_{t_id}"] = {
                "mean": avg_val,
                "mean_scaled": avg_val * 1e5,
                "std": float(np.std(all_scores)) if all_scores else 0.0,
                "count": len(all_scores)
            }

        print("\n" + "="*70)
        print("VGG Gram Style Distance Matrix (Lower is Better, Scaled x1e5)")
        print("-" * 70)
        print(f"{'Sub-Directory':<20} | {'Target Style':<12} | {'Avg VGG Dist':<15}")
        print("-" * 70)
        
        for sd in all_subdirs:
            for t_id in range(num_styles):
                if summary["per_subdirectory"][sd][f"style_{t_id}"]["count"] > 0:
                    avg = summary["per_subdirectory"][sd][f"style_{t_id}"]["mean_scaled"]
                    print(f"{sd:<20} | Style {t_id:<8} | {avg:.5f}")
                else:
                    print(f"{sd:<20} | Style {t_id:<8} | N/A")
            print("-" * 70)
        print("="*70)
        print(f"[Done] Cache saved in: {self.cache_root.resolve()}")
        
        # 保存到文件
        if save_dir:
            save_path = Path(save_dir) / "vgg_results.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            print(f"\n✅ VGG results saved to: {save_path}")
        
        return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--ref_dir", type=str, default=None)
    args = parser.parse_args()

    ref_dir = args.ref_dir.strip('"').strip("'") if args.ref_dir else None
    Evaluator(args.ckpt, ref_root=ref_dir).evaluate(bs=args.bs)