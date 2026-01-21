import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import AutoencoderKL
from PIL import Image
from pathlib import Path
import json
import clip
from tqdm import tqdm
import os
import argparse
import numpy as np
from collections import defaultdict

# 导入模型定义
from SAFlow import SAFModel

class MatrixImageDataset(Dataset):
    def __init__(self, root_dir, size=512):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Error: Directory not found: {self.root_dir}")
            
        self.files = []
        for f in self.root_dir.rglob('*'):
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                # 记录相对路径的父目录，作为子目录标签
                sub_dir = str(f.parent.relative_to(self.root_dir))
                self.files.append((f, sub_dir))
        
        print(f"Total images found: {len(self.files)}")
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # map to [-1, 1] for VAE
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, sub_dir = self.files[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), sub_dir
        except Exception as e:
            return torch.zeros(3, 512, 512), "error"

class Evaluator:
    def __init__(self, ckpt_path, config_path="config.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)

        # 1. Load VAE
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(self.device)
        self.vae.eval()

        # 2. Load CLIP Model (ViT-B/32 is standard for evaluation)
        print("Loading CLIP model...")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

        # CLIP normalization constants
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)

        # 3. Load SAFModel
        self.model = SAFModel(**self.cfg['model']).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)
        self.model.eval()

    def preprocess_for_clip(self, images):
        """
        Input: Tensor [-1, 1] (B, C, H, W)
        Output: Tensor normalized for CLIP (B, C, 224, 224)
        """
        # 1. Denormalize to [0, 1]
        images = (images + 1.0) / 2.0
        
        # 2. Resize to 224x224 (CLIP input size)
        images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)
        
        # 3. Normalize with CLIP mean/std
        images = (images - self.clip_mean) / self.clip_std
        return images

    @torch.no_grad()
    def process_batch(self, imgs, tgt_id):
        # 1. VAE Encode
        x_c = self.vae.encode(imgs.to(self.device)).latent_dist.sample() * 0.18215
        
        # 2. Flow Matching Inference
        steps = 10
        dt = 1.0 / steps
        x_t = x_c.clone()
        s_ids = torch.full((x_c.size(0),), tgt_id, device=self.device, dtype=torch.long)
        
        for i in range(steps):
            t = torch.ones(x_c.size(0), device=self.device) * (i * dt)
            v_pred = self.model(x_t, x_c, t, s_ids)
            x_t = x_t + v_pred * dt
            
        # 3. VAE Decode -> Pixel Space [-1, 1]
        imgs_gen = torch.clamp(self.vae.decode(x_t / 0.18215).sample, -1, 1)
        
        # 4. Calculate CLIP Content Score
        # Preprocess both source and gen images for CLIP
        clip_src = self.preprocess_for_clip(imgs.to(self.device))
        clip_gen = self.preprocess_for_clip(imgs_gen)
        
        # Encode with CLIP image encoder
        feat_src = self.clip_model.encode_image(clip_src)
        feat_gen = self.clip_model.encode_image(clip_gen)
        
        # Normalize features
        feat_src = feat_src / feat_src.norm(dim=-1, keepdim=True)
        feat_gen = feat_gen / feat_gen.norm(dim=-1, keepdim=True)
        
        # Cosine Similarity
        similarity = (feat_src * feat_gen).sum(dim=-1)
        
        return similarity.cpu().numpy()

    def evaluate(self, data_dir, batch_size=3, save_dir=None):
        dataset = MatrixImageDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        num_styles = self.cfg['model'].get('num_styles', 2)
        # 存储结构: matrix[sub_dir][tgt_style] = [scores]
        matrix_results = defaultdict(lambda: defaultdict(list))
        
        print("Calculating CLIP Content Consistency Matrix...")
        
        for imgs, sub_dirs in tqdm(dataloader):
            for tgt_id in range(num_styles):
                batch_scores = self.process_batch(imgs, tgt_id)
                for i, sub_dir in enumerate(sub_dirs):
                    if sub_dir != "error":
                        matrix_results[sub_dir][tgt_id].append(float(batch_scores[i]))

        # 计算汇总统计
        summary = {
            "metric": "CLIP",
            "description": "Content similarity (higher is better, max 1.0)",
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
                    "std": float(np.std(scores)) if scores else 0.0,
                    "count": len(scores)
                }
                overall_by_style[t_id].extend(scores)
        
        # 全局平均
        for t_id in range(num_styles):
            all_scores = overall_by_style[t_id]
            summary["overall_average"][f"style_{t_id}"] = {
                "mean": float(np.mean(all_scores)) if all_scores else 0.0,
                "std": float(np.std(all_scores)) if all_scores else 0.0,
                "count": len(all_scores)
            }

        # 打印详细矩阵表格
        print("\n" + "="*70)
        print("CLIP Content Score Matrix (Higher is better, Max 1.0)")
        print(f"Metric: Semantic Similarity between Source and Output")
        print("-" * 70)
        print(f"{'Sub-Directory':<20} | {'Target Style':<12} | {'Avg CLIP Score':<15}")
        print("-" * 70)
        
        for sd in all_subdirs:
            for t_id in range(num_styles):
                avg = summary["per_subdirectory"][sd][f"style_{t_id}"]["mean"]
                print(f"{sd:<20} | Style {t_id:<8} | {avg:.5f}")
            print("-" * 70)
        print("="*70)
        
        # 保存到文件
        if save_dir:
            save_path = Path(save_dir) / "clip_results.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            print(f"\n✅ CLIP results saved to: {save_path}")
        
        return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--bs", type=int, default=3)
    args = parser.parse_args()

    # 自动读取 config.json 中的 image_path
    with open("config.json", 'r', encoding='utf-8') as f:
        target_dir = json.load(f).get("inference", {}).get("image_path", "").strip('"').strip("'")

    if not target_dir:
        print("Error: 'inference.image_path' not set in config.json")
        exit(1)

    Evaluator(args.ckpt).evaluate(target_dir, args.bs)