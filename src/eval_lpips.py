import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import AutoencoderKL
from PIL import Image
from pathlib import Path
import json
import lpips
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
            
        # 递归获取所有图片并记录其所属的直接父目录名称
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
            transforms.Normalize([0.5], [0.5])
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

        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(self.device)
        self.vae.eval()

        self.loss_fn_lpips = lpips.LPIPS(net='alex').to(self.device)
        self.loss_fn_lpips.eval()

        self.model = SAFModel(**self.cfg['model']).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)
        self.model.eval()

    @torch.no_grad()
    def process_batch(self, imgs, tgt_id):
        # Latent 编解码与推理过程
        x_c = self.vae.encode(imgs.to(self.device)).latent_dist.sample() * 0.18215
        
        # Flow Matching 推理
        steps = 10
        dt = 1.0 / steps
        x_t = x_c.clone()
        s_ids = torch.full((x_c.size(0),), tgt_id, device=self.device, dtype=torch.long)
        
        for i in range(steps):
            t = torch.ones(x_c.size(0), device=self.device) * (i * dt)
            v_pred = self.model(x_t, x_c, t, s_ids)
            x_t = x_t + v_pred * dt
            
        # 解码回像素空间
        imgs_gen = torch.clamp(self.vae.decode(x_t / 0.18215).sample, -1, 1)
        return self.loss_fn_lpips(imgs.to(self.device), imgs_gen, normalize=True).view(-1).cpu().numpy()

    def evaluate(self, data_dir, batch_size=2):
        dataset = MatrixImageDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        num_styles = self.cfg['model'].get('num_styles', 2)
        # 存储结构: matrix[sub_dir][tgt_style] = [scores]
        matrix_results = defaultdict(lambda: defaultdict(list))
        
        for imgs, sub_dirs in tqdm(dataloader, desc="Calculating LPIPS Matrix"):
            for tgt_id in range(num_styles):
                batch_scores = self.process_batch(imgs, tgt_id)
                for i, sub_dir in enumerate(sub_dirs):
                    if sub_dir != "error":
                        matrix_results[sub_dir][tgt_id].append(float(batch_scores[i]))

        # 打印详细矩阵表格
        print("\n" + "="*60)
        print(f"{'Sub-Directory':<20} | {'Target Style':<12} | {'Avg LPIPS':<10}")
        print("-" * 60)
        
        all_subdirs = sorted(matrix_results.keys())
        for sd in all_subdirs:
            for t_id in range(num_styles):
                scores = matrix_results[sd][t_id]
                avg = np.mean(scores) if scores else 0
                print(f"{sd:<20} | Style {t_id:<8} | {avg:.5f}")
            print("-" * 60)
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--bs", type=int, default=3)
    args = parser.parse_args()

    with open("config.json", 'r', encoding='utf-8') as f:
        target_dir = json.load(f).get("inference", {}).get("image_path", "").strip('"').strip("'")

    Evaluator(args.ckpt).evaluate(target_dir, args.bs)