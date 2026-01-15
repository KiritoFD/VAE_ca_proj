import torch
from torch.utils.data import Dataset
import os
import random
from pathlib import Path


class RandomPairDataset(Dataset):
    """
    随机配对数据集：内容图和目标图来自不同类别
    """
    def __init__(self, content_dir, style_root, num_classes=None):
        """
        Args:
            content_dir: 内容latent的文件夹（每个子文件夹为一个类别）
            style_root: 风格latent的文件夹（每个子文件夹为一个类别）
            num_classes: 只使用图片最多的N个类别，None表示使用全部
        """
        self.content_root = Path(content_dir)
        self.style_root = Path(style_root)

        # 收集所有类别
        all_content_classes = sorted([d for d in self.content_root.iterdir() if d.is_dir()])
        all_style_classes = sorted([d for d in self.style_root.iterdir() if d.is_dir()])

        # 统计每个类别的图片数量
        class_counts = []
        for style_dir in all_style_classes:
            style_files = list(style_dir.glob("*.pt"))
            class_counts.append((style_dir, len(style_files)))

        # 只用图片最多的N个类别
        if num_classes is not None:
            class_counts.sort(key=lambda x: x[1], reverse=True)
            class_counts = class_counts[:num_classes]
            print(f"\n[Dataset] Using top {num_classes} classes with most images:")
            for i, (style_dir, count) in enumerate(class_counts):
                print(f"  Rank {i+1}: {style_dir.name} ({count} images)")

        # 重新组织类别
        self.style_classes = [x[0] for x in class_counts]
        self.content_classes = [c for c in all_content_classes if c in self.style_classes]
        # 保证类别顺序一致
        self.class_name_to_idx = {cls.name: idx for idx, cls in enumerate(self.style_classes)}

        # 构建目标样本列表：(文件路径, 类别ID)
        self.samples = []
        for class_id, style_dir in enumerate(self.style_classes):
            style_files = list(style_dir.glob("*.pt"))
            for fpath in style_files:
                self.samples.append((fpath, class_id))

        print(f"\n[Dataset Summary - Random Pair Mode]")
        print(f"Total samples: {len(self.samples)}")
        print(f"Using {len(self.style_classes)} classes:")
        for i, style_dir in enumerate(self.style_classes):
            count = sum(1 for s in self.samples if s[1] == i)
            print(f"  Class {i} ({style_dir.name}): {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 目标风格图
        target_path, target_label = self.samples[idx]
        x_style = torch.load(target_path, map_location='cpu')

        # 随机选择一个不同类别
        other_class_ids = [i for i in range(len(self.style_classes)) if i != target_label]
        content_class_id = random.choice(other_class_ids)
        content_class_dir = self.style_classes[content_class_id]
        content_files = list(content_class_dir.glob("*.pt"))
        content_path = random.choice(content_files)
        x_content = torch.load(content_path, map_location='cpu')

        # 确保形状正确 [4, 64, 64]
        if x_content.dim() == 4:
            x_content = x_content.squeeze(0)
        if x_style.dim() == 4:
            x_style = x_style.squeeze(0)
        
        # SD latent scaling (保持与VAE编码一致)
        x_content = x_content * 0.18215
        x_style = x_style * 0.18215

        return x_content, x_style, target_label
