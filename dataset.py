import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

class Stage1Dataset(Dataset):
    def __init__(self, data_root, num_classes=None):
        self.data_root = Path(data_root)
        # 获取子目录作为类别
        self.classes = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        if num_classes is not None:
            self.classes = self.classes[:num_classes]
            
        self.class_to_id = {cls.name: i for i, cls in enumerate(self.classes)}
        self.id_to_class = {i: name for name, i in self.class_to_id.items()}
        self.files_by_class = {i: [] for i in range(len(self.classes))}
        self.all_files = []
        
        for cls_dir in self.classes:
            cls_id = self.class_to_id[cls_dir.name]
            # 搜索编码后的 latent 文件
            files = sorted(list(cls_dir.glob("*.pt")))
            if files:
                self.files_by_class[cls_id] = files
                self.all_files.extend([(f, cls_id) for f in files])
        
        print(f"✅ [Dataset] Stage 1 加载完成。类别: {self.class_to_id}")

    def __len__(self):
        return len(self.all_files)

    def load_latent(self, path):
        """严格维度控制：确保输出一定是 [4, 64, 64]"""
        data = torch.load(path, map_location='cpu')
        # 处理字典包装
        if isinstance(data, dict):
            data = data.get('content') or data.get('z') or list(data.values())[0]
        
        # 彻底移除所有 batch 维度 [1, 1, 4, 64, 64] -> [4, 64, 64]
        if data.ndim > 3:
            # 找到最后三个维度
            data = data.view(-1, *data.shape[-3:])[0]
        
        return data

    def __getitem__(self, idx):
        path, src_label = self.all_files[idx]
        x_c = self.load_latent(path)
        
        # 随机选取一个目标风格类别 (用于训练跨域)
        target_label = random.choice([i for i in range(len(self.classes)) if i != src_label])
        style_path = random.choice(self.files_by_class[target_label])
        x_s = self.load_latent(style_path)
        
        return x_c, x_s, torch.tensor(target_label), torch.tensor(src_label)

class Stage2Dataset(Dataset):
    def __init__(self, reflow_dir):
        self.reflow_dir = Path(reflow_dir)
        self.pairs = sorted(list(self.reflow_dir.glob("pair_*.pt")))
        print(f"✅ [Dataset] Stage 2 加载完成。配对数: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        data = torch.load(self.pairs[idx], map_location='cpu')
        x_c, z, s_id = data['content'], data['z'], data['style_label']
        
        # 同样进行维度修剪
        if x_c.ndim > 3: x_c = x_c.view(-1, *x_c.shape[-3:])[0]
        if z.ndim > 3: z = z.view(-1, *z.shape[-3:])[0]
            
        return x_c, z, s_id