import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

class Stage1Dataset(Dataset):
    """
    Stage 1 数据集：随机配对 (Independent Coupling)
    用于学习内容流形到风格流形的初始映射
    """
    def __init__(self, data_root, num_classes=None):
        self.data_root = Path(data_root)
        
        # 列出所有包含 .pt 文件的子目录作为类别
        self.classes = sorted([d for d in self.data_root.iterdir() if d.is_dir() and list(d.glob("*.pt"))])
        if num_classes is not None:
            self.classes = self.classes[:num_classes]
        
        # 建立类别名到索引的映射
        self.class_to_id = {cls.name: i for i, cls in enumerate(self.classes)}
        
        # 收集每个类别的文件列表
        self.class_files = {}
        self.all_files = []
        for cls_dir in self.classes:
            files = sorted(list(cls_dir.glob("*.pt")))
            if files:
                self.class_files[cls_dir.name] = files
                self.all_files.extend(files)
        
        if not self.all_files:
            raise ValueError(f"在 {data_root} 中未找到任何 .pt 文件，请检查路径。")
            
        print(f"[Stage1Dataset] 成功加载 {len(self.all_files)} 个样本，共 {len(self.class_files)} 个类别")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        # 1. 加载内容图 (Content)
        content_file = self.all_files[idx]
        x_content = torch.load(content_file, map_location='cpu')
        
        # 2. 随机选择风格图 (Style)
        content_class_name = content_file.parent.name
        other_class_names = [c for c in self.class_files.keys() if c != content_class_name]
        
        if other_class_names:
            style_class_name = random.choice(other_class_names)
            style_file = random.choice(self.class_files[style_class_name])
        else:
            # 如果只有一个类别，则从同类中随机选一张不同的
            style_file = random.choice(self.class_files[content_class_name])
        
        x_style = torch.load(style_file, map_location='cpu')
        
        # 🔴 关键修复 1：维度挤压
        # 将 [1, 4, 64, 64] 转换为 [4, 64, 64]，防止 DataLoader 产生 5D 张量
        if x_content.dim() > 3:
            x_content = x_content.squeeze()
        if x_style.dim() > 3:
            x_style = x_style.squeeze()
            
        # 🔴 关键修复 2：移除二次缩放
        # 因为 encode_sd1.5.py 已乘过 0.18215，这里直接返回原始读取值
        # 若再次乘以 0.18215，会导致数值分布过小，Loss 异常
        
        # 获取风格对应的标签 ID
        style_label = self.class_to_id[style_file.parent.name]
        
        return x_content, x_style, torch.tensor(style_label, dtype=torch.long)


class Stage2Dataset(Dataset):
    """
    Stage 2 数据集：Reflow 生成的伪数据对
    (Content, Z) 其中 Z 是 Stage 1 模型生成的确定性映射结果
    """
    def __init__(self, reflow_dir):
        self.reflow_dir = Path(reflow_dir)
        self.pairs = sorted(list(self.reflow_dir.glob("pair_*.pt")))
        
        if not self.pairs:
            print(f"⚠️ 警告：在 {reflow_dir} 中未找到配对数据，请确认 Stage 1 生成步骤已完成。")

        print(f"[Stage2Dataset] 成功加载 {len(self.pairs)} 组 Reflow 配对数据")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        data = torch.load(self.pairs[idx], map_location='cpu')
        
        x_content = data['content']
        z_target = data['z']
        style_label = data['style_label']

        # 同样进行维度检查，确保万无一失
        if x_content.dim() > 3:
            x_content = x_content.squeeze()
        if z_target.dim() > 3:
            z_target = z_target.squeeze()

        # 🔴 移除二次缩放
        # Stage 1 生成的 Z 本身就是基于已缩放数据产生的，无需再次处理
        return x_content, z_target, style_label