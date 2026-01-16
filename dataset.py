import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

class Stage1Dataset(Dataset):
    """
    Stage 1 数据集：严格独立耦合 (Strict Independent Coupling)
    逻辑：强制要求 Content 和 Style 来自完全不同的类别，模拟 E[T_source -> T_target]
    """
    def __init__(self, data_root, num_classes=None):
        self.data_root = Path(data_root)
        
        # 1. 扫描所有包含 .pt 文件的子文件夹 (作为不同的风格类别)
        # 例如: data_root/trainA, data_root/trainB
        self.classes = sorted([d for d in self.data_root.iterdir() if d.is_dir() and list(d.glob("*.pt"))])
        
        if num_classes is not None:
            self.classes = self.classes[:num_classes]
        
        # 🔴 核心校验：必须至少有 2 个类别才能进行风格迁移
        if len(self.classes) < 2:
            raise ValueError(
                f"❌ 数据集配置错误：只找到了 {len(self.classes)} 个类别 ({[d.name for d in self.classes]})。\n"
                f"Reflow 训练要求至少 2 个不同的风格类别（如 trainA 和 trainB）以进行跨域配对。\n"
                f"请检查 config.json 中的 'data_root' 是否指向了包含子文件夹的父目录。"
            )

        # 2. 建立索引：类别名 -> 文件列表
        self.class_files = {}
        self.all_files = [] # 用于 __getitem__ 的主索引
        
        for cls_dir in self.classes:
            files = sorted(list(cls_dir.glob("*.pt")))
            if files:
                self.class_files[cls_dir.name] = files
                self.all_files.extend(files)
        
        # 建立类别名 -> ID 的映射
        self.class_to_id = {cls.name: i for i, cls in enumerate(self.classes)}
        
        print(f"[Stage1Dataset] ✅ 成功加载 {len(self.all_files)} 个样本。")
        print(f"                包含类别: {list(self.class_files.keys())}")
        print(f"               🔒 策略: 严格跨域配对 (Strict Cross-Domain Pairing)")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        # 1. 加载内容图 (Content)
        content_file = self.all_files[idx]
        # 使用 squeeze() 修复 [1, 4, 64, 64] -> [4, 64, 64] 维度问题
        x_content = torch.load(content_file, map_location='cpu')
        if x_content.dim() > 3: x_content = x_content.squeeze()
        
        # 获取内容图所属的类别名
        content_class_name = content_file.parent.name
        
        # 2. 强制选择异类风格图 (Strict Cross-Domain)
        # 找出所有"不是当前类别"的类别
        other_class_names = [c for c in self.class_files.keys() if c != content_class_name]
        
        # 理论上基于 __init__ 的校验，这里永远不会为空
        # 随机选一个目标风格类别
        target_class_name = random.choice(other_class_names)
        
        # 从该目标类别中随机选一张图
        style_file = random.choice(self.class_files[target_class_name])
        x_style = torch.load(style_file, map_location='cpu')
        if x_style.dim() > 3: x_style = x_style.squeeze()
        
        # 3. 获取目标风格的 Label ID
        style_label = self.class_to_id[target_class_name]
        
        # ⚠️ 注意：此处不进行 * 0.18215 缩放，因为 encode_sd1.5.py 已处理过
        return x_content, x_style, torch.tensor(style_label, dtype=torch.long)


class Stage2Dataset(Dataset):
    """
    Stage 2 数据集：Reflow 生成的伪数据对
    读取 (Content, Z) 进行直线轨迹拟合
    """
    def __init__(self, reflow_dir):
        self.reflow_dir = Path(reflow_dir)
        self.pairs = sorted(list(self.reflow_dir.glob("pair_*.pt")))
        
        if not self.pairs:
            print(f"⚠️ [Stage2Dataset] 警告：在 {reflow_dir} 中未找到配对数据。")
        else:
            print(f"[Stage2Dataset] ✅ 成功加载 {len(self.pairs)} 组 Reflow 配对数据。")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        data = torch.load(self.pairs[idx], map_location='cpu')
        
        x_content = data['content']
        z_target = data['z']
        style_label = data['style_label']

        # 维度修复
        if x_content.dim() > 3: x_content = x_content.squeeze()
        if z_target.dim() > 3: z_target = z_target.squeeze()

        return x_content, z_target, style_label