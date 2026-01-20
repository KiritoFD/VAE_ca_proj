import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import numpy as np
from PIL import Image
from torchvision import transforms

class Stage1Dataset(Dataset):
    """
    Stage 1 数据集: 通用加载器
    能读取：
    1. 预处理好的 Latents (.npy / .pt) -> 训练极快，IO压力小
    2. 原始图片 (.jpg / .png) -> 训练时实时 Encode，IO压力大
    """
    def __init__(self, root_dir, num_classes):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"❌ [Dataset] 路径不存在: {self.root}")

        # 1. 定义类别映射
        # 自动扫描子目录
        subdirs = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        
        # 默认映射表，如果你的文件夹叫 trainA/trainB，会自动匹配
        # 如果你的文件夹叫 class0/class1，也会自动匹配
        self.class_map = {
            'trainA': 0, 'testA': 0, 'class0': 0, 'A': 0, 'monet': 0, 'photo': 1,
            'trainB': 1, 'testB': 1, 'class1': 1, 'B': 1, 'art': 0
        }
        
        self.all_files = [] # list of (path, class_id)
        self.files_by_class = {} # dict {class_id: [paths]}

        print(f"🔍 [Dataset] Scanning {self.root}...")
        
        # 2. 遍历目录
        found_any = False
        for d in self.root.iterdir():
            if not d.is_dir(): continue
            
            # 确定类别 ID
            cid = -1
            if d.name in self.class_map:
                cid = self.class_map[d.name]
            else:
                # 如果文件夹名字不在映射表里，尝试按字母顺序分配
                # 这只是一个兜底策略
                pass
            
            if cid == -1: continue # 跳过未知文件夹

            if cid not in self.files_by_class:
                self.files_by_class[cid] = []

            # 3. 核心修复：递归搜索所有可能的后缀
            # 使用 rglob (recursive glob) 防止文件在子文件夹里
            extensions = ['*.npy', '*.pt', '*.jpg', '*.jpeg', '*.png', '*.bmp']
            files = []
            for ext in extensions:
                # case-insensitive search on Windows usually works with glob, 
                # but rglob is safer for nested structures
                files.extend(list(d.rglob(ext)))
                # 尝试大写后缀
                files.extend(list(d.rglob(ext.upper())))
            
            # 去重 (防止大小写重复匹配)
            files = sorted(list(set(files)))
            
            for f in files:
                self.all_files.append((f, cid))
                self.files_by_class[cid].append(f)
            
            if len(files) > 0:
                print(f"   📂 Found Class {cid} ({d.name}): {len(files)} files")
                found_any = True
        
        if not found_any:
            print(f"⚠️ [Dataset] 警告: 在 {self.root} 下找到了文件夹 {subdirs}，但没找到任何文件！")
            print(f"   请检查: 1. 文件夹内是否有 .npy/.jpg 文件？")
            print(f"           2. 文件夹名是否是 trainA/trainB？")
        else:
            print(f"✅ [Dataset] Stage 1 加载完成。总数: {len(self.all_files)}")

        # 预定义 Transform (仅当读取图片时使用)
        self.transform = transforms.Compose([
            transforms.Resize(512), 
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def load_latent(self, path):
        """智能加载函数"""
        path = Path(path)
        ext = path.suffix.lower()
        
        if ext == '.npy':
            # 加载 Numpy 格式 Latent
            arr = np.load(path)
            return torch.from_numpy(arr)
            
        elif ext == '.pt':
            # 加载 PyTorch 格式 Latent
            return torch.load(path)
            
        elif ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            # 加载原始图片 -> 转换成 Tensor
            # 注意：这里返回的是 Pixel Tensor [3, 512, 512]
            # 在 train.py 里，如果是 Pixel，需要 VAE Encode
            # 但为了统一接口，我们在这里假设 train.py 会处理 encode，或者我们在这里无法 encode (没有 vae)
            # 通常我们在 dataset 里只返回 tensor。
            # ⚠️ 重要: 如果你的 train.py 期望直接拿到 latent，这里读取图片会导致形状不对。
            # 既然你指向了 latents 文件夹，说明你主要是想读 .npy。
            # 如果读到了图片，do_inference 里的 vae.decode 会出错（因为输入已经是 pixel）。
            
            # 为了兼容性，这里返回 Pixel Tensor。
            # train.py 需要判断：如果输入是 [4, 64, 64] -> Latent
            # 如果输入是 [3, 512, 512] -> Pixel -> 需要 VAE Encode
            img = Image.open(path).convert("RGB")
            return self.transform(img)
            
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        path_c, cls_c = self.all_files[idx]
        
        # 1. 加载 Content
        x_c = self.load_latent(path_c)
        
        # 2. 随机采样 Style
        target_cls = np.random.choice(list(self.files_by_class.keys()))
        if len(self.files_by_class[target_cls]) > 0:
            path_s = np.random.choice(self.files_by_class[target_cls])
            x_s = self.load_latent(path_s)
        else:
            # Fallback (极少见情况)
            x_s = x_c.clone()
        
        return x_c, x_s, torch.tensor(target_cls), torch.tensor(cls_c)


class Stage2Dataset(Dataset):
    """
    Stage 2 数据集: 读取 Reflow 生成的 .pt 配对数据
    """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.pt_files = sorted(list(self.data_dir.glob("*.pt")))
        
        if len(self.pt_files) == 0:
            print(f"❌ [Dataset] 在 {self.data_dir} 下未找到 .pt 文件！")
            self.indices = []
            return

        print(f"🔍 [Dataset] 索引 Stage 2 数据 ({len(self.pt_files)} 文件)...")
        
        self.indices = [] 
        # 快速索引
        for i, pt_file in enumerate(self.pt_files):
            try:
                # 读取 header 以获取 batch size (通常文件名里不带信息，需读取)
                # 为了速度，假设所有 batch size 一样，或者只读第一个
                # 这里为了稳健，简单遍历一次（很快）
                # 优化: 假设每个文件 batch size = training batch size (e.g. 4)
                # 只有最后一个文件可能小。
                # 正确做法：读取文件获取大小
                data = torch.load(pt_file, map_location="cpu")
                bs = data['z0'].size(0)
                for j in range(bs):
                    self.indices.append((i, j))
            except:
                pass
                
        print(f"✅ [Dataset] Stage 2 加载完成。样本数: {len(self.indices)}")
        
        self.current_file_idx = -1
        self.current_data = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, row_idx = self.indices[idx]
        
        if file_idx != self.current_file_idx:
            self.current_data = torch.load(self.pt_files[file_idx], map_location="cpu")
            self.current_file_idx = file_idx
            
        z0 = self.current_data['z0'][row_idx]
        z1 = self.current_data['z1'][row_idx]
        t_id = self.current_data['t_id'][row_idx]
        
        return z0, z1, t_id