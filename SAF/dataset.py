import torch
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import AutoencoderKL
from pathlib import Path
import os
import json
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np

# ================= 常量定义 =================
SCALING_FACTOR = 0.18215 

# ================= 第一部分：预处理工具 =================
def preprocess_dataset(cfg, device='cuda'):
    """
    读取 config -> 遍历 raw_data_root -> VAE 编码 -> 保存到 data_root
    """
    # 1. 从配置读取路径
    src_root = Path(cfg['data']['raw_data_root'])
    dst_root = Path(cfg['data']['data_root'])
    
    if not src_root.exists():
        raise FileNotFoundError(f"❌ [Config Error] 原始数据路径不存在: {src_root}")

    print(f"🚀 [Preprocess] 启动预处理流程")
    print(f"   📂 原始图片: {src_root}")
    print(f"   💾 输出目标: {dst_root}")

    # 2. 强制加载 FT-MSE VAE (无回退)
    print("   ⏳ 正在加载 VAE: stabilityai/sd-vae-ft-mse ...")
    # 如果这里报错，说明环境里没下载好，或者网络不通，直接让它抛出异常，不给回退机会
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    vae.requires_grad_(False)
    vae.float() # 使用 FP32 保证编码精度

    # 3. 定义预处理 (假设图片已经是 256x256，不做 Resize)
    img_transform = transforms.Compose([
        transforms.ToTensor(),             # [0, 255] -> [0.0, 1.0]
        transforms.Normalize([0.5], [0.5]) # [0.0, 1.0] -> [-1.0, 1.0]
    ])

    # 4. 扫描并处理
    subdirs = [d for d in src_root.iterdir() if d.is_dir()]
    total_files = 0
    
    for subdir in subdirs:
        # 在目标目录创建同名子文件夹
        target_dir = dst_root / subdir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 扫描常见图片格式
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in extensions:
            files.extend(list(subdir.glob(ext)))
            files.extend(list(subdir.glob(ext.upper())))
        
        files = sorted(list(set(files)))
        print(f"   👉 正在处理类别目录: {subdir.name} ({len(files)} 张)")

        for img_path in tqdm(files, desc=f"Encoding {subdir.name}"):
            save_path = target_dir / (img_path.stem + ".pt")
            
            # 跳过已存在的 (支持断点续传)
            if save_path.exists():
                continue

            try:
                # 读取图片
                img = Image.open(img_path).convert("RGB")
                
                # 简单校验一下尺寸 (可选，防止混入脏数据)
                if img.size != (256, 256):
                    # 如果你非常确定全是256，这行可以注释掉；否则最好Resize一下防止报错
                    img = img.resize((256, 256), Image.BICUBIC)

                # 转 Tensor
                pixel_tensor = img_transform(img).unsqueeze(0).to(device)
                
                # VAE 编码
                with torch.no_grad():
                    # Encode -> Sample -> Scale
                    dist = vae.encode(pixel_tensor).latent_dist
                    latent = dist.sample() * SCALING_FACTOR
                
                # 保存为 [4, 32, 32] (CPU Tensor)
                torch.save(latent.squeeze(0).cpu(), save_path)
                total_files += 1
                
            except Exception as e:
                print(f"   ❌ [Error] {img_path.name}: {e}")

    print(f"✅ [Preprocess] 预处理完成！共生成 {total_files} 个 Latent 文件。")


# ================= 第二部分：数据集加载器 =================
class Stage1Dataset(Dataset):
    """
    训练用 Dataset：只读取 data_root 下的 .pt 文件
    """
    def __init__(self, data_root, num_classes=2):
        self.root = Path(data_root)
        if not self.root.exists():
            raise FileNotFoundError(f"❌ 数据集路径不存在: {self.root}\n👉 请先运行 'python src/dataset.py' 进行预处理！")

        # 硬编码的类别映射，适配 monet2photo
        self.class_map = {
            'trainA': 0, 'testA': 0, 'monet': 0, 
            'trainB': 1, 'testB': 1, 'photo': 1
        }
        
        self.all_files = [] 
        self.files_by_class = {} 

        print(f"🔍 [Dataset] 扫描 Latent 数据: {self.root}")
        
        # 遍历子文件夹
        for d in self.root.iterdir():
            if not d.is_dir(): continue
            
            cid = self.class_map.get(d.name, -1)
            if cid == -1: continue 

            if cid not in self.files_by_class: 
                self.files_by_class[cid] = []

            # 收集 .pt 文件
            files = sorted(list(d.glob("*.pt")))
            
            for f in files:
                self.all_files.append((f, cid))
                self.files_by_class[cid].append(f)
            
            if len(files) > 0:
                print(f"   📂 类别 {cid} ({d.name}): {len(files)} 个文件")

        if len(self.all_files) == 0:
            raise RuntimeError(f"❌ 在 {self.root} 下未找到 .pt 文件！\n请检查 config.json 中的 'data_root' 是否正确，或是否已运行预处理。")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        path_c, cls_c = self.all_files[idx]
        
        # 1. 加载 Content Latent [4, 32, 32]
        # weights_only=True 是新版 PyTorch 的安全建议
        x_c = torch.load(path_c, map_location='cpu', weights_only=True)
        
        # 2. 随机采样 Style Latent
        target_cls = np.random.choice(list(self.files_by_class.keys()))
        if len(self.files_by_class[target_cls]) > 0:
            path_s = np.random.choice(self.files_by_class[target_cls])
            x_s = torch.load(path_s, map_location='cpu', weights_only=True)
        else:
            x_s = x_c.clone()
        
        return x_c, x_s, torch.tensor(target_cls), torch.tensor(cls_c)


class Stage2Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.pt_files = sorted(list(self.data_dir.glob("*.pt")))
        self.indices = []
        
        if len(self.pt_files) > 0:
            print(f"🔍 [Dataset] Stage 2 索引 ({len(self.pt_files)} files)...")
            try:
                # 读取首个文件推断 Batch Size
                data = torch.load(self.pt_files[0], map_location="cpu", weights_only=True)
                bs = data['z0'].size(0)
                for i in range(len(self.pt_files)):
                    for j in range(bs):
                        self.indices.append((i, j))
            except Exception as e:
                print(f"⚠️ [Dataset] 索引出错: {e}")

        self.current_file_idx = -1
        self.current_data = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, row_idx = self.indices[idx]
        
        if file_idx != self.current_file_idx:
            self.current_data = torch.load(self.pt_files[file_idx], map_location="cpu", weights_only=True)
            self.current_file_idx = file_idx
        
        # 简单越界保护
        if row_idx >= self.current_data['z0'].size(0): row_idx = 0
            
        z0 = self.current_data['z0'][row_idx]
        z1 = self.current_data['z1'][row_idx]
        t_id = self.current_data['t_id'][row_idx]
        return z0, z1, t_id


# ================= 脚本入口 =================
if __name__ == "__main__":
    # 读取根目录下的 config.json
    config_path = Path("config.json")
    if not config_path.exists():
        # 尝试向上找一级，防止用户在 src 目录下运行
        config_path = Path("../config.json")
    
    if not config_path.exists():
        raise FileNotFoundError("❌ 找不到 config.json，请确保在项目根目录运行，或配置文件存在。")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # 执行预处理
    preprocess_dataset(cfg)