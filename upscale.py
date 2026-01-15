import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ================= 配置 =================
# 输入文件夹 (你的256图片目录)
INPUT_DIR = r"F:\monet2photo\monet2photo\testB"

# 输出文件夹 (脚本会自动创建)
OUTPUT_DIR = f"{INPUT_DIR}_512"

# 目标尺寸
TARGET_SIZE = 512

# 使用 FP16 (半精度) 加速，如果显卡报错改成 False
USE_FP16 = True 
# =======================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    # 1. 初始化 RealESRGAN 模型
    # 我们使用 x4plus 模型，它是最通用的高质量模型
    print("⏳ 正在加载/下载 Real-ESRGAN x4plus 模型...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,             # 如果显存不够（OOM），把这个改成 400 或 256
        tile_pad=10,
        pre_pad=0,
        half=USE_FP16,      
        gpu_id=0 if torch.cuda.is_available() else None
    )

    # 2. 准备路径
    input_root = Path(INPUT_DIR)
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    # 获取所有图片
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    img_files = [f for f in input_root.rglob('*') if f.suffix.lower() in extensions]
    
    print(f"📂 发现 {len(img_files)} 张图片，准备超分...")
    print(f"   输入: {input_root}")
    print(f"   输出: {output_root}")

    # 3. 批量处理
    for img_path in tqdm(img_files, desc="Upscaling"):
        try:
            # 保持相对目录结构 (例如 trainA/001.jpg -> output/trainA/001.jpg)
            rel_path = img_path.relative_to(input_root)
            save_path = output_root / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 跳过已存在的
            if save_path.exists():
                continue

            # 读取图片 (使用 OpenCV 读取，支持 RealESRGANer)
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"⚠️ 无法读取: {img_path}")
                continue

            # --- 核心步骤 A: AI 超分 ---
            # input (256x256) -> output (1024x1024)
            # outscale=4 表示放大4倍
            output, _ = upsampler.enhance(img, outscale=4)

            # --- 核心步骤 B: 高质量下采样 ---
            # 1024x1024 -> 512x512
            # 先转回 PIL 方便做高质量 Resize
            output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            output_final = output_pil.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

            # 保存 (使用高质量 JPEG 参数)
            output_final.save(save_path, quality=95)

        except Exception as e:
            print(f"\n❌ 处理失败 {img_path.name}: {e}")

    print(f"\n✅ 处理完成！高质量图片已保存在: {OUTPUT_DIR}")
    print("💡 下一步：请使用 encode_sd1.5.py 对这个新文件夹进行编码。")

if __name__ == "__main__":
    main()