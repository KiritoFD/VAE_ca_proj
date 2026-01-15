import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL
from torchvision import transforms

from SAFlow import SAFModel
import json

def load_config():
    with open("config.json", 'r') as f: 
        return json.load(f)

@torch.no_grad()
def main():
    # 1. 配置
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image_path = "test.jpg"
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    if not Path(input_image_path).exists():
        print("❌ 找不到 test.jpg，请放一张图片在根目录")
        return

    # 2. 加载模型
    print("Loading Model...")
    model = SAFModel(**cfg['model']).to(device)
    
    # 🔴 优先加载 Reflow 最终模型
    ckpt_dir = Path(cfg['checkpoint']['save_dir'])
    final_path = ckpt_dir / "saf_final_reflowed.pt"
    
    if final_path.exists():
        print(f"Loading final reflowed model: {final_path}")
        model.load_state_dict(torch.load(final_path, map_location=device))
    else:
        # Fallback：找最新的 checkpoint
        ckpts = sorted(list(ckpt_dir.glob("stage*.pt")), key=lambda x: x.stat().st_mtime)
        if not ckpts:
            print("❌ 没有找到模型权重，请先训练！")
            return
        latest_ckpt = ckpts[-1]
        print(f"Loading checkpoint: {latest_ckpt.name}")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    
    model.eval()

    # 3. 加载 VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        subfolder="vae"
    ).to(device)
    vae.eval()

    # 4. 处理输入图片
    print("Processing Input...")
    raw_img = Image.open(input_image_path).convert("RGB").resize((512, 512))
    tf = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = tf(raw_img).unsqueeze(0).to(device)

    # 编码 -> 缩放
    latent_c = vae.encode(img_tensor).latent_dist.sample() * 0.18215

    # 5. 生成 (Euler 20步)
    print("Generating...")
    x_t = latent_c.clone()
    x_cond = latent_c
    style_id = torch.tensor([0], device=device)
    dt = 1.0 / 20

    for i in range(20):
        t = torch.tensor([i * dt], device=device)
        v = model(x_t, x_cond, t, style_id)
        x_t = x_t + dt * v
    
    # 6. 解码 -> 保存
    print("Decoding...")
    decoded = vae.decode(x_t / 0.18215).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()[0]
    result_img = Image.fromarray((decoded * 255).astype('uint8'))
    
    save_path = output_dir / "result_hd.jpg"
    result_img.save(save_path)
    print(f"✅ 完成！结果已保存至: {save_path}")

if __name__ == "__main__":
    main()