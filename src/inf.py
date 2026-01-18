import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL
from torchvision import transforms
import time
from contextlib import nullcontext

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
    use_fp16 = device.type == 'cuda'
    amp_ctx = torch.cuda.amp.autocast if use_fp16 else nullcontext
    print("Using FP16 inference" if use_fp16 else "Using FP32 inference")
    input_image_path = "test.jpg"
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    if not Path(input_image_path).exists():
        print("❌ 找不到 test.jpg，请放一张图片在根目录")
        return

    start_time = time.time()

    # helper: report GPU memory usage
    def report_mem(prefix: str):
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"{prefix} - GPU mem: allocated={allocated:.1f}MB reserved={reserved:.1f}MB peak={peak_alloc:.1f}MB")
        else:
            print(f"{prefix} - CUDA not available")

    # 2. 加载模型
    print("Loading Model...")
    model = SAFModel(**cfg['model']).to(device)
    if use_fp16:
        model.half()
    
    # 🔴 使用指定 checkpoint 文件
    ckpt_path = Path(r"g:\GitHub\VAE_ca_proj\checkpoints\stage1_epoch10.pt")
    if not ckpt_path.exists():
        print(f"❌ 找不到 checkpoint: {ckpt_path}")
        return
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    model_loaded_time = time.time()
    print(f"Model load time: {(model_loaded_time - start_time):.3f}s")
    report_mem("After model load")

    # 3. 加载 VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        subfolder="vae"
    ).to(device)
    if use_fp16:
        vae.half()
    vae.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    vae_loaded_time = time.time()
    print(f"VAE load time: {(vae_loaded_time - model_loaded_time):.3f}s")
    report_mem("After VAE load")

    # 4. 处理输入图片
    print("Processing Input...")
    raw_img = Image.open(input_image_path).convert("RGB").resize((512, 512))
    tf = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = tf(raw_img).unsqueeze(0).to(device)
    if use_fp16:
        img_tensor = img_tensor.half()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    input_processed_time = time.time()
    print(f"Input processing time: {(input_processed_time - vae_loaded_time):.3f}s")
    report_mem("After input processed")

    # 编码 -> 缩放
    with amp_ctx():
        latent_c = vae.encode(img_tensor).latent_dist.sample() * 0.18215

    # 5. 生成 (Euler 20步)
    print("Generating...")
    x_t = latent_c.clone()
    x_cond = latent_c
    style_id = torch.tensor([0], device=device)
    dt = 1.0 / 20

    # reset peak memory before generation to capture gen peak specifically
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    report_mem("Before generation (peak reset)")

    gen_start = time.time()
    with amp_ctx():
        for i in range(20):
            t = torch.tensor([i * dt], device=device)
            v = model(x_t, x_cond, t, style_id)
            x_t = x_t + dt * v
    if device.type == 'cuda':
        torch.cuda.synchronize()
    gen_end = time.time()
    total_gen = gen_end - gen_start
    print(f"Generation total time: {total_gen:.3f}s, avg per step: {(total_gen/20):.4f}s")
    report_mem("After generation")

    # 6. 解码 -> 保存
    print("Decoding...")
    with amp_ctx():
        decoded = vae.decode(x_t / 0.18215).sample
    # ensure safe float32 math for postprocessing
    decoded = decoded.float()
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()[0]
    result_img = Image.fromarray((decoded * 255).astype('uint8'))
    
    save_path = output_dir / "result_hd.jpg"
    result_img.save(save_path)
    # ensure time is recorded regardless of device
    if device.type == 'cuda':
        torch.cuda.synchronize()
    decoded_time = time.time()
    print(f"Decoding+save time: {(decoded_time - gen_end):.3f}s")
    report_mem("After decode+save")

    total_time = decoded_time - start_time
    print(f"✅ 完成！结果已保存至: {save_path}")
    print(f"Total run time: {total_time:.3f}s")

if __name__ == "__main__":
    main()