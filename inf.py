import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np

from SAFlow import SAFModel
from config import Config


@torch.no_grad()
def generate_style_transfer_cfg(
    content_latent, 
    target_style_id, 
    model, 
    device, 
    steps=25, 
    cfg_scale=4.0
):
    """
    SA-Flow v2 推理：使用 CFG (Classifier-Free Guidance) 进行锐化生成
    
    Args:
        content_latent: [1, 4, 64, 64] - 内容图的latent
        target_style_id: int - 目标风格ID
        model: SAFModel
        device: cuda/cpu
        steps: int - 积分步数
        cfg_scale: float - CFG 强度 (3.0-5.0 推荐)
    
    Returns:
        stylized_latent: [1, 4, 64, 64]
    """
    model.eval()
    content_latent = content_latent.to(device)
    style_tensor = torch.tensor([target_style_id], dtype=torch.long, device=device)
    
    # 🔴 v2: 从纯噪声开始 (与训练一致)
    x_t = torch.randn_like(content_latent)
    
    # 准备条件
    cond_input = content_latent
    uncond_input = torch.zeros_like(content_latent)  # 空条件 (用于CFG)
    
    dt = 1.0 / steps
    
    # 欧拉积分 + CFG
    for i in range(steps):
        t_current = torch.tensor([i * dt], device=device)
        
        # A. 有条件预测 (看着内容图画)
        v_cond = model(x_t, cond_input, t_current, style_tensor)
        
        # B. 无条件预测 (盲画)
        v_uncond = model(x_t, uncond_input, t_current, style_tensor)
        
        # C. CFG 外推 (Extrapolation)
        # 公式: v_uncond + cfg_scale * (v_cond - v_uncond)
        # 作用: 放大"内容图"带来的特征，强力抑制模糊
        v_final = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        # 更新位置
        x_t = x_t + dt * v_final
    
    return x_t


@torch.no_grad()
def teleport_latent(content_latent, target_style_id, steps, model, device, noise_strength=1.0):
    """
    兼容性包装：调用新的 CFG 采样
    (保留旧接口以兼容现有代码)
    """
    return generate_style_transfer_cfg(
        content_latent, 
        target_style_id, 
        model, 
        device, 
        steps=steps, 
        cfg_scale=4.0  # 默认 CFG scale
    )


def load_vae_encoder_decoder():
    """加载SD1.5的VAE"""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae"
    )
    return vae


def image_to_latent(image_path, vae, device):
    """将图片编码为latent"""
    from torchvision import transforms
    
    img = Image.open(image_path).convert("RGB")
    img = img.resize((512, 512))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    vae = vae.to(device)
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()
        latent = latent * 0.18215
    
    return latent


def latent_to_image(latent, vae, device):
    """将latent解码为图片"""
    vae = vae.to(device)
    latent = latent / 0.18215
    
    with torch.no_grad():
        image = vae.decode(latent).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image)


def find_checkpoint(checkpoint_dir, prefer="best"):
    checkpoint_dir = Path(checkpoint_dir)
    if prefer == "best":
        best_path = checkpoint_dir / "SAF_best.pt"
        if best_path.exists():
            return best_path
    
    checkpoints = list(checkpoint_dir.glob("SAF_epoch*.pt"))
    if checkpoints:
        import re
        epoch_numbers = []
        for ckpt in checkpoints:
            match = re.search(r'epoch(\d+)', ckpt.name)
            if match:
                epoch_numbers.append((int(match.group(1)), ckpt))
        
        if epoch_numbers:
            latest = max(epoch_numbers, key=lambda x: x[0])
            return latest[1]
    return None


def load_model_from_checkpoint(checkpoint_path, model, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    return model


def main():
    # ========== 加载配置 ==========
    config = Config("config.json")
    
    model_cfg = config.model
    inf_cfg = config.inference
    ckpt_cfg = config.checkpoint
    
    # ========== 推理参数 ==========
    CHECKPOINT_PATH = "auto"
    INPUT_IMAGE = "test.jpg"
    TARGET_STYLE_ID = 1
    STEPS = inf_cfg.get('steps', 25)
    CFG_SCALE = inf_cfg.get('cfg_scale', 4.0)  # 🔴 新增 CFG 参数
    OUTPUT_ROOT = "inference_results"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========== 加载模型 ==========
    print("Loading SA-Flow v2 model...")
    model = SAFModel(**model_cfg).to(device)
    
    if CHECKPOINT_PATH == "auto":
        checkpoint_dir = Path(ckpt_cfg['save_dir'])
        checkpoint_path = find_checkpoint(checkpoint_dir, prefer="best")
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found in '{checkpoint_dir}' directory")
    else:
        checkpoint_path = Path(CHECKPOINT_PATH)
    
    model = load_model_from_checkpoint(checkpoint_path, model, device)
    model.eval()
    
    # ========== 加载VAE ==========
    print("Loading VAE...")
    vae = load_vae_encoder_decoder()
    
    # ========== 执行推理 ==========
    if not Path(INPUT_IMAGE).exists():
        print(f"⚠️ Input image {INPUT_IMAGE} not found. Please place a test image.")
        return

    print("Encoding input image...")
    content_latent = image_to_latent(INPUT_IMAGE, vae, device)
    
    input_base = Path(INPUT_IMAGE).stem
    style_str = f"style_{TARGET_STYLE_ID}"
    subdir = Path(OUTPUT_ROOT) / f"{input_base}_{style_str}_cfg{CFG_SCALE}"
    subdir.mkdir(parents=True, exist_ok=True)
    output_image_path = subdir / "output.jpg"
    
    print(f"Transferring to style {TARGET_STYLE_ID} with {STEPS} steps (CFG={CFG_SCALE})...")
    stylized_latent = generate_style_transfer_cfg(
        content_latent,
        TARGET_STYLE_ID,
        model,
        device,
        steps=STEPS,
        cfg_scale=CFG_SCALE
    )
    
    print("Decoding output image...")
    output_image = latent_to_image(stylized_latent, vae, device)
    output_image.save(output_image_path)
    print(f"✅ Saved result to {output_image_path}")
    print(f"💡 Tip: Try different CFG scales (3.0, 5.0, 7.0) for sharpness control")


if __name__ == "__main__":
    main()