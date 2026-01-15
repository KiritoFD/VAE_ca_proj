import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np

from SAFlow import SAFModel
from config import Config


@torch.no_grad()
def teleport_latent(content_latent, target_style_id, steps, model, device, noise_strength=1.0):
    """
    使用欧拉法求解ODE，实现条件风格生成 (SA-Flow Mapping)
    
    Args:
        content_latent: [1, 4, 64, 64] - 内容图的latent（作为结构控制）
        target_style_id: int - 目标风格ID
        steps: int - 积分步数（推荐 10-20 步即可）
        model: 训练好的 SAFlowModel (SAFModel)
        device: cuda/cpu
        noise_strength: float - 噪声强度
    
    Returns:
        stylized_latent: [1, 4, 64, 64] - 风格化后的latent
    """
    model.eval()
    
    content_latent = content_latent.to(device)
    style_id = torch.tensor([target_style_id], dtype=torch.long, device=device)
    
    # 1. 起点
    noise = torch.randn_like(content_latent)
    x_t = content_latent * (1 - noise_strength) + noise * noise_strength
    
    # 2. 结构条件
    cond_latent = content_latent
    
    # 3. 计算步长
    dt = 1.0 / steps
    
    # 4. 欧拉法迭代
    for i in range(steps):
        # 当前时间
        t_current = torch.tensor([i * dt], device=device)
        
        # 预测速度: SA-Flow 会根据局部卷积特征保持结构
        velocity = model(x_t, cond_latent, t_current, style_id)
        
        # 更新位置 (标准欧拉步)
        x_t = x_t + dt * velocity
    
    return x_t


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
    TARGET_STYLE_ID = 1  # 目标风格ID
    STEPS = inf_cfg.get('steps', 20)      # SA-Flow 步数少，20步足够
    NOISE_STRENGTH = inf_cfg.get('noise_strength', 0.8)
    OUTPUT_ROOT = "inference_results"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========== 加载模型 ==========
    print("Loading SA-Flow model...")
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
    subdir = Path(OUTPUT_ROOT) / f"{input_base}_{style_str}"
    subdir.mkdir(parents=True, exist_ok=True)
    output_image_path = subdir / "output.jpg"
    
    print(f"Transferring to style {TARGET_STYLE_ID} with {STEPS} steps...")
    stylized_latent = teleport_latent(
        content_latent,
        TARGET_STYLE_ID,
        STEPS,
        model,
        device,
        noise_strength=NOISE_STRENGTH
    )
    
    print("Decoding output image...")
    output_image = latent_to_image(stylized_latent, vae, device)
    output_image.save(output_image_path)
    print(f"Saved result to {output_image_path}")


if __name__ == "__main__":
    main()