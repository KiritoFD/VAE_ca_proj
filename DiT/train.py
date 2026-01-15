import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
from tqdm import tqdm
import glob
import re
import torch.cuda.amp as amp
from PIL import Image
import numpy as np
import random

from models.dit_model import DiTModel
from dataset import RandomPairDataset
from config import Config


def find_best_checkpoint(checkpoint_dir):
    """Find the best checkpoint (dit_best.pt)"""
    best_path = checkpoint_dir / "dit_best.pt"
    if best_path.exists():
        return best_path
    return None


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint by epoch number"""
    checkpoints = list(checkpoint_dir.glob("dit_epoch*.pt"))
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find the maximum
    epoch_numbers = []
    for ckpt in checkpoints:
        match = re.search(r'epoch(\d+)', ckpt.name)
        if match:
            epoch_numbers.append((int(match.group(1)), ckpt))
    
    if epoch_numbers:
        latest = max(epoch_numbers, key=lambda x: x[0])
        return latest[1]
    return None


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, device='cpu'):
    """
    Load checkpoint and return starting epoch and best loss
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full checkpoint and state_dict only
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from epoch {start_epoch} with loss {best_loss:.4f}")
    else:
        # Just state dict
        model.load_state_dict(checkpoint)
        start_epoch = 0
        best_loss = float('inf')
        print("Loaded model state dict only (no training state)")
    
    return start_epoch, best_loss


def compute_loss(model, batch, device):
    """
    计算Flow Matching损失（条件生成模式）
    注意：此函数内部不再需要手动转移到device，因为在autocast外部已经处理
    """
    x_content, x_style, style_label = batch
    x_content = x_content.to(device, non_blocking=True)  # 结构条件（与x_style相同）
    x_style = x_style.to(device, non_blocking=True)      # 重建目标
    style_label = style_label.to(device, non_blocking=True)
    
    batch_size = x_content.shape[0]
    
    # 1. 起点改为纯噪声（而不是内容图）
    x_0 = torch.randn_like(x_style)
    x_1 = x_style
    
    # 2. 采样随机时间 t ~ U(0, 1)
    t = torch.rand(batch_size, device=device)
    
    # 3. 线性插值：x_t = (1-t)*x_0 + t*x_1
    t_expanded = t[:, None, None, None]
    x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
    
    # 4. 计算真值速度：v_true = x_1 - x_0
    v_true = x_1 - x_0
    
    # 5. 模型预测速度
    v_pred = model(x_t, x_content, t, style_label)
    
    # 6. MSE损失
    loss = nn.functional.mse_loss(v_pred, v_true)
    
    return loss


def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler):
    """加入 AMP 的训练循环"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
    
    for step, batch in pbar:
        optimizer.zero_grad()
        
        # 开启混合精度上下文
        with torch.cuda.amp.autocast():
            loss = compute_loss(model, batch, device)
        
        # 使用 Scaler 进行反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪 (先 unscale)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # ================= [DEBUG START] =================
        # 每 100 个 batch 检查一次位置编码的梯度
        if step % 100 == 0 and step > 0:
            if hasattr(model, 'module'):
                pos_param = model.module.pos_embed
            else:
                pos_param = model.pos_embed
                
            if pos_param.grad is not None:
                grad_mean = pos_param.grad.abs().mean().item()
                if grad_mean < 1e-8:
                    print(f"\n⚠️ [WARNING] PosEmbed Gradient is extremely small: {grad_mean}")
            else:
                print("\n❌ [ERROR] PosEmbed has NO GRADIENT!")
        # ================= [DEBUG END] =================
        
        # 更新权重
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    
    # ================= [EPOCH DEBUG START] =================
    if hasattr(model, 'module'):
        pos_param = model.module.pos_embed
        patch_embed = model.module.patch_embed
    else:
        pos_param = model.pos_embed
        patch_embed = model.patch_embed
    
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        x_content, x_style, style_label = sample_batch
        x_content = x_content.to(device)
        x_in = torch.cat([x_content, x_content], dim=1)
        x_tokens = patch_embed(x_in)
        
        x_mean = x_tokens.abs().mean().item()
        x_std = x_tokens.std().item()
    model.train()
    
    pos_mean = pos_param.abs().mean().item()
    pos_std = pos_param.std().item()
    
    print(f"\n{'='*50}")
    print(f"[Epoch {epoch} Summary - Position Embedding Analysis]")
    print(f"Token Features:    Mean={x_mean:.4f}, Std={x_std:.4f}")
    print(f"Pos Embeddings:    Mean={pos_mean:.4f}, Std={pos_std:.4f}")
    print(f"Signal-to-Pos Ratio: {x_mean / (pos_mean + 1e-6):.2f}")
    if x_mean / (pos_mean + 1e-6) > 50:
        print(f"⚠️  Ratio > 50: Position info might be getting drowned out")
    else:
        print(f"✓  Ratio looks reasonable")
    print(f"{'='*50}\n")
    # ================= [EPOCH DEBUG END] =================
    
    return avg_loss


def run_inference_samples(model, vae, eval_configs, epoch, config, device):
    """
    支持多个评测配置：每个配置从指定目录随机选一张图片，对指定风格生成结果
    推理结果保存到规范化子文件夹：training_samples/epoch_{epoch:03d}/eval{eval_idx}/style_{style_id}/
    """
    from torchvision import transforms

    model.eval()
    root_dir = Path("training_samples")
    root_dir.mkdir(exist_ok=True)

    inf_cfg = config.inference
    steps = inf_cfg.get('steps', 50)
    noise_strength = inf_cfg.get('noise_strength', 0.8)

    for eval_idx, eval_cfg in enumerate(eval_configs):
        img_dir = eval_cfg["img_dir"]
        target_styles = eval_cfg["target_styles"]
        img_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            img_files.extend(glob.glob(os.path.join(img_dir, ext)))
        img_files = sorted(img_files)
        if not img_files:
            print(f"⚠️ No images found in {img_dir}, skipping eval {eval_idx}")
            continue
        img_path = random.choice(img_files)
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            latent = vae.encode(img_tensor).latent_dist.sample()
            latent = latent * 0.18215

        # 规范化输出目录
        eval_subdir = root_dir / f"epoch_{epoch:03d}" / f"eval{eval_idx}"
        eval_subdir.mkdir(parents=True, exist_ok=True)
        # 保存原图
        img.save(eval_subdir / "input.jpg")

        for style_id in target_styles:
            style_subdir = eval_subdir / f"style_{style_id}"
            style_subdir.mkdir(parents=True, exist_ok=True)
            style_tensor = torch.tensor([style_id], dtype=torch.long, device=device)
            noise = torch.randn_like(latent)
            x_t = latent * (1 - noise_strength) + noise * noise_strength
            cond_latent = latent
            dt = 1.0 / steps
            with torch.no_grad():
                for i in range(steps):
                    t_current = torch.tensor([i * dt], device=device)
                    velocity = model(x_t, cond_latent, t_current, style_tensor)
                    x_t = x_t + dt * velocity
                decoded = vae.decode(x_t / 0.18215).sample
            output = (decoded[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            output = (output * 255).astype(np.uint8)
            out_img = Image.fromarray(output)
            out_img.save(style_subdir / "output.jpg")
        print(f"✅ Saved eval {eval_idx} samples for epoch {epoch} to {eval_subdir}")

    model.train()


def main():
    # ========== 加载配置 ==========
    config = Config("config.json")
    config.print_config()
    
    # 解包配置
    model_cfg = config.model
    train_cfg = config.training
    data_cfg = config.data
    ckpt_cfg = config.checkpoint
    
    # 保存路径
    CHECKPOINT_DIR = Path(ckpt_cfg['save_dir'])
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if train_cfg['use_amp']:
        print(f"Mixed Precision Training (AMP): Enabled")
    
    # ========== 初始化 AMP Scaler ==========
    scaler = torch.cuda.amp.GradScaler() if train_cfg['use_amp'] else None
    
    # ========== 数据加载 ==========
    print("Loading dataset...")
    dataset = RandomPairDataset(
        data_cfg['content_dir'], 
        data_cfg['style_root'], 
        num_classes=data_cfg.get('num_classes')
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True
    )
    
    # ========== 模型初始化 ==========
    print("Initializing model...")
    model = DiTModel(**model_cfg).to(device)
    
    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # ========== 加载VAE（用于推理） ==========
    print("Loading VAE for inference...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae"
    ).to(device)
    vae.eval()
    
    # ========== 优化器 ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg['num_epochs'],
        eta_min=1e-6
    )
    
    # ========== 加载检查点 ==========
    start_epoch = 0
    best_loss = float('inf')
    
    resume_from = ckpt_cfg.get('resume_from')
    if resume_from:
        checkpoint_path = None
        
        if resume_from == "auto":
            checkpoint_path = find_best_checkpoint(CHECKPOINT_DIR)
            if checkpoint_path is None:
                checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
        elif resume_from == "best":
            checkpoint_path = find_best_checkpoint(CHECKPOINT_DIR)
        elif resume_from == "latest":
            checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
        
        if checkpoint_path and checkpoint_path.exists():
            start_epoch, best_loss = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler, device
            )
        else:
            print("No checkpoint found, starting from scratch")
    
    # 如果是从头开始，初始化scheduler
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
    
    # ========== 训练循环 ==========
    print(f"Starting training from epoch {start_epoch + 1}...")
    
    # 推理配置
    eval_configs = config._config.get("eval_samples", [])
    inference_every_n_epochs = train_cfg.get('inference_every_n_epochs', 5)
    
    for epoch in range(start_epoch + 1, train_cfg['num_epochs'] + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch, scaler)
        scheduler.step()
        print(f"Epoch {epoch}/{train_cfg['num_epochs']} - Avg Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        # ========== 每N个epoch运行推理 ==========
        if epoch % inference_every_n_epochs == 0 and eval_configs:
            run_inference_samples(model, vae, eval_configs, epoch, config, device)

        # 保存检查点
        save_every = ckpt_cfg.get('save_every_n_epochs', 10)
        if epoch % save_every == 0 or avg_loss < best_loss:
            checkpoint_path = CHECKPOINT_DIR / f"dit_epoch{epoch}_loss{avg_loss:.4f}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': avg_loss,
                'config': config._config,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = CHECKPOINT_DIR / "dit_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'loss': avg_loss,
                    'config': config._config,
                }, best_path)
                print(f"New best model saved: {best_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
