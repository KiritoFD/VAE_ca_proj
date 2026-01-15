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

from SAFlow import SAFModel
from dataset import RandomPairDataset
from config import Config


def find_best_checkpoint(checkpoint_dir):
    """Find the best checkpoint (SAF_best.pt)"""
    best_path = checkpoint_dir / "SAF_best.pt"
    if best_path.exists():
        return best_path
    return None


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint by epoch number"""
    checkpoints = list(checkpoint_dir.glob("SAF_epoch*.pt"))
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
    计算Flow Matching损失（内容到风格的映射模式）
    同时返回诊断指标：方向对齐度、强度比、结构保留度
    """
    x_content, x_style, style_label = batch
    x_content = x_content.to(device, non_blocking=True)
    x_style = x_style.to(device, non_blocking=True)
    style_label = style_label.to(device, non_blocking=True)
    
    batch_size = x_content.shape[0]
    
    # 🔴 核心修改：映射模式 - 从内容流形到风格流形
    x_0 = x_content  # 起点：内容图
    x_1 = x_style    # 终点：风格图
    
    # 采样随机时间 t ~ U(0, 1)
    t = torch.rand(batch_size, device=device)
    
    # 线性插值：x_t = (1-t)*x_0 + t*x_1
    t_expanded = t[:, None, None, None]
    # 添加微量噪声增强数值稳定性
    x_t = (1 - t_expanded) * x_0 + t_expanded * x_1 + torch.randn_like(x_0) * 0.01
    
    # 真值速度场：v_true = x_1 - x_0
    v_true = x_1 - x_0
    
    # 模型预测速度
    v_pred = model(x_t, x_content, t, style_label)
    
    # MSE损失
    loss = nn.functional.mse_loss(v_pred, v_true)
    
    # --- 🔍 诊断指标计算 ---
    with torch.no_grad():
        # 1. 方向对齐度 (Cosine Similarity)
        # 越接近1.0说明模型知道"往哪画"
        v_true_flat = v_true.reshape(batch_size, -1)
        v_pred_flat = v_pred.reshape(batch_size, -1)
        cos_sim = nn.functional.cosine_similarity(v_true_flat, v_pred_flat, dim=1).mean().item()
        
        # 2. 强度比 (Magnitude Ratio)
        # 越接近1.0说明画得越"用力"，低于0.5会导致模糊
        mag_true = torch.norm(v_true_flat, dim=1).mean().item()
        mag_pred = torch.norm(v_pred_flat, dim=1).mean().item()
        mag_ratio = mag_pred / (mag_true + 1e-6)
        
        # 3. 结构保留度
        # 确保预测改动没有破坏原图结构
        struct_corr = nn.functional.cosine_similarity(
            x_content.reshape(batch_size, -1), 
            v_pred.reshape(batch_size, -1),
            dim=1
        ).mean().item()
        
        # 4. 速度场标准差
        v_true_std = v_true.std().item()

    debug_info = {
        "cos_sim": cos_sim,      # 方向是否正确
        "mag_ratio": mag_ratio,  # 强度是否足够（关键！）
        "struct_corr": struct_corr,
        "v_true_std": v_true_std
    }
    
    return loss, debug_info


def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler):
    """增强版训练循环：实时监控速度场质量"""
    model.train()
    total_metrics = {"loss": 0.0, "cos_sim": 0.0, "mag_ratio": 0.0, "struct_corr": 0.0}
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
    
    for step, batch in pbar:
        # 🔴 第一个batch的数据检查
        if step == 0 and epoch == 1:
            x_content, x_style, _ = batch
            print(f"\n🔍 DEBUG: Data Statistics Check:")
            print(f"   Content - Mean: {x_content.mean():.4f}, Std: {x_content.std():.4f}, Range: [{x_content.min():.4f}, {x_content.max():.4f}]")
            print(f"   Style   - Mean: {x_style.mean():.4f}, Std: {x_style.std():.4f}, Range: [{x_style.min():.4f}, {x_style.max():.4f}]")
            expected_std = 0.18215
            if abs(x_content.std() - expected_std) > 0.1:
                print(f"   ⚠️  WARNING: Std should be ~{expected_std}, but got {x_content.std():.4f}")
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with torch.cuda.amp.autocast():
            loss, debug = compute_loss(model, batch, device)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # 累计指标
        total_metrics["loss"] += loss.item()
        total_metrics["cos_sim"] += debug["cos_sim"]
        total_metrics["mag_ratio"] += debug["mag_ratio"]
        total_metrics["struct_corr"] += debug["struct_corr"]
        
        # 🔴 实时监控进度条 - 关键指标
        pbar.set_postfix({
            "L": f"{loss.item():.4f}",
            "Cos": f"{debug['cos_sim']:.3f}",  # 方向对吗？
            "Mag": f"{debug['mag_ratio']:.3f}"  # 够清晰吗？（关键！）
        })
        
        # 每100步打印详细统计
        if step > 0 and step % 100 == 0:
            print(f"\n[Step {step}] V_True_Std: {debug['v_true_std']:.4f} | Structure_Corr: {debug['struct_corr']:.3f}")
    
    avg_metrics = {k: v / len(dataloader) for k, v in total_metrics.items()}
    return avg_metrics


def run_inference_samples(model, vae, eval_configs, epoch, config, device):
    """
    运行推理采样
    """
    from torchvision import transforms

    model.eval()
    root_dir = Path("training_samples")
    root_dir.mkdir(exist_ok=True)

    inf_cfg = config.inference
    steps = inf_cfg.get('steps', 20) # SA-Flow 推荐 20 步
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
            # 注意：SA-Flow 依然使用 SD1.5 的 Latent 空间，保持 scaling
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
            
            # 准备初始噪声
            noise = torch.randn_like(latent)
            x_t = latent * (1 - noise_strength) + noise * noise_strength
            cond_latent = latent
            
            dt = 1.0 / steps
            with torch.no_grad():
                for i in range(steps):
                    t_current = torch.tensor([i * dt], device=device)
                    # 调用 SA-Flow 模型
                    velocity = model(x_t, cond_latent, t_current, style_tensor)
                    # 欧拉积分
                    x_t = x_t + dt * velocity
                
                # 解码
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
    print("Initializing SA-Flow Model...")
    # 注意：这里的 SAFModel 类其实已经是 SA-Flow 架构了
    model = SAFModel(**model_cfg).to(device)
    
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
    
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
    
    # ========== 训练循环 ==========
    print(f"Starting training from epoch {start_epoch + 1}...")
    
    eval_configs = config._config.get("eval_samples", [])
    inference_every_n_epochs = train_cfg.get('inference_every_n_epochs', 5)
    
    for epoch in range(start_epoch + 1, train_cfg['num_epochs'] + 1):
        avg_metrics = train_one_epoch(model, dataloader, optimizer, device, epoch, scaler)
        scheduler.step()
        
        # 🔴 输出完整训练指标
        print(f"\nEpoch {epoch}/{train_cfg['num_epochs']} Summary:")
        print(f"  Loss: {avg_metrics['loss']:.4f}")
        print(f"  Direction (CosSim): {avg_metrics['cos_sim']:.3f} {'✅' if avg_metrics['cos_sim'] > 0.7 else '⚠️'}")
        print(f"  Strength (MagRatio): {avg_metrics['mag_ratio']:.3f} {'✅ Clear' if avg_metrics['mag_ratio'] > 0.7 else '⚠️ Blurry'}")
        print(f"  Structure (Corr): {avg_metrics['struct_corr']:.3f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # 每N个epoch运行推理
        if epoch % inference_every_n_epochs == 0 and eval_configs:
            run_inference_samples(model, vae, eval_configs, epoch, config, device)

        # 保存检查点
        save_every = ckpt_cfg.get('save_every_n_epochs', 10)
        if epoch % save_every == 0 or avg_metrics['loss'] < best_loss:
            checkpoint_path = CHECKPOINT_DIR / f"SAF_epoch{epoch}_loss{avg_metrics['loss']:.4f}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': avg_metrics['loss'],
                'metrics': avg_metrics,  # 🔴 保存诊断指标
                'config': config._config,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            if avg_metrics['loss'] < best_loss:
                best_loss = avg_metrics['loss']
                best_path = CHECKPOINT_DIR / "SAF_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'loss': avg_metrics['loss'],
                    'metrics': avg_metrics,
                    'config': config._config,
                }, best_path)
                print(f"New best model saved: {best_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()