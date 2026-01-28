import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from dataset import LatentDataset
from inference import decode_latent, encode_image, load_vae
from losses import (
    GeometricFreeEnergyLoss,
    StructureAnchoredLoss,
    VelocityRegularizationLoss,
    VelocitySmoothnessLoss,
    TrajectoryMSELoss,
)
from model import LGTUNet, count_parameters
from physics import generate_latent, get_dynamic_epsilon, integrate_ode, invert_latent

logger = logging.getLogger(__name__)


class RunningMean:
    """Simple running mean for adaptive loss normalization."""
    def __init__(self, momentum=0.99, device='cuda'):
        self.mean = torch.tensor(1.0, device=device)
        self.momentum = momentum

    def update(self, val):
        # Detach to prevent gradient tracking on the normalizer itself
        val_detached = val.detach()
        self.mean = self.momentum * self.mean + (1 - self.momentum) * val_detached
        return max(self.mean.item(), 1e-6)  # Avoid division by zero


class LGTTrainer:
    """Trainer orchestrating optimization, losses, and evaluation."""

    def __init__(self, config: Dict, device: torch.device = torch.device('cuda'), config_path: Optional[str] = None):
        self.config = config
        self.device = device
        self.config_path = Path(config_path) if config_path is not None else None

        # 🔥 Infra Optimization: Enable Tensor Cores (Ampere+)
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

        self.model = LGTUNet(
            latent_channels=config['model']['latent_channels'],
            base_channels=config['model']['base_channels'],
            style_dim=config['model']['style_dim'],
            time_dim=config['model']['time_dim'],
            num_styles=config['model']['num_styles'],
            num_encoder_blocks=config['model']['num_encoder_blocks'],
            num_decoder_blocks=config['model']['num_decoder_blocks'],
        ).to(device, memory_format=torch.channels_last)  # 🔥 Infra Optimization: Channels Last
        self.model.compute_avg_style_embedding()

        if config['training'].get('use_compile', False):
            try:
                self.model = torch.compile(self.model, mode='default', fullgraph=False)
                logger.info("✓ Model compiled with torch.compile")
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(f"torch.compile failed: {exc}")

        logger.info(f"Model parameters: {count_parameters(self.model):,}")

        self.energy_loss = GeometricFreeEnergyLoss(
            num_styles=config['model']['num_styles'],
            w_style=config['loss']['w_style'],
            swd_scales=config['loss'].get('swd_scales', [1, 3, 5, 7, 15]),
            swd_scale_weights=config['loss'].get('swd_scale_weights', [1.0, 5.0, 5.0, 5.0, 3.0]),
            num_projections=config['loss'].get('num_projections', 64),
            max_samples=config['loss'].get('max_samples', 4096),
        ).to(device)

        # 找到这一段（约 53 行）并替换：
        # ✅ 修复：优先读取 'w_mse'，并将默认值降为 1.0
        self.structure_weight = config['loss'].get('w_mse') or config['loss'].get('structure_weight', 1.0)

        # ✅ 修复：将 edge_boost 默认值从 9.0 降为 1.5
        edge_boost = config['loss'].get('edge_boost', 1.5)

        self.struc_loss = StructureAnchoredLoss(
            weight=1.0,  # 🔥 Note: Weight is now applied in trainer, this is a dummy value
            edge_boost=edge_boost,
        ).to(device)

        logger.info(
            f"✓ Laplacian Structure Lock enabled (weight={self.structure_weight}, edge_boost={edge_boost})"
        )

        self.layout_weight = config['loss'].get('w_layout', 2.0)
        self.layout_loss = TrajectoryMSELoss(weight=1.0).to(device)  # 🔥 Note: Weight is now applied in trainer
        logger.info(f"✓ Layout Preservation enabled (weight={self.layout_weight})")

        vel_smooth_weight = config['loss'].get('vel_smooth_weight', 0.2)
        self.vel_smooth_loss = VelocitySmoothnessLoss(weight=vel_smooth_weight).to(device)
        logger.info(f"✓ Velocity Smoothness Loss enabled (weight={vel_smooth_weight})")

        self.use_velocity_reg = config['loss'].get('use_velocity_reg', False)
        self.vel_reg_loss = None
        if self.use_velocity_reg:
            vel_reg_weight = config['loss'].get('vel_reg_weight', 0.1)
            self.vel_reg_loss = VelocityRegularizationLoss(weight=vel_reg_weight).to(device)
            logger.info(f"✓ Velocity Regularization enabled (weight={vel_reg_weight})")
        else:
            logger.info("Velocity Regularization disabled")

        self.num_epochs = config['training']['num_epochs']

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-5),
            betas=(0.9, 0.999),
            fused=False,  # 🔥 Fix: Fused AdamW conflicts with GradScaler.unscale_
        )

        # 🔥 Scheduler Strategy: Linear Warmup -> Cosine Decay
        # Replaces manual warmup loop for a global, holistic strategy
        if 'warmup_epochs' in config['training']:
            self.warmup_epochs = config['training']['warmup_epochs']
        else:
            p_cfg = config['training'].get('phased_training', {})
            self.warmup_epochs = max(1, int(self.num_epochs * p_cfg.get('lr_warmup_ratio', 0.05)))
        min_lr = config['training'].get('min_learning_rate', 1e-6)

        scheduler_warmup = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_epochs
        )
        scheduler_cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.num_epochs - self.warmup_epochs),
            eta_min=min_lr
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[self.warmup_epochs]
        )

        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # 🔥 Adaptive Loss Normalization (Self-Learning Scales)
        self.use_adaptive_norm = config['training'].get('use_adaptive_loss_norm', False)
        if self.use_adaptive_norm:
            self.norm_mse = RunningMean(device=device)
            self.norm_style = RunningMean(device=device)
            logger.info("✓ Adaptive Loss Normalization enabled (Running Mean Scaling)")

        self.label_drop_prob = config['training'].get('label_drop_prob', 0.1)
        self.use_avg_for_uncond = config['training'].get('use_avg_style_for_uncond', True)
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)
        self.effective_batch_size = config['training']['batch_size'] * self.accumulation_steps

        if self.accumulation_steps > 1:
            logger.info(
                f"Gradient accumulation: {self.accumulation_steps} steps | "
                f"Effective batch size: {self.effective_batch_size}"
            )

        self.global_step = 0
        self.alpha_warmup_steps = config['training'].get('alpha_warmup_steps', 1000)
        logger.info(f"Alpha warmup schedule: 0 → 1.0 over {self.alpha_warmup_steps} steps")

        self.epsilon = config['training'].get('epsilon', 0.01)
        self.ode_steps = config['training'].get('ode_integration_steps', 5)

        self.style_weights = torch.tensor(
            config['loss'].get('style_weights', [1.0] * config['model']['num_styles']),
            device=device,
            dtype=torch.float32,
        )
        logger.info(f"Per-style loss weights: {self.style_weights.tolist()}")

        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = config['training'].get('save_interval', 5)

        self.eval_interval = config['training'].get('eval_interval', 5)
        self.full_eval_interval = config['training'].get('full_eval_interval', self.eval_interval)
        logger.info(f"Full external evaluation interval: {self.full_eval_interval}")
        self.test_image_dir = Path(config['training'].get('test_image_dir', 'test_images'))
        self.inference_dir = self.checkpoint_dir / 'inference'
        self.inference_dir.mkdir(exist_ok=True)

        self.vae = load_vae(device)

        self.style_indices_cache = None
        self.dataset_ref: Optional[LatentDataset] = None

        self.log_dir = self.checkpoint_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_file, 'w') as f:
            f.write('epoch,loss_total,loss_style_swd,loss_style_swd_weighted,loss_mse,loss_mse_weighted,loss_layout,loss_mse_raw,learning_rate,epoch_time\n')

        self.start_epoch = 1
        self._maybe_resume(config['training'].get('resume_checkpoint'))

        from diffusers import DDPMScheduler

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )

    # ------------------------------------------------------------------
    # Initialization utilities
    # ------------------------------------------------------------------
    def _maybe_resume(self, resume_checkpoint: Optional[str]) -> None:
        latest_ckpt = None
        if resume_checkpoint:
            latest_ckpt = Path(resume_checkpoint)
            logger.info(f"Overriding resume checkpoint: {resume_checkpoint}")
        else:
            latest_ckpt = find_latest_checkpoint(self.checkpoint_dir)

        if latest_ckpt is None:
            logger.info("No checkpoint found; starting fresh")
            return

        try:
            resume_info = load_checkpoint(
                checkpoint_path=latest_ckpt,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                current_config=self.config,
                device=self.device,
            )
            self.start_epoch = resume_info.get('start_epoch', 1)
            self.global_step = resume_info.get('global_step', 0)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error(f"Failed to resume from {latest_ckpt}: {exc}")

        # Note: We rely on the loaded scheduler state to determine the correct LR.
        # Forcing LR to config['learning_rate'] here would reset it to max_lr,
        # which is incorrect if resuming in the middle of training.
        logger.info(f"✓ Resumed training state. Current LR will be determined by scheduler.")

    def build_style_indices_cache(self, dataset: LatentDataset) -> None:
        logger.info("Building style indices cache...")
        self.style_indices_cache = dataset.style_indices
        for style_id, indices in self.style_indices_cache.items():
            logger.info(f"  Style {style_id}: {len(indices)} samples")
        self.dataset_ref = dataset
        logger.info("✓ Style indices cache built")

    def sample_style_batch(self, target_style_ids: torch.Tensor) -> torch.Tensor:
        if self.dataset_ref is None:
            raise RuntimeError("Dataset reference is not set; call build_style_indices_cache first")

        device = target_style_ids.device
        b = target_style_ids.shape[0]
        style_latents = torch.empty((b, *self.dataset_ref.latents_tensor.shape[1:]), device=device)
        target_cpu = target_style_ids.detach().cpu()

        for style_id in range(self.config['model']['num_styles']):
            if style_id not in self.style_indices_cache:
                continue
            indices = self.style_indices_cache[style_id]
            mask = target_cpu == style_id
            count = int(mask.sum().item())
            if count == 0:
                continue
            # Convert list to tensor for random indexing
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            rand_indices = indices_tensor[torch.randint(len(indices), (count,))]
            selected = self.dataset_ref.latents_tensor[rand_indices]
            style_latents[mask.to(device)] = selected.to(device, non_blocking=True)

        return style_latents

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def _get_current_alpha(self) -> float:
        return min(self.global_step / max(self.alpha_warmup_steps, 1), 1.0)

    def _update_lora_alpha(self, global_step: int) -> None:
        current_alpha = min(global_step / max(self.alpha_warmup_steps, 1), 1.0)
        for module in self.model.modules():
            if hasattr(module, 'alpha'):
                module.alpha = current_alpha
        if global_step % 250 == 0 and global_step < self.alpha_warmup_steps:
            logger.debug(f"Alpha warmup: step {global_step}/{self.alpha_warmup_steps}, alpha={current_alpha:.4f}")

    def _get_thermodynamic_schedule(self, epoch: int):
        """
        🔥 Thermodynamic Cosine Schedule (Global & Smooth)
        
        Replaces the crude piecewise linear schedule with physics-inspired cosine curves.
        - Structure (Solid Phase): Cosine Decay (High -> Low)
        - Style (Liquid Phase): Cosine Warmup (Low -> High)
        """
        p_cfg = self.config['training'].get('phased_training', {})
        if not p_cfg:
            return 1.0, 1.0, 1.0

        # Normalized time t in [0, 1]
        t = min(epoch / self.num_epochs, 1.0)
        
        # 1. Structure Schedule: Cosine Decay
        # Starts strong to lock content, decays to allow artistic deformation
        mse_start = p_cfg.get('m_mse_start', 2.0)
        mse_end = p_cfg.get('m_mse_end', 0.5)
        m_mse = mse_end + 0.5 * (mse_start - mse_end) * (1 + np.cos(t * np.pi))
        
        # Layout follows MSE but usually stays a bit higher to keep composition
        m_layout = m_mse * p_cfg.get('layout_ratio', 0.8)

        # 2. Style Schedule: Cosine Warmup
        # Starts weak to prevent noise overfitting, rises to dominate late training
        style_start = p_cfg.get('m_style_start', 0.1)
        style_end = p_cfg.get('m_style_end', 2.0)
        m_style = style_start + 0.5 * (style_end - style_start) * (1 - np.cos(t * np.pi))

        return m_mse, m_layout, m_style

    def compute_energy_loss(self, batch: Dict, epoch: int, multipliers: tuple = (1.0, 1.0, 1.0)) -> Dict[str, torch.Tensor]:
        device = self.device
        m_mse, m_layout, m_style = multipliers

        # 🔥 Infra Optimization: Channels Last for faster Convolutions
        latent = batch['latent'].to(device, non_blocking=True, memory_format=torch.channels_last)
        style_id = batch['style_id'].to(device, non_blocking=True)
        latent_deformed = batch.get('latent_deformed')
        if latent_deformed is not None:
            latent_deformed = latent_deformed.to(device, non_blocking=True, memory_format=torch.channels_last)

        b = latent.shape[0]
        indices = torch.randperm(b, device=device)
        style_id_tgt = style_id[indices]
        style_latents = self.sample_style_batch(style_id_tgt)

        use_elastic = self.config['training'].get('use_elastic_deform', False)
        elastic_styles = self.config['training'].get('elastic_styles', [1])
        if use_elastic and latent_deformed is not None:
            mask = torch.zeros_like(style_id_tgt, dtype=torch.bool)
            for s in elastic_styles:
                mask |= style_id_tgt == s
            x_src_target = torch.where(mask.view(-1, 1, 1, 1), latent_deformed, latent)
        else:
            x_src_target = latent

        x0 = torch.randn_like(latent)

        sigma_noise = 0.1
        noise_injection = torch.randn_like(x_src_target) * sigma_noise
        x_target_noisy = x_src_target + noise_injection

        epsilon_dynamic = get_dynamic_epsilon(
            epoch=epoch,
            target_epsilon=self.epsilon,
            warmup_epochs=self.config['training'].get('epsilon_warmup_epochs', 10),
        )
        t = torch.rand(b, device=device) * (1 - epsilon_dynamic) + epsilon_dynamic

        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        _ = (t * (num_train_timesteps - 1)).long()  # retained for schedule parity
        t_expand = t.view(-1, 1, 1, 1)

        x_t = (1 - t_expand) * x0 + t_expand * x_target_noisy

        drop_label = False
        if self.model.training and torch.rand(1).item() < self.label_drop_prob:
            drop_label = True

        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            v_pred = self.model(x_t, t, style_id_tgt, use_avg_style=drop_label)

        with torch.no_grad():
            # 🔥 修复：Flow Matching 的真实目标是位移矢量，而非噪声本身
            # v_target = (x_target_noisy - x0) 是从初始噪声到目标艺术风格的向量场
            # 这与 Diffusion（预测噪声）的范式本质不同，解决"色块"的根本原因
            v_target = x_target_noisy - x0  # [B, C, H, W] 位移矢量

        # 从config读取structure_warmup_epochs，默认150个epoch缓慢施加结构锁，之后保持全强度
        structure_warmup = self.config['training'].get('structure_warmup_epochs') or self.config['loss'].get('structure_warmup_epochs', 150)
        loss_mse_unweighted = self.struc_loss(v_pred, v_target, latent, current_epoch=epoch, total_warmup_epochs=structure_warmup)
        
        # Adaptive Normalization for MSE
        norm_factor_mse = self.norm_mse.update(loss_mse_unweighted) if self.use_adaptive_norm else 1.0
        loss_mse = (self.structure_weight * m_mse / norm_factor_mse) * loss_mse_unweighted

        # Layout Loss (Low-frequency structure)
        loss_layout_unweighted = self.layout_loss(v_pred, v_target)
        loss_layout = self.layout_weight * m_layout * loss_layout_unweighted
        loss_mse_raw = F.mse_loss(v_pred, v_target).detach()

        # 🔥 Optimization: Reuse v_pred if steps=1 (Compute Reuse)
        # If we are conditional (not dropping label), v_pred IS the velocity we want.
        # Saves 1 full forward pass -> ~2x speedup for this step.
        if self.ode_steps == 1 and not drop_label:
            # Euler integration: x1 = xt + v * (1-t)
            dt = (1.0 - t).view(-1, 1, 1, 1)
            x_1 = x_t + v_pred * dt
        else:
            x_1 = integrate_ode(
                model=self.model,
                x_t=x_t,
                t_start=t,
                style_id=style_id_tgt,
                steps=self.ode_steps,
                use_checkpoint=self.config['training'].get('use_gradient_checkpointing', True),
                use_amp=self.use_amp,
                amp_dtype=torch.bfloat16,
                training=self.model.training,
            )
            
        loss_dict = self.energy_loss(x_1, style_latents, style_ids=style_id_tgt)

        style_weight_batch = self.style_weights[style_id_tgt].mean()
        loss_smooth = self.vel_smooth_loss(v_pred)

        # Adaptive Normalization for Style
        norm_factor_style = self.norm_style.update(loss_dict['style_swd']) if self.use_adaptive_norm else 1.0
        
        loss_style_weighted = style_weight_batch * (self.energy_loss.w_style * m_style / norm_factor_style) * loss_dict['style_swd']
        loss_dict['style_swd_weighted'] = loss_style_weighted
        loss_dict['mse_weighted'] = loss_mse
        loss_dict['layout'] = loss_layout
        loss_dict['mse_raw'] = loss_mse_raw

        total = loss_style_weighted + loss_mse + loss_smooth + loss_layout

        if self.use_velocity_reg and self.vel_reg_loss is not None:
            loss_reg = self.vel_reg_loss(v_pred)
            loss_dict['velocity_reg'] = loss_reg
            total = total + loss_reg

        loss_dict['smooth'] = loss_smooth
        loss_dict['mse'] = loss_mse
        loss_dict['total'] = total
        loss_dict['style_id_tgt'] = style_id_tgt

        return loss_dict

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()

        if self.style_indices_cache is None:
            self.build_style_indices_cache(dataloader.dataset)

        total_loss = 0.0
        total_style_swd = 0.0
        total_style_swd_weighted = 0.0
        total_mse = 0.0
        total_mse_weighted = 0.0
        total_layout = 0.0
        total_mse_raw = 0.0
        total_smooth = 0.0
        total_vel_reg = 0.0
        num_batches = 0
        accum_counter = 0

        import sys as _sys

        use_tqdm = _sys.stderr.isatty()
        from tqdm import tqdm

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.num_epochs}", disable=not use_tqdm, leave=use_tqdm)
        self.optimizer.zero_grad(set_to_none=True)

        # 🔥 使用新的热力学调度器
        multipliers = self._get_thermodynamic_schedule(epoch)
        

        for step_idx, batch in enumerate(pbar, start=1):
            torch.compiler.cudagraph_mark_step_begin()

            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                loss_dict = self.compute_energy_loss(batch, epoch, multipliers=multipliers)
                loss = loss_dict['total'] / self.accumulation_steps

            self.scaler.scale(loss).backward()
            accum_counter += 1

            if accum_counter >= self.accumulation_steps:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                self._update_lora_alpha(self.global_step)
                self.global_step += 1

            total_loss += loss.item() * self.accumulation_steps
            total_style_swd += loss_dict['style_swd'].item()
            total_style_swd_weighted += loss_dict['style_swd_weighted'].item()
            total_mse += loss_dict['mse'].item()
            total_mse_weighted += loss_dict['mse_weighted'].item()
            total_layout += loss_dict['layout'].item()
            total_mse_raw += loss_dict['mse_raw'].item()
            if 'smooth' in loss_dict:
                total_smooth += loss_dict['smooth'].item()
            if 'velocity_reg' in loss_dict:
                total_vel_reg += loss_dict['velocity_reg'].item()
            num_batches += 1

            if use_tqdm:
                postfix_dict = {
                    'loss': f"{loss.item():.4f}",
                    'swd': f"{loss_dict['style_swd'].item():.4f}",
                    'swd*w': f"{loss_dict['style_swd_weighted'].item():.4f}",
                    'mse': f"{loss_dict['mse'].item():.4f}",
                    'mse*w': f"{loss_dict['mse_weighted'].item():.4f}",
                    'mse_raw': f"{loss_dict['mse_raw'].item():.4f}",
                    'α': f"{self._get_current_alpha():.3f}",
                }
                pbar.set_postfix(postfix_dict)

        avg_loss = total_loss / max(num_batches, 1)
        avg_style_swd = total_style_swd / max(num_batches, 1)
        avg_style_swd_weighted = total_style_swd_weighted / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)
        avg_mse_weighted = total_mse_weighted / max(num_batches, 1)
        avg_layout = total_layout / max(num_batches, 1)
        avg_mse_raw = total_mse_raw / max(num_batches, 1)
        avg_smooth = total_smooth / max(num_batches, 1)
        avg_vel_reg = total_vel_reg / max(num_batches, 1) if self.use_velocity_reg else 0.0

        metrics = {
            'loss': avg_loss,
            'style_swd': avg_style_swd,
            'style_swd_weighted': avg_style_swd_weighted,
            'mse': avg_mse,
            'mse_weighted': avg_mse_weighted,
            'layout': avg_layout,
            'mse_raw': avg_mse_raw,
            'smooth': avg_smooth,
            'num_batches': num_batches,
        }
        if self.use_velocity_reg:
            metrics['velocity_reg'] = avg_vel_reg

        return metrics

    # ------------------------------------------------------------------
    # Evaluation utilities
    # ------------------------------------------------------------------
    def get_test_images_by_style(self) -> Dict[int, tuple]:
        test_dir = Path(self.test_image_dir)
        if not test_dir.exists():
            logger.warning(f"Test image directory not found: {test_dir}")
            return {}

        style_images = {}
        style_subdirs = self.config['data'].get('style_subdirs', [])

        for style_id, style_name in enumerate(style_subdirs):
            style_dir = test_dir / style_name
            if not style_dir.exists():
                logger.warning(f"Style directory not found: {style_dir}")
                continue

            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            images = []
            for ext in image_extensions:
                images.extend(style_dir.glob(f"*{ext}"))
                images.extend(style_dir.glob(f"*{ext.upper()}"))

            if images:
                test_image = sorted(images)[0]
                style_images[style_id] = (style_name, test_image)
                logger.info(f"  Test image for {style_name}: {test_image.name}")

        return style_images

    @torch.no_grad()
    def evaluate_and_infer(self, epoch: int) -> None:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Inference Evaluation (Epoch {epoch})")
        logger.info(f"{'='*80}")

        epoch_inference_dir = self.inference_dir / ('epoch_-1' if epoch == -1 else f"epoch_{epoch:04d}")
        epoch_inference_dir.mkdir(parents=True, exist_ok=True)

        test_images = self.get_test_images_by_style()
        if not test_images:
            logger.warning("No test images found. Skipping inference.")
            return

        self.model.eval()
        num_styles = self.config['model']['num_styles']

        temp_ckpt = self.checkpoint_dir / "temp_eval.pt"
        torch.save(self.model.state_dict(), temp_ckpt)

        try:
            for src_style_id, (src_style_name, src_image_path) in test_images.items():
                logger.info(f"\nProcessing source: {src_style_name}")
                try:
                    image = Image.open(src_image_path).convert('RGB')
                    image = image.resize((256, 256))
                    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
                    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
                    image_tensor = image_tensor * 2.0 - 1.0
                    image_tensor = image_tensor.to(self.device)
                except Exception as exc:  # pragma: no cover - IO heavy
                    logger.error(f"Failed to load {src_image_path}: {exc}")
                    continue

                latent_src = encode_image(self.vae, image_tensor, self.device)

                if epoch == -1:
                    try:
                        image_orig = decode_latent(self.vae, latent_src, self.device)
                        output_filename = f"{src_style_name}_original.jpg"
                        output_path = epoch_inference_dir / output_filename
                        from torchvision.utils import save_image

                        save_image(image_orig, output_path)
                        logger.info(f"    ✓ Saved original: {output_filename}")
                    except Exception as exc:  # pragma: no cover - IO heavy
                        logger.error(f"    ✗ Failed to save original: {exc}")
                    continue

                for tgt_style_id in range(num_styles):
                    tgt_style_name = self.config['data'].get('style_subdirs', [])[tgt_style_id]
                    logger.info(f"  → {src_style_name} to {tgt_style_name}")
                    try:
                        latent_x0 = invert_latent(self.model, latent_src, src_style_id)
                        latent_tgt = generate_latent(self.model, latent_x0, tgt_style_id)
                        image_out = decode_latent(self.vae, latent_tgt, self.device)
                        output_filename = f"{src_style_name}_to_{tgt_style_name}.jpg"
                        output_path = epoch_inference_dir / output_filename
                        from torchvision.utils import save_image

                        save_image(image_out, output_path)
                        logger.info(f"    ✓ Saved: {output_filename}")
                    except Exception as exc:  # pragma: no cover - IO heavy
                        logger.error(f"    ✗ Failed: {exc}")
                        continue
        finally:
            if temp_ckpt.exists():
                temp_ckpt.unlink()

        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Inference completed. Results saved to: {epoch_inference_dir}")
        logger.info(f"{'='*80}\n")

    def run_full_evaluation(self, epoch: int, timeout: int = 3600) -> None:
        logger.info(f"Starting full external evaluation for epoch {epoch}")

        ckpt_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        temp_ckpt = None
        if not ckpt_path.exists():
            temp_ckpt = self.checkpoint_dir / f"epoch_{epoch:04d}_eval_temp.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'config': self.config,
                'metrics': {},
            }
            try:
                torch.save(checkpoint, temp_ckpt)
                ckpt_to_use = temp_ckpt
                logger.info(f"Saved temporary checkpoint for evaluation: {temp_ckpt}")
            except Exception as exc:  # pragma: no cover - IO heavy
                logger.error(f"Failed to write temporary checkpoint for evaluation: {exc}")
                return
        else:
            ckpt_to_use = ckpt_path

        eval_out_dir = self.checkpoint_dir / 'full_eval' / f'epoch_{epoch:04d}'
        eval_out_dir.mkdir(parents=True, exist_ok=True)

        run_script = Path(__file__).resolve().parent / 'run_evaluation.py'
        cmd = [sys.executable, str(run_script), '--checkpoint', str(ckpt_to_use), '--output', str(eval_out_dir)]
        if self.config_path is not None:
            cmd += ['--config', str(self.config_path)]
        num_steps = self.config.get('inference', {}).get('num_steps', None)
        if num_steps is not None:
            cmd += ['--num_steps', str(num_steps)]

        logger.info(f"Running external evaluation command: {' '.join(cmd)}")

        proc = None
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                logger.error(f"External evaluation exited with code {proc.returncode}")
                logger.error(proc.stdout)
                logger.error(proc.stderr)
            else:
                logger.info(f"External evaluation completed for epoch {epoch}; output at {eval_out_dir}")
        except subprocess.TimeoutExpired:
            logger.error(f"External evaluation timed out after {timeout} seconds for epoch {epoch}")
        except Exception as exc:  # pragma: no cover - external call
            logger.error(f"Failed to run external evaluation for epoch {epoch}: {exc}")

        summary_src = eval_out_dir / 'summary.json'
        if summary_src.exists():
            dst = self.log_dir / f'eval_epoch_{epoch:04d}.json'
            try:
                shutil.copy(summary_src, dst)
                logger.info(f"Saved evaluation summary to {dst}")
            except Exception as exc:  # pragma: no cover - IO heavy
                logger.error(f"Failed to copy evaluation summary: {exc}")

        out_log = self.log_dir / f'eval_epoch_{epoch:04d}.log'
        try:
            with open(out_log, 'w', encoding='utf-8') as f:
                if proc is not None:
                    f.write('STDOUT\n')
                    f.write(proc.stdout or '')
                    f.write('\n\nSTDERR\n')
                    f.write(proc.stderr or '')
            logger.info(f"Saved external eval logs to {out_log}")
        except Exception as exc:  # pragma: no cover - IO heavy
            logger.error(f"Failed to write external eval logs: {exc}")

        if temp_ckpt is not None and temp_ckpt.exists():
            try:
                temp_ckpt.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Checkpointing/logging
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        save_checkpoint(
            checkpoint_dir=self.checkpoint_dir,
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            metrics=metrics,
            global_step=self.global_step,
        )

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        current_lr = self.optimizer.param_groups[0]['lr']
        with open(self.log_file, 'a') as f:
            f.write(
                f"{epoch},{metrics['loss']:.6f},{metrics['style_swd']:.6f},{metrics['style_swd_weighted']:.6f},"
                f"{metrics['mse']:.6f},{metrics['mse_weighted']:.6f},{metrics['layout']:.6f},{metrics['mse_raw']:.6f},"
                f"{current_lr:.2e},{metrics.get('epoch_time', 0.0):.2f}\n"
            )

        epoch_log_path = self.log_dir / 'epoch_logs.jsonl'
        epoch_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'loss': float(metrics['loss']),
            'style_swd': float(metrics['style_swd']),
            'style_swd_weighted': float(metrics['style_swd_weighted']),
            'mse': float(metrics['mse']),
            'mse_weighted': float(metrics['mse_weighted']),
            'layout': float(metrics['layout']),
            'mse_raw': float(metrics['mse_raw']),
            'learning_rate': current_lr,
            'epoch_time': metrics.get('epoch_time', 0.0),
            'num_batches': metrics.get('num_batches'),
        }
        with open(epoch_log_path, 'a', encoding='utf-8') as ef:
            ef.write(json.dumps(epoch_entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Convenience helpers for main loop
    # ------------------------------------------------------------------
    def on_training_start(self, dataloader: DataLoader) -> None:
        if self.style_indices_cache is None:
            self.build_style_indices_cache(dataloader.dataset)
        
        # 🔥 Initialize Style LUT Cache for Multi-Style SWD Loss
        logger.info("🔥 Initializing Style LUT Cache for Multi-Style SWD...")
        style_prototypes = {}
        
        # Collect representative latent for each style
        for style_id in range(self.config['model']['num_styles']):
            if hasattr(dataloader.dataset, 'style_indices') and style_id in self.style_indices_cache:
                # Take first sample of this style
                idx = self.style_indices_cache[style_id][0]
                latent_sample = dataloader.dataset.latents_tensor[idx].unsqueeze(0)  # [1, 4, 32, 32]
            else:
                # Fallback: random initialization
                logger.warning(f"⚠️  No style {style_id} samples found, using random init")
                latent_sample = torch.randn(1, 4, 32, 32)
            
            style_prototypes[style_id] = latent_sample
        
        # Initialize LUT cache (this pre-computes all sorted projections)
        self.energy_loss.initialize_cache(style_prototypes, self.device)
        
        # Save originals
        orig_dir = self.inference_dir / 'epoch_-1'
        if not orig_dir.exists() or not any(orig_dir.iterdir()):
            logger.info("Saving original test images to inference/epoch_-1")
            self.evaluate_and_infer(-1)
        else:
            logger.info("Original test images already saved in inference/epoch_-1; skipping.")

    def step_scheduler(self) -> None:
        self.scheduler.step()

    def set_train(self) -> None:
        self.model.train()

    def set_eval(self) -> None:
        self.model.eval()
