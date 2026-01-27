"""
LGT Training Script with Geometric Free Energy Optimization

Key differences from OT-CFM:
- Loss: Energy E(x_1) instead of velocity MSE
- Training: Integrate ODE to terminal state, compute energy
- Physics: Optimize geometric free energy landscape

Training process:
1. Sample x0 ~ N(0,I), t ~ U(ε, 1)
2. Construct x_t = (1-t)x0 + t*x_src (content-anchored path)
3. Integrate ODE from t to 1: dx/ds = v(x,s,style)
4. Compute energy E(x_1) = w_style*SWD(x_1, x_style) + w_content*SSM(x_1, x_src)
5. Backprop through ODE trajectory
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image

from model import LGTUNet, count_parameters
from losses import (
    GeometricFreeEnergyLoss, 
    TrajectoryMSELoss, 
    VelocityRegularizationLoss, 
    VelocitySmoothnessLoss, 
    DistilledStructureLoss,
    StructureAnchoredLoss  # 🔥 New Laplacian-based loss
)
from inference import LGTInference, load_vae, encode_image, decode_latent

import subprocess
import sys
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def elastic_deform(x, alpha=10, sigma=3):
    """
    Apply random elastic deformation to the latent.
    Differentiable augmentation for structure breaking.
    
    This allows the model to learn non-rigid transformations by providing
    warped training targets. The deformation is smooth (via averaging) and
    bounded (via alpha parameter).
    
    Args:
        x: [B, C, H, W] input latent tensor
        alpha: displacement magnitude (pixels)
        sigma: smoothness parameter (simulated via avg pooling iterations)
    
    Returns:
        x_deformed: [B, C, H, W] warped latent
    """
    B, C, H, W = x.shape
    device = x.device
    
    # Create random displacement fields
    dx = torch.rand(B, 1, H, W, device=device) * 2 - 1
    dy = torch.rand(B, 1, H, W, device=device) * 2 - 1
    
    # Smooth the displacement field (Gaussian Blur simulation via AvgPool)
    # Using AvgPool repeatedly to approximate Gaussian smoothing efficiently
    for _ in range(3):
        dx = F.avg_pool2d(dx, kernel_size=3, stride=1, padding=1)
        dy = F.avg_pool2d(dy, kernel_size=3, stride=1, padding=1)
        
    # Scale the displacement
    flow = torch.cat([dx, dy], dim=1) * alpha
    
    # Create normalized grid [-1, 1]
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack([x_grid, y_grid], dim=-1).float()  # [H, W, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
    
    # Normalize grid to [-1, 1] for grid_sample
    grid_norm = grid.clone()
    grid_norm[..., 0] = 2.0 * grid_norm[..., 0] / (W - 1) - 1.0
    grid_norm[..., 1] = 2.0 * grid_norm[..., 1] / (H - 1) - 1.0
    
    # Add flow (normalized)
    # flow values are in pixels, need to normalize to [-1, 1] range
    flow_norm = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    flow_norm[..., 0] = 2.0 * flow_norm[..., 0] / (W - 1)
    flow_norm[..., 1] = 2.0 * flow_norm[..., 1] / (H - 1)
    
    sample_grid = grid_norm + flow_norm
    
    # Warp
    x_deformed = F.grid_sample(x, sample_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    return x_deformed


class InMemoryLatentDataset(Dataset):
    """
    Balanced In-memory dataset.
    Loads all latents, but samples styles uniformly (1/N probability) regardless of image count.
    """
    
    def __init__(self, data_root, num_styles, style_subdirs=None, device='cuda'):
        self.data_root = Path(data_root)
        self.num_styles = num_styles
        self.device = device
        self.style_subdirs = style_subdirs or [f"style{i}" for i in range(num_styles)]

        # 1. Load data separated by style
        self.style_indices = {} # {style_id: [tensor_idx_1, tensor_idx_2, ...]}
        self.latents_list = []
        self.styles_list = []
        
        current_idx = 0
        logger.info("Loading latents with Balanced Sampling strategy...")
        
        for style_id, subdir in enumerate(self.style_subdirs):
            style_path = self.data_root / subdir
            if not style_path.exists():
                logger.warning(f"Style dir not found: {style_path}")
                continue

            latent_files = sorted(style_path.glob("*.pt"))
            count = len(latent_files)
            logger.info(f"  Style {style_id} ({subdir}): {count} images")
            
            # Record indices for this style
            # indices [current, current+count) belong to this style
            self.style_indices[style_id] = list(range(current_idx, current_idx + count))
            
            # Load files
            for lf in latent_files:
                latent = torch.load(lf, map_location='cpu')
                if latent.ndim == 4: latent = latent.squeeze(0)
                self.latents_list.append(latent)
                self.styles_list.append(style_id)
                current_idx += 1

        # 2. Stack to GPU
        self.latents_tensor = torch.stack(self.latents_list).to(self.device)
        self.styles_tensor = torch.tensor(self.styles_list, dtype=torch.long).to(self.device)
        
        # 3. Handle Scaling (Check only once globally)
        std_original = self.latents_tensor.std().item()
        scaling_factor = 0.18215
        if std_original < 0.5:
            logger.info(f"⚠️ Raw VAE latents detected (std={std_original:.4f}). Rescaling...")
            self.latents_tensor = self.latents_tensor / scaling_factor
        
        # 4. Set virtual length (make epoch long enough to cover the largest class)
        max_count = max([len(idxs) for idxs in self.style_indices.values()])
        self.virtual_length = max_count * num_styles * num_styles
        
    def __len__(self):
        return self.virtual_length
    
    def __getitem__(self, _):
        # 🔥 Magic happens here: IGNORE the input index (_)
        
        # 1. Randomly pick a style (Uniform probability 1/4)
        style_id = torch.randint(0, self.num_styles, (1,)).item()
        
        # 2. Randomly pick an image from THAT style
        # (Even if Style A has 100 images and Style B has 6000, 
        #  we pick Style A 25% of the time, just reusing its images more often)
        if style_id in self.style_indices and len(self.style_indices[style_id]) > 0:
            idx = random.choice(self.style_indices[style_id])
        else:
            # Fallback if a style folder is empty
            idx = random.randint(0, len(self.latents_tensor)-1)
            style_id = self.styles_tensor[idx].item()
            
        return {
            'latent': self.latents_tensor[idx],
            'style_id': self.styles_tensor[idx] # Should match style_id above
        }

class LGTTrainer:
    """
    Trainer for LGT model with geometric free energy optimization.
    """
    
    def __init__(self, config, device='cuda', config_path=None):
        self.config = config
        self.device = device
        self.config_path = Path(config_path) if config_path is not None else None
        
        # GPU optimization for fixed input size (32x32 latents)
        torch.backends.cudnn.benchmark = True
        
        # Create model
        self.model = LGTUNet(
            latent_channels=config['model']['latent_channels'],
            base_channels=config['model']['base_channels'],
            style_dim=config['model']['style_dim'],
            time_dim=config['model']['time_dim'],
            num_styles=config['model']['num_styles'],
            num_encoder_blocks=config['model']['num_encoder_blocks'],
            num_decoder_blocks=config['model']['num_decoder_blocks']
        ).to(device)
        
        # Compute average style embedding before training
        self.model.compute_avg_style_embedding()
        
        # Compile model if enabled (PyTorch 2.0+)
        if config['training'].get('use_compile', False):
            try:
                self.model = torch.compile(
                    self.model,
                    mode='default',
                    fullgraph=False
                )
                logger.info("✓ Model compiled with torch.compile (reduce-overhead mode)")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        
        # Create loss functions
        # Loss 1: Energy (Style SWD Only - Content Loss Removed)
        self.energy_loss = GeometricFreeEnergyLoss(
            w_style=config['loss']['w_style'],
            swd_scales=config['loss'].get('swd_scales', [2, 4, 8]),
            swd_scale_weights=config['loss'].get('swd_scale_weights', [2.0, 5.0, 5.0])
        ).to(device)
        
        # Loss 2: Structure Lock
        # 🔥 [UPDATED] Use Laplacian-based StructureAnchoredLoss for RTX 4070 optimization
        use_laplacian_lock = config['loss'].get('use_laplacian_structure_lock', True)
        
        if use_laplacian_lock:
            self.struc_loss = StructureAnchoredLoss(
                weight=config['loss'].get('structure_weight', 5.0),
                edge_boost=config['loss'].get('edge_boost', 9.0)
            ).to(device)
            logger.info(f"✓ Laplacian Structure Lock enabled (weight={config['loss'].get('structure_weight', 5.0)}, edge_boost={config['loss'].get('edge_boost', 9.0)})")
        else:
            # Fallback to CNN proxy-based loss
            self.struc_loss = DistilledStructureLoss(
                proxy_path="structure_proxy.pt",
                base_weight=config['loss'].get('structure_base_weight', 1.0),
                structure_weight=config['loss'].get('structure_weight', 15.0),
                device=device
            ).to(device)
            logger.info(f"✓ Distilled Structure Loss enabled (CNN proxy)")
        
        # Loss 2.5: Velocity Smoothness (The Anti-Flicker)
        vel_smooth_weight = config['loss'].get('vel_smooth_weight', 0.2)
        self.vel_smooth_loss = VelocitySmoothnessLoss(weight=vel_smooth_weight).to(device)
        logger.info(f"✓ Velocity Smoothness Loss enabled (weight={vel_smooth_weight})")
        
        # Loss 3: Velocity Regularization
        self.use_velocity_reg = config['loss'].get('use_velocity_reg', False)
        self.vel_reg_loss = None
        if self.use_velocity_reg:
            vel_reg_weight = config['loss'].get('vel_reg_weight', 0.1)
            self.vel_reg_loss = VelocityRegularizationLoss(weight=vel_reg_weight).to(device)
            logger.info(f"✓ Velocity Regularization enabled (weight={vel_reg_weight})")
        else:
            logger.info("Velocity Regularization disabled")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training'].get('min_learning_rate', 1e-6)
        )
        
        # AMP scaler for mixed precision
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        
        # Training config
        self.num_epochs = config['training']['num_epochs']
        self.label_drop_prob = config['training'].get('label_drop_prob', 0.1)
        self.use_avg_for_uncond = config['training'].get('use_avg_style_for_uncond', True)
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)
        self.effective_batch_size = config['training']['batch_size'] * self.accumulation_steps
        
        if self.accumulation_steps > 1:
            logger.info(
                f"Gradient accumulation: {self.accumulation_steps} steps | "
                f"Effective batch size: {self.effective_batch_size}"
            )
        
        # 🔥 [CRITICAL] Global step counter for Alpha warmup scheduling
        # ================================================================
        # This must be initialized BEFORE any training loop to track progress
        # for Hyper-LoRA alpha parameter scheduling.
        self.global_step = 0
        self.alpha_warmup_steps = config['training'].get('alpha_warmup_steps', 1000)
        logger.info(f"Alpha warmup schedule: 0 → 1.0 over {self.alpha_warmup_steps} steps")
        
        # Epsilon for time sampling (avoid t=0 singularity)
        self.epsilon = config['training'].get('epsilon', 0.01)
        
        # ODE integration steps for computing terminal state
        self.ode_steps = config['training'].get('ode_integration_steps', 5)
        
        # Per-style dynamic loss weighting (LGT++ enhancement)
        # Apply greater pressure to difficult "quenching" tasks (→photo)
        self.style_weights = torch.tensor(
            config['loss'].get('style_weights', [1.0] * config['model']['num_styles']),
            device=device,
            dtype=torch.float32
        )
        logger.info(f"Per-style loss weights: {self.style_weights.tolist()}")
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = config['training'].get('save_interval', 5)
        
        # Evaluation and inference
        self.eval_interval = config['training'].get('eval_interval', 5)
        self.full_eval_interval = config['training'].get('full_eval_interval', self.eval_interval)
        logger.info(f"Full external evaluation interval: {self.full_eval_interval}")
        self.test_image_dir = Path(config['training'].get('test_image_dir', 'test_images'))
        self.inference_dir = self.checkpoint_dir / 'inference'
        self.inference_dir.mkdir(exist_ok=True)
        
        # Load VAE for inference
        self.vae = load_vae(device)
        
        # Style indices cache for efficient sampling (will be set by train())
        self.style_indices_cache = None
        self.dataset_ref = None
        
        # Logging
        self.log_dir = self.checkpoint_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Initialize log file (Clean version: removed content_ssm)
        with open(self.log_file, 'w') as f:
            f.write('epoch,loss_total,loss_style_swd,loss_mse,learning_rate,epoch_time\n')
        
        # Resume checkpoint support - Auto-resume from highest epoch
        self.start_epoch = 1
        
        # 🔥 Check for migration checkpoint (for Hyper-LoRA upgrade)
        migration_ckpt = config['training'].get('migration_checkpoint')
        if migration_ckpt:
            logger.info(f"Performing weight migration from: {migration_ckpt}")
            load_checkpoint_with_migration(self.model, migration_ckpt, device)
            # Don't set start_epoch - this is a fresh training with migrated weights
        else:
            # Auto-resume from highest epoch checkpoint if exists
            epoch_checkpoints = sorted(self.checkpoint_dir.glob('epoch_*.pt'))
            if epoch_checkpoints:
                latest_ckpt = epoch_checkpoints[-1]  # Last one is highest epoch
                logger.info(f"Found checkpoint {latest_ckpt.name}, auto-resuming training from it")
                self.load_checkpoint(latest_ckpt)
            else:
                # Manual resume if specified in config
                resume_ckpt = config['training'].get('resume_checkpoint')
                if resume_ckpt:
                    self.load_checkpoint(resume_ckpt)
    
    def build_style_indices_cache(self, dataset):
        """
        Build cache mapping {style_id: [idx0, idx1, ...]} for O(1) random sampling.
        This is called once at the start of training.
        """
        logger.info("Building style indices cache...")
        self.style_indices_cache = {}
        
        for style_id in range(self.config['model']['num_styles']):
            # Find all indices belonging to this style
            indices = (dataset.styles_tensor == style_id).nonzero(as_tuple=True)[0]
            self.style_indices_cache[style_id] = indices
            logger.info(f"  Style {style_id}: {len(indices)} samples")
        
        self.dataset_ref = dataset
        logger.info("✓ Style indices cache built")
    
    def sample_style_batch(self, target_style_ids):
        """
        Sample real latents from target style distribution using GPU vectorization.

        Args:
            target_style_ids: [B] tensor of target style IDs

        Returns:
            style_latents: [B, 4, H, W] real latents from target styles
        """
        B = target_style_ids.shape[0]
        device = target_style_ids.device

        # Initialize result tensor
        style_latents = torch.empty((B, *self.dataset_ref.latents_tensor.shape[1:]), device=device)

        # Process each style ID
        for style_id in range(self.config['model']['num_styles']):
            mask = (target_style_ids == style_id)
            count = mask.sum().item()

            if count > 0:
                indices = self.style_indices_cache[style_id]
                rand_indices = indices[torch.randint(len(indices), (count,), device=device)]
                style_latents[mask] = self.dataset_ref.latents_tensor[rand_indices]

        return style_latents
    
    def get_dynamic_epsilon(self, epoch):
        """
        Dynamic epsilon warmup to avoid t=0 singularity.
        Gradually increase from 0 to target epsilon.
        """
        warmup_epochs = self.config['training'].get('epsilon_warmup_epochs', 10)
        target_epsilon = self.epsilon
        
        if epoch < warmup_epochs:
            return target_epsilon * (epoch / warmup_epochs)
        else:
            return target_epsilon
    
    def integrate_ode(self, x_t, t_start, style_id, use_avg_style=False):
        """
        Integrate ODE from t_start to 1.0 to get terminal state x_1.
        
        Uses simple Euler integration for efficiency.
        dx/dt = v(x, t, style)
        
        Optimization: Uses gradient checkpointing to reduce memory from O(steps) to O(1).
        
        Args:
            x_t: [B, 4, H, W] starting state at time t_start
            t_start: [B] starting times
            style_id: [B] style IDs
            use_avg_style: bool, use average style (for CFG)
        
        Returns:
            x_1: [B, 4, H, W] terminal state at t=1
        """
        from torch.utils.checkpoint import checkpoint
        
        x = x_t.clone()
        t = t_start.clone()
        
        # Number of integration steps
        num_steps = self.ode_steps
        use_checkpoint = self.config['training'].get('use_gradient_checkpointing', True)
        
        # Define single-step function for checkpointing
        def step_func(x_in, t_in, style_id_in):
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                return self.model(x_in, t_in, style_id_in, use_avg_style=use_avg_style)
        
        for _ in range(num_steps):
            # Compute remaining time
            t_remaining = 1.0 - t
            dt = t_remaining / num_steps
            
            # Velocity at current state (with or without checkpointing)
            if use_checkpoint and self.model.training:
                # Gradient checkpointing: trade compute for memory
                # Reduces activation memory from O(steps) to O(1)
                v = checkpoint(step_func, x, t, style_id, use_reentrant=False)
            else:
                v = step_func(x, t, style_id)
            
            # Euler step
            x = x + v * dt.view(-1, 1, 1, 1)
            t = t + dt
        
        return x
    
    def compute_energy_loss(self, x_src, style_id_src, style_id_tgt, epoch, real_style_latents, noise_scheduler):
        """
        Compute hybrid loss: MSE (Supervisor) + Energy (Artist).
        
        🔥 CRITICAL CHANGE: Now receives noise_scheduler to compute correct targets.
        
        Clean version: Content Loss completely removed.
        
        This implements Hybrid Dynamics with Classifier-Free Guidance training:
        1. Supervisor Task (MSE): High-freq clarity + source brightness preservation
        2. Artist Task (Energy/SWD): Texture/brushstroke matching
        3. CFG Training (Label Dropping): Learn unconditional distribution for inference
        4. Conditional Deformation: Only deform painting (Style 1), keep photos (Style 0) intact
        
        Args:
            x_src: [B, 4, H, W] source content latents
            style_id_src: [B] source style IDs
            style_id_tgt: [B] target style IDs
            epoch: current epoch for epsilon scheduling
            real_style_latents: [B, 4, H, W] real latents from target style distribution
            noise_scheduler: 🔥 NEW: DDPMScheduler or equivalent for computing targets
        
        Returns:
            loss_dict: dictionary with total loss and components
        """
        B = x_src.shape[0]
        device = x_src.device
        
        # 1. Sample noise from standard normal
        x0 = torch.randn_like(x_src)
        
        # 2. Conditional Deformation (only for painting, not photos)
        use_elastic = self.config['training'].get('use_elastic_deform', False)
        elastic_alpha = self.config['training'].get('elastic_alpha', 1.0)
        
        if use_elastic and self.model.training:
            x_deformed = elastic_deform(x_src, alpha=elastic_alpha)
            is_painting = (style_id_tgt == 1).view(-1, 1, 1, 1).float()
            x_src_target = (1 - is_painting) * x_src + is_painting * x_deformed
        else:
            x_src_target = x_src
        
        # 2.5 Noise Injection
        sigma_noise = 0.1
        noise_injection = torch.randn_like(x_src_target) * sigma_noise
        x_target_noisy = x_src_target + noise_injection
        
        # 3. Time sampling
        epsilon = self.get_dynamic_epsilon(epoch)
        t = torch.rand(B, device=device) * (1 - epsilon) + epsilon
        
        # 🔥 CRITICAL: Convert to timesteps for scheduler
        # Assuming T=1000 as typical for DDPM schedulers
        num_train_timesteps = noise_scheduler.config.num_train_timesteps
        timesteps = (t * (num_train_timesteps - 1)).long()  # Convert [0,1] to [0, T-1]
        t_expand = t.view(-1, 1, 1, 1)
        
        # 4. Trajectory construction: x_t = (1-t)*x0 + t*x_target_noisy
        x_t = (1 - t_expand) * x0 + t_expand * x_target_noisy
        
        # 5. Label Dropping (For CFG Training)
        drop_label = False
        if self.model.training and torch.rand(1).item() < self.label_drop_prob:
            drop_label = True
        
        # ==========================================
        # Task A: Supervisor (Structure Lock)
        # ==========================================
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            v_pred = self.model(x_t, t, style_id_tgt, use_avg_style=drop_label)
        
        # 🔥 CRITICAL: Compute v_target using scheduler (NOT locally!)
        # The scheduler knows the correct prediction type (epsilon, v_prediction, sample, etc)
        with torch.no_grad():
            if noise_scheduler.config.prediction_type == "epsilon":
                # Model predicts noise (ε-prediction)
                v_target = x0  # Ground truth noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                # Model predicts velocity (v-prediction)
                # v = (α_t * noise - σ_t * x_src) for some schedule (α, σ)
                # For simplicity, approximate: v ≈ noise (this is scheduler-dependent)
                v_target = x0
            elif noise_scheduler.config.prediction_type == "sample":
                # Model predicts the clean sample (score-based)
                v_target = x_src_target
            else:
                # Fallback: assume epsilon prediction
                v_target = x0
        
        # Use StructureAnchoredLoss with scheduler-provided target
        # 🔥 NEW SIGNATURE: (pred, target, clean_latents)
        loss_mse = self.struc_loss(v_pred, v_target, x_src)
        
        # ==========================================
        # Task B: Artist (Energy - SWD Texture Loss)
        # ==========================================
        x_1 = self.integrate_ode(x_t, t, style_id_tgt, use_avg_style=False)
        loss_dict = self.energy_loss(x_1, real_style_latents)
        
        # ==========================================
        # Total Loss Fusion
        # ==========================================
        style_weight_batch = self.style_weights[style_id_tgt].mean()
        loss_smooth = self.vel_smooth_loss(v_pred)
        
        loss_dict['total'] = (
            style_weight_batch * self.energy_loss.w_style * loss_dict['style_swd'] +
            loss_mse + 
            loss_smooth
        )
        
        loss_dict['mse'] = loss_mse
        loss_dict['smooth'] = loss_smooth
        
        # Velocity Regularization
        if self.use_velocity_reg and self.vel_reg_loss is not None:
            loss_reg = self.vel_reg_loss(v_pred)
            loss_dict['velocity_reg'] = loss_reg
            loss_dict['total'] = loss_dict['total'] + loss_reg
        
        return loss_dict
    
    def train_epoch(self, dataloader, epoch, noise_scheduler):
        """Train for one epoch with gradient accumulation support."""
        self.model.train()
        
        total_loss = 0.0
        total_style_swd = 0.0
        total_mse = 0.0
        total_smooth = 0.0
        total_vel_reg = 0.0
        num_batches = 0
        accum_counter = 0
        
        import sys
        use_tqdm = sys.stderr.isatty()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.num_epochs}", disable=not use_tqdm, leave=use_tqdm)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for step_idx, batch in enumerate(pbar, start=1):
            torch.compiler.cudagraph_mark_step_begin()

            latent = batch['latent'].to(self.device, non_blocking=True)
            style_id = batch['style_id'].to(self.device, non_blocking=True)
            
            B = latent.shape[0]
            indices = torch.randperm(B, device=self.device)
            style_id_tgt = style_id[indices]
            
            style_latents = self.sample_style_batch(style_id_tgt)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                # 🔥 NEW: Pass noise_scheduler to compute_energy_loss
                loss_dict = self.compute_energy_loss(
                    latent, style_id, style_id_tgt, epoch, style_latents, noise_scheduler
                )
                loss = loss_dict['total'] / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            accum_counter += 1
            
            if accum_counter >= self.accumulation_steps:
                # 🔥🔥🔥 CRITICAL FIX: Gradient Clipping (Essential for training from scratch)
                # ================================================================
                # When model starts with random weights (output ≈ 0) and target is
                # noise (Std ≈ 1.0), gradients can be huge. Clipping bounds them.
                # This is standard practice for Transformer/Diffusion models.
                # 
                # Effect: Prevents single outlier loss from dominating gradient update.
                # Typical max_norm: 1.0 for stable training, 0.5 for very aggressive regularization.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                
                # 🔥 Alpha Warmup
                self._update_lora_alpha(self.global_step)
                self.global_step += 1
            
            total_loss += loss.item() * self.accumulation_steps
            total_style_swd += loss_dict['style_swd'].item()
            total_mse += loss_dict['mse'].item()
            if 'smooth' in loss_dict:
                total_smooth += loss_dict['smooth'].item()
            if 'velocity_reg' in loss_dict:
                total_vel_reg += loss_dict['velocity_reg'].item()
            num_batches += 1
            
            if use_tqdm:
                postfix_dict = {
                    'loss': f"{loss.item():.4f}",
                    'swd': f"{loss_dict['style_swd'].item():.4f}",
                    'mse': f"{loss_dict['mse'].item():.4f}",
                    'α': f"{self._get_current_alpha():.3f}"
                }
                pbar.set_postfix(postfix_dict)
        
        avg_loss = total_loss / num_batches
        avg_style_swd = total_style_swd / num_batches
        avg_mse = total_mse / num_batches
        avg_smooth = total_smooth / num_batches
        avg_vel_reg = total_vel_reg / num_batches if self.use_velocity_reg else 0.0
        
        metrics = {
            'loss': avg_loss,
            'style_swd': avg_style_swd,
            'mse': avg_mse,
            'smooth': avg_smooth,
            'num_batches': num_batches
        }
        if self.use_velocity_reg:
            metrics['velocity_reg'] = avg_vel_reg
        
        return metrics
    
    def _get_current_alpha(self):
        """
        Compute current alpha value for Hyper-LoRA warmup schedule.
        
        Returns:
            alpha: float in [0.0, 1.0]
        """
        return min(self.global_step / max(self.alpha_warmup_steps, 1), 1.0)
    
    def _update_lora_alpha(self, global_step):
        """
        Update alpha parameter in all Hyper-LoRA layers.
        
        This is called every optimizer step to gradually enable LoRA contribution.
        
        Args:
            global_step: Current training step (0-indexed)
        """
        current_alpha = min(global_step / max(self.alpha_warmup_steps, 1), 1.0)
        
        # Iterate through all modules and update alpha if they have it
        for module in self.model.modules():
            if hasattr(module, 'alpha'):
                module.alpha = current_alpha
        
        # Debug logging every 250 steps during warmup
        if global_step % 250 == 0 and global_step < self.alpha_warmup_steps:
            logger.debug(f"Alpha warmup: step {global_step}/{self.alpha_warmup_steps}, alpha={current_alpha:.4f}")
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"✓ Saved checkpoint: {checkpoint_path}")
        # Intentionally do not save 'latest.pt' (removed per configuration)

    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint for resuming training."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle state_dict mismatch between compiled and non-compiled models
        model_state_dict = checkpoint['model_state_dict']
        
        # Check if checkpoint has _orig_mod prefix (from compiled model)
        if any(k.startswith('_orig_mod.') for k in model_state_dict.keys()):
            # Compiled model checkpoint - strip _orig_mod prefix for loading into non-compiled
            if not isinstance(self.model, torch.jit.ScriptModule):
                logger.info("Converting compiled model state_dict to non-compiled format")
                model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}
        elif isinstance(self.model, torch.jit.ScriptModule) or hasattr(self.model, '_orig_mod'):
            # Non-compiled checkpoint but model is compiled - add _orig_mod prefix
            logger.info("Converting non-compiled state_dict to compiled model format")
            model_state_dict = {f'_orig_mod.{k}': v for k, v in model_state_dict.items()}
        
        # Load model state with strict=False to handle minor mismatches
        try:
            self.model.load_state_dict(model_state_dict, strict=False)
            logger.info("✓ Model state loaded successfully")
        except RuntimeError as e:
            logger.error(f"Failed to load model state: {e}")
            raise
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Set starting epoch
        self.start_epoch = checkpoint['epoch'] + 1
        
        logger.info(f"✓ Resumed from epoch {checkpoint['epoch']}")
        logger.info(f"  Next epoch will be: {self.start_epoch}")
    
    def train(self, dataloader):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info("Starting LGT Training")
        logger.info("=" * 80)
        
        # 🔥 NEW: Initialize noise scheduler for correct target computation
        # This should match the scheduler used during preprocessing/VAE encoding
        from diffusers import DDPMScheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        if self.style_indices_cache is None:
            self.build_style_indices_cache(dataloader.dataset)

        orig_dir = self.inference_dir / 'epoch_-1'
        if not orig_dir.exists() or not any(orig_dir.iterdir()):
            logger.info("Saving original test images to inference/epoch_-1")
            self.evaluate_and_infer(-1)
        else:
            logger.info("Original test images already saved in inference/epoch_-1; skipping.")
        
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # Announce epoch start
            logger.info(f"Starting epoch {epoch}/{self.num_epochs}")
            
            # Train one epoch
            metrics = self.train_epoch(dataloader, epoch, noise_scheduler)
            
            # Learning rate step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = metrics.get('epoch_time', 0.0)
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch}/{self.num_epochs} completed in {epoch_time:.1f}s | "
                f"Loss: {metrics['loss']:.4f} | "
                f"SWD: {metrics['style_swd']:.4f} | "
                f"MSE: {metrics['mse']:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Write to CSV (append epoch_time)
            with open(self.log_file, 'a') as f:
                f.write(
                    f"{epoch},{metrics['loss']:.6f},"
                    f"{metrics['style_swd']:.6f},{metrics['mse']:.6f},"
                    f"{current_lr:.2e},{epoch_time:.2f}\n"
                )
            
            # Append structured epoch log to JSONL
            epoch_log_path = self.log_dir / 'epoch_logs.jsonl'
            epoch_entry = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'loss': float(metrics['loss']),
                'style_swd': float(metrics['style_swd']),
                'mse': float(metrics['mse']),
                'learning_rate': current_lr,
                'epoch_time': epoch_time,
                'num_batches': metrics.get('num_batches')
            }
            with open(epoch_log_path, 'a', encoding='utf-8') as ef:
                ef.write(json.dumps(epoch_entry, ensure_ascii=False) + "\n")
            
            # Save checkpoint
            if epoch % self.save_interval == 0 or epoch == self.num_epochs:
                self.save_checkpoint(epoch, metrics)
            
            # Run evaluation and inference
            if epoch % self.eval_interval == 0 or epoch == self.num_epochs:
                self.evaluate_and_infer(epoch)
                # Run full external evaluation script (run_evaluation.py) using current checkpoint
                if self.full_eval_interval is not None and (epoch % self.full_eval_interval == 0 or epoch == self.num_epochs):
                    try:
                        self._run_full_evaluation(epoch)
                    except Exception as e:
                        logger.error(f"Full external evaluation failed for epoch {epoch}: {e}")
                else:
                    logger.debug(f"Skipping full external evaluation at epoch {epoch} (full_eval_interval={self.full_eval_interval})")
        
        logger.info("=" * 80)
        logger.info("✓ Training completed!")
        logger.info("=" * 80)
    
    def get_test_images_by_style(self):
        """
        Find one image from each style subdirectory in test_image_dir.
        
        Returns:
            dict: {style_id: (style_name, image_path)}
        """
        test_dir = Path(self.test_image_dir)
        if not test_dir.exists():
            logger.warning(f"Test image directory not found: {test_dir}")
            return {}
        
        style_images = {}
        
        # Get style subdirectories
        style_subdirs = self.config['data'].get('style_subdirs', [])
        
        for style_id, style_name in enumerate(style_subdirs):
            style_dir = test_dir / style_name
            if not style_dir.exists():
                logger.warning(f"Style directory not found: {style_dir}")
                continue
            
            # Find images
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            images = []
            for ext in image_extensions:
                images.extend(style_dir.glob(f"*{ext}"))
                images.extend(style_dir.glob(f"*{ext.upper()}"))
            
            if images:
                # Take first image
                test_image = sorted(images)[0]
                style_images[style_id] = (style_name, test_image)
                logger.info(f"  Test image for {style_name}: {test_image.name}")
        
        return style_images
    
    @torch.no_grad()
    def evaluate_and_infer(self, epoch):
        """
        Evaluate model and perform style transfer inference on test images.
        
        Process:
        1. Load one image per style
        2. For each image, transfer to all other styles
        3. Save results to inference/epoch_XXXX/
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Inference Evaluation (Epoch {epoch})")
        logger.info(f"{'='*80}")
        
        # Create inference directory for this epoch (special-case -1 -> 'epoch_-1')
        if epoch == -1:
            epoch_inference_dir = self.inference_dir / 'epoch_-1'
        else:
            epoch_inference_dir = self.inference_dir / f"epoch_{epoch:04d}"
        epoch_inference_dir.mkdir(parents=True, exist_ok=True) 
        
        # Get test images
        test_images = self.get_test_images_by_style()
        if not test_images:
            logger.warning("No test images found. Skipping inference.")
            return
        
        self.model.eval()
        num_styles = self.config['model']['num_styles']

        # Create inference engine
        temp_ckpt = self.checkpoint_dir / "temp_eval.pt"
        torch.save(self.model.state_dict(), temp_ckpt)

        try:
            # For each test image (source style)
            for src_style_id, (src_style_name, src_image_path) in test_images.items():
                logger.info(f"\nProcessing source: {src_style_name}")

                # Load and preprocess image
                try:
                    image = Image.open(src_image_path).convert('RGB')
                    image = image.resize((256, 256))
                    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
                    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
                    image_tensor = image_tensor * 2.0 - 1.0
                    image_tensor = image_tensor.to(self.device)
                except Exception as e:
                    logger.error(f"Failed to load {src_image_path}: {e}")
                    continue

                # Encode to latent
                latent_src = encode_image(self.vae, image_tensor, self.device)

                # If epoch == -1, save original decoded image and continue (only save originals once)
                if epoch == -1:
                    try:
                        image_orig = decode_latent(self.vae, latent_src, self.device)
                        output_filename = f"{src_style_name}_original.jpg"
                        output_path = epoch_inference_dir / output_filename
                        from torchvision.utils import save_image
                        save_image(image_orig, output_path)
                        logger.info(f"    ✓ Saved original: {output_filename}")
                    except Exception as e:
                        logger.error(f"    ✗ Failed to save original: {e}")
                    continue

                # Transfer to all target styles (including same style for self-comparison)
                for tgt_style_id in range(num_styles):

                    tgt_style_name = self.config['data'].get('style_subdirs', [])[tgt_style_id]
                    logger.info(f"  → {src_style_name} to {tgt_style_name}")

                    try:
                        # Integrate ODE with source style (inversion)
                        latent_x0 = self._invert_latent(latent_src, src_style_id)

                        # Integrate ODE with target style (generation)
                        latent_tgt = self._generate_latent(latent_x0, tgt_style_id)

                        # Decode to image
                        image_out = decode_latent(self.vae, latent_tgt, self.device)

                        # Save result
                        output_filename = f"{src_style_name}_to_{tgt_style_name}.jpg"
                        output_path = epoch_inference_dir / output_filename

                        # Convert and save
                        from torchvision.utils import save_image
                        save_image(image_out, output_path)

                        logger.info(f"    ✓ Saved: {output_filename}")

                    except Exception as e:
                        logger.error(f"    ✗ Failed: {e}")
                        continue

        finally:
            # Cleanup temp checkpoint
            if temp_ckpt.exists():
                temp_ckpt.unlink()

        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Inference completed. Results saved to: {epoch_inference_dir}")
        logger.info(f"{'='*80}\n")

    def _run_full_evaluation(self, epoch, timeout=3600):
        """Run external evaluation script `run_evaluation.py` with the current checkpoint.

        - Ensures a full checkpoint file exists (creates a temporary checkpoint if needed)
        - Runs `run_evaluation.py` using the same Python executable
        - Copies `summary.json` into the trainer log dir as `eval_epoch_{epoch:04d}.json`
        - Saves stdout/stderr into `logs/eval_epoch_{epoch:04d}.log`
        """
        logger.info(f"Starting full external evaluation for epoch {epoch}")

        # Determine checkpoint path for this epoch
        ckpt_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        temp_ckpt = None
        if not ckpt_path.exists():
            # Create a temporary full checkpoint containing config & states
            temp_ckpt = self.checkpoint_dir / f"epoch_{epoch:04d}_eval_temp.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'config': self.config,
                'metrics': {}
            }
            try:
                torch.save(checkpoint, temp_ckpt)
                ckpt_to_use = temp_ckpt
                logger.info(f"Saved temporary checkpoint for evaluation: {temp_ckpt}")
            except Exception as e:
                logger.error(f"Failed to write temporary checkpoint for evaluation: {e}")
                return
        else:
            ckpt_to_use = ckpt_path

        # Prepare output directory
        eval_out_dir = self.checkpoint_dir / 'full_eval' / f'epoch_{epoch:04d}'
        eval_out_dir.mkdir(parents=True, exist_ok=True)

        # Build command
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
        except Exception as e:
            logger.error(f"Failed to run external evaluation for epoch {epoch}: {e}")

        # If summary.json exists, copy it to log dir with epoch-specific name
        summary_src = eval_out_dir / 'summary.json'
        if summary_src.exists():
            dst = self.log_dir / f'eval_epoch_{epoch:04d}.json'
            try:
                shutil.copy(summary_src, dst)
                logger.info(f"Saved evaluation summary to {dst}")
            except Exception as e:
                logger.error(f"Failed to copy evaluation summary: {e}")

        # Save stdout/stderr to a log file
        out_log = self.log_dir / f'eval_epoch_{epoch:04d}.log'
        try:
            with open(out_log, 'w', encoding='utf-8') as f:
                if proc is not None:
                    f.write('STDOUT\n')
                    f.write(proc.stdout or '')
                    f.write('\n\nSTDERR\n')
                    f.write(proc.stderr or '')
            logger.info(f"Saved external eval logs to {out_log}")
        except Exception as e:
            logger.error(f"Failed to write external eval logs: {e}")

        # Cleanup temporary checkpoint if we created one
        if temp_ckpt is not None and temp_ckpt.exists():
            try:
                temp_ckpt.unlink()
            except Exception:
                pass

    def _invert_latent(self, latent, style_id, num_steps=15):
        """
        Invert latent from terminal state to noise (reverse ODE).
        
        Args:
            latent: [B, 4, H, W] terminal latent
            style_id: style ID for inversion
            num_steps: number of ODE steps
        
        Returns:
            x0: [B, 4, H, W] inverted noise
        """
        B = latent.shape[0]
        device = latent.device
        
        if isinstance(style_id, int):
            style_id = torch.full((B,), style_id, dtype=torch.long, device=device)
        
        x = latent.clone()
        dt = 1.0 / num_steps
        
        # Reverse integration
        for step_idx in range(num_steps):
            t_forward = 1.0 - step_idx * dt
            t = torch.full((B,), t_forward, device=device)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                v = self.model(x, t, style_id, use_avg_style=False)
            
            x = x - v * dt
        
        return x
    
    def _generate_latent(self, latent, style_id, num_steps=15):
        """
        Generate latent from noise to terminal state (forward ODE).
        
        Args:
            latent: [B, 4, H, W] noise
            style_id: target style ID
            num_steps: number of ODE steps
        
        Returns:
            x1: [B, 4, H, W] generated latent
        """
        B = latent.shape[0]
        device = latent.device
        
        if isinstance(style_id, int):
            style_id = torch.full((B,), style_id, dtype=torch.long, device=device)
        
        x = latent.clone()
        dt = 1.0 / num_steps
        
        # Forward integration with Langevin dynamics
        for step_idx in range(num_steps):
            t_current = step_idx * dt
            t = torch.full((B,), t_current, device=device)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                v = self.model(x, t, style_id, use_avg_style=False)
            
            # Deterministic step
            x = x + v * dt
            
            # Stochastic step (Langevin) for t > 0.5
            if t_current > 0.5:
                sigma = 0.1  # Temperature
                noise = torch.randn_like(x)
                x = x + sigma * np.sqrt(dt) * noise
        
        return x


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LGT Training with Resume Support')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override resume checkpoint if provided
    if args.resume:
        config['training']['resume_checkpoint'] = args.resume
        logger.info(f"Overriding resume checkpoint: {args.resume}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset = InMemoryLatentDataset(
        data_root=config['data']['data_root'],
        num_styles=config['model']['num_styles'],
        style_subdirs=config['data'].get('style_subdirs', None)
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # In-memory dataset, no need for workers
        pin_memory=False,
        drop_last=True
    )
    
    # Create trainer (pass config path for external evaluation usage)
    trainer = LGTTrainer(config, device=device, config_path=str(config_path))
    
    # Start training
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
