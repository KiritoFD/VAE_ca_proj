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
from losses import GeometricFreeEnergyLoss, VelocityRegularizationLoss
from inference import LGTInference, load_vae, encode_image, decode_latent


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InMemoryLatentDataset(Dataset):
    """
    In-memory dataset for VAE latents.
    Loads all latents into RAM for maximum training speed.
    """
    
    def __init__(self, data_root, num_styles, style_subdirs=None):
        """
        Args:
            data_root: Path to latent files
            num_styles: Number of style classes
            style_subdirs: List of subdirectory names for each style
        """
        self.data_root = Path(data_root)
        self.num_styles = num_styles
        
        if style_subdirs is None:
            style_subdirs = [f"style{i}" for i in range(num_styles)]
        
        self.style_subdirs = style_subdirs
        
        # Load all latents into memory
        self.latents_list = []
        self.styles_list = []
        
        logger.info("Loading latents into memory...")
        for style_id, subdir in enumerate(style_subdirs):
            style_path = self.data_root / subdir
            if not style_path.exists():
                logger.warning(f"Style directory not found: {style_path}")
                continue
            
            latent_files = sorted(style_path.glob("*.pt"))
            logger.info(f"  Style {style_id} ({subdir}): {len(latent_files)} files")
            
            for latent_file in latent_files:
                latent = torch.load(latent_file, map_location='cpu')
                
                # Ensure correct shape [4, H, W]
                if latent.ndim == 4:
                    latent = latent.squeeze(0)
                
                self.latents_list.append(latent)
                self.styles_list.append(style_id)
        
        # Stack into tensors
        self.latents_tensor = torch.stack(self.latents_list)
        self.styles_tensor = torch.tensor(self.styles_list, dtype=torch.long)
        
        # Convert to channels_last for better GPU performance
        self.latents_tensor = self.latents_tensor.contiguous(
            memory_format=torch.channels_last
        )
        
        logger.info(f"✓ Loaded {len(self)} latents into memory")
        logger.info(f"  Shape: {self.latents_tensor.shape}")
        logger.info(f"  Memory: {self.latents_tensor.nbytes / 1e9:.2f} GB")
    
    def __len__(self):
        return len(self.latents_tensor)
    
    def __getitem__(self, idx):
        return {
            'latent': self.latents_tensor[idx],
            'style_id': self.styles_tensor[idx]
        }


class LGTTrainer:
    """
    Trainer for LGT model with geometric free energy optimization.
    """
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        
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
                self.model = torch.compile(self.model)
                logger.info("✓ Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        
        # Create loss functions
        self.energy_loss = GeometricFreeEnergyLoss(
            w_style=config['loss']['w_style'],
            w_content=config['loss']['w_content'],
            patch_size=config['loss']['patch_size'],
            num_projections=config['loss']['num_projections'],
            max_samples=config['loss']['max_samples']
        ).to(device)
        
        # Optional velocity regularization
        self.use_vel_reg = config['loss'].get('use_velocity_reg', False)
        if self.use_vel_reg:
            self.vel_reg = VelocityRegularizationLoss(
                weight=config['loss'].get('vel_reg_weight', 0.01)
            ).to(device)
        
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
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Training config
        self.num_epochs = config['training']['num_epochs']
        self.label_drop_prob = config['training'].get('label_drop_prob', 0.1)
        self.use_avg_for_uncond = config['training'].get('use_avg_style_for_uncond', True)
        
        # Epsilon for time sampling (avoid t=0 singularity)
        self.epsilon = config['training'].get('epsilon', 0.01)
        
        # ODE integration steps for computing terminal state
        self.ode_steps = config['training'].get('ode_integration_steps', 5)
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = config['training'].get('save_interval', 5)
        
        # Evaluation and inference
        self.eval_interval = config['training'].get('eval_interval', 5)
        self.test_image_dir = Path(config['training'].get('test_image_dir', 'test_images'))
        self.inference_dir = self.checkpoint_dir / 'inference'
        self.inference_dir.mkdir(exist_ok=True)
        
        # Load VAE for inference
        self.vae = load_vae(device)
        
        # Logging
        self.log_dir = self.checkpoint_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Initialize log file (add epoch_time column)
        with open(self.log_file, 'w') as f:
            f.write('epoch,loss_total,loss_style_swd,loss_content_ssm,learning_rate,epoch_time\n')
        
        # Resume checkpoint support
        self.start_epoch = 1
        resume_ckpt = config['training'].get('resume_checkpoint')
        if resume_ckpt:
            self.load_checkpoint(resume_ckpt)
    
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
    
    def compute_energy_loss(self, x_src, style_id_src, style_id_tgt, epoch):
        """
        Compute geometric free energy loss.
        
        Training procedure:
        1. Sample x0 ~ N(0, I), t ~ U(ε, 1)
        2. Construct path x_t = (1-t)*x0 + t*x_src (anchored to content)
        3. Integrate to x_1
        4. Sample style reference x_style
        5. Compute E(x_1) = w_style*SWD(x_1, x_style) + w_content*SSM(x_1, x_src)
        
        Args:
            x_src: [B, 4, H, W] source content latents
            style_id_src: [B] source style IDs
            style_id_tgt: [B] target style IDs
            epoch: current epoch for epsilon scheduling
        
        Returns:
            loss_dict: dictionary with loss components
        """
        B = x_src.shape[0]
        device = x_src.device
        
        # Sample noise x0 ~ N(0, I)
        x0 = torch.randn_like(x_src)
        
        # Sample time t ~ U(ε, 1)
        epsilon = self.get_dynamic_epsilon(epoch)
        t = torch.rand(B, device=device) * (1 - epsilon) + epsilon
        
        # Construct path x_t = (1-t)*x0 + t*x_src
        t_expand = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x_src
        
        # Label dropping for CFG training
        use_avg_style = torch.rand(1).item() < self.label_drop_prob
        
        # Integrate ODE to terminal state x_1
        x_1 = self.integrate_ode(x_t, t, style_id_tgt, use_avg_style=use_avg_style)
        
        # For style reference, we need another sample from target style
        # In practice, we can use a different sample from the batch or same sample
        # Here we use the same x_src but with target style as reference
        # (This assumes x_src will be transformed to match target style distribution)
        x_style_ref = x_src  # Placeholder - in real training, sample from target style
        
        # Compute geometric free energy
        # E(x_1) = w_style*SWD(x_1, x_style) + w_content*SSM(x_1, x_src)
        loss_dict = self.energy_loss(x_1, x_style_ref, x_src)
        
        # Optional: Add velocity regularization
        if self.use_vel_reg:
            # Compute velocity at intermediate point for regularization
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                v_mid = self.model(x_t, t, style_id_tgt, use_avg_style=False)
            vel_reg_loss = self.vel_reg(v_mid)
            loss_dict['total'] = loss_dict['total'] + vel_reg_loss
            loss_dict['vel_reg'] = vel_reg_loss
        
        return loss_dict
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_style_swd = 0.0
        total_content_ssm = 0.0
        num_batches = 0
        
        import sys
        # Disable tqdm when not running in an interactive terminal (prevents polluted log files)
        use_tqdm = sys.stderr.isatty()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.num_epochs}", disable=not use_tqdm, leave=use_tqdm)
        
        for step_idx, batch in enumerate(pbar, start=1):
            latent = batch['latent'].to(self.device, non_blocking=True)
            style_id = batch['style_id'].to(self.device, non_blocking=True)
            
            # For style transfer training, we need source and target styles
            # Here we randomly shuffle to create style transfer pairs
            B = latent.shape[0]
            indices = torch.randperm(B)
            style_id_tgt = style_id[indices]
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                loss_dict = self.compute_energy_loss(
                    latent, style_id, style_id_tgt, epoch
                )
                loss = loss_dict['total']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_style_swd += loss_dict['style_swd'].item()
            total_content_ssm += loss_dict['content_ssm'].item()
            num_batches += 1
            
            # Update progress bar or log periodically if disabled
            if use_tqdm:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'swd': f"{loss_dict['style_swd'].item():.4f}",
                    'ssm': f"{loss_dict['content_ssm'].item():.4f}"
                })
            else:
                # Log structured progress every N steps to keep logs readable
                if step_idx % 100 == 0 or step_idx == len(dataloader):
                    logger.info(
                        f"Epoch {epoch} Step {step_idx}/{len(dataloader)} | "
                        f"loss={loss.item():.4f} | "
                        f"swd={loss_dict['style_swd'].item():.6f} | "
                        f"ssm={loss_dict['content_ssm'].item():.4f}"
                    )
        
        # Compute epoch averages
        avg_loss = total_loss / num_batches
        avg_style_swd = total_style_swd / num_batches
        avg_content_ssm = total_content_ssm / num_batches
        
        return {
            'loss': avg_loss,
            'style_swd': avg_style_swd,
            'content_ssm': avg_content_ssm
        }
    
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
        
        # Save as latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint for resuming training."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
        
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # Announce epoch start
            logger.info(f"Starting epoch {epoch}/{self.num_epochs}")
            
            # Train one epoch
            metrics = self.train_epoch(dataloader, epoch)
            
            # Learning rate step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = metrics.get('epoch_time', 0.0)
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch}/{self.num_epochs} completed in {epoch_time:.1f}s | "
                f"Loss: {metrics['loss']:.4f} | "
                f"SWD: {metrics['style_swd']:.4f} | "
                f"SSM: {metrics['content_ssm']:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Write to CSV (append epoch_time)
            with open(self.log_file, 'a') as f:
                f.write(
                    f"{epoch},{metrics['loss']:.6f},"
                    f"{metrics['style_swd']:.6f},{metrics['content_ssm']:.6f},"
                    f"{current_lr:.2e},{epoch_time:.2f}\n"
                )
            
            # Append structured epoch log to JSONL
            epoch_log_path = self.log_dir / 'epoch_logs.jsonl'
            epoch_entry = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'loss': float(metrics['loss']),
                'style_swd': float(metrics['style_swd']),
                'content_ssm': float(metrics['content_ssm']),
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
        
        # Create inference directory for this epoch
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
                
                # Transfer to all target styles
                for tgt_style_id in range(num_styles):
                    if tgt_style_id == src_style_id:
                        continue  # Skip same style
                    
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
        pin_memory=True,
        drop_last=True
    )
    
    # Create trainer
    trainer = LGTTrainer(config, device=device)
    
    # Start training
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
