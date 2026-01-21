"""
训练脚本：基于 OT-CFM 的风格迁移模型
使用最优传输流匹配 (Optimal Transport Conditional Flow Matching)

优化方案：
1. 全内存数据集（消除IO瓶颈）
2. 移除所有手动显存管理（消除GPU同步停顿）
3. 优化内存布局（channels_last预转换）
4. Pin Memory + Non-blocking Transfer（CPU-GPU异步流水线）
5. Torch Compile（消除Python解释器开销，算子融合）
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import gc
from PIL import Image
import random
import logging
import csv
from datetime import datetime

from model import create_model


class ReflowDataset(Dataset):
    """
    Reflow缓存数据集 - 用于stage2训练
    
    Reflow数据包含两个部分：
    - x0: 初始噪声向量
    - x1: 目标图像的latent
    
    用于优化ODE求解路径，提高生成质量
    """
    def __init__(self, reflow_data_root, num_styles=None):
        super().__init__()
        self.data_root = Path(reflow_data_root)
        
        if not self.data_root.exists():
            raise ValueError(f"Reflow data directory not found: {reflow_data_root}")
        
        print(f"\n⚡ Loading Reflow dataset from {reflow_data_root}...")
        
        # 读取所有reflow对
        pairs = []
        for pair_file in sorted(self.data_root.glob("pair_*.pt")):
            try:
                pair_data = torch.load(pair_file, map_location='cpu', weights_only=True)
                # pair_data 应该包含 x0, x1, style_id
                pairs.append(pair_data)
            except Exception as e:
                print(f"❌ Failed to load {pair_file}: {e}")
        
        if not pairs:
            raise ValueError(f"No reflow pairs found in {reflow_data_root}")
        
        print(f"✅ Loaded {len(pairs)} reflow pairs")
        
        # 堆叠数据
        self.x0_list = []
        self.x1_list = []
        self.style_ids = []
        
        for pair in pairs:
            if isinstance(pair, dict):
                self.x0_list.append(pair['x0'])
                self.x1_list.append(pair['x1'])
                self.style_ids.append(pair.get('style_id', 0))
            elif isinstance(pair, tuple):
                self.x0_list.append(pair[0])
                self.x1_list.append(pair[1])
                self.style_ids.append(pair[2] if len(pair) > 2 else 0)
        
        self.x0_tensor = torch.stack(self.x0_list).contiguous(memory_format=torch.channels_last)
        self.x1_tensor = torch.stack(self.x1_list).contiguous(memory_format=torch.channels_last)
        self.style_tensor = torch.tensor(self.style_ids, dtype=torch.long)
        
        memory_mb = (self.x0_tensor.numel() + self.x1_tensor.numel()) * 4 / (1024**2)
        print(f"Memory usage: {memory_mb:.2f} MB")
        print(f"Data shape: x0={self.x0_tensor.shape}, x1={self.x1_tensor.shape}")
    
    def __len__(self):
        return len(self.style_tensor)
    
    def __getitem__(self, idx):
        return {
            'x0': self.x0_tensor[idx],
            'x1': self.x1_tensor[idx],
            'style_id': self.style_tensor[idx]
        }


class InMemoryLatentDataset(Dataset):
    """
    全内存Latent数据集 - 极速版
    
    设计理念：
    - 训练开始前一次性加载所有数据到RAM
    - 消除训练时的IO瓶颈
    - 预验证数据尺寸，避免运行时插值
    """
    def __init__(self, data_root, num_styles=None):
        super().__init__()
        self.data_root = Path(data_root)
        
        # 读取元数据
        metadata_path = self.data_root.parent / "wikiart_dataset" / "metadata.json"
        if metadata_path.exists():
            print(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            valid_styles = metadata["valid_styles"]
            style_to_id = metadata["style_to_id"]
            
            if num_styles is not None:
                max_style_id = max(style_to_id.values())
                if max_style_id >= num_styles:
                    raise ValueError(
                        f"Metadata contains style_id={max_style_id} but num_classes={num_styles}. "
                        f"Please update config.json 'num_classes' to {max_style_id + 1} or regenerate metadata."
                    )
        else:
            valid_styles = sorted([d.name for d in self.data_root.iterdir() if d.is_dir()])
            style_to_id = {s: i for i, s in enumerate(valid_styles)}
            print("⚠ Metadata not found, using folder names for style mapping")
            
            if num_styles is not None and len(valid_styles) > num_styles:
                raise ValueError(
                    f"Found {len(valid_styles)} styles but num_classes={num_styles}. "
                    f"Update config.json or reduce the number of style folders."
                )
        
        self.style_to_id = style_to_id
        self.num_expected_styles = num_styles
        
        print("\n⚡ Loading all latents into RAM for maximum training speed...")
        print("This may take 30-60 seconds but will eliminate all IO bottlenecks.\n")
        
        latents_list = []
        styles_list = []
        failed_files = []
        
        # 预加载所有数据
        for style_name in valid_styles:
            style_dir = self.data_root / style_name
            if not style_dir.exists():
                continue
            
            style_id = style_to_id[style_name]
            latent_files = list(style_dir.glob("*.pt"))
            
            for fpath in tqdm(latent_files, desc=f"Loading {style_name}", leave=False):
                try:
                    latent = torch.load(fpath, map_location='cpu', weights_only=True)
                    
                    # 严格验证尺寸 - 如果不是32x32，跳过并记录
                    if latent.shape != (4, 32, 32):
                        failed_files.append((fpath, latent.shape))
                        continue
                    
                    latents_list.append(latent)
                    styles_list.append(style_id)
                    
                except Exception as e:
                    print(f"❌ Failed to load {fpath}: {e}")
        
        if len(latents_list) == 0:
            raise ValueError(f"No valid latent files found in {data_root}!")
        
        # 转换为大Tensor（一次性操作，训练时零拷贝）
        print("\n📦 Stacking tensors into single array...")
        self.latents_tensor = torch.stack(latents_list)  # [N, 4, 32, 32]
        self.styles_tensor = torch.tensor(styles_list, dtype=torch.long)  # [N]
        
        # 预转换为channels_last格式（避免训练时转换）
        self.latents_tensor = self.latents_tensor.contiguous(memory_format=torch.channels_last)
        
        # 统计信息
        memory_mb = self.latents_tensor.element_size() * self.latents_tensor.numel() / (1024**2)
        unique_styles = sorted(set(styles_list))
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(self.styles_tensor)}")
        print(f"   Unique styles: {unique_styles}")
        print(f"   Memory usage: {memory_mb:.2f} MB")
        print(f"   Tensor shape: {self.latents_tensor.shape}")
        
        if failed_files:
            print(f"\n⚠️  Warning: {len(failed_files)} files skipped due to wrong shape:")
            for fpath, shape in failed_files[:5]:  # 只显示前5个
                print(f"   - {fpath.name}: {shape} (expected [4, 32, 32])")
            if len(failed_files) > 5:
                print(f"   ... and {len(failed_files) - 5} more")
            print(f"\n💡 Tip: Re-run preprocess_latents.py with target_size=256 to fix this.")
        
        if num_styles is not None and max(unique_styles) >= num_styles:
            raise ValueError(
                f"Dataset contains style_id={max(unique_styles)} but model expects num_classes={num_styles}"
            )
    
    def __len__(self):
        return len(self.styles_tensor)
    
    def __getitem__(self, idx):
        """极速返回 - 无IO，无变换"""
        return {
            'latent': self.latents_tensor[idx],  # 已经是 channels_last
            'style_id': self.styles_tensor[idx]
        }


class OTCFMTrainer:
    """
    OT-CFM 训练器 - 优化版
    
    关键优化：
    1. 移除所有torch.cuda.empty_cache()调用（消除GPU同步停顿）
    2. 移除手动del操作（交给PyTorch自动管理）
    3. 简化训练循环（减少CPU开销）
    """
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        
        # 从配置读取 num_classes 用于验证
        self.expected_num_classes = config['data']['num_classes']
        
        # 训练配置
        train_cfg = config['training']
        self.batch_size = train_cfg['batch_size']
        self.learning_rate = train_cfg['learning_rate']
        self.num_epochs = train_cfg.get('stage1_epochs', 200)
        self.use_amp = train_cfg.get('use_amp', True)
        self.label_drop_prob = train_cfg.get('label_drop_prob', 0.10)
        
        # CFG策略
        self.use_avg_style_for_uncond = train_cfg.get('use_avg_style_for_uncond', True)
        
        # 动态epsilon
        self.dynamic_epsilon = train_cfg.get('dynamic_epsilon', True)
        self.epsilon_warmup_epochs = train_cfg.get('epsilon_warmup_epochs', 100)
        self.current_epoch = 0
        
        # 推理配置
        self.eval_step = train_cfg.get('eval_step', 10)
        self.inference_cfg = config.get('inference', {})
        
        # 断点续传配置
        self.resume_checkpoint = train_cfg.get('resume_checkpoint', '')
        self.save_interval = train_cfg.get('save_interval', 10)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6
        )
        
        # AMP Scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        # Checkpoint
        self.save_dir = Path(config['checkpoint']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 推理输出目录
        self.inference_dir = self.save_dir / "inference"
        self.inference_dir.mkdir(exist_ok=True)
        
        # VAE 解码器（推理用）
        self.vae = None
        self._init_vae()
        
        # 数据集引用（稍后在train方法中初始化）
        self.dataset = None
        
        # 记录起始epoch
        self.start_epoch = 1
        
        # 日志系统
        self._init_logging()
    
    def _init_logging(self):
        """初始化日志系统"""
        # 创建logs目录
        self.log_dir = self.save_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # 时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 文本日志文件
        self.log_file = self.log_dir / f"training_{timestamp}.log"
        self.csv_file = self.log_dir / f"training_{timestamp}.csv"
        
        # 配置logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # CSV日志头
        self.csv_headers = [
            'epoch', 'stage', 'avg_loss', 'learning_rate', 'epsilon',
            'inference_time', 'checkpoint_saved', 'notes'
        ]
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writeheader()
    
    def _log_training_info(self, epoch, avg_loss, learning_rate, epsilon, inference_time=None):
        """
        记录训练信息到日志
        
        Args:
            epoch: 当前epoch
            avg_loss: 平均loss
            learning_rate: 学习率
            epsilon: 当前epsilon值
            inference_time: 推理耗时（秒）
        """
        msg = f"Epoch {epoch:3d}/{self.num_epochs} | Loss: {avg_loss:.6f} | LR: {learning_rate:.2e} | ε: {epsilon:.4f}"
        if inference_time:
            msg += f" | Inference: {inference_time:.1f}s"
        self.logger.info(msg)
        
        # 写入CSV
        row = {
            'epoch': epoch,
            'stage': 'stage1',
            'avg_loss': f"{avg_loss:.6f}",
            'learning_rate': f"{learning_rate:.2e}",
            'epsilon': f"{epsilon:.4f}",
            'inference_time': f"{inference_time:.1f}s" if inference_time else "—",
            'checkpoint_saved': '✓',
            'notes': ''
        }
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writerow(row)
    
    def _init_vae(self):
        """初始化 VAE 用于解码"""
        try:
            from diffusers import AutoencoderKL
            print("Loading VAE decoder for inference...")
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
            self.vae = self.vae.to(self.device)
            self.vae.eval()
            print("✓ VAE decoder loaded")
        except Exception as e:
            print(f"⚠️  Failed to load VAE: {e}")
            print("   Inference will only save latent files")
            self.vae = None
    
    def _latent_to_image(self, latent_tensor):
        """
        将latent解码为图片
        
        Args:
            latent_tensor: [B, 4, H, W] latent 张量
        
        Returns:
            PIL Image 或 None
        """
        if self.vae is None:
            return None
        
        try:
            with torch.no_grad():
                # 解码：latent -> 图片
                decoded = self.vae.decode(latent_tensor / 0.18215).sample
                # 转到 [0, 1]
                decoded = (decoded + 1.0) / 2.0
                decoded = torch.clamp(decoded, 0, 1)
                
                # 转为 PIL Image
                img = decoded[0].cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                return Image.fromarray(img)
        except Exception as e:
            print(f"   Error decoding latent: {e}")
            return None
    
    def _prepare_inference_samples(self):
        """从数据集中为每个风格类别选择一张代表图片用于推理"""
        # 使用已加载的数据集，而不是从磁盘读取
        if not hasattr(self, 'dataset') or self.dataset is None:
            return  # 如果数据集还未准备，延迟初始化
        
        self.inference_samples = {}
        unique_styles = set()
        
        # 从内存数据集中为每个style找一个样本
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            style_id = item['style_id'].item()
            
            # 找到style_id对应的名称
            for style_name, sid in self.dataset.style_to_id.items():
                if sid == style_id and style_name not in unique_styles:
                    latent = item['latent']
                    self.inference_samples[style_name] = {
                        'latent': latent,
                        'style_id': style_id
                    }
                    unique_styles.add(style_name)
                    break
            
            if len(unique_styles) == len(self.dataset.style_to_id):
                break  # 已收集所有style
        
        if self.inference_samples:
            print(f"✓ Prepared {len(self.inference_samples)} inference samples")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载checkpoint进行断点续传
        
        Args:
            checkpoint_path: checkpoint文件路径
        
        Returns:
            bool: 是否成功加载
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            self.logger.info(f"📥 Loading checkpoint: {checkpoint_path.name}")
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("  ✓ Model state loaded")
            
            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("  ✓ Optimizer state loaded")
            
            # 加载学习率调度器状态
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("  ✓ Scheduler state loaded")
            
            # 加载AMP scaler状态
            if 'scaler_state_dict' in checkpoint and self.use_amp:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.logger.info("  ✓ AMP Scaler state loaded")
            
            # 恢复训练进度
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.current_epoch = checkpoint.get('epoch', 0)
            
            self.logger.info(f"✅ Resume from epoch {self.start_epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @torch.no_grad()
    def run_inference(self, epoch):
        """
        运行推理：每个源图片转换到所有目标风格
        使用OT逻辑：从源latent通过风格转换场到达目标风格
        
        Args:
            epoch: 当前epoch数
        """
        if not self.inference_samples:
            self.logger.warning("Skip inference: no samples prepared")
            return
        
        import time
        inference_start = time.time()
        
        self.model.eval()
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"🎨 Running Inference at Epoch {epoch}")
        self.logger.info(f"{'='*60}")
        
        # 推理参数
        num_steps = self.inference_cfg.get('num_inference_steps', 15)
        cfg_scale = self.inference_cfg.get('cfg_scale', 2.0)
        use_cfg = self.inference_cfg.get('use_cfg', True)
        
        # 为本次epoch创建子目录
        epoch_dir = self.inference_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计推理结果
        total_generated = 0
        
        # 遍历每个源图片
        for src_style, src_data in self.inference_samples.items():
            self.logger.info(f"📷 Source: {src_style}")
            
            # 直接使用内存中的latent
            src_latent = src_data['latent'].unsqueeze(0).to(self.device, memory_format=torch.channels_last)
            src_style_id = src_data['style_id']
            
            # 保存原始图片
            src_img = self._latent_to_image(src_latent)
            if src_img is not None:
                src_img.save(str(epoch_dir / f"00_src_{src_style}.png"))
                self.logger.info(f"  📸 Source image: 00_src_{src_style}.png")
            
            # 转换到每个目标风格（包括自身）
            for tgt_style, tgt_data in self.inference_samples.items():
                tgt_style_id = tgt_data['style_id']
                
                # OT-CFM的正确推理：从源latent开始，经过velocity field进行风格转换
                # x(t=0) = x_src，通过t从0到1的积分，得到x(t=1)在目标风格空间中的对应
                x = src_latent.clone()  # 从源latent开始！而不是随机噪声
                
                # ODE求解
                dt = 1.0 / num_steps
                tgt_id_tensor = torch.tensor([tgt_style_id], dtype=torch.long, device=self.device)
                
                for step in range(num_steps):
                    t = torch.full((1,), step * dt, device=self.device)
                    
                    if use_cfg:
                        # Classifier-Free Guidance
                        # 为了避免torch.compile的CUDA Graph复用问题，显式克隆输出
                        v_cond = self.model(x, t, tgt_id_tensor, use_avg_style=False).clone()
                        
                        # 标记CUDA Graph的新步骤
                        if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                            torch.compiler.cudagraph_mark_step_begin()
                        
                        v_uncond = self.model(x, t, tgt_id_tensor, use_avg_style=True)
                        v = v_uncond + cfg_scale * (v_cond - v_uncond)
                    else:
                        v = self.model(x, t, tgt_id_tensor, use_avg_style=False)
                    
                    x = x + v * dt
                
                # 保存生成的图片
                gen_img = self._latent_to_image(x)
                if gen_img is not None:
                    gen_img.save(str(epoch_dir / f"{src_style}_to_{tgt_style}.png"))
                    total_generated += 1
        
        inference_time = time.time() - inference_start
        self.logger.info(f"✅ Inference completed in {inference_time:.1f}s ({total_generated} images)")
        self.logger.info(f"{'='*60}")
        
        self.model.train()
        
        return inference_time
    
    def get_dynamic_epsilon(self):
        """动态epsilon调整"""
        if not self.dynamic_epsilon:
            return 0.0
        epsilon = min(0.1, self.current_epoch / self.epsilon_warmup_epochs)
        return epsilon
    
    def compute_otcfm_loss(self, x1, style_id):
        """计算 OT-CFM 损失"""
        batch_size = x1.size(0)
        
        # 1. 采样 x0 ~ N(0, I)
        x0 = torch.randn_like(x1)
        
        # 2. 采样时间 t ~ Uniform(ε, 1)
        epsilon = self.get_dynamic_epsilon()
        t = torch.rand(batch_size, device=self.device) * (1.0 - epsilon) + epsilon
        
        # 3. 构造路径 x_t = (1-t)*x0 + t*x1
        t_expanded = t[:, None, None, None]
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # 4. 计算目标速度场 u_t = x1 - x0
        u_t = x1 - x0
        
        # 5. Label Dropping for CFG
        drop_mask = torch.rand(batch_size, device=self.device) < self.label_drop_prob
        
        if self.use_avg_style_for_uncond and drop_mask.any():
            v_pred = self.model(x_t, t, style_id, use_avg_style=False)
            
            if drop_mask.sum() > 0:
                x_t_drop = x_t[drop_mask]
                t_drop = t[drop_mask]
                style_id_drop = style_id[drop_mask]
                v_pred_drop = self.model(x_t_drop, t_drop, style_id_drop, use_avg_style=True)
                v_pred[drop_mask] = v_pred_drop
        else:
            v_pred = self.model(x_t, t, style_id, use_avg_style=False)
        
        # 6. MSE Loss
        loss = F.mse_loss(v_pred, u_t)
        return loss
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch - 极致优化版"""
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0
        epsilon = self.get_dynamic_epsilon()
        
        # leave=False 保持控制台清爽
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.num_epochs}", leave=False)
        
        for batch in pbar:
            # non_blocking=True 实现 CPU-GPU 异步传输
            latent = batch['latent'].to(self.device, non_blocking=True)
            style_id = batch['style_id'].to(self.device, non_blocking=True)
            
            # 验证范围
            if style_id.max().item() >= self.expected_num_classes:
                raise ValueError(
                    f"Batch contains style_id={style_id.max().item()} but expected num_classes={self.expected_num_classes}"
                )
            
            # 数据已经是 channels_last，无需转换
            
            # 训练步骤
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                loss = self.compute_otcfm_loss(latent, style_id)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 记录
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ε': f"{epsilon:.3f}"})
            
            # 🟢 移除所有手动内存管理：
            # - 不要 del latent, style_id, loss
            # - 不要 torch.cuda.empty_cache()
            # - 不要 gc.collect()
            # PyTorch 会自动管理这些！
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def train(self, dataloader):
        """完整训练流程"""
        import time
        train_start_time = time.time()
        
        self.logger.info("="*80)
        self.logger.info("🚀 OT-CFM Training Started")
        self.logger.info("="*80)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"GPU: {torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"AMP: {self.use_amp}")
        self.logger.info(f"Compile: {self.model.__class__.__name__}")
        self.logger.info(f"Dataset size: {len(dataloader.dataset)}")
        self.logger.info(f"Batches per epoch: {len(dataloader)}")
        self.logger.info("="*80)
        
        # 尝试加载断点
        checkpoint_path = Path(self.resume_checkpoint) if self.resume_checkpoint else None
        if checkpoint_path and checkpoint_path.exists():
            # 如果指定了checkpoint路径且存在，加载它
            self.load_checkpoint(checkpoint_path)
        else:
            # 自动查找最新的checkpoint
            ckpt_files = sorted(self.save_dir.glob("stage1_epoch*.pt"))
            if ckpt_files:
                latest_ckpt = ckpt_files[-1]
                self.logger.info(f"📂 Found latest checkpoint: {latest_ckpt.name}")
                self.load_checkpoint(latest_ckpt)
            else:
                self.logger.info("No checkpoint found, starting fresh training")
        
        # 保存dataset引用并初始化inference样本
        self.dataset = dataloader.dataset
        self._prepare_inference_samples()
        
        # 初始化平均风格嵌入
        if self.start_epoch == 1:
            self.logger.info("Initializing average style embedding...")
            self.model.initialize_avg_style_embedding()
            self.logger.info("✓ Average style embedding initialized")
        else:
            self.logger.info(f"⏭️  Resuming from epoch {self.start_epoch}")
        
        self.logger.info("")
        
        # 训练循环
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            epoch_start = time.time()
            avg_loss = self.train_epoch(dataloader, epoch)
            epoch_time = time.time() - epoch_start
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            epsilon = self.get_dynamic_epsilon()
            
            # 记录训练信息
            self._log_training_info(epoch, avg_loss, current_lr, epsilon)
            
            # 定期推理
            inference_time = None
            if epoch % self.eval_step == 0 or epoch == self.num_epochs:
                inference_time = self.run_inference(epoch)
                self.logger.info("")
            
            # 保存checkpoint
            if epoch % self.save_interval == 0 or epoch == self.num_epochs:
                self.save_checkpoint(epoch)
                self.logger.info("")
        
        # 保存final checkpoint
        final_path = self.save_dir / "stage1_final.pt"
        if not final_path.exists():
            self.save_checkpoint(self.num_epochs, is_final=True)
        
        # 总结
        total_time = time.time() - train_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info("="*80)
        self.logger.info(f"✅ Stage1 Training completed!")
        self.logger.info(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        self.logger.info(f"Checkpoint dir: {self.save_dir}")
        self.logger.info(f"Logs: {self.log_dir}")
        self.logger.info("="*80)
    
    def train_stage2(self, reflow_dataloader):
        """Stage2: Reflow 训练 - 基于stage1模型进行重流程优化"""
        import time
        stage2_start_time = time.time()
        
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("🚀 OT-CFM Stage2 (Reflow) Training Started")
        self.logger.info("="*80)
        
        # 获取stage2配置
        train_cfg = self.config['training']
        stage2_epochs = train_cfg.get('stage2_epochs', 50)
        stage2_lr = train_cfg.get('stage2_learning_rate', self.learning_rate * 0.1)
        
        # 调整为stage2配置
        self.num_epochs = stage2_epochs
        self.start_epoch = 1
        self.current_epoch = 0
        
        # 重置优化器和调度器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=stage2_lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=stage2_epochs,
            eta_min=1e-6
        )
        
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Stage2 Learning rate: {stage2_lr}")
        self.logger.info(f"Dataset size: {len(reflow_dataloader.dataset)}")
        self.logger.info(f"Batches per epoch: {len(reflow_dataloader)}")
        self.logger.info("="*80)
        
        self.logger.info("")
        
        # Stage2训练循环
        for epoch in range(1, stage2_epochs + 1):
            epoch_start = time.time()
            avg_loss = self.train_epoch(reflow_dataloader, epoch)
            epoch_time = time.time() - epoch_start
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            epsilon = self.get_dynamic_epsilon()
            
            # 记录训练信息（标记为stage2）
            msg = f"Epoch {epoch:3d}/{stage2_epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | ε: {epsilon:.4f}"
            self.logger.info(msg)
            
            # 写入CSV
            row = {
                'epoch': epoch,
                'stage': 'stage2',
                'avg_loss': f"{avg_loss:.6f}",
                'learning_rate': f"{current_lr:.2e}",
                'epsilon': f"{epsilon:.4f}",
                'inference_time': "—",
                'checkpoint_saved': '✓' if epoch % train_cfg.get('stage2_save_interval', 5) == 0 else '—',
                'notes': ''
            }
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writerow(row)
            
            # 定期推理
            if epoch % self.eval_step == 0 or epoch == stage2_epochs:
                inference_time = self.run_inference(epoch)
                self.logger.info("")
            
            # 保存stage2 checkpoint
            if epoch % train_cfg.get('stage2_save_interval', 5) == 0 or epoch == stage2_epochs:
                stage2_ckpt_path = self.save_dir / f"stage2_epoch{epoch}.pt"
                checkpoint = {
                    'epoch': epoch,
                    'stage': 2,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
                    'config': self.config,
                    'training_info': {
                        'current_epoch': epoch,
                        'learning_rate': current_lr,
                        'epsilon': epsilon,
                    }
                }
                torch.save(checkpoint, str(stage2_ckpt_path))
                file_size_mb = stage2_ckpt_path.stat().st_size / (1024 ** 2)
                self.logger.info(f"💾 Stage2 checkpoint saved: {stage2_ckpt_path.name} ({file_size_mb:.1f}MB)")
                self.logger.info("")
        
        # 保存final stage2 checkpoint
        final_stage2_path = self.save_dir / "stage2_final.pt"
        checkpoint = {
            'epoch': stage2_epochs,
            'stage': 2,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'config': self.config,
            'training_info': {
                'current_epoch': stage2_epochs,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epsilon': self.get_dynamic_epsilon(),
            }
        }
        torch.save(checkpoint, str(final_stage2_path))
        self.logger.info(f"💾 Final stage2 checkpoint saved: {final_stage2_path.name}")
        
        # 总结
        total_time = time.time() - stage2_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info("="*80)
        self.logger.info(f"✅ Stage2 (Reflow) Training completed!")
        self.logger.info(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        self.logger.info("="*80)
    
    def save_checkpoint(self, epoch, is_final=False):
        """
        保存checkpoint
        
        Args:
            epoch: 当前epoch数
            is_final: 是否为最终checkpoint
        """
        if is_final:
            checkpoint_path = self.save_dir / "stage1_final.pt"
        else:
            checkpoint_path = self.save_dir / f"stage1_epoch{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'config': self.config,
            'training_info': {
                'current_epoch': epoch,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epsilon': self.get_dynamic_epsilon(),
            }
        }
        
        # 转为字符串确保兼容性
        torch.save(checkpoint, str(checkpoint_path))
        file_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path.name} ({file_size_mb:.1f}MB)")
        
        if not is_final:
            self._cleanup_old_checkpoints(keep_last=3)
    
    def _cleanup_old_checkpoints(self, keep_last=3):
        """
        清理旧的checkpoint文件，只保留最近的N个
        
        Args:
            keep_last: 保留最近的几个checkpoint
        """
        ckpt_files = sorted(
            self.save_dir.glob("stage1_epoch*.pt"),
            key=lambda x: int(x.stem.split('epoch')[-1])
        )
        
        if len(ckpt_files) <= keep_last:
            return  # 文件不足，无需清理
        
        # 保留最后N个即可
        to_remove = ckpt_files[:-keep_last]
        
        for ckpt in to_remove:
            try:
                ckpt.unlink()
            except Exception:
                pass  # 静默失败，不影响训练


def main():
    # 加载配置
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 底层优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # ========== Stage1: OT-CFM Training ==========
    print("\n" + "="*80)
    print("STAGE1: OT-CFM Training")
    print("="*80)
    
    data_cfg = config['data']
    train_cfg = config['training']
    
    # Stage1: 使用latent数据集
    print("\nLoading Stage1 dataset...")
    dataset_stage1 = InMemoryLatentDataset(
        data_root=data_cfg['data_root'],
        num_styles=data_cfg['num_classes']
    )
    print(f"Dataset size: {len(dataset_stage1)}")
    
    # 创建模型
    print("\nCreating model...")
    model = create_model(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")
    
    expected_num_classes = data_cfg['num_classes']
    print(f"Model configured for num_classes: {expected_num_classes}")
    
    # Torch Compile
    if train_cfg.get('use_compile', True):
        try:
            print("\n🚀 Compiling model with torch.compile...")
            print("   This may take 1-2 minutes on first run but will be cached.")
            model = torch.compile(model, mode="reduce-overhead")
            print("✅ Model compiled successfully!")
        except Exception as e:
            print(f"⚠️  Torch compile failed: {e}")
            print("   Continuing with eager mode (no performance loss if PyTorch < 2.0)")
    else:
        print("✓ Model ready (native PyTorch eager mode)")
    
    # DataLoader - Stage1
    dataloader_stage1 = DataLoader(
        dataset_stage1,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Batches per epoch: {len(dataloader_stage1)}")
    
    # 训练器
    trainer = OTCFMTrainer(config, model, device)
    
    # Stage1训练
    trainer.train(dataloader_stage1)
    
    # ========== Stage2: Reflow Training ==========
    stage2_enabled = train_cfg.get('enable_stage2', False)
    
    if stage2_enabled:
        print("\n" + "="*80)
        print("STAGE2: Reflow Training")
        print("="*80)
        
        reflow_data_dir = train_cfg.get('reflow_data_dir', 'data_reflow_cache')
        
        try:
            # 加载reflow数据集
            print(f"\nLoading Stage2 (Reflow) dataset...")
            dataset_stage2 = ReflowDataset(
                reflow_data_root=reflow_data_dir,
                num_styles=data_cfg['num_classes']
            )
            
            # DataLoader - Stage2
            dataloader_stage2 = DataLoader(
                dataset_stage2,
                batch_size=train_cfg['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True
            )
            
            # Stage2训练
            trainer.train_stage2(dataloader_stage2)
            
            print("\n" + "="*80)
            print("✅ Both Stage1 and Stage2 training completed!")
            print("="*80)
            
        except Exception as e:
            print(f"\n⚠️  Stage2 skipped: {e}")
            print("   Check that reflow data directory exists and contains pair_*.pt files")
    else:
        print("\nℹ️  Stage2 disabled in config. Set 'enable_stage2': true to enable reflow training.")


if __name__ == "__main__":
    main()
