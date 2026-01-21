import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import json
import random
import os
import sys
import logging
import time
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
import torch.nn.functional as F
import gc  # 🟢 新增：垃圾回收

from SAFlow import SAFModel
from dataset import Stage1Dataset, Stage2Dataset

# -----------------------------------------------------------------------------
# 硬件加速与全局设置
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ================= 超参数 =================
EVAL_STEP = 1           # 每多少个 Epoch 进行一次推理验证
MAX_GRAD_NORM = 1.0     # 梯度裁剪阈值
IDENTITY_PROB = 0.15    # Stage 1 训练恒等映射的概率
# =========================================

# 🟢 新增：递归更新字典
def deep_update(source, overrides):
    """
    递归更新字典，确保子层级的配置不会被暴力覆盖，而是按键更新。
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

# 🟢 [命名] 频谱幅度损失 (Spectral Amplitude Distance)
def compute_spectral_loss(v_pred, v_gt):
    """
    计算频域损失 (Spectral Amplitude Loss)，强制纹理细节对齐。
    基于帕塞瓦尔定理，在频域计算 MSE 以解决空域 MSE 对高频不敏感的问题。
    """
    # 1. 强制转 float32 避免 FFT 精度溢出
    v_pred = v_pred.float()
    v_gt = v_gt.float()

    # 2. FFT 变换 (Real-to-Complex)
    # norm='ortho' 保证能量守恒
    fft_pred = torch.fft.rfft2(v_pred, norm='ortho')
    fft_gt = torch.fft.rfft2(v_gt, norm='ortho')
    
    # 3. 计算幅度谱 (Amplitude Spectrum)
    amp_pred = torch.abs(fft_pred)
    amp_gt = torch.abs(fft_gt)
    
    # 4. 频域 MSE
    return F.mse_loss(amp_pred, amp_gt)

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # 🟢 修改：使用唯一的 logger 名称避免重名冲突
        self.logger = logging.getLogger(f"LSFM_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # 🟢 新增：防止重复打印
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_dir / "train.log", encoding='utf-8')
            ch = logging.StreamHandler(sys.stdout)
            fmt = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S')
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def info(self, msg): 
        self.logger.info(msg)

class LSFMTrainer:
    # 🟢 核心修改：接收 config_override 参数
    def __init__(self, config_override=None):
        # 1. 加载基础配置
        config_path = Path("config.json")
        if not config_path.exists(): config_path = Path("../config.json")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        
        # 2. 🟢 深度应用配置覆盖 (支持任意参数修改)
        if config_override:
            self.cfg = deep_update(self.cfg, config_override)
        
        self.device = torch.device("cuda")
        self.ckpt_dir = Path(self.cfg['checkpoint']['save_dir'])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志
        self.logger = TrainingLogger(self.ckpt_dir / "logs")
        
        # 🟢 新增：保存本次实验的完整配置，方便回溯
        with open(self.ckpt_dir / "experiment_config.json", "w", encoding='utf-8') as f:
            json.dump(self.cfg, f, indent=4, ensure_ascii=False)
        
        self.reflow_dir = Path(self.cfg['training'].get('reflow_data_dir', 'data/reflow_pairs'))
        
        # 🟢 读取可配置的损失权重
        self.transfer_weight = self.cfg['training'].get('transfer_loss_weight', 1.0)

        # 🟢 读取标签丢弃概率（CFG 训练必需）
        self.label_drop_prob = self.cfg['training'].get('label_drop_prob', 0.15)
        self.null_class_id = self.cfg['model']['num_styles']  # N+1 是空类别
        
        self.logger.info("="*50)
        self.logger.info(f"🚀 Initializing Experiment")
        self.logger.info(f"📂 Save Dir  : {self.ckpt_dir}")
        self.logger.info(f"Device      : {self.device}")
        self.logger.info(f"Batch Size  : {self.cfg['training']['batch_size']}")
        self.logger.info(f"Resolution  : 256x256")
        self.logger.info(f"⚡ LR: {self.cfg['training']['learning_rate']} | Weight: {self.transfer_weight}")
        self.logger.info(f"🎲 Label Drop Prob: {self.label_drop_prob} (Null ID: {self.null_class_id})")
        self.logger.info("="*50)

        # 2. 加载 VAE (FT-MSE)
        self.logger.info("[Init] Loading VAE (stabilityai/sd-vae-ft-mse)...")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)
        self.vae.eval().requires_grad_(False).float()

        # 3. 移除 LPIPS 以节省显存 (我们现在用 Spectral Loss 替代它)
        # self.lpips = ... (Deleted)

        # 4. 初始化训练数据集
        self.logger.info("[Init] Loading Training Dataset...")
        self.train_ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data']['num_classes'])
        self.logger.info(f"[Init] Dataset Ready. Total Latents: {len(self.train_ds)}")

        # 5. 推理预处理
        self.infer_transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def get_model(self):
        model = SAFModel(**self.cfg['model']).to(self.device, memory_format=torch.channels_last)
        self.logger.info("[Model] Compiling network with torch.compile (max-autotune)...")
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            self.logger.info(f"[Model] Compile warning: {e}")
        return model

    def safe_load(self, model, state_dict, strict=True):
        clean_dict = self.clean_sd(state_dict)
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(clean_dict, strict=strict)
        else:
            model.load_state_dict(clean_dict, strict=strict)

    def clean_sd(self, sd):
        if list(sd.keys())[0].startswith("_orig_mod."):
            return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        return sd

    # -----------------------------------------------------------------------------
    # 🟢 新增：检查点恢复逻辑
    # -----------------------------------------------------------------------------
    def find_latest_checkpoint(self, stage="stage1"):
        """查找最新的训练检查点"""
        ckpt_pattern = f"{stage}_epoch*.pt"
        ckpts = list(self.ckpt_dir.glob(ckpt_pattern))
        if not ckpts:
            return None, 0
        
        # 提取 epoch 数字并排序
        import re
        ckpt_epochs = []
        for ckpt in ckpts:
            match = re.search(r'epoch(\d+)', ckpt.name)
            if match:
                ckpt_epochs.append((int(match.group(1)), ckpt))
        
        if not ckpt_epochs:
            return None, 0
        
        # 返回最新的检查点
        latest_epoch, latest_path = max(ckpt_epochs, key=lambda x: x[0])
        return latest_path, latest_epoch

    def load_training_state(self, checkpoint_path, model, optimizer, scheduler):
        """加载完整的训练状态"""
        self.logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 🟢 新增：向后兼容旧格式检查点
        # 旧格式：直接是 state_dict (没有包裹在字典中)
        # 新格式：{'model_state_dict': ..., 'optimizer_state_dict': ..., ...}
        if 'model_state_dict' not in checkpoint:
            # 这是旧格式，直接当作 state_dict 加载
            self.logger.info("⚠️  Old checkpoint format detected (no training state)")
            self.safe_load(model, checkpoint)
            return 0, float('inf')  # 从头开始训练
        
        # 加载模型权重
        self.safe_load(model, checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("✅ Optimizer state restored")
            except Exception as e:
                self.logger.info(f"⚠️  Failed to load optimizer state: {e}")
        
        # 加载调度器状态
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("✅ Scheduler state restored")
            except Exception as e:
                self.logger.info(f"⚠️  Failed to load scheduler state: {e}")
        
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        self.logger.info(f"✅ Resumed from epoch {start_epoch}, best_loss={best_loss:.6f}")
        return start_epoch, best_loss

    # -----------------------------------------------------------------------------
    # Stage 1: Latent Structure Flow Matching
    # -----------------------------------------------------------------------------
    def construct_target_lsfm(self, x_c, x_s):
        x_c, x_s = x_c.float(), x_s.float()
        B, C, H, W = x_c.size()
        eps = 1e-5
        
        zc = x_c.view(B, C, -1)
        zs = x_s.view(B, C, -1)
        mu_c, mu_s = zc.mean(2, keepdim=True), zs.mean(2, keepdim=True)
        zc, zs = zc - mu_c, zs - mu_s
        
        cc = torch.bmm(zc, zc.transpose(1, 2)) / (H*W-1)
        cs = torch.bmm(zs, zs.transpose(1, 2)) / (H*W-1)
        
        try:
            uc, sc, _ = torch.linalg.svd(cc)
            us, ss, _ = torch.linalg.svd(cs)
        except: return x_c 
        
        c_inv = uc @ torch.diag_embed(1.0/torch.sqrt(sc.clamp(min=eps))) @ uc.transpose(1,2)
        s_mat = us @ torch.diag_embed(torch.sqrt(ss.clamp(min=eps))) @ us.transpose(1,2)
        z_wct = (s_mat @ c_inv @ zc + mu_s).view(B, C, H, W)
        
        fc = torch.fft.rfft2(z_wct, norm='ortho')
        fs = torch.fft.rfft2(x_s, norm='ortho')
        target = torch.fft.irfft2(torch.abs(fs) * torch.exp(1j * torch.angle(fc)), s=(H, W), norm='ortho')
        return target

    def run_stage1(self):
        # 🟢 修改：检查 final 而非跳过训练
        final_ckpt = self.ckpt_dir / "stage1_final.pt"
        
        self.logger.info("="*50)
        self.logger.info(">>> Starting Stage 1: LSFM Training")
        self.logger.info("="*50)
        
        model = self.get_model()
        dl = DataLoader(self.train_ds, batch_size=self.cfg['training']['batch_size'], 
                        shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        
        total_epochs = self.cfg['training']['stage1_epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, 
            T_max=total_epochs, 
            eta_min=1e-6
        )

        # 🟢 新增：断点续训逻辑
        start_epoch = 0
        best_loss = float('inf')
        
        # 优先检查是否已完成训练
        if final_ckpt.exists():
            self.logger.info("[Stage 1] ✅ Final checkpoint exists. Skipping training.")
            return
        
        # 查找最新的中间检查点
        latest_ckpt, latest_epoch = self.find_latest_checkpoint("stage1")
        if latest_ckpt:
            start_epoch, best_loss = self.load_training_state(
                latest_ckpt, model, opt, scheduler
            )

        # 🟢 修改：从 start_epoch + 1 开始训练
        for epoch in range(start_epoch + 1, total_epochs + 1):
            model.train()
            epoch_loss = 0.0
            start_time = time.time()
            
            pbar = tqdm(dl, desc=f"[S1] Epoch {epoch}/{total_epochs}", leave=False)
            
            for x_c, x_s, t_id, s_id in pbar:
                x_c = x_c.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                x_s = x_s.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                t_id, s_id = t_id.to(self.device, non_blocking=True), s_id.to(self.device, non_blocking=True)

                with torch.no_grad():
                    # 🟢 Identity 采样
                    if random.random() < IDENTITY_PROB:
                        x_s, t_id = x_c.clone(), s_id.clone()
                    
                    # 🟢 关键：Label Dropping (CFG 训练)
                    # 以一定概率将 t_id 替换为 null_class_id
                    drop_mask = torch.rand(t_id.shape[0], device=self.device) < self.label_drop_prob
                    t_id_dropped = torch.where(drop_mask, 
                                               torch.full_like(t_id, self.null_class_id), 
                                               t_id)
                    
                    target = self.construct_target_lsfm(x_c, x_s).to(memory_format=torch.channels_last)
                    is_id = (s_id == t_id).view(-1, 1, 1, 1).float()
                    target = is_id * x_c + (1 - is_id) * target

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    v_gt = target - x_c
                    
                    t = torch.rand(x_c.size(0), device=self.device)
                    x_t = (1 - t.view(-1,1,1,1)) * x_c + t.view(-1,1,1,1) * target
                    
                    # 🟢 使用 dropped 的 label 训练
                    v_pred = model(x_t, x_c, t, t_id_dropped)
                    
                    # 🟢 基础 MSE
                    loss_mse_raw = F.mse_loss(v_pred, v_gt, reduction='none')
                    loss_mse_per_sample = loss_mse_raw.mean(dim=[1, 2, 3])
                    
                    # 🟢 辅助损失仅用于非空类别的转换任务
                    is_transfer = (s_id != t_id).float()
                    is_not_null = (t_id_dropped != self.null_class_id).float()
                    apply_aux_loss = (is_transfer * is_not_null).bool()
                    
                    if apply_aux_loss.any():
                        loss_spec = compute_spectral_loss(
                            v_pred[apply_aux_loss], 
                            v_gt[apply_aux_loss]
                        )
                        
                        v_pred_flat = v_pred[apply_aux_loss].flatten(1)
                        v_gt_flat = v_gt[apply_aux_loss].flatten(1)
                        cos_sim = F.cosine_similarity(v_pred_flat, v_gt_flat, dim=1, eps=1e-6)
                        loss_dir = (1 - cos_sim).mean()
                    else:
                        loss_spec = torch.tensor(0.0, device=self.device)
                        loss_dir = torch.tensor(0.0, device=self.device)
                    
                    # 🟢 加权 MSE
                    sample_weights = 1.0 + is_transfer * (self.transfer_weight - 1.0)
                    weighted_mse = (loss_mse_per_sample * sample_weights).mean()
                    
                    # 🟢 总损失
                    loss = weighted_mse + 0.1 * loss_spec + 0.1 * loss_dir

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = epoch_loss / len(dl)
            elapsed = time.time() - start_time
            
            # 🟢 更新 best_loss
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            current_lr = scheduler.get_last_lr()[0]
            self.logger.info(f"[S1] Epoch {epoch:03d} | Avg Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
            
            scheduler.step()

            # 🟢 修改：保存包含训练状态的检查点
            if epoch % EVAL_STEP == 0:
                self.save_ckpt(model, opt, scheduler, epoch, avg_loss, best_loss, "stage1")
                self.do_inference(model, epoch, "stage1")
        
        # 🟢 最后保存 final 检查点
        self.logger.info("[Stage 1] Training Completed. Saving final checkpoint...")
        torch.save({
            'epoch': total_epochs,
            'model_state_dict': self.clean_sd(model.state_dict()),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'config': self.cfg
        }, final_ckpt)

    # -----------------------------------------------------------------------------
    # Intermediate: Reflow Data Generation
    # -----------------------------------------------------------------------------
    @torch.no_grad()
    def generate_reflow_data(self):
        if self.reflow_dir.exists() and len(list(self.reflow_dir.glob("*.pt"))) > 0:
            return

        self.logger.info("="*50)
        self.logger.info(">>> Generating Reflow Data (Latent ODE Sampling)")
        self.logger.info("="*50)
        
        model = self.get_model()
        
        # 🟢 修复：加载时处理新旧格式
        stage1_ckpt = torch.load(self.ckpt_dir / "stage1_final.pt", map_location=self.device)
        if 'model_state_dict' in stage1_ckpt:
            self.safe_load(model, stage1_ckpt['model_state_dict'])
        else:
            self.safe_load(model, stage1_ckpt)
        
        model.eval()
        
        self.reflow_dir.mkdir(parents=True, exist_ok=True)
        dl = DataLoader(self.train_ds, batch_size=self.cfg['training']['batch_size'], 
                        shuffle=False, num_workers=8, pin_memory=True)
        
        steps, dt = 10, 0.1
        count = 0
        
        for x_c, _, _, _ in tqdm(dl, desc="Generating Pairs"):
            x_c = x_c.to(self.device, memory_format=torch.channels_last, non_blocking=True)
            for tid in range(self.cfg['data']['num_classes']):
                t_vec = torch.full((x_c.size(0),), tid, device=self.device, dtype=torch.long)
                x_t = x_c.clone()
                for i in range(steps):
                    t = torch.ones(x_c.size(0), device=self.device) * (i * dt)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        v = model(x_t, x_c, t, t_vec)
                    x_t = x_t + v.float() * dt
                
                torch.save({'z0': x_c.cpu(), 'z1': x_t.cpu(), 't_id': t_vec.cpu()}, 
                           self.reflow_dir / f"b{count}_c{tid}.pt")
            count += 1
            
        self.logger.info(f"[Reflow] Generation Finished. Saved to {self.reflow_dir}")

    # -----------------------------------------------------------------------------
    # Stage 2: Reflow (Distillation)
    # -----------------------------------------------------------------------------
    def run_stage2(self):
        # 🟢 同样的断点续训逻辑
        final_ckpt = self.ckpt_dir / "stage2_final.pt"
        
        self.logger.info("="*50)
        self.logger.info(">>> Starting Stage 2: Distillation (Reflow)")
        self.logger.info("="*50)
        
        model = self.get_model()
        
        # 🟢 修复：加载 stage1_final 时处理新旧格式
        stage1_ckpt = torch.load(self.ckpt_dir / "stage1_final.pt", map_location=self.device)
        if 'model_state_dict' in stage1_ckpt:
            self.safe_load(model, stage1_ckpt['model_state_dict'], strict=False)
        else:
            self.safe_load(model, stage1_ckpt, strict=False)
        
        ds = Stage2Dataset(self.reflow_dir)
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], 
                        shuffle=True, num_workers=8, pin_memory=True, 
                        persistent_workers=True, drop_last=True)
        
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        total_epochs = self.cfg['training']['stage2_epochs']
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, 
            T_max=total_epochs, 
            eta_min=1e-6
        )
        
        # 🟢 断点续训
        start_epoch = 0
        best_loss = float('inf')
        
        if final_ckpt.exists():
            self.logger.info("[Stage 2] ✅ Final checkpoint exists. Skipping training.")
            return
        
        latest_ckpt, latest_epoch = self.find_latest_checkpoint("stage2")
        if latest_ckpt:
            start_epoch, best_loss = self.load_training_state(
                latest_ckpt, model, opt, scheduler
            )
        
        for epoch in range(start_epoch + 1, total_epochs + 1):
            model.train()
            epoch_loss = 0.0
            start_time = time.time()
            
            pbar = tqdm(dl, desc=f"[S2] Epoch {epoch}/{total_epochs}", leave=False)
            
            for z0, z1, t_id in pbar:
                z0 = z0.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                z1 = z1.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                t_id = t_id.to(self.device, non_blocking=True)
                
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    t = torch.rand(z0.size(0), device=self.device)
                    x_t = (1-t.view(-1,1,1,1)) * z0 + t.view(-1,1,1,1) * z1
                    v_pred = model(x_t, z0, t, t_id)
                    loss = F.mse_loss(v_pred, z1 - z0)
                
                loss.backward()
                opt.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = epoch_loss / len(dl)
            elapsed = time.time() - start_time
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            current_lr = scheduler.get_last_lr()[0]
            self.logger.info(f"[S2] Epoch {epoch:03d} | Avg Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
            
            scheduler.step()
            
            if epoch % EVAL_STEP == 0:
                self.save_ckpt(model, opt, scheduler, epoch, avg_loss, best_loss, "stage2")
                self.do_inference(model, epoch, "stage2", steps_override=4)
        
        # 保存 final
        torch.save({
            'epoch': total_epochs,
            'model_state_dict': self.clean_sd(model.state_dict()),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'config': self.cfg
        }, final_ckpt)

    # -----------------------------------------------------------------------------
    # Inference / Validation
    # -----------------------------------------------------------------------------

    @torch.no_grad()
    def do_inference(self, model, epoch, stage, steps_override=None):
        model.eval()
        
        with open("config.json", 'r', encoding='utf-8') as f:
            fresh_cfg = json.load(f)
        
        inf_cfg = self.cfg.get('inference', {})
        # 🟢 强行清洗字符串：确保 monet2photo 中只有一个 p
        raw_path = inf_cfg.get('image_path', '').replace("monet2pphoto", "monet2photo")
        test_root = Path(raw_path)
        
        if not test_root.exists():
            self.logger.info(f"[Inference] ❌ Path not found: {test_root}")
            model.train()
            return
        
        # 🟢 [修改] 推理结果保存到 checkpoint 目录下的 inf 子目录
        save_root = self.ckpt_dir / "inf" / stage / f"ep{epoch}"
        save_root.mkdir(parents=True, exist_ok=True)
        
        steps = steps_override if steps_override else inf_cfg.get('num_inference_steps', 5)
        cfg_scale = inf_cfg.get('cfg_scale', 2.0)
        use_cfg = inf_cfg.get('use_cfg', True)
        latent_clamp = inf_cfg.get('latent_clamp', 3.0)
        
        dt = 1.0 / steps

        # 遍历子目录
        subdirs = [d for d in test_root.iterdir() if d.is_dir()]
        if not subdirs: subdirs = [test_root]

        self.logger.info(f"[Inference] Starting Epoch {epoch} | Steps: {steps}")

        for subdir in subdirs:
            files = []
            for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg']:
                files.extend(list(subdir.glob(ext)))
            
            files = sorted(files)[:2] # 每个子类只跑2张，节省时间
            if not files: continue

            for img_p in files:
                try:
                    img = Image.open(img_p).convert("RGB")
                    pixel = self.infer_transform(img).unsqueeze(0).to(self.device)
                    z_c = self.vae.encode(pixel).latent_dist.sample() * 0.18215
                    z_c = z_c.to(memory_format=torch.channels_last)
                    
                    self.save_img(pixel, save_root / f"{subdir.name}_{img_p.stem}_orig.jpg")
                    
                    for tid in range(self.cfg['data']['num_classes']):
                        z_t = z_c.clone()
                        t_vec = torch.tensor([tid], device=self.device)
                        
                        # 🟢 CFG 推理
                        for k in range(steps):
                            t_step = torch.ones(1, device=self.device) * (k * dt)
                            
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                if use_cfg:
                                    # 条件预测
                                    v_cond = model(z_t, z_c, t_step, t_vec)
                                    
                                    # 无条件预测
                                    null_vec = torch.tensor([self.null_class_id], device=self.device)
                                    v_uncond = model(z_t, z_c, t_step, null_vec)
                                    
                                    # 🟢 CFG 公式: v = v_uncond + scale * (v_cond - v_uncond)
                                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                                else:
                                    v = model(z_t, z_c, t_step, t_vec)
                            
                            # 欧拉积分
                            z_t = z_t + v.float() * dt
                            
                            # 🟢 数值裁剪（防止爆炸）
                            if latent_clamp > 0:
                                z_t = z_t.clamp(-latent_clamp, latent_clamp)
                        
                        # 🟢 关键：除以缩放因子，保证解码清晰度
                        res_pixel = self.vae.decode(z_t.float() / 0.18215).sample
                        self.save_img(res_pixel, save_root / f"{subdir.name}_{img_p.stem}_to_S{tid}.jpg")
                    
                except Exception as e:
                    self.logger.info(f"[Inference] Error processing {img_p.name}: {e}")

        self.logger.info(f"[Inference] Successfully finished. Output: {save_root}")
        model.train()

    def save_ckpt(self, model, optimizer, scheduler, epoch, loss, best_loss, stage):
        """保存包含完整训练状态的检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.clean_sd(model.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'best_loss': best_loss,
            'config': self.cfg
        }
        
        path = self.ckpt_dir / f"{stage}_epoch{epoch}.pt"
        torch.save(checkpoint, path)
        
        # 🟢 保留最近 3 个检查点
        ckpts = sorted(list(self.ckpt_dir.glob(f"{stage}_epoch*.pt")), key=os.path.getmtime)
        if len(ckpts) > 3: 
            for old_ckpt in ckpts[:-3]:
                os.remove(old_ckpt)
                self.logger.info(f"🗑️  Removed old checkpoint: {old_ckpt.name}")

    def save_img(self, tensor, path):
        img = (tensor.cpu().permute(0,2,3,1).numpy()[0] * 0.5 + 0.5).clip(0, 1)
        Image.fromarray((img * 255).astype('uint8')).save(path)

    def run_pipeline(self):
        self.run_stage1()
        self.generate_reflow_data()
        self.run_stage2()

if __name__ == "__main__":
    LSFMTrainer().run_pipeline()