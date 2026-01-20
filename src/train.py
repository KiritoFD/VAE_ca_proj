import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from pathlib import Path
import json
import random
import re
import shutil
import os
import sys
import logging
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
import torch.nn.functional as F
import lpips

# 项目模块导入
from SAFlow import SAFModel
from dataset import Stage1Dataset, Stage2Dataset

# ================= Hyperparameters =================
EVAL_STEP = 1           
MAX_GRAD_NORM = 1.0     
IDENTITY_PROB = 0.15    
# ===================================================

# 🟢 [优化 1] 开启 Ada 架构硬件加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("LSFM_Trainer")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_dir / "train.log", encoding='utf-8')
            fh.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)

class LSFMTrainer:
    def __init__(self):
        if not os.path.exists("config.json"):
            raise FileNotFoundError("Error: config.json not found.")
        with open("config.json", 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = Path(self.cfg['checkpoint']['save_dir'])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_root = self.ckpt_dir / "visualizations"
        self.vis_root.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.ckpt_dir / "logs"
        self.logger = TrainingLogger(self.log_dir)
        self.reflow_data_dir = Path(self.cfg['training'].get('reflow_data_dir', 'data/reflow_pairs'))
        
        self.logger.info(f"🚀 Initializing Trainer on {self.device} (WSL2 Optimized)")
        self.logger.info("Precision: Using BFloat16 (BF16) + TF32.")
        
        self.logger.info("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae = self.vae.float() 

        self.logger.info("Loading LPIPS...")
        self.loss_fn_lpips = lpips.LPIPS(net='alex').to(self.device).eval()
        self.loss_fn_lpips.requires_grad_(False)

        self.logger.info("Initializing Global Visualization Dataset...")
        vis_root = self.cfg['data']['data_root']
        self.vis_ds = Stage1Dataset(vis_root, self.cfg['data']['num_classes'])
        self.logger.info(f"Dataset loaded: {len(self.vis_ds)} samples.")

    def get_model(self):
        # 🟢 [优化 2] 强制 Channels Last 内存格式
        model = SAFModel(**self.cfg['model']).to(self.device, memory_format=torch.channels_last)
        
        # 🟢 [优化 3] 启用 WSL2 下的编译加速 (Triton)
        # max-autotune 会尝试更多内核组合，虽然第一次启动稍慢，但运行速度最快
        self.logger.info("⚡ Compiling model with torch.compile (max-autotune)...")
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            self.logger.warning(f"Compilation failed: {e}. Falling back to eager execution.")
        
        return model
    
    def load_checkpoint(self, model, stage_prefix):
        user_ckpt = self.cfg['training'].get('resume_checkpoint', "")
        target_ckpt = None

        if user_ckpt and os.path.exists(user_ckpt) and stage_prefix in user_ckpt:
            self.logger.info(f"Resuming from: {user_ckpt}")
            target_ckpt = Path(user_ckpt)
        else:
            ckpts = list(self.ckpt_dir.glob(f"{stage_prefix}_epoch*.pt"))
            if ckpts:
                target_ckpt = max(ckpts, key=lambda p: int(re.search(r'epoch(\d+)', p.name).group(1)))
                self.logger.info(f"Auto-resuming from: {target_ckpt.name}")

        if target_ckpt:
            state_dict = torch.load(target_ckpt, map_location=self.device)
            if list(state_dict.keys())[0].startswith("_orig_mod."):
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            match = re.search(r'epoch(\d+)', target_ckpt.name)
            return int(match.group(1)) + 1 if match else 1
        return 1

    def make_balanced_sampler(self, dataset):
        try:
            targets = [cls_id for _, cls_id in dataset.all_files]
            class_counts = np.bincount(targets)
            class_weights = 1. / (class_counts + 1e-6)
            sample_weights = np.array([class_weights[t] for t in targets])
            return WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
        except: return None

    # ================= Stage 1 Logic =================

    def construct_target_lsfm(self, x_c, x_s):
        x_c, x_s = x_c.float(), x_s.float()
        B, C, H, W = x_c.size()
        eps = 1e-5

        zc_flat = x_c.view(B, C, -1)
        zs_flat = x_s.view(B, C, -1)
        mu_c = zc_flat.mean(2, keepdim=True)
        mu_s = zs_flat.mean(2, keepdim=True)
        zc_centered = zc_flat - mu_c
        zs_centered = zs_flat - mu_s
        
        cov_c = torch.bmm(zc_centered, zc_centered.transpose(1, 2)) / (H*W-1)
        cov_s = torch.bmm(zs_centered, zs_centered.transpose(1, 2)) / (H*W-1)
        
        try:
            Uc, Sc, _ = torch.linalg.svd(cov_c)
            Us, Ss, _ = torch.linalg.svd(cov_s)
        except:
            return x_c # Fallback
            
        Sc, Ss = torch.clamp(Sc, min=eps), torch.clamp(Ss, min=eps)
        C_inv = Uc @ torch.diag_embed(1.0/torch.sqrt(Sc)) @ Uc.transpose(1, 2)
        S_mat = Us @ torch.diag_embed(torch.sqrt(Ss)) @ Us.transpose(1, 2)
        z_wct = (S_mat @ C_inv @ zc_centered + mu_s).view(B, C, H, W)

        freq_c = torch.fft.rfft2(z_wct, norm='ortho')
        freq_s = torch.fft.rfft2(x_s, norm='ortho')
        new_freq = torch.abs(freq_s) * torch.exp(1j * torch.angle(freq_c))
        return torch.fft.irfft2(new_freq, s=(H, W), norm='ortho')

    def run_stage1(self):
        if self.check_stage1_completion(): return

        self.logger.info("Starting Stage 1: LSFM Training...")
        model = self.get_model()
        start_epoch = self.load_checkpoint(model, "stage1")
        
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data']['num_classes'])
        sampler = self.make_balanced_sampler(ds)
        bs = self.cfg['training']['batch_size']
        
        # 🟢 [优化 4] WSL2 下开启多进程加载 (num_workers=8) + pin_memory
        dl = DataLoader(ds, batch_size=bs, sampler=sampler, drop_last=True, 
                        num_workers=8, pin_memory=True, persistent_workers=True)
        
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        total_epochs = self.cfg['training']['stage1_epochs']

        for epoch in range(start_epoch, total_epochs + 1):
            model.train()
            epoch_loss = 0.0
            steps = 0
            
            pbar = tqdm(dl, desc=f"Stage1 Epoch {epoch}")
            for x_c, x_s, t_id, s_id in pbar:
                # 🟢 [优化 5] Non-blocking 传输 + Channels Last
                x_c = x_c.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                x_s = x_s.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                t_id, s_id = t_id.to(self.device, non_blocking=True), s_id.to(self.device, non_blocking=True)
                
                if random.random() < IDENTITY_PROB:
                    x_s, t_id = x_c.clone(), s_id.clone()
                
                with torch.no_grad():
                    target_wct = self.construct_target_lsfm(x_c, x_s).to(memory_format=torch.channels_last)
                    is_id = (s_id == t_id).view(-1, 1, 1, 1).float()
                    x_target = is_id * x_c + (1 - is_id) * target_wct
                    s_id_target = t_id
                
                opt.zero_grad(set_to_none=True) # 微小提速
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    t = torch.rand(x_c.size(0), device=self.device)
                    x_t = (1 - t.view(-1,1,1,1)) * x_c + t.view(-1,1,1,1) * x_target
                    v_pred = model(x_t, x_c, t, s_id_target)
                    loss = F.mse_loss(v_pred, x_target - x_c)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()

                epoch_loss += loss.item()
                steps += 1
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            self.logger.info(f"Epoch {epoch} finished. Avg Loss: {epoch_loss/steps:.6f}")
            if epoch % EVAL_STEP == 0:
                self.save_checkpoint(model, epoch, "stage1")
                self.do_inference(model, epoch, "stage1")

        torch.save(model.state_dict(), self.ckpt_dir / "stage1_final.pt")

    # ================= Intermediate: Reflow Generation =================

    def check_reflow_completion(self):
        return self.reflow_data_dir.exists() and len(list(self.reflow_data_dir.glob("*.pt"))) > 0

    @torch.no_grad()
    def generate_reflow_data(self):
        if self.check_reflow_completion(): return

        self.logger.info("Generating Reflow Data...")
        model = self.get_model() 
        sd = torch.load(self.ckpt_dir / "stage1_final.pt", map_location=self.device)
        if list(sd.keys())[0].startswith("_orig_mod."): sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        
        # 显存允许的情况下，Batch Size 开大
        inference_bs = self.cfg['training']['batch_size']
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data']['num_classes'])
        dl = DataLoader(ds, batch_size=inference_bs, shuffle=False, num_workers=8, pin_memory=True)
        
        self.reflow_data_dir.mkdir(parents=True, exist_ok=True)
        steps, dt = 10, 0.1
        count = 0
        
        for x_c, _, _, _ in tqdm(dl, desc="Generating"):
            x_c = x_c.to(self.device, memory_format=torch.channels_last, non_blocking=True)
            if x_c.shape[1] == 3:
                x_c = self.vae.encode(x_c).latent_dist.sample() * 0.18215
            
            num_cls = self.cfg['data']['num_classes']
            for t_idx in range(num_cls):
                t_id = torch.full((x_c.size(0),), t_idx, device=self.device, dtype=torch.long)
                x_t = x_c.clone()
                
                # FP32 Accumulation loop for precision
                for i in range(steps):
                    t_step = torch.ones(x_c.size(0), device=self.device) * (i * dt)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        v = model(x_t, x_c, t_step, t_id)
                    x_t = x_t + v.float() * dt
                
                torch.save({'z0': x_c.cpu(), 'z1': x_t.cpu(), 't_id': t_id.cpu()}, 
                           self.reflow_data_dir / f"batch_{count}_{t_idx}.pt")
            count += 1

    # ================= Stage 2 Logic =================

    def run_stage2(self):
        self.logger.info("Starting Stage 2: Distillation...")
        model = self.get_model()
        sd = torch.load(self.ckpt_dir / "stage1_final.pt", map_location=self.device)
        if list(sd.keys())[0].startswith("_orig_mod."): sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        
        start_epoch = self.load_checkpoint(model, "stage2")
        ds = Stage2Dataset(self.reflow_data_dir)
        # 🟢 Stage 2 数据通常较小，Shuffle 开销不大
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], shuffle=True, 
                        num_workers=8, pin_memory=True, persistent_workers=True)
        
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        total_epochs = self.cfg['training'].get('stage2_epochs', 50)

        for epoch in range(start_epoch, total_epochs + 1):
            model.train()
            epoch_loss = 0.0
            steps = 0
            
            pbar = tqdm(dl, desc=f"Stage2 Epoch {epoch}")
            for z0, z1, t_id in pbar:
                z0 = z0.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                z1 = z1.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                t_id = t_id.to(self.device, non_blocking=True)
                
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    t = torch.rand(z0.size(0), device=self.device)
                    x_t = (1 - t.view(-1,1,1,1)) * z0 + t.view(-1,1,1,1) * z1
                    v_pred = model(x_t, z0, t, t_id)
                    loss = F.mse_loss(v_pred, z1 - z0)

                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                steps += 1
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if epoch % EVAL_STEP == 0:
                self.save_checkpoint(model, epoch, "stage2")
                self.do_inference(model, epoch, "stage2", steps_override=4)

        torch.save(model.state_dict(), self.ckpt_dir / "stage2_final.pt")
        self.do_inference(model, total_epochs, "stage2", steps_override=4)

    # ================= Utils =================

    def check_stage1_completion(self):
        return (self.ckpt_dir / "stage1_final.pt").exists()

    def save_checkpoint(self, model, epoch, stage):
        path = self.ckpt_dir / f"{stage}_epoch{epoch}.pt"
        torch.save(model.state_dict(), path)
        # 保持只留最近5个
        ckpts = sorted(list(self.ckpt_dir.glob(f"{stage}_epoch*.pt")), key=os.path.getmtime)
        if len(ckpts) > 5:
            for old in ckpts[:-5]: os.remove(old)

    @torch.no_grad()
    def validate_metrics(self, model, epoch):
        # 简单验证逻辑，略
        pass

    @torch.no_grad()
    def do_inference(self, model, epoch, stage, steps_override=None):
        model.eval()
        save_dir = self.vis_root / stage / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        steps = steps_override or 20
        dt = 1.0 / steps
        
        for src_id in range(self.cfg['data']['num_classes']):
            files = self.vis_ds.files_by_class.get(src_id, [])
            if not files: continue
            # 只随机画 2 张
            for idx, path in enumerate(random.sample(files, min(2, len(files)))):
                x_c = self.vis_ds.load_latent(path).to(self.device)
                if x_c.dim() == 3: x_c = x_c.unsqueeze(0)
                if x_c.shape[1] == 3: x_c = self.vae.encode(x_c).latent_dist.sample() * 0.18215
                x_c = x_c.to(memory_format=torch.channels_last)
                
                # 画原图
                orig = self.vae.decode(x_c.float() / 0.18215).sample
                Image.fromarray(((orig.cpu().permute(0,2,3,1).numpy()[0]*0.5+0.5).clip(0,1)*255).astype('uint8')).save(save_dir / f"C{src_id}_{idx}_src.jpg")
                
                # 画转换图
                for tgt_id in range(self.cfg['data']['num_classes']):
                    tid = torch.tensor([tgt_id], device=self.device)
                    x_t = x_c.clone()
                    for i in range(steps):
                        t = torch.ones(1, device=self.device) * (i * dt)
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                            v = model(x_t, x_c, t, tid)
                        x_t = x_t + v.float() * dt
                    
                    res = self.vae.decode(x_t.float() / 0.18215).sample
                    Image.fromarray(((res.cpu().permute(0,2,3,1).numpy()[0]*0.5+0.5).clip(0,1)*255).astype('uint8')).save(save_dir / f"C{src_id}_{idx}_to_C{tgt_id}.jpg")
        model.train()

    def run_pipeline(self):
        self.run_stage1()
        self.generate_reflow_data()
        self.run_stage2()

if __name__ == "__main__":
    trainer = LSFMTrainer()
    trainer.run_pipeline()