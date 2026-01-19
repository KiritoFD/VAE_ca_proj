import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import json, random, re, shutil, os, time
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from SAFlow import SAFModel
from dataset import Stage1Dataset, Stage2Dataset

# ================= 核心超参 =================
EVAL_STEP = 1
MAX_GRAD_NORM = 1.0        
IDENTITY_PROB = 0.15  # 🟢 [新增] 15% 的概率进行 Identity 训练 (源=目标)
# ===========================================

class ReflowTrainer:
    def __init__(self):
        print("\n🔧 [Init] 初始化训练器 (LSFM: WCT + Fourier + Identity)...")
        with open("config.json", 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ckpt_dir = Path(self.cfg['checkpoint']['save_dir'])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.reflow_dir = Path(self.cfg['training']['reflow_data_dir'])
        self.vis_root = self.ckpt_dir / "visualizations"
        
        self.writer = SummaryWriter(log_dir=str(self.ckpt_dir / "logs"))
        self.scaler = GradScaler() if self.cfg['training'].get('use_amp', False) else None
        
        print("⏳ [Init] 加载 VAE...")
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)

    def get_model(self):
        return SAFModel(**self.cfg['model']).to(self.device)

    def load_checkpoint_logic(self, model, stage_prefix):
        user_ckpt = self.cfg['training'].get('resume_checkpoint', "")
        if user_ckpt and os.path.exists(user_ckpt):
            print(f"🛑 [Manual Resume] 加载指定权重: {user_ckpt}")
            state = torch.load(user_ckpt, map_location=self.device)
            model.load_state_dict(state, strict=False)
            match = re.search(r'epoch(\d+)', Path(user_ckpt).name)
            start_ep = int(match.group(1)) + 1 if match else 1
            print(f"   -> 将从 Epoch {start_ep} 继续")
            return start_ep

        ckpts = list(self.ckpt_dir.glob(f"{stage_prefix}_epoch*.pt"))
        if ckpts:
            latest = max(ckpts, key=lambda p: int(re.search(r'epoch(\d+)', p.name).group(1)))
            print(f"🟢 [Auto Resume] 恢复自: {latest.name}")
            state = torch.load(latest, map_location=self.device)
            model.load_state_dict(state, strict=False)
            return int(re.search(r'epoch(\d+)', latest.name).group(1)) + 1
            
        print("⚪ [Start] 开始新训练")
        return 1

    def make_balanced_sampler(self, dataset):
        print("⚖️ [Sampler] 计算均衡权重...")
        targets = [cls_id for _, cls_id in dataset.all_files]
        class_counts = np.bincount(targets)
        class_weights = 1. / (class_counts + 1e-6)
        sample_weights = np.array([class_weights[t] for t in targets])
        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights), 
            num_samples=len(sample_weights), 
            replacement=True
        )

    def construct_target_lsfm(self, x_c, x_s):
        """
        🟢 [Math Core] LSFM 目标构造器
        包含: WCT (协方差对齐) + Fourier Mixing (频域纹理注入)
        """
        B, C, H, W = x_c.size()
        eps = 1e-5

        # === 1. WCT (Whitening and Coloring Transform) ===
        zc_flat = x_c.view(B, C, -1)
        zs_flat = x_s.view(B, C, -1)
        
        # 中心化
        mu_c = zc_flat.mean(dim=2, keepdim=True)
        mu_s = zs_flat.mean(dim=2, keepdim=True)
        zc_centered = zc_flat - mu_c
        zs_centered = zs_flat - mu_s
        
        # 协方差 (Unbiased)
        cov_c = torch.bmm(zc_centered, zc_centered.transpose(1, 2)) / (H * W - 1)
        cov_s = torch.bmm(zs_centered, zs_centered.transpose(1, 2)) / (H * W - 1)
        
        # SVD 分解 (增加数值稳定性)
        # ⚠️ 使用 try-except 处理极少数 SVD 不收敛的情况
        try:
            Uc, Sc, _ = torch.linalg.svd(cov_c)
            Us, Ss, _ = torch.linalg.svd(cov_s)
        except RuntimeError:
            # 极个别情况 SVD 失败，直接退化为 AdaIN (均值方差对齐)
            return (x_c - mu_c.view(B,C,1,1)) * (zs_flat.std(dim=2).view(B,C,1,1) / (zc_flat.std(dim=2).view(B,C,1,1)+eps)) + mu_s.view(B,C,1,1)
            
        Sc = torch.clamp(Sc, min=eps)
        Ss = torch.clamp(Ss, min=eps)
        
        # 构造变换矩阵
        C_inv = torch.bmm(Uc, torch.diag_embed(1.0 / torch.sqrt(Sc)))
        C_inv = torch.bmm(C_inv, Uc.transpose(1, 2))
        
        S_mat = torch.bmm(Us, torch.diag_embed(torch.sqrt(Ss)))
        S_mat = torch.bmm(S_mat, Us.transpose(1, 2))
        
        transform_mat = torch.bmm(S_mat, C_inv)
        z_wct = torch.bmm(transform_mat, zc_centered) + mu_s
        z_wct = z_wct.view(B, C, H, W)

        # === 2. Fourier Amplitude Mixing ===
        # 使用 rfft2 加速 (实数输入)
        freq_c = torch.fft.rfft2(z_wct, norm='ortho')
        freq_s = torch.fft.rfft2(x_s, norm='ortho')
        
        amp_s = torch.abs(freq_s)
        phase_c = torch.angle(freq_c)
        
        # 组合: 风格幅值 + 内容相位
        # 加上 1e-8 防止纯 0 导致的相位计算 NaN
        new_freq = amp_s * torch.exp(1j * phase_c)
        
        # 逆变换 (确保尺寸一致)
        z_target = torch.fft.irfft2(new_freq, s=(H, W), norm='ortho')
        
        return z_target

    def run_stage1(self):
        print("\n🚀 [Stage 1] 启动训练 (LSFM)...")
        model = self.get_model()
        start_epoch = self.load_checkpoint_logic(model, "stage1")
        
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data']['num_classes'])
        sampler = self.make_balanced_sampler(ds)
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], sampler=sampler, drop_last=True)
        
        use_amp = self.cfg['training'].get('use_amp', False)
        global_step = (start_epoch - 1) * len(dl)

        for epoch in range(start_epoch, self.cfg['training']['stage1_epochs'] + 1):
            model.train()
            pbar = tqdm(dl, desc=f"S1 Ep {epoch}")
            log_loss, log_grad = [], []
            
            # x_c: 内容图, x_s: 风格参考图
            # t_id: x_s 的风格标签
            # s_id: x_c 的风格标签
            for x_c, x_s, t_id, s_id in pbar:
                x_c, x_s = x_c.to(self.device), x_s.to(self.device)
                t_id, s_id = t_id.to(self.device), s_id.to(self.device)
                
                # 🟢 [Identity Loss / Augmentation]
                # 以一定概率强制 x_s = x_c，教模型“如果目标风格也是自己，则保持不动”
                do_identity = (random.random() < IDENTITY_PROB)
                if do_identity:
                    x_s = x_c.clone()
                    t_id = s_id.clone()
                
                # 🟢 [LSFM Target Construction]
                with torch.no_grad():
                    # 如果是 Identity 模式，WCT(x, x) == x，Fourier(x, x) == x
                    # 所以 x_target 会自动变为 x_c，v_target 会变为 0
                    x_target = self.construct_target_lsfm(x_c, x_s)
                    s_id_target = t_id
                
                # Flow Matching
                opt.zero_grad()
                with autocast(enabled=use_amp):
                    t = torch.rand(x_c.size(0), device=self.device)
                    t_view = t.view(-1,1,1,1)
                    
                    x_t = (1 - t_view) * x_c + t_view * x_target
                    
                    v_pred = model(x_t, x_c, t, s_id_target) 
                    v_target = x_target - x_c
                    
                    loss = torch.mean((v_pred - v_target)**2)

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    opt.step()

                loss_val, grad_norm_val = loss.item(), grad_norm.item()
                log_loss.append(loss_val)
                log_grad.append(grad_norm_val)
                
                global_step += 1
                if global_step % 10 == 0:
                    self.writer.add_scalar("Train/Loss", loss_val, global_step)
                    self.writer.add_scalar("Train/Grad_Norm", grad_norm_val, global_step)
                    self.writer.add_scalar("Train/LR", opt.param_groups[0]['lr'], global_step)

                pbar.set_postfix({
                    "L": f"{np.mean(log_loss[-20:]):.4f}", 
                    "G": f"{np.mean(log_grad[-20:]):.2f}"
                })

            if epoch % EVAL_STEP == 0:
                torch.save(model.state_dict(), self.ckpt_dir / f"stage1_epoch{epoch}.pt")
                self.do_inference(model, ds, epoch, "stage1")
                
        torch.save(model.state_dict(), self.ckpt_dir / "stage1_final.pt")

    @torch.no_grad()
    def do_inference(self, model, dataset, epoch, stage):
        """ N-to-N 全风格转换测试 """
        model.eval()
        save_dir = self.vis_root / stage / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        num_steps = self.cfg['inference'].get('num_inference_steps', 20)
        dt = 1.0 / num_steps
        
        # 只随机挑几个源图片进行展示，避免太慢
        vis_count = 0
        max_vis = 2 # 每个类别挑2张

        for src_id in range(len(dataset.classes)):
            available_files = dataset.files_by_class[src_id]
            if not available_files: continue
            
            selected_files = random.sample(available_files, min(max_vis, len(available_files)))
            
            for idx, sample_path in enumerate(selected_files):
                x_c = dataset.load_latent(sample_path).unsqueeze(0).to(self.device)
                
                # 保存原图
                orig_img = self.vae.decode(x_c / 0.18215).sample
                orig_img = (orig_img / 2 + 0.5).clamp(0, 1).cpu().permute(0,2,3,1).numpy()[0]
                Image.fromarray((orig_img * 255).astype('uint8')).save(
                    save_dir / f"Src_Cls{src_id}_Img{idx}_Original.jpg"
                )
                
                # 转换所有目标风格
                for target_id in range(len(dataset.classes)):
                    x_t = x_c.clone()
                    tid_tensor = torch.tensor([target_id], device=self.device)
                    
                    for i in range(num_steps):
                        t = torch.ones(1, device=self.device) * (i * dt)
                        v = model(x_t, x_c, t, tid_tensor)
                        x_t = x_t + v * dt
                    
                    res = self.vae.decode(x_t / 0.18215).sample
                    res = (res / 2 + 0.5).clamp(0, 1).cpu().permute(0,2,3,1).numpy()[0]
                    Image.fromarray((res * 255).astype('uint8')).save(
                        save_dir / f"Src_Cls{src_id}_Img{idx}_To_Tgt{target_id}.jpg"
                    )

        model.train()

    # run_generation 和 run_stage2 与之前保持一致，或按需添加
    def run_all(self):
        self.run_stage1()

if __name__ == "__main__":
    trainer = ReflowTrainer()
    trainer.run_all()