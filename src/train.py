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
from scipy.optimize import linear_sum_assignment

from SAFlow import SAFModel
from dataset import Stage1Dataset, Stage2Dataset

# ================= 核心超参 =================
EVAL_STEP = 1
MAX_GRAD_NORM = 1.0        
NUM_SAMPLES_PER_CLASS = 1
# ===========================================

class ReflowTrainer:
    def __init__(self):
        print("\n🔧 [Init] 初始化训练器 (Final OT-CFM Edition)...")
        with open("config.json", 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ckpt_dir = Path(self.cfg['checkpoint']['save_dir'])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.reflow_dir = Path(self.cfg['training']['reflow_data_dir'])
        self.vis_root = self.ckpt_dir / "visualizations"
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.ckpt_dir / "logs"))
        print(f"📈 TensorBoard 日志目录: {self.ckpt_dir / 'logs'}")
        
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

    @torch.no_grad()
    def do_inference(self, model, dataset, epoch, stage):
        """
        N-to-N 全类别转换测试
        用于验证 Style ID 是否真正控制了生成结果。
        """
        model.eval()
        save_dir = self.vis_root / stage / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        num_steps = self.cfg['inference'].get('num_inference_steps', 20)
        dt = 1.0 / num_steps

        # 遍历每个源类别
        for src_id in range(len(dataset.classes)):
            available_files = dataset.files_by_class[src_id]
            if not available_files: continue
            
            # 随机抽 1 张源图
            sample_path = random.choice(available_files)
            x_c = dataset.load_latent(sample_path).unsqueeze(0).to(self.device)
            
            # 保存原图
            orig_img = self.vae.decode(x_c / 0.18215).sample
            orig_img = (orig_img / 2 + 0.5).clamp(0, 1).cpu().permute(0,2,3,1).numpy()[0]
            Image.fromarray((orig_img * 255).astype('uint8')).save(
                save_dir / f"Src_Cls{src_id}_Original.jpg"
            )
            
            # 转换为所有目标类别
            for target_id in range(len(dataset.classes)):
                x_t = x_c.clone()
                # 显式指定目标 Style ID
                tid_tensor = torch.tensor([target_id], device=self.device)
                
                # Flow Matching 推理
                for i in range(num_steps):
                    t = torch.ones(1, device=self.device) * (i * dt)
                    v = model(x_t, x_c, t, tid_tensor)
                    x_t = x_t + v * dt
                
                # 保存结果
                res = self.vae.decode(x_t / 0.18215).sample
                res = (res / 2 + 0.5).clamp(0, 1).cpu().permute(0,2,3,1).numpy()[0]
                Image.fromarray((res * 255).astype('uint8')).save(
                    save_dir / f"Src_Cls{src_id}_To_Tgt{target_id}.jpg"
                )

        model.train()

    def compute_ot_matching(self, x_c, x_s):
        """
        计算 OT 匹配索引
        返回: col_ind (重排索引), ot_cost (匹配代价)
        """
        B = x_c.size(0)
        x_c_flat = x_c.view(B, -1)
        x_s_flat = x_s.view(B, -1)
        
        # 计算距离矩阵 [B, B]
        dists = torch.cdist(x_c_flat, x_s_flat).cpu().numpy()
        
        # 匈牙利算法求解
        row_ind, col_ind = linear_sum_assignment(dists)
        ot_cost = dists[row_ind, col_ind].mean()
        
        return col_ind, ot_cost

    def run_stage1(self):
        print("\n🚀 [Stage 1] 启动训练 (OT-CFM Corrected)...")
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
            log_loss, log_ot, log_grad = [], [], []
            
            # x_c: 内容图
            # x_s: 随机采样的目标图 (来自 t_id)
            # t_id: x_s 的类别标签 (Target Style Label)
            # s_id: x_c 的类别标签 (Source Style Label)
            for x_c, x_s, t_id, s_id in pbar:
                x_c, x_s = x_c.to(self.device), x_s.to(self.device)
                t_id = t_id.to(self.device) 
                
                # 🟢 1. 计算 OT 匹配索引
                with torch.no_grad():
                     col_ind, ot_cost_val = self.compute_ot_matching(x_c, x_s)
                     
                     # 将 numpy 索引转为 tensor
                     idx = torch.from_numpy(col_ind).to(self.device)
                     
                     # 🔴 [关键修复] 同步重排 Image 和 Label
                     x_target = x_s[idx]      # 重排目标图片
                     s_id_target = t_id[idx]  # 重排目标标签 (注意这里用 t_id, 因为它是 x_s 的标签)
                
                # 2. Flow Matching
                opt.zero_grad()
                with autocast(enabled=use_amp):
                    t = torch.rand(x_c.size(0), device=self.device)
                    t_view = t.view(-1,1,1,1)
                    
                    # 插值路径: Source -> Target
                    x_t = (1 - t_view) * x_c + t_view * x_target
                    
                    # 预测: 此时传入的 Style ID 是与 x_target 对应的正确标签！
                    v_pred = model(x_t, x_c, t, s_id_target) 
                    v_target = x_target - x_c
                    
                    loss = torch.mean((v_pred - v_target)**2)

                # 3. 反向传播
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

                # 4. 日志记录
                loss_val = loss.item()
                grad_norm_val = grad_norm.item()
                log_loss.append(loss_val)
                log_ot.append(ot_cost_val)
                log_grad.append(grad_norm_val)
                
                global_step += 1
                if global_step % 10 == 0:
                    self.writer.add_scalar("Train/Loss", loss_val, global_step)
                    self.writer.add_scalar("Train/OT_Cost", ot_cost_val, global_step)
                    self.writer.add_scalar("Train/Grad_Norm", grad_norm_val, global_step)
                    self.writer.add_scalar("Train/LR", opt.param_groups[0]['lr'], global_step)

                pbar.set_postfix({
                    "L": f"{np.mean(log_loss[-20:]):.4f}", 
                    "OT": f"{np.mean(log_ot[-20:]):.2f}",
                    "G": f"{np.mean(log_grad[-20:]):.2f}"
                })

            if epoch % EVAL_STEP == 0:
                torch.save(model.state_dict(), self.ckpt_dir / f"stage1_epoch{epoch}.pt")
                self.do_inference(model, ds, epoch, "stage1")
                
        torch.save(model.state_dict(), self.ckpt_dir / "stage1_final.pt")

    @torch.no_grad()
    def run_generation(self):
        print("\n🌊 [Reflow] 准备生成数据...")
        if self.reflow_dir.exists(): shutil.rmtree(self.reflow_dir)
        self.reflow_dir.mkdir(parents=True, exist_ok=True)
        model = self.get_model()
        
        # 优先使用最好的权重，如果没有则用 final
        ckpt_path = self.ckpt_dir / "stage1_final.pt"
        if not ckpt_path.exists(): 
            print("⚠️ 未找到权重，跳过生成")
            return

        print(f"🔄 Loading: {ckpt_path.name}")
        state = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state, strict=False)
        model.eval()
        
        ds = Stage1Dataset(self.cfg['data']['data_root'])
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], shuffle=False)
        
        num_steps = self.cfg['inference'].get('num_inference_steps', 20)
        dt = 1.0 / num_steps
        
        cnt = 0
        for x_c, _, _, s_id in tqdm(dl, desc=f"Reflow Gen"):
            x_c, s_id = x_c.to(self.device), s_id.to(self.device)
            
            for target_id in range(len(ds.classes)):
                t_ids = torch.full((x_c.size(0),), target_id, dtype=torch.long, device=self.device)
                # 排除自身重构 (可选)
                mask = (s_id != t_ids)
                if mask.sum() == 0: continue
                
                curr_xc, curr_tid = x_c[mask], t_ids[mask]
                xt = curr_xc.clone()
                
                # Teacher 生成
                for i in range(num_steps):
                    t = torch.ones(curr_xc.size(0), device=self.device) * (i * dt)
                    v = model(xt, curr_xc, t, curr_tid)
                    xt = xt + v * dt
                
                for i in range(curr_xc.size(0)):
                    torch.save({'content': curr_xc[i].cpu(), 'z': xt[i].cpu(), 'style_label': curr_tid[i].cpu()}, 
                               self.reflow_dir / f"pair_{cnt}.pt")
                    cnt += 1

    def run_stage2(self):
        print("\n✨ [Stage 2] 启动训练...")
        model = self.get_model()
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        start_epoch = self.load_checkpoint_logic(model, "stage2")
        
        ds = Stage2Dataset(self.reflow_dir)
        if len(ds) == 0: 
            print("❌ Reflow 数据集为空！请检查 run_generation")
            return

        ds_infer = Stage1Dataset(self.cfg['data']['data_root'])
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], shuffle=True, drop_last=True)
        use_amp = self.cfg['training'].get('use_amp', False)

        for epoch in range(start_epoch, self.cfg['training']['stage2_epochs'] + 1):
            model.train()
            pbar = tqdm(dl, desc=f"S2 Ep {epoch}")
            for x_c, z, s_id in pbar:
                x_c, z, s_id = x_c.to(self.device), z.to(self.device), s_id.to(self.device)
                opt.zero_grad()
                with autocast(enabled=use_amp):
                    t = torch.rand(x_c.size(0), device=self.device).view(-1,1,1,1)
                    x_t = (1 - t) * x_c + t * z
                    v_pred = model(x_t, x_c, t.squeeze(), s_id)
                    loss = torch.mean((v_pred - (z - x_c))**2)

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    opt.step()
                    
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            if epoch % EVAL_STEP == 0:
                torch.save(model.state_dict(), self.ckpt_dir / f"stage2_epoch{epoch}.pt")
                self.do_inference(model, ds_infer, epoch, "stage2")
        torch.save(model.state_dict(), self.ckpt_dir / "saf_final_reflowed.pt")

    def run_all(self):
        if not (self.ckpt_dir / "stage1_final.pt").exists(): self.run_stage1()
        if not self.reflow_dir.exists() or not any(self.reflow_dir.iterdir()): self.run_generation()
        self.run_stage2()

if __name__ == "__main__":
    trainer = ReflowTrainer()
    trainer.run_all()