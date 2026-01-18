import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from pathlib import Path
import json, random, re, shutil, os
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL

from SAFlow import SAFModel
from dataset import Stage1Dataset, Stage2Dataset

# ================= 核心参数 =================
NUM_SAMPLES_PER_CLASS = 1  # 推理时每个类别选几张图
IDENTITY_RATE = 0.2        # 20% 概率学 Identity
EVAL_STEP = 1              # 每个 Epoch 结束后保存并推理
# ===========================================

class ReflowTrainer:
    def __init__(self):
        print("\n🔧 [Init] 初始化训练器...")
        with open("config.json", 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ckpt_dir = Path(self.cfg['checkpoint']['save_dir'])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.reflow_dir = Path(self.cfg['training']['reflow_data_dir'])
        self.vis_root = self.ckpt_dir / "visualizations"
        
        # 加载 VAE (冻结)
        print("⏳ [Init] 加载 VAE...")
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        # 准备类别权重张量
        self.id_loss_weights = torch.tensor(
            self.cfg['training'].get('identity_weights', [1.0] * self.cfg['data']['num_classes']),
            device=self.device
        )

    def get_model(self):
        return SAFModel(**self.cfg['model']).to(self.device)

    def resume_checkpoint(self, model, stage_prefix):
        ckpts = list(self.ckpt_dir.glob(f"{stage_prefix}_epoch*.pt"))
        if not ckpts: return 1
        latest = max(ckpts, key=lambda p: int(re.search(r'epoch(\d+)', p.name).group(1)))
        model.load_state_dict(torch.load(latest, map_location=self.device))
        epoch = int(re.search(r'epoch(\d+)', latest.name).group(1))
        print(f"🟢 [Resume] 已从 Epoch {epoch} 恢复权重")
        return epoch + 1

    def make_balanced_sampler(self, dataset):
        print("⚖️ [Sampler] 正在计算类别均衡权重...")
        targets = [cls_id for _, cls_id in dataset.all_files]
        class_counts = np.bincount(targets)
        class_weights = 1. / class_counts
        sample_weights = np.array([class_weights[t] for t in targets])
        
        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights), 
            num_samples=len(sample_weights), 
            replacement=True
        )

    @torch.no_grad()
    def do_inference(self, model, dataset, epoch, stage):
        model.eval()
        save_dir = self.vis_root / stage / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        num_classes = len(dataset.classes)
        print(f"🖼️ [Inference] 采样推理中...")
        
        for src_id in range(num_classes):
            available_files = dataset.files_by_class[src_id]
            selected_files = random.sample(available_files, min(NUM_SAMPLES_PER_CLASS, len(available_files)))
            
            for f_idx, sample_path in enumerate(selected_files):
                x_c = dataset.load_latent(sample_path).unsqueeze(0).to(self.device)
                
                # 原图参考
                orig = self.vae.decode(x_c / 0.18215).sample
                orig = (orig / 2 + 0.5).clamp(0, 1).cpu().permute(0,2,3,1).numpy()[0]
                Image.fromarray((orig * 255).astype('uint8')).save(save_dir / f"src_cls{src_id}_samp{f_idx}_orig.jpg")
                
                for target_id in range(num_classes):
                    x_t = x_c.clone()
                    tid_tensor = torch.tensor([target_id], device=self.device)
                    for i in range(20):
                        t = torch.ones(1, device=self.device) * (i / 20)
                        x_t = x_t + model(x_t, x_c, t, tid_tensor) * (1/20)
                    
                    res = self.vae.decode(x_t / 0.18215).sample
                    res = (res / 2 + 0.5).clamp(0, 1).cpu().permute(0,2,3,1).numpy()[0]
                    
                    suffix = "_ID" if src_id == target_id else ""
                    Image.fromarray((res * 255).astype('uint8')).save(
                        save_dir / f"src{src_id}_to{target_id}{suffix}.jpg"
                    )
        model.train()

    def run_stage1(self):
        print("\n🚀 [Stage 1] 开始均衡化训练...")
        model = self.get_model()
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data']['num_classes'])
        sampler = self.make_balanced_sampler(ds)
        
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], 
                        sampler=sampler, shuffle=False, drop_last=True)
        
        start_epoch = self.resume_checkpoint(model, "stage1")
        
        for epoch in range(start_epoch, self.cfg['training']['stage1_epochs'] + 1):
            model.train()
            pbar = tqdm(dl, desc=f"S1 Ep {epoch}")
            history = {'Lc': [], 'Ls': []}
            
            for x_c, x_s, t_id, s_id in pbar:
                x_c, x_s, t_id, s_id = x_c.to(self.device), x_s.to(self.device), t_id.to(self.device), s_id.to(self.device)
                B = x_c.size(0)
                
                is_self = torch.rand(B, device=self.device) < IDENTITY_RATE
                x_target = torch.where(is_self.view(-1,1,1,1), x_c, x_s)
                style_id = torch.where(is_self, s_id, t_id)
                
                opt.zero_grad()
                t = torch.rand(B, device=self.device)
                x_t = (1 - t.view(-1,1,1,1)) * x_c + t.view(-1,1,1,1) * x_target
                
                v_pred = model(x_t, x_c, t, style_id)
                v_target = x_target - x_c
                
                # 计算逐样本 MSE
                loss_elementwise = torch.mean((v_pred - v_target)**2, dim=[1,2,3])
                
                # 🔴 应用类别特定的 Identity 权重
                weights = torch.ones(B, device=self.device)
                if is_self.any():
                    # 仅在 identity 样本上应用 config 中的权重
                    weights[is_self] = self.id_loss_weights[s_id[is_self]]
                
                loss_weighted = loss_elementwise * weights
                loss = loss_weighted.mean()
                
                loss.backward()
                opt.step()
                
                # 监控
                ls_val = loss_elementwise[is_self].mean().item() if is_self.any() else 0
                lc_val = loss_elementwise[~is_self].mean().item() if (~is_self).any() else 0
                history['Ls'].append(ls_val); history['Lc'].append(lc_val)
                pbar.set_postfix({"Lc": f"{np.mean(history['Lc'][-50:]):.4f}", "Ls": f"{np.mean(history['Ls'][-50:]):.4f}"})

            if epoch % EVAL_STEP == 0:
                torch.save(model.state_dict(), self.ckpt_dir / f"stage1_epoch{epoch}.pt")
                self.do_inference(model, ds, epoch, "stage1")
                
        torch.save(model.state_dict(), self.ckpt_dir / "stage1_final.pt")

    @torch.no_grad()
    def run_generation(self):
        print("\n🌊 [Reflow] 数据合成中...")
        if self.reflow_dir.exists(): shutil.rmtree(self.reflow_dir)
        self.reflow_dir.mkdir(parents=True, exist_ok=True)
        model = self.get_model()
        model.load_state_dict(torch.load(self.ckpt_dir / "stage1_final.pt", map_location=self.device))
        model.eval()
        ds = Stage1Dataset(self.cfg['data']['data_root'])
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], shuffle=False)
        cnt = 0
        for x_c, _, _, s_id in tqdm(dl, desc="Synthesizing"):
            x_c, s_id = x_c.to(self.device), s_id.to(self.device)
            for target_id in range(len(ds.classes)):
                t_ids = torch.full((x_c.size(0),), target_id, dtype=torch.long, device=self.device)
                mask = (s_id != t_ids)
                if mask.sum() == 0: continue
                curr_xc, curr_tid = x_c[mask], t_ids[mask]
                xt = curr_xc.clone()
                for i in range(20):
                    t = torch.ones(curr_xc.size(0), device=self.device) * (i / 20)
                    xt = xt + model(xt, curr_xc, t, curr_tid) * (1/20)
                for i in range(curr_xc.size(0)):
                    torch.save({'content': curr_xc[i].cpu(), 'z': xt[i].cpu(), 'style_label': curr_tid[i].cpu()}, 
                               self.reflow_dir / f"pair_{cnt}.pt")
                    cnt += 1

    def run_stage2(self):
        print("\n✨ [Stage 2] 直线拉直训练...")
        model = self.get_model()
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        start_epoch = self.resume_checkpoint(model, "stage2")
        ds = Stage2Dataset(self.reflow_dir)
        ds_infer = Stage1Dataset(self.cfg['data']['data_root'])
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], shuffle=True, drop_last=True)
        for epoch in range(start_epoch, self.cfg['training']['stage2_epochs'] + 1):
            model.train()
            pbar = tqdm(dl, desc=f"S2 Ep {epoch}")
            for x_c, z, s_id in pbar:
                x_c, z, s_id = x_c.to(self.device), z.to(self.device), s_id.to(self.device)
                opt.zero_grad()
                t = torch.rand(x_c.size(0), device=self.device).view(-1,1,1,1)
                x_t = (1 - t) * x_c + t * z
                v_pred = model(x_t, x_c, t.squeeze(), s_id)
                loss = torch.mean((v_pred - (z - x_c))**2)
                loss.backward(); opt.step()
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