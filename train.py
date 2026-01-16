import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from pathlib import Path
import json
import os
import random
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
import re
EVAL_INTERVAL = 1
# 引用你的模型和数据集
from SAFlow import SAFModel
from dataset import Stage1Dataset, Stage2Dataset

def load_config():
    with open("config.json", 'r', encoding='utf-8') as f:
        return json.load(f)

class ReflowTrainer:
    def __init__(self):
        self.cfg = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 路径设置
        self.ckpt_dir = Path(self.cfg['checkpoint']['save_dir'])
        self.ckpt_dir.mkdir(exist_ok=True)
        self.reflow_dir = Path(self.cfg['training']['reflow_data_dir'])
        self.vis_root = self.ckpt_dir / "visualizations"
        self.vis_root.mkdir(exist_ok=True)
        
        # 加载 VAE
        print("⏳ Loading VAE for visualization...")
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="vae"
        ).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        print(f"🚀 Initialized Trainer on {self.device}")
        
    def get_model(self):
        return SAFModel(**self.cfg['model']).to(self.device)

    def resume_checkpoint(self, model, stage_prefix):
        """自动断点续传"""
        ckpts = list(self.ckpt_dir.glob(f"{stage_prefix}_epoch*.pt"))
        if not ckpts:
            print(f"⚪ No resume checkpoint for {stage_prefix}, starting from Epoch 1.")
            return 1

        def extract_epoch(p):
            match = re.search(r'epoch(\d+)', p.name)
            return int(match.group(1)) if match else 0
        
        latest_ckpt = max(ckpts, key=extract_epoch)
        latest_epoch = extract_epoch(latest_ckpt)
        print(f"🟢 Resuming {stage_prefix} from Epoch {latest_epoch} (File: {latest_ckpt.name})")
        model.load_state_dict(torch.load(latest_ckpt, map_location=self.device))
        return latest_epoch + 1

    def compute_loss(self, model, x_start, x_end, s_label, use_dropout=False):
        B = x_start.size(0)
        t = torch.rand(B, device=self.device)
        t_view = t.view(-1, 1, 1, 1)
        
        x_t = (1 - t_view) * x_start + t_view * x_end
        
        if use_dropout and random.random() < 0.1:
            x_cond = torch.zeros_like(x_start)
        else:
            x_cond = x_start

        v_pred = model(x_t, x_cond, t, s_label)
        v_target = x_end - x_start
        
        return nn.functional.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def decode_latent_to_image(self, latents):
        latents = latents / 0.18215 
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
        return (imgs * 255).astype('uint8')

    @torch.no_grad()
    def do_inference(self, model, x_content, epoch, stage_name):
        model.eval()
        save_dir = self.vis_root / stage_name / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)

        x_c = x_content[0:1].to(self.device) 
        num_styles = self.cfg['model']['num_styles']
        
        img_c = self.decode_latent_to_image(x_c)[0]
        Image.fromarray(img_c).save(save_dir / "content_source.jpg")
        
        for s_idx in range(num_styles):
            s_id = torch.tensor([s_idx], dtype=torch.long, device=self.device)
            x_t = x_c.clone()
            dt = 1.0 / 20 
            
            for step in range(20):
                t = torch.ones(1, device=self.device) * (step * dt)
                v = model(x_t, x_c, t, s_id)
                x_t = x_t + v * dt
            
            img_g = self.decode_latent_to_image(x_t)[0]
            Image.fromarray(img_g).save(save_dir / f"style_{s_idx}_result.jpg")

        print(f"🖼️ Inference done. Saved to {save_dir}")
        model.train()

    def make_balanced_sampler(self, dataset):
        print("⚖️ 计算类别权重，启用平衡采样...")
        targets = []
        for f in dataset.all_files:
            class_name = f.parent.name
            class_id = dataset.class_to_id[class_name]
            targets.append(class_id)
        targets = np.array(targets)
        
        class_counts = np.bincount(targets)
        print(f"   类别样本分布: {class_counts} (索引对应: {list(dataset.class_to_id.keys())})")
        
        class_weights = 1. / class_counts
        sample_weights = class_weights[targets]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler

    def run_stage1(self):
        print("\n🚀 [Stage 1] Independent Coupling Training...")
        model = self.get_model()
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        
        start_epoch = self.resume_checkpoint(model, "stage1")
        total_epochs = self.cfg['training']['stage1_epochs']
        
        if start_epoch > total_epochs:
            print("✅ Stage 1 already completed.")
            return

        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data']['num_classes'])
        sampler = self.make_balanced_sampler(ds)
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], 
                        sampler=sampler, shuffle=False, 
                        num_workers=self.cfg['training']['num_workers'], drop_last=True)
        
        vis_batch = next(iter(dl))
        
        # 🔴 定义探针检查频率：每个Epoch检查4次
        check_interval = max(1, len(dl) // 4)
        
        for epoch in range(start_epoch, total_epochs + 1):
            model.train()
            pbar = tqdm(dl, desc=f"S1 Epoch {epoch}/{total_epochs}")
            total_loss = 0
            smooth_loss = 0
            
            for step, (x_c, x_s, s_id) in enumerate(pbar):
                x_c, x_s, s_id = x_c.to(self.device), x_s.to(self.device), s_id.to(self.device)

                # ================= 🔍 增强版数据探针 (Periodic Probe) =================
                if step % check_interval == 0:
                    with torch.no_grad():
                        # 1. 基础数值统计
                        c_std = x_c.std().item()
                        s_std = x_s.std().item()
                        
                        # 2. 真实差异计算 (Target MSE)
                        diffs = (x_s - x_c).pow(2).view(x_c.size(0), -1).mean(dim=1)
                        avg_mse = diffs.mean().item()
                        
                        # 3. 统计“疑似同图”的数量 (MSE < 0.1)
                        suspicious_count = (diffs < 0.1).sum().item()
                        
                        tqdm.write(f"\n🔍 [探针 Step {step}] Avg Target MSE: {avg_mse:.4f} | Content Std: {c_std:.3f}")
                        
                        # 4. 实时报警逻辑
                        if c_std < 0.2:
                            tqdm.write(f"❌ [数值报警] Std过小 ({c_std:.4f})! 仍在进行二次缩放！")
                        elif suspicious_count > 0:
                            tqdm.write(f"⚠️ [逻辑报警] 本Batch有 {suspicious_count}/{x_c.size(0)} 张图差异过小! (疑似同类配对)")
                        elif avg_mse < 0.2:
                            tqdm.write(f"⚠️ [Loss报警] 理论 Loss 下限过低 ({avg_mse:.4f})! 请检查是否在训练恒等映射。")
                        else:
                            # 正常情况不刷屏，只显示简报
                            pass
                # ====================================================================
                
                opt.zero_grad()
                loss = self.compute_loss(model, x_c, x_s, s_id, use_dropout=True)
                loss.backward()
                opt.step()
                
                loss_val = loss.item()
                total_loss += loss_val
                
                if step == 0: smooth_loss = loss_val
                else: smooth_loss = 0.9 * smooth_loss + 0.1 * loss_val
                pbar.set_postfix({"loss": f"{smooth_loss:.4f}"})
            
            print(f"📊 Stage 1 Epoch {epoch} Avg Loss: {total_loss / len(dl):.6f}")
            
            if epoch % EVAL_INTERVAL == 0:
                self.do_inference(model, vis_batch[0], epoch, "stage1")
                torch.save(model.state_dict(), self.ckpt_dir / f"stage1_epoch{epoch}.pt")
                
        torch.save(model.state_dict(), self.ckpt_dir / "stage1_final.pt")

    # ... (run_generation, run_stage2, run_all 保持不变，可直接使用上一版的内容，或需要我再次完整贴出吗？) ...
    # 为了节省篇幅，这里假设你保留了之前版本的 run_generation 和 run_stage2
    # 如果你需要我再贴一遍完整的这部分，请告诉我。
    
    @torch.no_grad()
    def run_generation(self):
        # ... (同上一版) ...
        print("\n🌊 [Reflow] Data Synthesis...")
        self.reflow_dir.mkdir(exist_ok=True)
        model = self.get_model()
        
        s1_path = self.ckpt_dir / "stage1_final.pt"
        if not s1_path.exists(): raise FileNotFoundError("Stage 1 final model not found!")
            
        print(f"Loading {s1_path}...")
        model.load_state_dict(torch.load(s1_path, map_location=self.device))
        model.eval()
        
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data']['num_classes'])
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], shuffle=False)
        
        cnt = 0
        for x_c, _, s_id in tqdm(dl, desc="Synthesizing"):
            x_c, s_id = x_c.to(self.device), s_id.to(self.device)
            x_t = x_c.clone()
            dt = 1.0 / 20
            for i in range(20):
                t = torch.ones(x_c.size(0), device=self.device) * (i * dt)
                v = model(x_t, x_c, t, s_id)
                x_t = x_t + v * dt
            
            for i in range(x_c.size(0)):
                torch.save({
                    'content': x_c[i].cpu(), 
                    'z': x_t[i].cpu(), 
                    'style_label': s_id[i].cpu()
                }, self.reflow_dir / f"pair_{cnt}.pt")
                cnt += 1
        print(f"✅ Generated {cnt} pairs.")

    def run_stage2(self):
        # ... (同上一版) ...
        print("\n✨ [Stage 2] Straightening Training...")
        model = self.get_model()
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        
        start_epoch = self.resume_checkpoint(model, "stage2")
        total_epochs = self.cfg['training']['stage2_epochs']

        if start_epoch > total_epochs:
            print("✅ Stage 2 already completed.")
            return
            
        ds = Stage2Dataset(self.reflow_dir)
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], shuffle=True, 
                        num_workers=self.cfg['training']['num_workers'], drop_last=True)
        
        vis_batch = next(iter(dl))
        
        for epoch in range(start_epoch, total_epochs + 1):
            model.train()
            pbar = tqdm(dl, desc=f"S2 Epoch {epoch}/{total_epochs}")
            total_loss = 0
            smooth_loss = 0
            
            for step, (x_c, z, s_id) in enumerate(pbar):
                x_c, z, s_id = x_c.to(self.device), z.to(self.device), s_id.to(self.device)
                
                opt.zero_grad()
                loss = self.compute_loss(model, x_c, z, s_id, use_dropout=False)
                loss.backward()
                opt.step()
                
                loss_val = loss.item()
                total_loss += loss_val
                
                if step == 0: smooth_loss = loss_val
                else: smooth_loss = 0.9 * smooth_loss + 0.1 * loss_val
                pbar.set_postfix({"loss": f"{smooth_loss:.6f}"})
            
            print(f"📊 Stage 2 Epoch {epoch} Avg Loss: {total_loss / len(dl):.8f}")
            
            if epoch % EVAL_INTERVAL == 0:
                self.do_inference(model, vis_batch[0], epoch, "stage2")
                torch.save(model.state_dict(), self.ckpt_dir / f"stage2_epoch{epoch}.pt")
                
        torch.save(model.state_dict(), self.ckpt_dir / "saf_final_reflowed.pt")

    def run_all(self):
        if not (self.ckpt_dir / "stage1_final.pt").exists():
            self.run_stage1()
        if not self.reflow_dir.exists() or len(list(self.reflow_dir.glob("*.pt"))) == 0:
            self.run_generation()
        self.run_stage2()

if __name__ == "__main__":
    trainer = ReflowTrainer()
    trainer.run_all()