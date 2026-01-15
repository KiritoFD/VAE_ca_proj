import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
import os
import random
from PIL import Image
from diffusers import AutoencoderKL

# 确保引用正确的模型和数据集文件
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
        
        # 加载 VAE 用于可视化解码
        print("Loading VAE for visualization...")
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="vae"
        ).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        print(f"🚀 Initialized Trainer on {self.device}")
        
    def get_model(self):
        return SAFModel(**self.cfg['model']).to(self.device)

    def compute_loss(self, model, x_start, x_end, s_label, use_dropout=False):
        B = x_start.size(0)
        t = torch.rand(B, device=self.device)
        t_view = t.view(-1, 1, 1, 1)
        
        x_t = (1 - t_view) * x_start + t_view * x_end
        
        if use_dropout and random.random() < 0.1: # 建议Dropout概率调低到0.1-0.2
            x_cond = torch.zeros_like(x_start)
        else:
            x_cond = x_start

        v_pred = model(x_t, x_cond, t, s_label)
        v_target = x_end - x_start
        return nn.functional.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def decode_latent_to_image(self, latents):
        # 必须先除以缩放因子才能正确还原色彩
        latents = latents / 0.18215
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
        return (imgs * 255).astype('uint8')

    @torch.no_grad()
    def do_inference(self, model, x_content, x_style, epoch, stage_name):
        """
        每5个Epoch执行一次，并创建独立的子文件夹保存
        """
        model.eval()
        save_dir = self.vis_root / stage_name / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 取前2个样本展示
        num_samples = min(2, x_content.size(0))
        for i in range(num_samples):
            x_c = x_content[i:i+1].to(self.device)
            x_s = x_style[i:i+1].to(self.device)
            s_id = torch.zeros(1, dtype=torch.long, device=self.device)
            
            # Euler 推理过程
            x_t = x_c.clone()
            dt = 1.0 / 20
            for step in range(20):
                t = torch.ones(1, device=self.device) * (step * dt)
                v = model(x_t, x_c, t, s_id)
                x_t = x_t + v * dt
            
            # 解码
            img_c = self.decode_latent_to_image(x_c)[0]
            img_g = self.decode_latent_to_image(x_t)[0]
            img_s = self.decode_latent_to_image(x_s)[0]

            # 单独保存
            Image.fromarray(img_c).save(save_dir / f"sample_{i}_content.png")
            Image.fromarray(img_g).save(save_dir / f"sample_{i}_transformed.png")
            Image.fromarray(img_s).save(save_dir / f"sample_{i}_reference.png")
        
        print(f"🖼️ Saved inference results to {save_dir}")
        model.train()

    def run_stage1(self):
        print("\n🚀 [Stage 1] Independent Coupling Training...")
        model = self.get_model()
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data']['num_classes'])
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], shuffle=True, 
                        num_workers=self.cfg['training']['num_workers'], drop_last=True)
        
        # 固定一个batch用于观察变化
        vis_batch = next(iter(dl))
        
        for epoch in range(1, self.cfg['training']['stage1_epochs'] + 1):
            model.train()
            pbar = tqdm(dl, desc=f"S1 Epoch {epoch}")
            total_loss = 0
            
            for x_c, x_s, s_id in pbar:
                x_c, x_s, s_id = x_c.to(self.device), x_s.to(self.device), s_id.to(self.device)
                opt.zero_grad()
                loss = self.compute_loss(model, x_c, x_s, s_id, use_dropout=True)
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 打印 Epoch 平均 Loss
            print(f"📊 Stage 1 Epoch {epoch} Average Loss: {total_loss / len(dl):.6f}")
            
            # 每 5 轮推理一次
            if epoch % 5 == 0:
                self.do_inference(model, vis_batch[0], vis_batch[1], epoch, "stage1")
                torch.save(model.state_dict(), self.ckpt_dir / f"stage1_epoch{epoch}.pt")
                
        torch.save(model.state_dict(), self.ckpt_dir / "stage1_final.pt")

    @torch.no_grad()
    def run_generation(self):
        """生成Reflow伪数据"""
        print("\n🌊 [Reflow] Data Synthesis...")
        self.reflow_dir.mkdir(exist_ok=True)
        model = self.get_model()
        model.load_state_dict(torch.load(self.ckpt_dir / "stage1_final.pt", map_location=self.device))
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

    def run_stage2(self):
        print("\n✨ [Stage 2] Straightening Training...")
        model = self.get_model()
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        ds = Stage2Dataset(self.reflow_dir)
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], shuffle=True, 
                        num_workers=self.cfg['training']['num_workers'], drop_last=True)
        
        vis_batch = next(iter(dl))
        
        for epoch in range(1, self.cfg['training']['stage2_epochs'] + 1):
            model.train()
            pbar = tqdm(dl, desc=f"S2 Epoch {epoch}")
            total_loss = 0
            
            for x_c, z, s_id in pbar:
                x_c, z, s_id = x_c.to(self.device), z.to(self.device), s_id.to(self.device)
                opt.zero_grad()
                loss = self.compute_loss(model, x_c, z, s_id, use_dropout=False)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            
            print(f"📊 Stage 2 Epoch {epoch} Average Loss: {total_loss / len(dl):.8f}")
            
            if epoch % 5 == 0:
                self.do_inference(model, vis_batch[0], vis_batch[1], epoch, "stage2")
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