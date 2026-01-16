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
        self.ckpt_dir = Path(self.cfg['checkpoint']['save_dir'])
        self.ckpt_dir.mkdir(exist_ok=True)
        print(f"🚀 Initialized on {self.device}")

    def get_model(self):
        return SAFModel(**self.cfg['model']).to(self.device)

    def compute_loss(self, model, x_start, x_end, s_label, use_dropout=False):
        # ... (标准 Loss 计算) ...
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

    def run_stage1_debug(self):
        print("\n🕵️‍♂️ [DEBUG MODE] 正在检查 Stage 1 数据流...")
        
        # 1. 强制重新加载 Dataset
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data']['num_classes'])
        
        # 2. 打印 Dataset 内部状态
        print(f"Dataset 识别到的类别: {list(ds.class_files.keys())}")
        if len(ds.class_files) < 2:
            print("❌【致命错误】Dataset 只识别到了 1 个类别！必然导致同类配对！")
            return

        dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True) # 小 Batch 方便看
        
        model = self.get_model()
        model.train()
        
        print("\nWait... 正在抓取第一个 Batch 分析...")
        
        # 3. 抓取第一个 Batch 进行核磁共振
        for i, (x_c, x_s, s_id) in enumerate(dl):
            x_c, x_s = x_c.to(self.device), x_s.to(self.device)
            
            print(f"\n--- Batch {i} 分析 ---")
            
            # check A: 数值范围 (检查是否二次缩放)
            # 正常的 Latent 均值约为 0，标准差约为 1
            # 如果二次缩放，标准差会变成 ~0.18
            c_std = x_c.std().item()
            s_std = x_s.std().item()
            print(f"数值检查: Content Std={c_std:.4f}, Style Std={s_std:.4f}")
            
            if c_std < 0.2:
                print("❌【严重警告】数值过小！疑似在 Dataset 中进行了二次缩放 (* 0.18215)。请立即删除 Dataset 中的乘法操作！")
            else:
                print("✅ 数值范围正常 (未二次缩放)")

            # check B: 配对差异 (检查是否同图/同类)
            # 计算 Batch 里每一对 (Content, Style) 的像素平均差异
            diffs = (x_c - x_s).abs().view(x_c.size(0), -1).mean(dim=1)
            print(f"配对差异 (Pixel Diff): {diffs.tolist()}")
            
            low_diff_count = (diffs < 0.1).sum().item()
            if low_diff_count > 0:
                print(f"❌【逻辑错误】发现 {low_diff_count} 张图的内容和风格几乎一样 (Diff < 0.1)！")
                print("   说明 dataset.py 依然在进行同类配对，或者你的 trainA 和 trainB 里有重复图片！")
            else:
                print("✅ 配对逻辑正常 (所有图片差异显著，无同类配对)")

            # check C: 试跑 Loss
            loss = self.compute_loss(model, x_c, x_s, s_id, use_dropout=True)
            print(f"当前 Batch Loss: {loss.item():.4f}")
            
            if loss.item() < 0.1:
                print("❌ Loss 异常低！请结合上面的检查结果分析。")
            else:
                print("✅ Loss 正常 ( > 0.1 )")

            # 只跑一轮就退出，这是为了诊断
            print("\n🛑 诊断结束。请根据红色的 ❌ 信息修改代码。")
            break

if __name__ == "__main__":
    trainer = ReflowTrainer()
    trainer.run_stage1_debug()