import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from model import LGTUNet

def run_diagnostics():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔍 Starting System Diagnostics on {device}...\n")
    
    # ==========================================
    # 1. 数据量级检查 (Data Scale Check)
    # ==========================================
    print("[Test 1] Latent Data Statistics")
    try:
        with open('config.json', 'r') as f:
            cfg = json.load(f)
        data_root = Path(cfg['data']['data_root'])
        # 找一个真实数据
        sample_file = next(data_root.rglob("*.pt"))
        latent = torch.load(sample_file, map_location='cpu')
        
        print(f"   Sample: {sample_file}")
        print(f"   Shape:  {latent.shape}")
        print(f"   Mean:   {latent.mean().item():.4f}")
        print(f"   Std:    {latent.std().item():.4f}")
        
        if latent.std() > 3.0:
            print("   ❌ DIAGNOSIS: Data is UNSCALED (Raw VAE Output).")
            print("      Action: Must multiply by 0.18215 in DataLoader.")
        elif 0.8 < latent.std() < 1.5:
            print("   ✅ DIAGNOSIS: Data is SCALED (Standard Normal).")
        else:
            print("   ⚠️ DIAGNOSIS: Data scale is suspicious.")
            
    except Exception as e:
        print(f"   ❌ Skipped Data Check: {e}")

    print("-" * 50)

    # ==========================================
    # 2. 权重加载检查 (Weight Loading Check)
    # ==========================================
    print("\n[Test 2] Checkpoint Compatibility")
    checkpoint_path = "checkpoints/epoch_0100.pt"  # 你的旧权重路径
    
    if not os.path.exists(checkpoint_path):
        print(f"   ❌ Checkpoint not found at {checkpoint_path}")
        return

    # 初始化新模型 (带 Hyper-LoRA)
    print("   Initializing LGTUNet with Hyper-LoRA...")
    try:
        # 确保参数与你 config.json 一致
        model = LGTUNet(
            base_channels=cfg['model']['base_channels'], 
            style_dim=cfg['model']['style_dim'],
            num_styles=cfg['model']['num_styles']
        )
    except:
        model = LGTUNet() # Fallback default
        
    # 尝试加载
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    # 检查交集
    intersection = model_keys.intersection(ckpt_keys)
    missing_in_model = model_keys - ckpt_keys
    missing_in_ckpt = ckpt_keys - model_keys
    
    match_rate = len(intersection) / len(model_keys)
    print(f"   Keys in Model:      {len(model_keys)}")
    print(f"   Keys in Checkpoint: {len(ckpt_keys)}")
    print(f"   Direct Match Rate:  {match_rate:.2%}")
    
    # 检查关键层：Bottleneck Attention
    print("\n   Checking Critical Layer: 'bottleneck_attn'")
    attn_keys_model = [k for k in model_keys if "bottleneck_attn" in k]
    attn_keys_ckpt = [k for k in ckpt_keys if "bottleneck_attn" in k]
    
    print(f"   - Model expects: {len(attn_keys_model)} keys (Hyper-LoRA structure)")
    print(f"   - Ckpt provides: {len(attn_keys_ckpt)} keys (Standard structure)")
    
    # 检查是否需要迁移
    needs_migration = False
    for k in attn_keys_model:
        if "base_linear" in k and k not in ckpt_keys:
            needs_migration = True
            break
            
    if needs_migration:
        print("   ❌ DIAGNOSIS: Key Mismatch detected!")
        print("      Reason: Hyper-LoRA uses '.base_linear.weight', but checkpoint has '.weight'.")
        print("      Action: Must use 'load_checkpoint_with_migration' function.")
    elif match_rate > 0.9:
        print("   ✅ DIAGNOSIS: Weights look compatible.")
    else:
        print("   ⚠️ DIAGNOSIS: Low match rate, check model config.")

if __name__ == "__main__":
    run_diagnostics()