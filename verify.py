import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image
import numpy as np

# 直接把 dit_model.py 里的这两个类复制过来，或者 import
# from models.dit_model import PatchEmbed, FinalLayer

# 为了方便直接运行，我把这两个类的定义贴在这里：
class PatchEmbed(nn.Module):
    def __init__(self, latent_size=64, patch_size=2, in_channels=4, hidden_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        self.patch_size = patch_size
        self.out_channels = out_channels
    def forward(self, x):
        x = self.linear(x)
        B, N, _ = x.shape
        H = W = int(N ** 0.5)
        # ⚠️ 重点怀疑对象：这里的 reshape/permute 逻辑
        x = x.view(B, H, W, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, H * self.patch_size, W * self.patch_size)
        return x

def test_logic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("正在测试 Patchify <-> Unpatchify 闭环逻辑...")
    
    # 1. 造一个有明显图案的 Latent (左边亮，右边暗)
    latent = torch.zeros(1, 4, 64, 64).to(device)
    latent[:, :, :, :32] = 1.0  # 左半边为 1
    latent[:, :, :, 32:] = -1.0 # 右半边为 -1
    
    # 2. 初始化模块
    # 注意：为了测试还原，我们需要一个临时的 PatchEmbed (in_channels=4)
    # 实际模型里是 8，但这里我们只测 "切开->还原" 的位置逻辑
    patch_embed = PatchEmbed(in_channels=4, hidden_dim=768).to(device)
    final_layer = FinalLayer(hidden_dim=768, patch_size=2, out_channels=4).to(device)
    
    # 3. 强行让 Linear 变成“恒等映射” (Identity)
    # 这一步比较难，因为 PatchEmbed 是 Conv，FinalLayer 是 Linear。
    # 我们用一个简单的 trick：直接用 unfold 模拟 PatchEmbed 的输出
    
    # 模拟 PatchEmbed 的输出 (B, N, D)
    # 这里的 D 对应 P*P*C = 2*2*4 = 16
    patches = torch.nn.functional.unfold(latent, kernel_size=2, stride=2) # [1, 16, 1024]
    patches = patches.transpose(1, 2) # [1, 1024, 16]
    
    # 喂给 FinalLayer (我们要测的就是它能不能把这个 1024x16 还原回 64x64)
    # 我们需要临时修改 FinalLayer，跳过 linear，直接测后面的 reshape
    # 或者我们构造一个 input 使得 x.linear(input) = patches
    
    # 我们直接手动调用 FinalLayer 的后半部分逻辑：
    x = patches 
    B, N, _ = x.shape
    H = W = int(N ** 0.5)
    
    # === FinalLayer 的逻辑开始 ===
    x = x.view(B, H, W, 2, 2, 4)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, 4, 64, 64)
    # === FinalLayer 的逻辑结束 ===
    
    # 4. 验证
    diff = (x - latent).abs().sum()
    print(f"原始 Latent 与 还原 Latent 的差异值: {diff.item()}")
    
    if diff < 1e-5:
        print("✅ 恭喜！Patch 还原逻辑是正确的。问题可能出在模型训练或PosEmbed上。")
    else:
        print("❌ 严重错误！还原逻辑不对，图片被打散了！")
        print("请检查 FinalLayer 的 view/permute 顺序。")

if __name__ == "__main__":
    test_logic()