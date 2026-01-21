import torch
import torch.nn as nn
import math

# ==============================================================================
# 基础组件：针对 4070 Laptop 优化的算子
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb_scale', torch.exp(torch.arange(half_dim) * -emb))

    def forward(self, x):
        ex = x.unsqueeze(-1) * self.emb_scale
        return torch.cat((ex.sin(), ex.cos()), dim=-1)

class AdaGN(nn.Module):
    """
    自适应组归一化：只修改统计量(均值/方差)，严格控制风格注入，不破坏空间结构。
    """
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 2)
        )

    def forward(self, x, cond):
        # x: [B, C, H, W], cond: [B, cond_dim]
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        x = self.norm(x)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

class ResBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.gn1 = AdaGN(dim, cond_dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn2 = AdaGN(dim, cond_dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.act = nn.SiLU()
        
        # 显存优化：如果不匹配才做卷积
        self.shortcut = nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(self.act(self.gn1(x, cond)))
        h = self.conv2(self.act(self.gn2(h, cond)))
        return x + h

# ==============================================================================
# 主模型：纯向量场预测网络
# ==============================================================================

class StyleFlowNet(nn.Module):
    def __init__(self, in_channels=4, model_dim=128, num_styles=10):
        super().__init__()
        self.model_dim = model_dim
        
        # 1. 时间与风格嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_dim),
            nn.Linear(model_dim*2, model_dim*4),
            nn.SiLU(),
            nn.Linear(model_dim*4, model_dim*4)
        )
        # N+1 类用于 CFG (Null Token)
        self.style_emb = nn.Embedding(num_styles + 1, model_dim*4)
        
        # 2. U-Net 骨架 (为了Inversion稳定，保持对称性)
        self.in_conv = nn.Conv2d(in_channels, model_dim, 3, padding=1)
        
        dims = [model_dim, model_dim*2, model_dim*4]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Downsample Path
        curr = model_dim
        for d in dims[1:]:
            self.downs.append(nn.ModuleList([
                ResBlock(curr, model_dim*4),
                ResBlock(curr, model_dim*4),
                nn.Conv2d(curr, d, 3, stride=2, padding=1) # Downsample
            ]))
            curr = d
            
        # Bottleneck
        self.mid1 = ResBlock(curr, model_dim*4)
        self.mid2 = ResBlock(curr, model_dim*4)
        
        # Upsample Path
        for d in reversed(dims[:-1]):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(curr, d, 4, stride=2, padding=1), # Upsample
                ResBlock(d + d, model_dim*4), # Cat后通道翻倍
                ResBlock(d, model_dim*4)
            ]))
            curr = d
            
        self.out_norm = nn.GroupNorm(32, model_dim)
        self.out_conv = nn.Conv2d(model_dim, in_channels, 3, padding=1)
        
        # 零初始化最后一层，让训练初期流场接近0
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, style_id):
        """
        x: [B, 4, H, W] 当前流状态
        t: [B] 时间 (0~1)
        style_id: [B] 风格索引
        """
        # 条件处理
        t_emb = self.time_mlp(t)
        s_emb = self.style_emb(style_id)
        cond = t_emb + s_emb  # 融合时间与风格
        
        h = self.in_conv(x)
        skips = [h]
        
        for b1, b2, down in self.downs:
            h = b1(h, cond)
            h = b2(h, cond)
            skips.append(h)
            h = down(h)
            
        h = self.mid1(h, cond)
        h = self.mid2(h, cond)
        
        for up, b1, b2 in self.ups:
            h = up(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = b1(h, cond)
            h = b2(h, cond)
            
        h = self.out_norm(h)
        h = torch.nn.functional.silu(h)
        return self.out_conv(h)