import torch
import torch.nn as nn
import math


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """
    将 Latent (64x64) 转换为 Patch 序列
    使用 patch_size=2，序列长度从 4096 降为 1024
    """
    def __init__(self, latent_size=64, patch_size=2, in_channels=4, hidden_dim=768):
        super().__init__()
        self.num_patches = (latent_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class FinalLayer(nn.Module):
    """将序列还原回 Latent 空间 (Unpatchify)"""
    def __init__(self, hidden_dim, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        B, N, _ = x.shape
        H = W = int(N ** 0.5)
        
        x = x.view(B, H, W, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, H * self.patch_size, W * self.patch_size)
        return x


class SinusoidalTimeEmbedding(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class AdaLN(nn.Module):
    """自适应层归一化 (AdaLN-Zero)"""
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim, bias=True)
        )
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)
    
    def forward(self, x, cond):
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = AdaLN(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = AdaLN(hidden_dim, hidden_dim)
        
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim)
        )
        
    def forward(self, x, cond):
        x = x + self.attn(self.norm1(x, cond), self.norm1(x, cond), self.norm1(x, cond))[0]
        x = x + self.mlp(self.norm2(x, cond))
        return x


class DiTModel(nn.Module):
    """
    Latent Flow Matching Model (with Patchify & PosEmbed)
    """
    def __init__(
        self,
        latent_channels=4,
        latent_size=64,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        num_styles=4,
        mlp_ratio=4.0,
        patch_size=2
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        
        # 1. Patch Embedder (xt + x0 拼接，通道数 * 2)
        self.patch_embed = PatchEmbed(latent_size, patch_size, latent_channels * 2, hidden_dim)
        num_patches = self.patch_embed.num_patches
        
        # 2. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # 3. 条件 Embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.style_embed = nn.Embedding(num_styles, hidden_dim)
        
        self.cond_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio) 
            for _ in range(num_layers)
        ])
        
        # 5. Output Layer
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_layer = FinalLayer(hidden_dim, patch_size, latent_channels)

    def forward(self, xt, x0, t, style_id):
        """
        Args:
            xt: [B, 4, 64, 64] - Current state
            x0: [B, 4, 64, 64] - Content anchor
            t: [B] - Time steps
            style_id: [B] - Style class IDs
        Returns:
            v: [B, 4, 64, 64] - Predicted velocity
        """
        # 1. 拼接当前状态和结构条件
        x_in = torch.cat([xt, x0], dim=1)  # [B, 8, 64, 64]
        
        # 2. Patchify
        x = self.patch_embed(x_in)  # [B, 1024, D]
        
        # 3. 加上位置编码
        x = x + self.pos_embed
        
        # 4. 准备条件向量
        t_emb = self.time_embed(t)
        style_emb = self.style_embed(style_id)
        cond = self.cond_fusion(torch.cat([t_emb, style_emb], dim=-1))
        
        # 5. Transformer 处理
        for block in self.blocks:
            x = block(x, cond)
            
        # 6. Unpatchify
        x = self.final_norm(x)
        v = self.final_layer(x)
        
        return v