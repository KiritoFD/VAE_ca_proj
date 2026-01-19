import torch
import torch.nn as nn

class TimeAwareSpatialModulator(nn.Module):
    """
    时间感知空间调制器
    原理：根据时间 t 动态控制结构信息的注入强度。
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        # 时间门控: t_emb -> [0,1]
        self.time_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid() 
        )

    def forward(self, x, x_cond, t_emb):
        # 计算门控系数 alpha (Batch, Dim, 1, 1)
        alpha = self.time_gate(t_emb).unsqueeze(-1).unsqueeze(-1)
        cond_feat = self.conv(x_cond)
        # 调制: 原特征 + alpha * 结构特征
        return x + cond_feat * alpha

class AdaGN(nn.Module):
    """自适应组归一化：注入全局风格和时间信息"""
    def __init__(self, dim, emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, dim * 2)
        )

    def forward(self, x, emb):
        scale_shift = self.proj(emb)
        scale, shift = scale_shift.chunk(2, dim=1)
        x = self.norm(x)
        return x * (1 + scale[..., None, None]) + shift[..., None, None]

class SAFBlock(nn.Module):
    """MetaFormer 骨架 Block (ConvNeXt 风格)"""
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.ada_gn = AdaGN(dim, dim)
        self.spatial_mod = TimeAwareSpatialModulator(dim)
        
        # 大核卷积 (Structure)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                padding=kernel_size//2, groups=dim)
        
        # 倒瓶颈结构 (Feature Mixing)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, 1)
        self.act = nn.SiLU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x, x_cond, global_emb):
        shortcut = x
        x = self.ada_gn(x, global_emb)         # 1. 注入风格
        x = self.spatial_mod(x, x_cond, global_emb) # 2. 注入结构
        x = self.dwconv(x)                     # 3. 空间处理
        x = self.pwconv1(x)                    # 4. 通道混合
        x = self.act(x)
        x = self.pwconv2(x)
        return shortcut + x

class SAFModel(nn.Module):
    """SA-Flow v6 (OT-CFM Standard)"""
    def __init__(self, latent_channels=4, hidden_dim=256, num_layers=8, num_styles=2, kernel_size=7):
        super().__init__()
        
        # 全局嵌入 (Time + Style)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.style_embed = nn.Embedding(num_styles, hidden_dim)
        
        # 输入 Stem
        self.stem = nn.Conv2d(latent_channels, hidden_dim, 3, padding=1)
        
        # 条件编码器 (处理 x_content)
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
        
        # 主干
        self.blocks = nn.ModuleList([
            SAFBlock(hidden_dim, kernel_size) for _ in range(num_layers)
        ])
        
        # 输出头
        self.final_norm = nn.GroupNorm(32, hidden_dim)
        self.final = nn.Conv2d(hidden_dim, latent_channels, 3, padding=1)
        
        # 零初始化 (Flow Matching 最佳实践)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x_t, x_content, t, style_id):
        # 全局条件
        t_emb = self.time_mlp(t.view(-1, 1))
        s_emb = self.style_embed(style_id)
        global_emb = t_emb + s_emb 
        
        # 局部条件
        x_cond = self.cond_encoder(x_content)
        
        # 网络流动
        x = self.stem(x_t)
        for block in self.blocks:
            x = block(x, x_cond, global_emb)
            
        x = self.final_norm(x)
        return self.final(x)