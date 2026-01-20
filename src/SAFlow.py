import torch
import torch.nn as nn

class TimeAwareSpatialModulator(nn.Module):
    """
    时间感知空间调制器 (Time-Aware Spatial Modulator)
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        self.time_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid() 
        )

    def forward(self, x, x_cond, t_emb):
        # x, x_cond: [B, C, H, W] (Channels Last is preferred)
        alpha = self.time_gate(t_emb).unsqueeze(-1).unsqueeze(-1)
        cond_feat = self.conv(x_cond)
        return x + cond_feat * alpha

class AdaGN(nn.Module):
    """
    自适应组归一化 (Adaptive Group Norm)
    """
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
        # Broadcasting scale/shift for NCHW layout
        x = x * (1 + scale[..., None, None]) + shift[..., None, None]
        return x

class SAFBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.ada_gn = AdaGN(dim, dim)
        self.spatial_mod = TimeAwareSpatialModulator(dim)
        
        # Large Kernel Depthwise Conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                padding=kernel_size//2, groups=dim)
        
        # Inverted Bottleneck
        self.pwconv1 = nn.Conv2d(dim, dim * 4, 1)
        self.act = nn.SiLU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x, x_cond, global_emb):
        shortcut = x
        x = self.ada_gn(x, global_emb)
        x = self.spatial_mod(x, x_cond, global_emb)
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return shortcut + x

class SAFModel(nn.Module):
    def __init__(self, latent_channels=4, hidden_dim=256, num_layers=8, num_styles=2, kernel_size=7):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), 
            nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.style_embed = nn.Embedding(num_styles, hidden_dim)
        
        self.stem = nn.Conv2d(latent_channels, hidden_dim, 3, padding=1)
        
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
        
        self.blocks = nn.ModuleList([
            SAFBlock(hidden_dim, kernel_size) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.GroupNorm(32, hidden_dim)
        self.final = nn.Conv2d(hidden_dim, latent_channels, 3, padding=1)
        
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x_t, x_content, t, style_id):
        if t.dim() == 1:
            t = t.view(-1, 1)
        t_emb = self.time_mlp(t)
        s_emb = self.style_embed(style_id)
        global_emb = t_emb + s_emb 
        
        x_cond = self.cond_encoder(x_content)
        x = self.stem(x_t)
        
        for block in self.blocks:
            x = block(x, x_cond, global_emb)
            
        x = self.final_norm(x)
        return self.final(x)