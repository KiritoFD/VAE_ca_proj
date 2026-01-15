import torch
import torch.nn as nn
import math


class GlobalContextBlock(nn.Module):
    """
    SA-Flow v2 新增组件: 全局上下文块 (Global Context Block)
    解决卷积网络"只见树木不见森林"的问题，确保整体色调和笔触风格的一致性。
    """
    def __init__(self, dim):
        super().__init__()
        # 1. 上下文建模: 1x1 Conv -> Softmax -> Attention
        self.conv_mask = nn.Conv2d(dim, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)  # 对空间维度(HW)做Softmax

        # 2. 特征转换
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.LayerNorm([dim, 1, 1]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        batch, channel, height, width = x.size()
        
        # [B, C, H, W] -> [B, C, H*W]
        input_x = x.view(batch, channel, height * width)
        
        # 计算全局注意力掩码
        context_mask = self.conv_mask(x)  # [B, 1, H, W]
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)  # [B, 1, H*W]
        
        # 获取全局上下文 [B, C, H*W] @ [B, H*W, 1] -> [B, C, 1]
        context = torch.bmm(input_x, context_mask.permute(0, 2, 1))
        context = context.unsqueeze(-1)  # [B, C, 1, 1]
        
        # 融合: 原特征 + 全局特征
        return x + self.channel_add_conv(context)


class GatedDifferentialBlock(nn.Module):
    """
    SA-Flow v2 核心 Block
    集成: 大核微分流 + 全局上下文 + 门控混合
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        # 1. 局部微分流 (Local Flow) - 保持 7x7 大核
        self.local_flow = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                    padding=kernel_size//2, groups=dim)
        
        # 2. 全局上下文 (Global Context) - v2 新增
        self.gc_block = GlobalContextBlock(dim)
        
        # 3. 风格注入 (AdaGN)
        self.norm = nn.GroupNorm(32, dim) 
        self.style_proj = nn.Linear(dim, dim * 2) 
        
        # 4. 门控混合 (Gated Mixing)
        self.proj_1 = nn.Conv2d(dim, dim * 2, 1) 
        self.proj_2 = nn.Conv2d(dim, dim, 1)
        self.act = nn.SiLU()

    def forward(self, x, style_emb):
        shortcut = x
        
        # A. 风格注入 (AdaGN)
        style_params = self.style_proj(style_emb)
        mu, sigma = style_params.chunk(2, dim=-1)
        mu = mu.unsqueeze(-1).unsqueeze(-1)
        sigma = sigma.unsqueeze(-1).unsqueeze(-1)
        x = self.norm(x) * (1 + sigma) + mu
        
        # B. 局部与全局特征提取
        x = self.local_flow(x)
        x = self.gc_block(x)  # v2: 注入全局信息
        
        # C. 门控混合 (GLU)
        x_gate, x_val = self.proj_1(x).chunk(2, dim=1)
        x = self.act(x_gate) * x_val
        x = self.proj_2(x)
        
        return shortcut + x


class SinusoidalTimeEmbedding(nn.Module):
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


class SAFModel(nn.Module):
    """
    SA-Flow v2 Model (Structure-Aware Flow)
    专为流形映射设计，支持 CFG (Classifier-Free Guidance)
    
    训练模式: Noise -> Style (条件为 Content)
    推理模式: 支持 CFG 锐化
    """
    def __init__(
        self,
        latent_channels=4,
        latent_size=64,  # 占位，不依赖固定尺寸
        hidden_dim=384,
        num_layers=12,
        num_styles=2,
        kernel_size=7,
        **kwargs
    ):
        super().__init__()
        self.in_channels = latent_channels * 2  # x_t(4) + x_cond(4)
        self.hidden_dim = hidden_dim
        
        # 1. 嵌入层
        self.style_embed = nn.Embedding(num_styles, hidden_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Stem (处理 8 通道输入)
        self.stem = nn.Conv2d(self.in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # 3. Backbone (v2 Blocks with GC)
        self.blocks = nn.ModuleList([
            GatedDifferentialBlock(hidden_dim, kernel_size=kernel_size)
            for _ in range(num_layers)
        ])
        
        # 4. Final Head
        self.final_norm = nn.GroupNorm(32, hidden_dim)
        self.final_conv = nn.Conv2d(hidden_dim, latent_channels, kernel_size=3, padding=1)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out final layer for identity initialization
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)
        
        # Kaiming init for other convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m != self.final_conv:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_t, x_cond, t, style_id):
        """
        Args:
            x_t: [B, 4, H, W] - 当前流形状态 (噪声 -> 风格图)
            x_cond: [B, 4, H, W] - 结构条件 (内容图 或 全零用于CFG)
            t: [B] - 时间步
            style_id: [B] - 风格ID
        Returns:
            v: [B, 4, H, W] - 预测速度场
        """
        # 1. 准备全局条件
        t_emb = self.time_mlp(t)
        style_emb = self.style_embed(style_id)
        global_cond = t_emb + style_emb
        
        # 2. 拼接条件 (Early Fusion)
        x = torch.cat([x_t, x_cond], dim=1)
        x = self.stem(x)
        
        # 3. Backbone Flow
        for block in self.blocks:
            x = block(x, global_cond)
            
        # 4. 预测速度
        x = self.final_norm(x)
        v = self.final_conv(x)
        
        return v