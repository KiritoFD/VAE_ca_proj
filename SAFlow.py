import torch
import torch.nn as nn
import math

class GatedDifferentialBlock(nn.Module):
    """
    SA-Flow 核心组件：门控微分块
    利用大核卷积近似局部微分流，完全保留空间拓扑结构。
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        # 1. 局部微分流 (Local Differential Term)
        # Depthwise Conv: 极低参数量，捕捉局部纹理流向，保持拓扑
        self.local_flow = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                    padding=kernel_size//2, groups=dim)
        
        # 2. 全局风格势能 (Global Potential Term)
        # GroupNorm 保持空间结构，不像 LayerNorm 那样Flatten
        self.norm = nn.GroupNorm(32, dim) 
        self.style_proj = nn.Linear(dim, dim * 2) 
        
        # 3. 混合与非线性 (Flow Mixing)
        # 1x1 Conv 替代 Linear，实现通道间的信息交互
        self.proj_1 = nn.Conv2d(dim, dim * 2, 1) 
        self.proj_2 = nn.Conv2d(dim, dim, 1)
        self.act = nn.SiLU()
        
        # 🔴 移除 self.scale - 让 GroupNorm 和 Residual 自己平衡

    def forward(self, x, style_emb):
        # x: [B, C, H, W]
        shortcut = x
        
        # A. 注入全局风格 (AdaGN)
        # style_emb: [B, dim] -> [B, 2*dim]
        style_params = self.style_proj(style_emb)
        mu, sigma = style_params.chunk(2, dim=-1)
        
        # 广播到空间维度 [B, dim, 1, 1]
        mu = mu.unsqueeze(-1).unsqueeze(-1)
        sigma = sigma.unsqueeze(-1).unsqueeze(-1)
        
        # 调制：Norm后进行缩放和平移
        x = self.norm(x) * (1 + sigma) + mu
        
        # B. 局部空间建模 (No Attention, just Large Kernel Conv)
        x = self.local_flow(x)
        
        # C. 门控流体混合 (GLU)
        # 模拟流体力学中的非线性粘滞
        x_gate, x_val = self.proj_1(x).chunk(2, dim=1)
        x = self.act(x_gate) * x_val
        x = self.proj_2(x)
        
        # D. 欧拉积分步 (Residual Connection)
        # 🔴 直接相加,不乘极小的 scale
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
    SA-Flow Architecture (Structure-Aware Flow)
    替代原有的 Transformer 架构，专门针对 Image-to-Image Mapping 优化。
    保持了与原 DiTModel 相同的输入输出接口。
    """
    def __init__(
        self,
        latent_channels=4,
        latent_size=64, # 仅占位，SA-Flow 不依赖固定尺寸
        hidden_dim=384,
        num_layers=12,
        num_styles=2,
        kernel_size=7,
        **kwargs # 吞掉 config 中不再需要的 transformer 参数
    ):
        super().__init__()
        self.in_channels = latent_channels * 2 # xt + x_content
        self.hidden_dim = hidden_dim
        
        # 1. 风格嵌入 (Style Embedding)
        # 这就是方案一里提到的“风格身份证”
        self.style_embed = nn.Embedding(num_styles, hidden_dim)
        
        # 2. 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. 入口层 (Stem)
        # 直接在 Latent 空间卷积，保持 2D 结构
        self.stem = nn.Conv2d(self.in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # 4. 核心微分流块 (Differential Blocks)
        self.blocks = nn.ModuleList([
            GatedDifferentialBlock(hidden_dim, kernel_size=kernel_size)
            for _ in range(num_layers)
        ])
        
        # 5. 出口层 (Final Velocity Prediction)
        self.final_norm = nn.GroupNorm(32, hidden_dim)
        self.final_conv = nn.Conv2d(hidden_dim, latent_channels, kernel_size=3, padding=1)
        
        # 初始化
        self.initialize_weights()

    def initialize_weights(self):
        # 最后一层初始化为零，保证初始状态下模型输出接近零速度（恒等映射）
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)
        
        # 🔴 添加: 显式初始化中间层,确保梯度流动
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m != self.final_conv:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xt, x0, t, style_id):
        """
        Args:
            xt: [B, 4, H, W] - 当前流形状态
            x0: [B, 4, H, W] - 原始内容锚点 (结构条件)
            t: [B] - 时间步
            style_id: [B] - 风格 ID
        Returns:
            v: [B, 4, H, W] - 预测流速
        """
        # 1. 准备条件
        t_emb = self.time_mlp(t)                 # [B, dim]
        style_emb = self.style_embed(style_id)   # [B, dim]
        
        # 融合时间与风格 (简单的相加或拼接均可，这里选择相加作为全局 Condition)
        global_cond = t_emb + style_emb
        
        # 2. 拼接输入并进入特征空间
        x = torch.cat([xt, x0], dim=1)
        x = self.stem(x)
        
        # 3. 通过微分流块
        for block in self.blocks:
            x = block(x, global_cond)
            
        # 4. 预测速度场
        x = self.final_norm(x)
        v = self.final_conv(x)
        
        return v