import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码 - 标准 Transformer 时间编码"""
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


class AdaGN(nn.Module):
    """
    自适应组归一化 (Adaptive Group Normalization)
    核心：让风格/时间条件直接控制每层特征的分布（均值+方差）
    这是确保条件信号不被忽略的关键组件
    """
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, affine=False)
        
        # 🟢 关键：从条件预测 scale & shift
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 2)
        )
        
    def forward(self, x, cond):
        """
        Args:
            x: [B, C, H, W]
            cond: [B, cond_dim] - 风格+时间的融合嵌入
        """
        # 1. 标准归一化（零均值单位方差）
        x = self.norm(x)
        
        # 2. 从条件预测调制参数
        scale_shift = self.modulation(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        
        # 3. 应用调制（广播到空间维度）
        # 🟢 公式: x = x * (1 + scale) + shift
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        return x


class ContentFusion(nn.Module):
    """
    内容融合模块 - 确保生成图保留原图结构
    使用门控机制动态平衡内容注入强度
    """
    def __init__(self, dim):
        super().__init__()
        self.content_proj = nn.Sequential(
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        
        # 🟢 时间感知门控：早期多注入内容，后期多注入风格
        self.time_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, x_content, time_emb):
        """
        Args:
            x: 当前特征 [B, C, H, W]
            x_content: 原图特征 [B, C, H, W]
            time_emb: 时间嵌入 [B, dim]
        """
        # 计算时间门控系数
        alpha = self.time_gate(time_emb)[:, :, None, None]
        
        # 提取内容特征并融合
        content_feat = self.content_proj(x_content)
        return x + content_feat * alpha


class SAFBlock(nn.Module):
    """
    SA-Flow Block v2 (AdaGN-Based)
    核心改进：所有归一化层都换成 AdaGN，强制注入条件信号
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        
        # 🟢 路径1: 自适应归一化 + 空间卷积
        self.ada_gn1 = AdaGN(dim, dim)
        self.dwconv = nn.Conv2d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            groups=dim  # Depthwise Conv
        )
        
        # 🟢 路径2: Inverted Bottleneck (ConvNeXt style)
        self.ada_gn2 = AdaGN(dim, dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, 1)
        self.act = nn.SiLU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, 1)
        
        # 🟢 内容融合
        self.content_fusion = ContentFusion(dim)
        
        # Layer Scale (稳定深层训练)
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1) * 1e-6)
        
    def forward(self, x, x_content, global_cond, time_emb):
        """
        Args:
            x: [B, C, H, W]
            x_content: [B, C, H, W] 原图特征
            global_cond: [B, dim] 风格+时间嵌入
            time_emb: [B, dim] 时间嵌入（用于门控）
        """
        shortcut = x
        
        # 🟢 Stage 1: 条件归一化 + 空间建模
        x = self.ada_gn1(x, global_cond)
        x = self.dwconv(x)
        
        # 🟢 Stage 2: 条件归一化 + 通道混合
        x = self.ada_gn2(x, global_cond)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # 🟢 Stage 3: 内容注入
        x = self.content_fusion(x, x_content, time_emb)
        
        # 🟢 残差连接 + Layer Scale
        return shortcut + x * self.gamma


class SAFModel(nn.Module):
    """
    SA-Flow v2: AdaGN-Based Conditional Flow Matching
    
    核心改进:
    1. 全局使用 AdaGN 替代普通归一化
    2. 支持 Classifier-Free Guidance (CFG)
    3. 独立的内容编码器
    4. 时间感知的内容融合
    """
    def __init__(
        self, 
        latent_channels=4, 
        hidden_dim=256, 
        num_layers=8, 
        num_styles=2, 
        kernel_size=7,
        **kwargs
    ):
        super().__init__()
        
        # 🟢 核心：Null Class 支持 (CFG 必需)
        self.num_styles = num_styles
        self.null_class_id = num_styles  # 最后一个 ID 是空类别
        
        # 🟢 时间嵌入 (正弦编码)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 🟢 风格嵌入 (支持 N+1 个类别)
        self.style_embed = nn.Embedding(num_styles + 1, hidden_dim)
        
        # 🟢 输入投影
        self.stem = nn.Conv2d(latent_channels, hidden_dim, 3, padding=1)
        
        # 🟢 独立的内容编码器
        self.content_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(16, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
        
        # 🟢 主干网络 (全部使用 AdaGN)
        self.blocks = nn.ModuleList([
            SAFBlock(hidden_dim, kernel_size) 
            for _ in range(num_layers)
        ])
        
        # 🟢 输出层
        self.final_norm = nn.GroupNorm(32, hidden_dim)
        self.final = nn.Conv2d(hidden_dim, latent_channels, 3, padding=1)
        
        # 🟢 零初始化最后一层（标准 Diffusion 实践）
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        
    def forward(self, x_t, x_content, t, style_id):
        """
        Args:
            x_t: [B, 4, H, W] - 当前 Flow 状态
            x_content: [B, 4, H, W] - 原图 Latent
            t: [B] 或 [B, 1] - 时间步 (0~1)
            style_id: [B] - 风格 ID (0~N 或 null_class_id)
        """
        # 1. 时间嵌入
        if t.dim() == 1:
            t = t.view(-1, 1)
        t_emb = self.time_embed(t.squeeze(-1))  # [B, dim]
        
        # 2. 风格嵌入
        s_emb = self.style_embed(style_id)  # [B, dim]
        
        # 3. 🟢 全局条件融合
        global_cond = t_emb + s_emb
        
        # 4. 编码内容
        x_cond = self.content_encoder(x_content)
        
        # 5. 输入投影
        x = self.stem(x_t)
        
        # 6. 主干网络（每一层都注入条件）
        for block in self.blocks:
            x = block(x, x_cond, global_cond, t_emb)
        
        # 7. 输出速度场
        x = self.final_norm(x)
        v = self.final(x)
        
        return v