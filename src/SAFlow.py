import torch
import torch.nn as nn

class TimeAwareSpatialModulator(nn.Module):
    """
    时间感知空间调制器 (Time-Aware Spatial Modulator)
    
    原理：
    根据时间步 t 动态控制结构信息 (Content) 的注入强度。
    - 在 t 较小（接近原图）时，可能允许更多结构通过。
    - 在 t 较大（接近目标）时，调节结构与风格的平衡。
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        
        # 用于提取 Condition (x_content) 的特征
        self.conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        
        # 时间门控: 将时间嵌入映射为 [0, 1] 的门控系数
        self.time_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid() 
        )

    def forward(self, x, x_cond, t_emb):
        """
        x: 当前流特征 (Main Stream)
        x_cond: 内容图的特征 (Condition Stream)
        t_emb: 时间嵌入 (Global Time Embedding)
        """
        # 计算门控系数 alpha: [Batch, Dim, 1, 1]
        alpha = self.time_gate(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        # 处理条件特征
        cond_feat = self.conv(x_cond)
        
        # 调制: 原特征 + alpha * 结构特征
        # 这种残差式的注入方式能极好地保留原图的几何结构
        return x + cond_feat * alpha

class AdaGN(nn.Module):
    """
    自适应组归一化 (Adaptive Group Norm)
    
    原理：
    类似于 StyleGAN 的 AdaIN，通过全局条件 (Style + Time) 
    预测归一化层的缩放 (Scale) 和偏移 (Shift) 参数。
    这是风格注入的主要通道。
    """
    def __init__(self, dim, emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, dim * 2) # 输出 scale 和 shift
        )

    def forward(self, x, emb):
        # 从全局嵌入预测仿射参数
        scale_shift = self.proj(emb)
        scale, shift = scale_shift.chunk(2, dim=1)
        
        # 执行归一化与调制
        x = self.norm(x)
        x = x * (1 + scale[..., None, None]) + shift[..., None, None]
        return x

class SAFBlock(nn.Module):
    """
    Structure-Aware Flow Block
    
    结合了 MetaFormer/ConvNeXt 的设计哲学：
    1. AdaGN: 注入全局风格
    2. Spatial Modulator: 注入局部结构
    3. Depthwise Conv: 处理空间信息
    4. Pointwise Conv / MLP: 混合通道信息
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.ada_gn = AdaGN(dim, dim)
        self.spatial_mod = TimeAwareSpatialModulator(dim)
        
        # 大核深度卷积 (Large Kernel Depthwise Conv) - 增强感受野
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                padding=kernel_size//2, groups=dim)
        
        # 倒瓶颈结构 (Inverted Bottleneck)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, 1)
        self.act = nn.SiLU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x, x_cond, global_emb):
        shortcut = x
        
        # 1. 全局风格注入 (Global Style Injection)
        x = self.ada_gn(x, global_emb)
        
        # 2. 局部结构注入 (Local Structure Injection)
        x = self.spatial_mod(x, x_cond, global_emb)
        
        # 3. 空间混合 (Spatial Mixing)
        x = self.dwconv(x)
        
        # 4. 通道混合 (Channel Mixing / FFN)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # 残差连接
        return shortcut + x

class SAFModel(nn.Module):
    """
    SA-Flow v6: Structure-Aware Flow Matching Model
    """
    def __init__(self, latent_channels=4, hidden_dim=256, num_layers=8, num_styles=2, kernel_size=7):
        super().__init__()
        
        # ================= 全局条件嵌入 =================
        # 时间嵌入 MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), 
            nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 风格嵌入 (Learnable Style Embeddings)
        self.style_embed = nn.Embedding(num_styles, hidden_dim)
        
        # ================= 图像特征处理 =================
        # 输入 Stem (处理 noisy latents x_t)
        self.stem = nn.Conv2d(latent_channels, hidden_dim, 3, padding=1)
        
        # 条件编码器 (处理 clean content latents x_c)
        # 将 x_c 映射到与主干相同的维度，作为结构引导
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
        
        # ================= 主干网络 =================
        self.blocks = nn.ModuleList([
            SAFBlock(hidden_dim, kernel_size) for _ in range(num_layers)
        ])
        
        # ================= 输出头 =================
        self.final_norm = nn.GroupNorm(32, hidden_dim)
        self.final = nn.Conv2d(hidden_dim, latent_channels, 3, padding=1)
        
        # 零初始化 (Zero Initialization)
        # 使得初始输出接近 0，Flow Matching 训练更稳定
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x_t, x_content, t, style_id):
        """
        Args:
            x_t: 当前时刻的噪声化潜变量 [B, 4, H, W]
            x_content: 原始内容潜变量 [B, 4, H, W] (用于保持结构)
            t: 时间步 [B] 或 [B, 1] (0~1)
            style_id: 目标风格索引 [B]
        Returns:
            v_pred: 预测的速度向量 [B, 4, H, W]
        """
        # 1. 准备全局条件 (Time + Style)
        # t: [B] -> [B, 1] -> [B, Dim]
        if t.dim() == 1:
            t = t.view(-1, 1)
        t_emb = self.time_mlp(t)
        
        s_emb = self.style_embed(style_id)
        
        # 简单的加和融合 (也可以做 concat)
        global_emb = t_emb + s_emb 
        
        # 2. 准备局部条件 (Content Structure)
        x_cond = self.cond_encoder(x_content)
        
        # 3. 主干前向传播
        x = self.stem(x_t)
        
        for block in self.blocks:
            # 每个 Block 都接收:
            # - x: 当前流
            # - x_cond: 结构引导 (Spatial Modulator 用)
            # - global_emb: 风格引导 (AdaGN 用)
            x = block(x, x_cond, global_emb)
            
        # 4. 输出预测
        x = self.final_norm(x)
        return self.final(x)