"""
LGT (Latent Geometric Thermodynamics) Model Architecture

Core Design:
- Encoder: Standard convolutions for structure extraction
- Bottleneck: 8×8 resolution forcing semantic compression
- Decoder: StyleDynamicConv for texture generation
- Conditioning: Time (sinusoidal) + Style (learned embeddings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP projection."""
    
    def __init__(self, dim=256, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t):
        """
        Args:
            t: [B] tensor of timesteps in [0, 1]
        Returns:
            emb: [B, dim] timestep embeddings
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return self.mlp(emb)


class StyleEmbedding(nn.Module):
    """Learnable style embeddings with average embedding for CFG."""
    
    def __init__(self, num_styles, style_dim=256):
        super().__init__()
        self.num_styles = num_styles
        self.style_dim = style_dim
        
        # Learnable embeddings for each style
        self.embeddings = nn.Embedding(num_styles, style_dim)
        
        # Average embedding for unconditional generation (CFG)
        self.register_buffer('avg_embedding', torch.zeros(style_dim))
        self.avg_computed = False
    
    def compute_avg_embedding(self):
        """Compute average of all style embeddings (call before training)."""
        with torch.no_grad():
            self.avg_embedding.copy_(self.embeddings.weight.mean(dim=0))
        self.avg_computed = True
    
    def forward(self, style_id, use_avg=False):
        """
        Args:
            style_id: [B] style class IDs
            use_avg: bool, use average embedding for unconditional
        Returns:
            emb: [B, style_dim] style embeddings
        """
        if use_avg:
            if not self.avg_computed:
                self.compute_avg_embedding()
            return self.avg_embedding.unsqueeze(0).expand(style_id.shape[0], -1)
        else:
            return self.embeddings(style_id)


class StyleDynamicConv(nn.Module):
    """
    Dynamic Depthwise Convolution conditioned on style.
    HyperNet generates convolution kernels from style embeddings.
    
    Optimized: Uses grouped convolution to process entire batch in one kernel launch.
    """
    
    def __init__(self, in_channels, kernel_size=3, style_dim=256, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # HyperNet: style_dim → depthwise conv weights
        # Output shape: [C, 1, K, K] for depthwise conv
        self.hypernet = nn.Linear(style_dim, in_channels * kernel_size * kernel_size)
        
        # Initialize with small weights to start near identity
        nn.init.zeros_(self.hypernet.weight)
        nn.init.zeros_(self.hypernet.bias)
    
    def forward(self, x, style_emb):
        """
        Args:
            x: [B, C, H, W] input features
            style_emb: [B, style_dim] style conditioning
        Returns:
            out: [B, C, H, W] dynamically filtered features
        
        Optimization: Fold batch into channels, use one grouped conv (B*C groups).
        This replaces B separate kernel launches with 1 large launch.
        """
        B, C, H, W = x.shape
        K = self.kernel_size
        
        # Generate dynamic weights: [B, C*K*K] -> [B*C, 1, K, K]
        weights = self.hypernet(style_emb)
        weights = weights.view(B * C, 1, K, K)
        
        # Reshape input: [B, C, H, W] -> [1, B*C, H, W]
        # This is a zero-copy stride operation
        x_reshaped = x.view(1, B * C, H, W)
        
        # Single grouped convolution: groups=B*C
        # Each of the B*C input channels gets its own 1x1xKxK kernel
        out = F.conv2d(x_reshaped, weights, padding=self.padding, groups=B * C)
        
        # Reshape back: [1, B*C, H, W] -> [B, C, H, W]
        return out.view(B, C, H, W)


class ResidualBlock(nn.Module):
    """Standard residual block with GroupNorm."""
    
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.GroupNorm(32, channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class StyleResidualBlock(nn.Module):
    """
    Residual block with StyleDynamicConv for decoder.
    Structure: StyleDynamicConv → GroupNorm → SiLU → Conv1x1 → Add Residual
    """
    
    def __init__(self, channels, kernel_size=3, style_dim=256):
        super().__init__()
        self.style_conv = StyleDynamicConv(channels, kernel_size, style_dim)
        self.norm = nn.GroupNorm(32, channels)
        self.conv = nn.Conv2d(channels, channels, 1)  # Pointwise
        self.act = nn.SiLU()
    
    def forward(self, x, style_emb):
        residual = x
        x = self.style_conv(x, style_emb)
        x = self.act(self.norm(x))
        x = self.conv(x)
        return self.act(x + residual)


class LGTUNet(nn.Module):
    """
    Latent Geometric Thermodynamics U-Net.
    
    Architecture:
    - Encoder: Standard conv, extract structure (32×32 → 8×8)
    - Bottleneck: Minimal resolution (8×8), semantic skeleton only
    - Decoder: StyleDynamicConv, generate texture distributions (8×8 → 32×32)
    
    Conditioning:
    - Time: Sinusoidal embedding (continuous t ∈ [0,1])
    - Style: Learned embeddings (discrete style classes)
    """
    
    def __init__(
        self,
        latent_channels=4,
        base_channels=128,
        style_dim=256,
        time_dim=256,
        num_styles=2,
        num_encoder_blocks=2,
        num_decoder_blocks=3,
        shared_style_mlp=True
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.style_dim = style_dim
        
        # Conditioning embeddings
        self.time_embed = TimestepEmbedding(time_dim)
        self.style_embed = StyleEmbedding(num_styles, style_dim)
        
        # Conditioning fusion: time + style → unified conditioning
        self.cond_fusion = nn.Sequential(
            nn.Linear(time_dim + style_dim, style_dim * 2),
            nn.SiLU(),
            nn.Linear(style_dim * 2, style_dim)
        )
        
        # ============ ENCODER (Structure Extraction) ============
        # Input projection: 4ch → base_channels
        self.input_proj = nn.Conv2d(latent_channels, base_channels, 3, padding=1)
        
        # Encoder blocks at 32×32
        self.encoder_blocks_32 = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_encoder_blocks)
        ])
        
        # Downsample: 32×32 → 16×16
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        
        # Encoder blocks at 16×16
        self.encoder_blocks_16 = nn.ModuleList([
            ResidualBlock(base_channels * 2) for _ in range(num_encoder_blocks)
        ])
        
        # Downsample: 16×16 → 8×8 (BOTTLENECK)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        
        # ============ BOTTLENECK (8×8) ============
        # At this resolution, high-freq texture is lost, only semantic skeleton remains
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )
        
        # ============ DECODER (Texture Generation with Style) ============
        # Upsample: 8×8 → 16×16
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        
        # Decoder blocks at 16×16 (with StyleDynamicConv)
        self.decoder_blocks_16 = nn.ModuleList([
            StyleResidualBlock(base_channels * 2, style_dim=style_dim) for _ in range(num_decoder_blocks)
        ])
        
        # Upsample: 16×16 → 32×32
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        
        # Decoder blocks at 32×32 (with StyleDynamicConv)
        self.decoder_blocks_32 = nn.ModuleList([
            StyleResidualBlock(base_channels, style_dim=style_dim) for _ in range(num_decoder_blocks)
        ])
        
        # Output projection: base_channels → 4ch
        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, 3, padding=1)
        )
        
        # Initialize output projection to near-zero for stability
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)
    
    def forward(self, x, t, style_id, use_avg_style=False):
        """
        Args:
            x: [B, 4, H, W] latent at time t (typically H=W=32)
            t: [B] timestep in [0, 1]
            style_id: [B] style class ID
            use_avg_style: bool, use average style embedding (for CFG)
        
        Returns:
            v: [B, 4, H, W] velocity field
        """
        # Embed conditioning signals
        t_emb = self.time_embed(t)  # [B, time_dim]
        s_emb = self.style_embed(style_id, use_avg=use_avg_style)  # [B, style_dim]
        
        # Fuse time and style into unified conditioning
        cond_emb = self.cond_fusion(torch.cat([t_emb, s_emb], dim=-1))  # [B, style_dim]
        
        # ============ ENCODER ============
        h = self.input_proj(x)
        
        # 32×32 resolution
        for block in self.encoder_blocks_32:
            h = block(h)
        h_32 = h  # Skip connection
        
        # Downsample to 16×16
        h = self.down1(h)
        
        # 16×16 resolution
        for block in self.encoder_blocks_16:
            h = block(h)
        h_16 = h  # Skip connection
        
        # Downsample to 8×8 (bottleneck)
        h = self.down2(h)
        
        # ============ BOTTLENECK ============
        h = self.bottleneck(h)
        
        # ============ DECODER (Style-conditioned) ============
        # Upsample to 16×16
        h = self.up1(h)
        h = h + h_16  # Skip connection
        
        # 16×16 resolution with style dynamics
        for block in self.decoder_blocks_16:
            h = block(h, cond_emb)
        
        # Upsample to 32×32
        h = self.up2(h)
        h = h + h_32  # Skip connection
        
        # 32×32 resolution with style dynamics
        for block in self.decoder_blocks_32:
            h = block(h, cond_emb)
        
        # Output projection
        v = self.output_proj(h)
        
        return v
    
    def compute_avg_style_embedding(self):
        """Compute average style embedding (call once before training)."""
        self.style_embed.compute_avg_embedding()


# Model size calculation utility
def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model instantiation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LGTUNet(
        latent_channels=4,
        base_channels=128,
        style_dim=256,
        time_dim=256,
        num_styles=2,
        num_encoder_blocks=2,
        num_decoder_blocks=3
    ).to(device)
    
    # Compute average style embedding
    model.compute_avg_style_embedding()
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 32, 32, device=device)
    t = torch.rand(batch_size, device=device)
    style_id = torch.randint(0, 2, (batch_size,), device=device)
    
    with torch.no_grad():
        v_cond = model(x, t, style_id, use_avg_style=False)
        v_uncond = model(x, t, style_id, use_avg_style=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Conditional output shape: {v_cond.shape}")
    print(f"Unconditional output shape: {v_uncond.shape}")
    print("✓ Model test passed!")
