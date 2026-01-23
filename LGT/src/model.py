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


class AdaGN(nn.Module):
    """
    Adaptive Group Normalization (AdaGN) - Direct statistical moment control.
    
    Physics motivation:
    - Style transfer = distribution matching (Wasserstein distance minimization)
    - Distribution is fully characterized by moments (mean, variance)
    - AdaGN directly modulates these moments via scale (γ) and shift (β)
    
    Mathematical formulation:
    1. Normalize: x_norm = (x - μ) / σ  (zero mean, unit variance)
    2. Affine transform: y = γ(style) * x_norm + β(style)
    
    Key advantage over dynamic convolution:
    - O(1) parameters per style (2C scalars) vs O(C*K²) kernel weights
    - Direct gradient flow to moment statistics
    - Identity initialization (γ=1, β=0) guarantees non-zero initial gradients
    """
    
    def __init__(self, num_channels, style_dim=256, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        
        # Group normalization (base operation)
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)
        
        # Style-conditioned affine parameters
        # Predicts scale (γ) and shift (β) from style embedding
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, num_channels * 2)  # 2C for scale + shift
        )
        
        # CRITICAL: Identity initialization for immediate gradient flow
        # Initialize final layer to output [1, 1, ..., 1, 0, 0, ..., 0]
        # First C values → scale=1.0, Last C values → shift=0.0
        with torch.no_grad():
            # Zero out all weights and biases
            self.style_mlp[-1].weight.zero_()
            self.style_mlp[-1].bias.zero_()
            
            # Set bias for first C outputs to 1.0 (scale=1.0)
            self.style_mlp[-1].bias[:num_channels] = 1.0
            # Last C outputs remain 0.0 (shift=0.0)
    
    def forward(self, x, style_emb):
        """
        Args:
            x: [B, C, H, W] input features
            style_emb: [B, style_dim] style conditioning
        
        Returns:
            out: [B, C, H, W] style-modulated features
        """
        # Step 1: Normalize to zero mean, unit variance
        x_norm = self.group_norm(x)  # [B, C, H, W]
        
        # Step 2: Predict style-dependent scale and shift
        style_params = self.style_mlp(style_emb)  # [B, 2C]
        
        # Split into scale (γ) and shift (β)
        scale, shift = style_params.chunk(2, dim=1)  # Each [B, C]
        
        # Reshape for broadcasting: [B, C] → [B, C, 1, 1]
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        
        # Step 3: Apply affine transformation
        out = scale * x_norm + shift
        
        return out


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
    Residual block with AdaGN for decoder (thermodynamic texture synthesis).
    
    Architecture: Conv3x3 → AdaGN → SiLU → Conv1x1 → Residual
    
    Key design principles:
    1. Standard Conv3x3 (NOT depthwise) - enables RGB channel mixing for color transformation
    2. AdaGN after convolution - modulates feature statistics (mean/variance)
    3. Pointwise Conv1x1 - channel-wise feature recombination
    4. Residual connection - preserves content topology
    
    Physics interpretation:
    - Conv3x3: Spatial interaction (diffusion operator)
    - AdaGN: Statistical control (temperature/pressure modulation)
    - Residual: Conservative field (content anchoring)
    """
    
    def __init__(self, channels, kernel_size=3, style_dim=256):
        super().__init__()
        
        # Standard 3×3 convolution with channel mixing (9× more parameters than depthwise)
        # Critical for color transformation (e.g., blue sky → golden sunset)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        
        # Adaptive Group Normalization - style-conditioned moment control
        self.ada_gn = AdaGN(channels, style_dim=style_dim)
        
        # Pointwise convolution for channel recombination
        self.conv2 = nn.Conv2d(channels, channels, 1)
        
        self.act = nn.SiLU()
    
    def forward(self, x, style_emb):
        """
        Args:
            x: [B, C, H, W] input features
            style_emb: [B, style_dim] style conditioning
        
        Returns:
            out: [B, C, H, W] style-modulated features with residual
        """
        residual = x
        
        # Spatial convolution (feature extraction)
        x = self.conv1(x)
        
        # Style-conditioned normalization (moment modulation)
        x = self.ada_gn(x, style_emb)
        x = self.act(x)
        
        # Channel recombination
        x = self.conv2(x)
        
        # Residual connection (content preservation)
        return self.act(x + residual)


class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention for global context modeling.
    
    Physics motivation:
    - Enables long-range dependencies for coherent "crystallization" during quenching
    - Allows model to use global semantic context ("this is a tree") when sharpening
    - Critical for annealing→quenching asymmetry
    """
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.head_dim = channels // num_heads
        
        # Layer normalization before attention
        self.norm1 = nn.LayerNorm(channels)
        
        # QKV projections
        self.qkv = nn.Linear(channels, channels * 3)
        
        # Output projection
        self.out_proj = nn.Linear(channels, channels)
        
        # Layer normalization after attention
        self.norm2 = nn.LayerNorm(channels)
        
        # Initialize output projection to zero for residual stability
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input features
        
        Returns:
            out: [B, C, H, W] attention-modulated features
        """
        B, C, H, W = x.shape
        N = H * W
        
        # Reshape: [B, C, H, W] → [B, N, C]
        x_flat = x.view(B, C, N).transpose(1, 2)  # [B, N, C]
        
        # Residual connection
        residual = x_flat
        
        # Pre-normalization
        x_norm = self.norm1(x_flat)  # [B, N, C]
        
        # QKV projection and reshape
        qkv = self.qkv(x_norm)  # [B, N, 3*C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, N, N]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ v  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection
        out = residual + out
        
        # Post-normalization
        out = self.norm2(out)
        
        # Reshape back: [B, N, C] → [B, C, H, W]
        out = out.transpose(1, 2).view(B, C, H, W)
        
        return out


class StyleGate(nn.Module):
    """
    Style-conditioned gating for skip connections (Maxwell's Demon).
    
    Physics motivation:
    - Acts as information filter based on target style
    - Annealing (→painting): Gate half-open, preserve structural edges
    - Quenching (→photo): Gate auto-closes, blocks source brush strokes
    
    Architecture: style_emb → MLP → Sigmoid → Channel-wise gate
    """
    
    def __init__(self, channels, style_dim=256):
        super().__init__()
        self.channels = channels
        
        # MLP: style_dim → style_dim → channels
        self.gate_mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, channels),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # CRITICAL: Initialize final layer bias to 0
        # Sigmoid(0) = 0.5 → half-open gate at initialization
        with torch.no_grad():
            self.gate_mlp[-2].weight.zero_()
            self.gate_mlp[-2].bias.zero_()
    
    def forward(self, x, style_emb):
        """
        Args:
            x: [B, C, H, W] skip connection features
            style_emb: [B, style_dim] style conditioning
        
        Returns:
            gated_x: [B, C, H, W] style-gated features
        """
        # Compute channel-wise gate: [B, C]
        gate = self.gate_mlp(style_emb)  # [B, C]
        
        # Reshape for broadcasting: [B, C] → [B, C, 1, 1]
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        
        # Apply gate
        return x * gate


class LGTUNet(nn.Module):
    """
    LGT++ U-Net with Self-Attention and Style-Gated Skip Connections.
    
    Architecture:
    - Encoder: Standard conv, extract structure (32×32 → 8×8)
    - Bottleneck: Self-Attention at 8×8 for global context
    - Decoder: AdaGN-based blocks with style-gated skips (8×8 → 32×32)
    
    Key innovations (LGT++):
    1. Self-Attention in bottleneck: Long-range dependency modeling
    2. StyleGate skip connections: Adaptive entropy flow control
    3. AdaGN in decoder: Direct statistical moment modulation
    
    Physics:
    - Supports asymmetric phase transitions (annealing ↔ quenching)
    - Entropy control via information gating (Maxwell's Demon)
    - Multi-scale energy minimization via attention + local conv
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
        self.bottleneck_conv = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )
        
        # Self-Attention for global context (LGT++ enhancement)
        self.bottleneck_attn = SelfAttention(
            channels=base_channels * 4,
            num_heads=8
        )
        
        # ============ DECODER (Texture Generation with Style) ============
        # Upsample: 8×8 → 16×16
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        
        # Style-gated skip connection for 16×16 (LGT++ enhancement)
        self.skip_gate_16 = StyleGate(base_channels * 2, style_dim=style_dim)
        
        # Decoder blocks at 16×16 (with StyleDynamicConv)
        self.decoder_blocks_16 = nn.ModuleList([
            StyleResidualBlock(base_channels * 2, style_dim=style_dim) for _ in range(num_decoder_blocks)
        ])
        
        # Upsample: 16×16 → 32×32
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        
        # Style-gated skip connection for 32×32 (LGT++ enhancement)
        self.skip_gate_32 = StyleGate(base_channels, style_dim=style_dim)
        
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
        # Convolutional processing
        h = self.bottleneck_conv(h)
        
        # Self-Attention for global context (LGT++ enhancement)
        h = self.bottleneck_attn(h)
        
        # ============ DECODER (Style-conditioned) ============
        # Upsample to 16×16
        h = self.up1(h)
        
        # Style-gated skip connection (LGT++ enhancement)
        h_16_gated = self.skip_gate_16(h_16, cond_emb)
        h = h + h_16_gated
        
        # 16×16 resolution with style dynamics
        for block in self.decoder_blocks_16:
            h = block(h, cond_emb)
        
        # Upsample to 32×32
        h = self.up2(h)
        
        # Style-gated skip connection (LGT++ enhancement)
        h_32_gated = self.skip_gate_32(h_32, cond_emb)
        h = h + h_32_gated
        
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
