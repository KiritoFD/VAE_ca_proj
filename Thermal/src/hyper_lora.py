"""
Hyper-LoRA Modules for Dynamic Style-Conditioned Attention.

Theory: Instead of fixed projection weights, we generate low-rank deltas
conditioned on style embeddings. This allows the model to dynamically
adjust its attention patterns based on the target style.

Optimization for RTX 4070:
- Zero-initialized hypernets ensure identity behavior at start
- Grouped generation for batch efficiency
- Contiguous memory layout for Tensor Core utilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleHyperLinear(nn.Module):
    """
    Hyper-LoRA Linear Layer - OPTIMIZED FOR RTX 4070 WITH DIAGNOSTIC MONITORING.
    
    🔥 CRITICAL IMPROVEMENTS:
    1. Rank=16: Aligns with GPU memory bus width (128-bit GDDR6)
    2. Wider HyperNet: style_dim → style_dim (vs style_dim//2)
    3. Better expressiveness: Can capture complex style variations
    4. Built-in diagnostics: Monitor LoRA contribution ratio
    """
    def __init__(self, in_features, out_features, style_dim=256, rank=16, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # 🔥 Debug flag (controlled externally by trainer)
        self.debug_trigger = False
        self.debug_name = f"Linear({in_features}→{out_features})"
        
        # Base weights (can be frozen or finetuned)
        self.base_linear = nn.Linear(in_features, out_features)
        
        # 🔥 OPTIMIZED HyperNetwork for 4070
        # ================================================================
        # Increased hidden dim from style_dim//2 to style_dim
        # This allows HyperNet to capture more complex style patterns
        # without significant memory overhead (still < 1MB on 4070)
        self.hyper_gen = nn.Sequential(
            nn.Linear(style_dim, style_dim),  # 256→256 (not 256→128)
            nn.SiLU(),
            nn.Linear(style_dim, (in_features + out_features) * rank)
        )
        
        # Zero-Init: Ensures Identity behavior at training start
        # Critical for seamless checkpoint migration
        nn.init.zeros_(self.hyper_gen[-1].weight)
        nn.init.zeros_(self.hyper_gen[-1].bias)

    def forward(self, x, style_emb):
        """
        Args:
            x: [B, N, in_features] input tensor (N = spatial dim)
            style_emb: [B, style_dim] style embedding
        
        Returns:
            out: [B, N, out_features] transformed tensor
        """
        base_out = self.base_linear(x)
        
        # Grouped Generation Optimization
        # Only compute hypernet for unique styles in the batch
        unique_styles, inv_idx = torch.unique(style_emb, dim=0, return_inverse=True)
        lora_params = self.hyper_gen(unique_styles)[inv_idx]
        
        B = x.shape[0]
        split = self.in_features * self.rank
        
        # Reshape & Contiguous for Tensor Cores
        # 🔥 Rank=16 optimization: ensures clean register allocation on 4070
        A = lora_params[:, :split].reshape(B, self.rank, self.in_features).contiguous()
        B_mat = lora_params[:, split:].reshape(B, self.out_features, self.rank).contiguous()
        
        # Compute: (x @ A.T) @ B_mat.T
        # bmm is heavily optimized on NVIDIA GPUs via Tensor Cores
        low_rank = torch.bmm(x, A.transpose(1, 2))  # [B, N, rank]
        delta = torch.bmm(low_rank, B_mat.transpose(1, 2))  # [B, N, out]
        
        # 🔥 DEBUG: Monitor LoRA contribution ratio
        # ================================================================
        # Theory: LoRA contribution should be in [0.01, 0.2] (1-20% of base output)
        # - If ratio < 0.001: HyperNet is dead (not learning)
        # - If ratio > 0.5: LoRA is overwhelming base weights
        # 
        # Healthy range indicates good balance between base and adaptive components
        if self.debug_trigger:
            with torch.no_grad():
                base_norm = base_out.norm().item()
                delta_norm = (self.alpha * delta).norm().item()
                ratio = delta_norm / (base_norm + 1e-8)
                
                if ratio < 0.001:
                    print(f"⚠️  [{self.debug_name}] DEAD LoRA: ratio={ratio:.6f} (base={base_norm:.3f} delta={delta_norm:.6f})")
                elif ratio > 0.5:
                    print(f"⚠️  [{self.debug_name}] EXPLODING LoRA: ratio={ratio:.4f} (base={base_norm:.3f} delta={delta_norm:.3f})")
                elif ratio >= 0.01 and ratio <= 0.2:
                    print(f"✓ [{self.debug_name}] Healthy LoRA: ratio={ratio:.4f} (base={base_norm:.3f} delta={delta_norm:.3f})")
                else:
                    print(f"ℹ  [{self.debug_name}] LoRA ratio={ratio:.4f} (base={base_norm:.3f} delta={delta_norm:.3f})")
        
        return base_out + self.alpha * delta


class StyleLoRAAttention(nn.Module):
    """
    Style-Conditioned Attention with Hyper-LoRA projections.
    
    Optimized for RTX 4070 with higher Rank for better style expressiveness.
    """
    def __init__(self, channels, style_dim=256, num_heads=8, rank=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.channels = channels
        
        self.norm = nn.LayerNorm(channels)
        
        # 🔥 INCREASED RANK: 8→32 for better attention capacity on 4070
        # This allows the model to learn more diverse style-conditioned attention patterns
        self.q_proj = StyleHyperLinear(channels, channels, style_dim, rank=rank)
        self.k_proj = StyleHyperLinear(channels, channels, style_dim, rank=rank)
        self.v_proj = StyleHyperLinear(channels, channels, style_dim, rank=rank)
        self.out_proj = StyleHyperLinear(channels, channels, style_dim, rank=rank)

    def forward(self, x, style_emb):
        """
        Args:
            x: [B, C, H, W] input feature map
            style_emb: [B, style_dim] style embedding
        
        Returns:
            out: [B, C, H, W] attended feature map
        """
        B, C, H, W = x.shape
        
        # Layout transformation: [B, C, H, W] -> [B, N, C]
        x_flat = x.permute(0, 2, 3, 1).reshape(B, -1, C).contiguous()
        
        res = x_flat
        x_norm = self.norm(x_flat)
        
        # Dynamic Projections with Hyper-LoRA
        q = self.q_proj(x_norm, style_emb)
        k = self.k_proj(x_norm, style_emb)
        v = self.v_proj(x_norm, style_emb)
        
        # Reshape for multi-head attention
        q = q.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        
        out = self.out_proj(out, style_emb)
        
        # Residual + restore layout
        return (out + res).transpose(1, 2).reshape(B, C, H, W).contiguous()


def migrate_attention_weights(old_state_dict, new_state_dict, prefix="bottleneck_attn"):
    """
    Migrate weights from standard attention to Hyper-LoRA attention.
    
    Maps: {prefix}.{q,k,v,out}_proj.{weight,bias}
    To:   {prefix}.{q,k,v,out}_proj.base_linear.{weight,bias}
    
    Args:
        old_state_dict: checkpoint state dict
        new_state_dict: model state dict (will be modified in-place)
        prefix: attention module prefix
    
    Returns:
        migrated_count: number of successfully migrated keys
    """
    migrated_count = 0
    proj_names = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
    
    for proj in proj_names:
        for suffix in ['weight', 'bias']:
            old_key = f"{prefix}.{proj}.{suffix}"
            new_key = f"{prefix}.{proj}.base_linear.{suffix}"
            
            if old_key in old_state_dict and new_key in new_state_dict:
                if old_state_dict[old_key].shape == new_state_dict[new_key].shape:
                    new_state_dict[new_key] = old_state_dict[old_key]
                    migrated_count += 1
    
    return migrated_count
