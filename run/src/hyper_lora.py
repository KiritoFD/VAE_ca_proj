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
    Hyper-LoRA Linear Layer.
    
    Theory: y = x @ W_base.T + alpha * (x @ A.T @ B.T)
    
    Where A and B are dynamically generated from style embedding.
    Zero-initialization ensures the model starts as identity (base_linear only).
    """
    def __init__(self, in_features, out_features, style_dim=256, rank=8, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Base weights (can be frozen or finetuned)
        self.base_linear = nn.Linear(in_features, out_features)
        
        # HyperNetwork: Generates A and B matrices from style embedding
        self.hyper_gen = nn.Sequential(
            nn.Linear(style_dim, style_dim // 2),
            nn.SiLU(),
            nn.Linear(style_dim // 2, (in_features + out_features) * rank)
        )
        
        # Zero-Init: Ensures Identity behavior at training start
        # Critical for seamless checkpoint migration
        nn.init.zeros_(self.hyper_gen[-1].weight)
        nn.init.zeros_(self.hyper_gen[-1].bias)

    def forward(self, x, style_emb):
        """
        Args:
            x: [B, N, in_features] input tensor
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
        A = lora_params[:, :split].reshape(B, self.rank, self.in_features).contiguous()
        B_mat = lora_params[:, split:].reshape(B, self.out_features, self.rank).contiguous()
        
        # Compute: (x @ A.T) @ B_mat.T
        low_rank = torch.bmm(x, A.transpose(1, 2))  # [B, N, rank]
        delta = torch.bmm(low_rank, B_mat.transpose(1, 2))  # [B, N, out]
        
        return base_out + self.alpha * delta


class StyleLoRAAttention(nn.Module):
    """
    Style-Conditioned Attention with Hyper-LoRA projections.
    
    Replaces standard linear projections with dynamically generated ones.
    This allows the attention mechanism to adapt its behavior based on
    the target style, enabling more expressive style transfer.
    """
    def __init__(self, channels, style_dim=256, num_heads=8, rank=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.channels = channels
        
        self.norm = nn.LayerNorm(channels)
        
        # Hyper-LoRA projections (dynamically conditioned on style)
        self.q_proj = StyleHyperLinear(channels, channels, style_dim, rank)
        self.k_proj = StyleHyperLinear(channels, channels, style_dim, rank)
        self.v_proj = StyleHyperLinear(channels, channels, style_dim, rank)
        self.out_proj = StyleHyperLinear(channels, channels, style_dim, rank)

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
