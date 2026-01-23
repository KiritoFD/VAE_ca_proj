"""
LGT Geometric Free Energy Loss Functions

Core losses implementing geometric thermodynamics:
1. Patch-Sliced Wasserstein Distance (Patch-SWD): Style distribution matching
2. Cosine Self-Similarity Matrix (Cosine-SSM): Content topology preservation

Both losses operate in FP32 for numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSlicedWassersteinLoss(nn.Module):
    """
    Patch-Sliced Wasserstein Distance for style distribution matching.
    
    Pipeline:
    1. Unfold: Extract 3×3 patches → [B, N_patches, 36] (4×3×3=36)
    2. Sample: Random sample 4096 points for memory efficiency
    3. Project: Radon transform with 64 random projections
    4. Sort: Quantile alignment
    5. Metric: MSE between sorted distributions
    
    Critical: All ops in FP32 for stability.
    """
    
    def __init__(
        self,
        patch_size=3,
        num_projections=64,
        max_samples=4096,
        use_fp32=True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_projections = num_projections
        self.max_samples = max_samples
        self.use_fp32 = use_fp32
    
    def forward(self, x_pred, x_style):
        """
        Args:
            x_pred: [B, 4, H, W] predicted latent
            x_style: [B, 4, H, W] style reference latent
        
        Returns:
            loss: scalar SWD loss
        """
        original_dtype = x_pred.dtype
        
        # Convert to FP32 for numerical stability
        # Use autocast context instead of manual .float() for portability
        with torch.amp.autocast('cuda', enabled=False):
            x_pred = x_pred.float()
            x_style = x_style.float()
            
            B, C, H, W = x_pred.shape
            
            # Step 1: Unfold to extract 3×3 patches
            # Output: [B, C*K*K, N_patches] where N_patches = H*W for k=3, pad=1
            x_pred_patches = F.unfold(x_pred, kernel_size=self.patch_size, padding=self.patch_size//2)
            x_style_patches = F.unfold(x_style, kernel_size=self.patch_size, padding=self.patch_size//2)
            
            # Reshape to [B, N_patches, C*K*K]
            # Memory optimization: unfold output may have non-contiguous stride layout
            # transpose changes only strides (no copy), but reshape after requires contiguous memory
            x_pred_patches = x_pred_patches.transpose(1, 2).contiguous()  # [B, N_patches, 36]
            x_style_patches = x_style_patches.transpose(1, 2).contiguous()  # [B, N_patches, 36]
            
            N_patches = x_pred_patches.shape[1]
            feature_dim = x_pred_patches.shape[2]  # 36 for 4ch×3×3
            
            # Step 2: Sample patches for memory efficiency
            # Flatten batch and patches: [B*N_patches, 36]
            x_pred_flat = x_pred_patches.reshape(-1, feature_dim)
            x_style_flat = x_style_patches.reshape(-1, feature_dim)
            
            total_samples = x_pred_flat.shape[0]
            
            if total_samples > self.max_samples:
                # Random sampling without replacement
                indices = torch.randperm(total_samples, device=x_pred.device)[:self.max_samples]
                x_pred_sampled = x_pred_flat[indices]
                x_style_sampled = x_style_flat[indices]
            else:
                x_pred_sampled = x_pred_flat
                x_style_sampled = x_style_flat
            
            # Step 3: Random projections (Radon transform)
            # Generate random projection matrix: [36, num_projections]
            theta = torch.randn(
                feature_dim,
                self.num_projections,
                device=x_pred.device,
                dtype=x_pred.dtype
            )
            theta = theta / theta.norm(dim=0, keepdim=True)  # Normalize columns
            
            # Project: [N_sampled, 36] @ [36, 64] → [N_sampled, 64]
            # Contiguous memory layout ensures optimal gemm performance
            proj_pred = x_pred_sampled @ theta
            proj_style = x_style_sampled @ theta
            
            # Step 4: Sort for quantile alignment
            proj_pred_sorted, _ = torch.sort(proj_pred, dim=0)
            proj_style_sorted, _ = torch.sort(proj_style, dim=0)
            
            # Step 5: Compute SWD as MSE between sorted distributions
            loss = F.mse_loss(proj_pred_sorted, proj_style_sorted)
        
        return loss


class CosineSSMLoss(nn.Module):
    """
    Cosine Self-Similarity Matrix for content topology preservation.
    
    Pipeline:
    1. Reshape: [B, C, H, W] → [B, C, N] where N=H*W
    2. Normalize: L2-normalize along channel dimension (magnitude invariant)
    3. Gram Matrix: Compute A = Z^T @ Z (cosine similarity between all spatial positions)
    4. Metric: Frobenius norm ||A_pred - A_src||_F^2
    
    Physical meaning: Locks angular relationships between feature points,
    allows style-induced scaling but preserves topological structure.
    
    Critical: L2-normalization and matmul in FP32 for stability.
    
    Optimization: Optional Monte Carlo sampling to reduce O(N^2) memory cost.
    """
    
    def __init__(self, use_fp32=True, normalize_by_num_elements=True, max_spatial_samples=None):
        super().__init__()
        self.use_fp32 = use_fp32
        self.normalize_by_num_elements = normalize_by_num_elements
        # If set, randomly sample spatial positions to reduce memory from O(N^2) to O(S^2)
        # For 32×32 latents with max_spatial_samples=256: 256² = 65K vs 1024² = 1M (16x reduction)
        self.max_spatial_samples = max_spatial_samples
    
    def forward(self, x_pred, x_src):
        """
        Args:
            x_pred: [B, 4, H, W] predicted latent
            x_src: [B, 4, H, W] source content latent
        
        Returns:
            loss: scalar SSM loss
        """
        # Convert to FP32 for numerical stability
        # Use autocast context for portability across different GPU types
        with torch.amp.autocast('cuda', enabled=False):
            x_pred = x_pred.float()
            x_src = x_src.float()
            
            B, C, H, W = x_pred.shape
            N = H * W
            
            # Step 1: Reshape to [B, C, N]
            z_pred = x_pred.view(B, C, N)
            z_src = x_src.view(B, C, N)
            
            # Optional: Monte Carlo sampling to reduce memory (O(N^2) -> O(S^2))
            if self.max_spatial_samples is not None and N > self.max_spatial_samples:
                indices = torch.randperm(N, device=x_pred.device)[:self.max_spatial_samples]
                z_pred = z_pred[..., indices]
                z_src = z_src[..., indices]
                N = self.max_spatial_samples
            
            # Step 2: L2-normalize along channel dimension (critical for cosine similarity)
            # This removes magnitude/intensity, keeping only directional info
            z_pred = F.normalize(z_pred, p=2, dim=1)  # [B, C, N]
            z_src = F.normalize(z_src, p=2, dim=1)    # [B, C, N]
            
            # Step 3: Compute Gram matrix (Self-Similarity Matrix)
            # A[i,j] = cosine similarity between spatial position i and j
            # A = Z^T @ Z: [B, N, C] @ [B, C, N] → [B, N, N]
            A_pred = torch.bmm(z_pred.transpose(1, 2), z_pred)  # [B, N, N]
            A_src = torch.bmm(z_src.transpose(1, 2), z_src)     # [B, N, N]
            
            # Step 4: Frobenius norm (sum of squared differences)
            diff = A_pred - A_src
            loss = torch.sum(diff ** 2)
            
            # Normalize by number of elements for scale invariance
            if self.normalize_by_num_elements:
                loss = loss / (B * N * N)
        
        return loss


class MultiScaleSWDLoss(nn.Module):
    """
    Multi-Scale Sliced Wasserstein Distance for unified annealing/quenching.
    
    Physics motivation:
    - Unifies optimization objective for both phase transitions
    - Matching photo's high-freq distribution → auto-sharpens (quenching)
    - Matching painting's low-freq distribution → auto-smooths (annealing)
    - No need for w_freq or asymmetric hard constraints
    
    Architecture:
    - Scale 1×1 (Pixel): Color palette distribution
    - Scale 3×3 (Texture): High-frequency details (brushstrokes/noise)
    - Scale 7×7 (Structure): Local structural patterns (edges/shapes)
    
    Default weights: [1.0, 1.0, 1.0] - let model balance frequency bands naturally
    """
    
    def __init__(
        self,
        scales=[1, 3, 7],
        scale_weights=[1.0, 1.0, 1.0],
        num_projections=64,
        max_samples=4096,
        use_fp32=True
    ):
        super().__init__()
        
        assert len(scales) == len(scale_weights), "scales and scale_weights must have same length"
        
        self.scales = scales
        self.scale_weights = scale_weights
        
        # Create SWD loss for each scale
        self.swd_losses = nn.ModuleList([
            PatchSlicedWassersteinLoss(
                patch_size=scale,
                num_projections=num_projections,
                max_samples=max_samples,
                use_fp32=use_fp32
            )
            for scale in scales
        ])
    
    def forward(self, x_pred, x_style):
        """
        Args:
            x_pred: [B, 4, H, W] predicted latent
            x_style: [B, 4, H, W] style reference latent
        
        Returns:
            loss: scalar weighted multi-scale SWD
            loss_dict: dictionary with per-scale losses
        """
        total_loss = 0.0
        loss_dict = {}
        
        for scale, weight, swd_loss in zip(self.scales, self.scale_weights, self.swd_losses):
            scale_loss = swd_loss(x_pred, x_style)
            total_loss = total_loss + weight * scale_loss
            loss_dict[f'swd_scale_{scale}'] = scale_loss
        
        loss_dict['swd_total'] = total_loss
        
        return total_loss, loss_dict


class GeometricFreeEnergyLoss(nn.Module):
    """
    Total geometric free energy combining style and content potentials.
    
    LGT++: Uses Multi-Scale SWD for unified annealing/quenching optimization.
    
    E_total = w_style * MultiScaleSWD(x, x_style) + w_content * SSM(x, x_src)
    
    This is the fundamental loss that replaces MSE velocity matching.
    """
    
    def __init__(
        self,
        w_style=1.0,
        w_content=1.0,
        patch_size=3,  # Deprecated: kept for backward compatibility
        num_projections=64,
        max_samples=4096,
        max_spatial_samples=None,
        use_multiscale_swd=True,  # LGT++ enhancement
        swd_scales=[1, 3, 7],
        swd_scale_weights=[1.0, 1.0, 1.0]
    ):
        super().__init__()
        
        self.w_style = w_style
        self.w_content = w_content
        self.use_multiscale_swd = use_multiscale_swd
        
        # Style potential: Multi-Scale SWD (LGT++) or Single-Scale SWD (legacy)
        if use_multiscale_swd:
            self.swd_loss = MultiScaleSWDLoss(
                scales=swd_scales,
                scale_weights=swd_scale_weights,
                num_projections=num_projections,
                max_samples=max_samples,
                use_fp32=True
            )
        else:
            # Legacy single-scale SWD
            self.swd_loss = PatchSlicedWassersteinLoss(
                patch_size=patch_size,
                num_projections=num_projections,
                max_samples=max_samples,
                use_fp32=True
            )
        
        # Content potential: Cosine-SSM with optional spatial sampling
        self.ssm_loss = CosineSSMLoss(
            use_fp32=True,
            max_spatial_samples=max_spatial_samples
        )
    
    def forward(self, x_pred, x_style, x_src):
        """
        Args:
            x_pred: [B, 4, H, W] predicted terminal state
            x_style: [B, 4, H, W] style reference
            x_src: [B, 4, H, W] source content
        
        Returns:
            loss_dict: dictionary with total loss and components
        """
        # Compute style potential (multi-scale or single-scale)
        if self.use_multiscale_swd:
            style_potential, swd_dict = self.swd_loss(x_pred, x_style)
        else:
            style_potential = self.swd_loss(x_pred, x_style)
            swd_dict = {}
        
        # Compute content potential
        content_potential = self.ssm_loss(x_pred, x_src)
        
        # Total free energy
        total_energy = self.w_style * style_potential + self.w_content * content_potential
        
        result = {
            'total': total_energy,
            'style_swd': style_potential,
            'content_ssm': content_potential
        }
        
        # Add per-scale losses if using multi-scale SWD
        result.update(swd_dict)
        
        return result


# Auxiliary losses for training stability (optional)

class VelocityRegularizationLoss(nn.Module):
    """
    L2 regularization on velocity field to prevent explosion.
    Optional: Can help early training stability.
    """
    
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, v):
        """
        Args:
            v: [B, 4, H, W] velocity field
        Returns:
            loss: scalar L2 norm
        """
        return self.weight * torch.mean(v ** 2)


class DivergenceRegularizationLoss(nn.Module):
    """
    Penalize divergence of velocity field.
    Encourages incompressibility (volume preservation).
    
    ∇·v ≈ 0
    
    Optional: Based on physical constraints.
    """
    
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, v):
        """
        Args:
            v: [B, 4, H, W] velocity field
        Returns:
            loss: scalar divergence penalty
        """
        # Compute divergence: ∂v_x/∂x + ∂v_y/∂y (approximate with finite differences)
        # For 4-channel latent, we compute per-channel divergence
        
        # Gradient along height (y direction)
        dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]
        # Gradient along width (x direction)
        dv_dx = v[:, :, :, 1:] - v[:, :, :, :-1]
        
        # Take minimum spatial dimension for broadcasting
        min_h = min(dv_dy.shape[2], dv_dx.shape[2])
        min_w = min(dv_dy.shape[3], dv_dx.shape[3])
        
        dv_dy = dv_dy[:, :, :min_h, :min_w]
        dv_dx = dv_dx[:, :, :min_h, :min_w]
        
        divergence = dv_dy + dv_dx
        
        return self.weight * torch.mean(divergence ** 2)


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    B, C, H, W = 4, 4, 32, 32
    x_pred = torch.randn(B, C, H, W, device=device)
    x_style = torch.randn(B, C, H, W, device=device)
    x_src = torch.randn(B, C, H, W, device=device)
    
    print("Testing Patch-SWD Loss...")
    swd_loss = PatchSlicedWassersteinLoss().to(device)
    swd_value = swd_loss(x_pred, x_style)
    print(f"  SWD Loss: {swd_value.item():.6f}")
    
    print("\nTesting Cosine-SSM Loss...")
    ssm_loss = CosineSSMLoss().to(device)
    ssm_value = ssm_loss(x_pred, x_src)
    print(f"  SSM Loss: {ssm_value.item():.6f}")
    
    print("\nTesting Geometric Free Energy Loss...")
    energy_loss = GeometricFreeEnergyLoss(w_style=1.0, w_content=1.0).to(device)
    loss_dict = energy_loss(x_pred, x_style, x_src)
    print(f"  Total Energy: {loss_dict['total'].item():.6f}")
    print(f"  Style SWD: {loss_dict['style_swd'].item():.6f}")
    print(f"  Content SSM: {loss_dict['content_ssm'].item():.6f}")
    
    print("\nTesting auxiliary losses...")
    v = torch.randn(B, C, H, W, device=device)
    vel_reg = VelocityRegularizationLoss().to(device)
    div_reg = DivergenceRegularizationLoss().to(device)
    print(f"  Velocity Reg: {vel_reg(v).item():.6f}")
    print(f"  Divergence Reg: {div_reg(v).item():.6f}")
    
    print("\n✓ All loss functions tested successfully!")
