"""
LGT Geometric Free Energy Loss Functions - Clean Version

Core principle: MSE maintains structure + brightness, SWD controls texture.
Removed: CosineSSMLoss, NeighborhoodMatchingLoss (deprecated Content Losses)

Loss functions:
1. PatchSlicedWassersteinLoss: Texture matching (with brightness normalization)
2. MultiScaleSWDLoss: Multi-scale texture matching
3. TrajectoryMSELoss: Structure + brightness supervision
4. GeometricFreeEnergyLoss: Unified wrapper for SWD only

All losses operate in FP32 for numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityRegularizationLoss(nn.Module):
    """
    Velocity Magnitude Regularization for Flow Matching.
    
    Purpose: Prevent model from learning excessively large velocity vectors,
    which causes brightness explosion and instability during inference (especially with CFG).
    
    Physics: Flow Matching assumes the velocity field v(x,t,c) is bounded in magnitude.
    If training data has small variance (e.g., VAE latents with std=0.18) but noise has std=1.0,
    the model learns to output huge velocities to bridge this gap. This causes:
    - Large gradient magnitudes → training instability
    - CFG amplifies these huge vectors → color saturation/clipping
    - Batch-wise brightness variance due to different energy scales
    
    Solution: Regularize L2 norm of velocity vectors during training.
    Loss = weight * mean(v_pred²) across spatial and batch dimensions.
    
    Trade-off: Too large regularization → slower learning, washed-out images.
    Recommended: weight ∈ [0.05, 0.2] for typical latent-space diffusion.
    """
    
    def __init__(self, weight=0.1):
        """
        Args:
            weight: Regularization strength (default 0.1 = 5-10% of total loss)
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, v_pred):
        """
        Compute velocity regularization loss.
        
        Args:
            v_pred: [B, C, H, W] Predicted velocity field from model
        
        Returns:
            loss: scalar regularization penalty
        """
        # L2 norm squared: sum across channel and spatial dims, mean over batch
        # This penalizes all velocity pixels equally regardless of position/channel
        return self.weight * torch.mean(v_pred ** 2)


class PatchSlicedWassersteinLoss(nn.Module):
    """
    Patch-Sliced Wasserstein Distance for style distribution matching.
    
    Update: Added 'normalize_patch' to decouple texture from brightness.
    When normalize_patch='mean', we compute SWD on contrast (patches - patch_mean),
    removing brightness information and preserving only texture details.
    This solves the "brightness explosion" problem by letting MSE control luminance
    while SWD controls high-frequency texture details.
    
    Pipeline:
    1. Unfold: Extract K×K patches → [B, N_patches, C*K*K]
    2. Normalize: Optionally subtract patch mean (brightness invariance)
    3. Sample: Random sample max_samples patches for memory efficiency
    4. Project: Radon transform with num_projections random projections
    5. Sort: Quantile alignment
    6. Metric: MSE between sorted distributions
    
    Critical: All ops in FP32 for stability.
    """
    
    def __init__(
        self,
        patch_size=3,
        num_projections=64,
        max_samples=4096,
        use_fp32=True,
        normalize_patch='mean'  # 🔥 New: 'mean' to remove brightness, 'none' to keep
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_projections = num_projections
        self.max_samples = max_samples
        self.use_fp32 = use_fp32
        self.normalize_patch = normalize_patch
    
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
            
            # Step 1: Unfold to extract K×K patches
            # Output: [B, C*K*K, N_patches] where N_patches = H*W for k=3, pad=1
            x_pred_patches = F.unfold(x_pred, kernel_size=self.patch_size, padding=self.patch_size//2)
            x_style_patches = F.unfold(x_style, kernel_size=self.patch_size, padding=self.patch_size//2)
            
            # Reshape to [B, N_patches, C*K*K]
            # Memory optimization: unfold output may have non-contiguous stride layout
            # transpose changes only strides (no copy), but reshape after requires contiguous memory
            x_pred_patches = x_pred_patches.transpose(1, 2).contiguous()  # [B, N_patches, C*K*K]
            x_style_patches = x_style_patches.transpose(1, 2).contiguous()  # [B, N_patches, C*K*K]
            
            # ==========================================
            # 🔥 Patch Brightness Normalization
            # ==========================================
            # Remove DC component (patch mean) to decouple texture from brightness.
            # Physics: "I care about texture contrast, not absolute luminance."
            # This lets MSE control brightness while SWD controls high-frequency details.
            if self.normalize_patch == 'mean':
                # Compute mean across feature dimension (all channels and spatial positions in patch)
                patch_mean = x_pred_patches.mean(dim=2, keepdim=True)  # [B, N_patches, 1]
                x_pred_patches = x_pred_patches - patch_mean
                
                patch_mean = x_style_patches.mean(dim=2, keepdim=True)  # [B, N_patches, 1]
                x_style_patches = x_style_patches - patch_mean
            
            N_patches = x_pred_patches.shape[1]
            feature_dim = x_pred_patches.shape[2]  # C*K*K (e.g., 36 for 4ch×3×3)
            
            # Step 2: Sample patches for memory efficiency
            # Flatten batch and patches: [B*N_patches, C*K*K]
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
            # Generate random projection matrix: [feature_dim, num_projections]
            theta = torch.randn(
                feature_dim,
                self.num_projections,
                device=x_pred.device,
                dtype=x_pred.dtype
            )
            theta = theta / theta.norm(dim=0, keepdim=True)  # Normalize columns
            
            # Project: [N_sampled, feature_dim] @ [feature_dim, num_projections] → [N_sampled, num_projections]
            # Contiguous memory layout ensures optimal gemm performance
            proj_pred = x_pred_sampled @ theta
            proj_style = x_style_sampled @ theta
            
            # Step 4: Sort for quantile alignment
            proj_pred_sorted, _ = torch.sort(proj_pred, dim=0)
            proj_style_sorted, _ = torch.sort(proj_style, dim=0)
            
            # Step 5: Compute SWD as MSE between sorted distributions
            loss = F.mse_loss(proj_pred_sorted, proj_style_sorted)
        
        return loss




class MultiScaleSWDLoss(nn.Module):
    """
    Multi-Scale Sliced Wasserstein Distance for unified annealing/quenching.
    
    🔥 Update: Automatically enables brightness normalization for scales > 1.
    This ensures high-frequency texture matching without brightness explosion.
    
    Physics motivation:
    - Unifies optimization objective for both phase transitions
    - Matching photo's high-freq distribution → auto-sharpens (quenching)
    - Matching painting's low-freq distribution → auto-smooths (annealing)
    - No need for w_freq or asymmetric hard constraints
    
    Architecture:
    - Scale 1×1 (Pixel): Color palette distribution (no normalization - patch_size=1 would collapse)
    - Scale 3×3 (Texture): High-frequency details (mean-normalized for brightness invariance)
    - Scale 7×7 (Structure): Local structural patterns (mean-normalized for texture focus)
    
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
                use_fp32=use_fp32,
                # 🔥 Key: Only normalize for scale > 1
                # Scale 1 (1×1 patch) normalization would yield all zeros
                # Scale 3, 7 get brightness normalization for texture focus
                normalize_patch='mean' if scale > 1 else 'none'
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


class TrajectoryMSELoss(nn.Module):
    """
    Trajectory Matching Loss (Flow Matching Objective).
    
    Standard Flow Matching objective: || v_theta(x_t) - (x_1 - x_0) ||^2
    
    Physics Interpretation:
    Acts as a 'Teacher Forcing' signal. It forces the vector field to 
    point precisely towards the target geometry (x_1), locking the 
    phase spectrum (edges/high-freqs) that SWD often misses.
    
    This is the 'Supervisor' component in Hybrid Dynamics, ensuring
    high-frequency clarity and geometric fidelity (plus source brightness).
    """
    def __init__(self, weight=5.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, v_pred, x_0, x_1):
        """
        Args:
            v_pred: [B, C, H, W] Predicted velocity field from model
            x_0:    [B, C, H, W] Starting noise
            x_1:    [B, C, H, W] Target data (deformed source or style target)
        
        Returns:
            loss: scalar MSE trajectory loss
        """
        # Optimal transport path has constant velocity: v = x_1 - x_0
        v_target = x_1 - x_0
        
        # Simple MSE is extremely efficient and memory-friendly
        # Backward pass is O(N) complexity
        return self.weight * F.mse_loss(v_pred, v_target)


class GeometricFreeEnergyLoss(nn.Module):
    """
    Simplified Energy Loss - Style (SWD) Only.
    
    Clean version: MSE handles structure + brightness, SWD handles texture only.
    Content Loss (SSM/NML) completely removed.
    
    Physics principle:
    - MSE: Supervisor (high-freq clarity, source brightness preservation)
    - SWD: Artist (texture/brushstroke matching)
    
    No overlap, no conflicts.
    """
    
    def __init__(
        self,
        w_style=60.0,
        swd_scales=[1, 3, 7],
        swd_scale_weights=[1.0, 1.0, 1.0],
        num_projections=64,
        max_samples=4096,
        **kwargs  # Accept but ignore deprecated parameters for backward compatibility
    ):
        super().__init__()
        
        self.w_style = w_style
        
        # Style potential: Multi-Scale SWD (mean-normalized)
        self.swd_loss = MultiScaleSWDLoss(
            scales=swd_scales,
            scale_weights=swd_scale_weights,
            num_projections=num_projections,
            max_samples=max_samples,
            use_fp32=True
        )
    
    def forward(self, x_pred, x_style):
        """
        Args:
            x_pred: [B, 4, H, W] predicted terminal state
            x_style: [B, 4, H, W] style reference
        
        Returns:
            loss_dict: dictionary with total loss and components
        """
        # Compute style potential only (texture matching)
        style_potential, swd_dict = self.swd_loss(x_pred, x_style)
        
        result = {
            'total': self.w_style * style_potential,
            'style_swd': style_potential
        }
        
        # Add per-scale losses
        result.update(swd_dict)
        
        return result

if __name__ == "__main__":
    # Test loss functions (clean version)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    B, C, H, W = 4, 4, 32, 32
    x_pred = torch.randn(B, C, H, W, device=device)
    x_style = torch.randn(B, C, H, W, device=device)
    x_0 = torch.randn(B, C, H, W, device=device)
    x_1 = torch.randn(B, C, H, W, device=device)
    v_pred = torch.randn(B, C, H, W, device=device)
    
    print("Testing Patch-SWD Loss (mean-normalized)...")
    swd_loss = PatchSlicedWassersteinLoss(normalize_patch='mean').to(device)
    swd_value = swd_loss(x_pred, x_style)
    print(f"  SWD Loss: {swd_value.item():.6f}")
    
    print("\nTesting Multi-Scale SWD Loss...")
    multi_swd = MultiScaleSWDLoss(scales=[2, 4, 8], scale_weights=[2.0, 5.0, 5.0]).to(device)
    multi_loss, scale_dict = multi_swd(x_pred, x_style)
    print(f"  Multi-Scale SWD Total: {multi_loss.item():.6f}")
    for scale, loss_val in scale_dict.items():
        if isinstance(loss_val, torch.Tensor):
            print(f"    {scale}: {loss_val.item():.6f}")
    
    print("\nTesting Geometric Free Energy Loss (SWD only)...")
    energy_loss = GeometricFreeEnergyLoss(w_style=60.0).to(device)
    loss_dict = energy_loss(x_pred, x_style)
    print(f"  Total Energy: {loss_dict['total'].item():.6f}")
    print(f"  Style SWD: {loss_dict['style_swd'].item():.6f}")
    
    print("\nTesting Trajectory MSE Loss...")
    mse_loss = TrajectoryMSELoss(weight=5.0).to(device)
    mse_value = mse_loss(v_pred, x_0, x_1)
    print(f"  MSE Loss: {mse_value.item():.6f}")
    
    print("\n✓ All loss functions tested successfully!")
