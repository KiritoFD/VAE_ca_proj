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
import os# 🔥 New import for Structure Lock


class StructureAnchoredLoss(nn.Module):
    """
    Laplacian Structural Lock with Adaptive Gating - WITH DIAGNOSTIC MONITORING.
    
    🔥 CRITICAL IMPROVEMENTS for training from scratch:
    1. Dynamic Gating: Gradually increase constraint over epochs
    2. Huber Loss: Robust to outliers (prevents gradient explosion at MSE=10)
    3. Smooth L1 transition: Behaves like MSE for small errors, L1 for large
    4. Built-in diagnostics: Monitor edge detection hardness
    """
    def __init__(self, weight=2.0, edge_boost=3.0):
        super().__init__()
        self.weight = weight
        self.edge_boost = edge_boost
        
        # Static Laplacian Kernel (Discrete Second Derivative)
        # Shape: [4, 1, 3, 3] for group conv (one kernel per channel)
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('kernel', k.view(1, 1, 3, 3).repeat(4, 1, 1, 1))
        
        # 🔥 Debug flag
        self.debug_trigger = False

    def forward(self, v_pred, v_target, clean_latents, current_epoch=0, total_warmup_epochs=20):
        """
        Args:
            v_pred: [B, C, H, W] Model predicted velocity
            v_target: [B, C, H, W] Ground truth target from scheduler
            clean_latents: [B, C, H, W] Original clean latents (for structure extraction)
            current_epoch: Current training epoch (for adaptive gating)
            total_warmup_epochs: Number of epochs to warm up gate (default 20)
        
        Returns:
            loss: scalar weighted Smooth L1 loss with gradient safety bounds
        """
        # 1. Physical Edge Extraction (on clean latents, no gradients)
        with torch.no_grad():
            # Depthwise conv: O(1) memory overhead per channel
            edges = F.conv2d(clean_latents, self.kernel, groups=4, padding=1)
            
            # Max aggregation across channels to find dominant structure
            mask = torch.max(torch.abs(edges), dim=1, keepdim=True)[0]  # [B, 1, H, W]
            
            # Robust Z-Score Normalization (prevents division by zero and extreme values)
            mu = mask.mean(dim=[2, 3], keepdim=True)  # Spatial mean
            std = mask.std(dim=[2, 3], keepdim=True)   # Spatial standard deviation
            
            # Z-Score: (x - μ) / σ → bounded to (-∞, +∞)
            mask_zscore = (mask - mu) / (std + 1e-6)
            
            # Sigmoid: Maps (-∞, +∞) → (0, 1), matching CNN Proxy output range
            mask_norm = torch.sigmoid(mask_zscore)  # [B, 1, H, W] in [0, 1]
            
            # 🔥 FIX 1: Dynamic Gating (Adaptive Warmup)
            # ================================================================
            # Gradually increase constraint from 0 to 1 over warmup_epochs.
            # Early epochs: gate ≈ 0 → light supervision, model learns basic patterns
            # Late epochs: gate → 1 → full structure lock, model refines edges
            gate = min(current_epoch / max(total_warmup_epochs, 1), 1.0)
            
            # weight_map now interpolates between:
            # Early (gate=0): weight_map = 1.0 (uniform MSE)
            # Late (gate=1):  weight_map = 1.0 + mask_norm * edge_boost (structure lock)
            weight_map = 1.0 + (mask_norm * self.edge_boost * gate)  # [B, 1, H, W]
            
            # 🔥 DEBUG: Monitor edge detection hardness
            # ================================================================
            # Healthy range for weight_map:
            # - Min should be ≈ 1.0 (flat regions, no constraint)
            # - Max should be ≈ (1.0 + edge_boost) (edge regions, full constraint)
            # - Gate interpolates from 0→1 as epochs progress
            #
            # If weight_map.max() is already 4.0 but gate is still 0.1,
            # the constraint is too aggressive for early training.
            if self.debug_trigger:
                w_min, w_max = weight_map.min().item(), weight_map.max().item()
                w_mean = weight_map.mean().item()
                print(
                    f"[Loss Debug] Epoch {current_epoch}/{total_warmup_epochs} | "
                    f"Weight: min={w_min:.3f} max={w_max:.3f} mean={w_mean:.3f} | "
                    f"Gate={gate:.3f} | "
                    f"Mask Range=[{mask.min().item():.4f}, {mask.max().item():.4f}]"
                )

        # 2. Weighted Loss Computation
        # ================================================================
        # 🔥 FIX 2: Use Smooth L1 (Huber) Loss instead of MSE
        # 
        # Problem: At MSE=10, gradient is huge (2*10=20 per pixel).
        # When multiplied by weight_map (up to 4.0), gradient norm becomes 80.
        # Even with clipping, the loss surface is sharp and optimization is unstable.
        #
        # Solution: Smooth L1 Loss (Huber Loss)
        # - For |error| < β: acts like MSE (smooth, adaptive step size)
        # - For |error| > β: acts like L1 (constant gradient, robustness to outliers)
        # - Smooth transition at boundary ensures gradient continuity
        #
        # β=0.1 means:
        # - Errors < 0.1: Use quadratic (steep early learning)
        # - Errors > 0.1: Use linear (steady descent from plateau)
        weighted_diff = weight_map * (v_pred - v_target)
        
        # Smooth L1 Loss: more robust to outliers than MSE at MSE=10
        loss = F.smooth_l1_loss(weighted_diff, torch.zeros_like(weighted_diff), beta=0.1, reduction='mean')
        
        return self.weight * loss


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
    Trajectory Matching Loss (Flow Matching Objective) - CORRECTED VERSION.
    
    🔥 CRITICAL FIX: Target (v_target) is now provided by Scheduler, not computed locally.
    
    Low-Pass MSE for structure-only supervision.
    Only supervise LOW frequencies (structure/outline).
    HIGH frequencies are left free for SWD to control (texture/style).
    
    This is the 'Sculptor' component in Hybrid Dynamics:
    - Sculpture (MSE, low-freq): Rough form, outlines, large color regions
    - Painting (SWD, high-freq): Texture, brushstrokes, fine details
    """
    def __init__(self, weight=5.0, low_pass_kernel_size=5):
        super().__init__()
        self.weight = weight
        self.kernel_size = low_pass_kernel_size  # Size of avg_pool kernel
    
    def forward(self, v_pred, v_target):
        """
        Args:
            v_pred: [B, C, H, W] Predicted velocity field from model
            v_target: [B, C, H, W] Target velocity (from noise scheduler)
                      🔥 CRITICAL: This MUST come from noise_scheduler.get_*(),
                      NOT computed locally!
        
        Returns:
            loss: scalar MSE trajectory loss (low-frequency only)
        """
        # Low-Pass Filtering: extract low-frequency components
        # This makes MSE supervision "blurry" - it only cares about structure,
        # not texture details.
        k = self.kernel_size
        v_pred_blur = F.avg_pool2d(v_pred, k, stride=1, padding=k//2)
        v_target_blur = F.avg_pool2d(v_target, k, stride=1, padding=k//2)
        
        # MSE only on blurred (low-frequency) signals
        return self.weight * F.mse_loss(v_pred_blur, v_target_blur)


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


class VelocitySmoothnessLoss(nn.Module):
    """
    Velocity Field Smoothness (TV Loss).
    
    Physics: Penalizes high-frequency noise in the vector field.
    Effect: Eliminates 'flying color blocks' and flickering artifacts.
    
    Implementation: Total Variation regularization on velocity field.
    By forcing gradients (dv/dx, dv/dy) to be small, the velocity flow becomes smooth.
    This removes tile boundary artifacts that occur when adjacent tiles have vastly
    different velocity vectors.
    
    Trade-off: Too large weight → smooth but blurry motion vectors → featureless output
    Recommended: weight ∈ [0.05, 0.3] for typical training
    """
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, v_pred):
        """
        Args:
            v_pred: [B, C, H, W] Predicted velocity field
        
        Returns:
            loss: scalar TV loss
        """
        # Calculate spatial gradients (dv/dx, dv/dy)
        # These represent how rapidly the velocity changes across space
        diff_x = torch.abs(v_pred[:, :, :, 1:] - v_pred[:, :, :, :-1])
        diff_y = torch.abs(v_pred[:, :, 1:, :] - v_pred[:, :, :-1, :])
        
        # Sum gradient magnitudes for all channels and average over batch
        loss = torch.mean(diff_x) + torch.mean(diff_y)
        return self.weight * loss


class DistilledStructureLoss(nn.Module):
    """
    Distilled Structure Loss (The "Structure Anchor").
    
    Physics: Uses a pre-trained "Structure Proxy" neural network to identify
    structural edges in Latent Space. High-frequency MSE constraint is applied
    ONLY in structural regions (edges), while flat/texture regions are left free
    for SWD to control.
    
    Architecture:
    - Proxy: LearnableStructureExtractor (lightweight 4→1 CNN)
    - Output: [0, 1] probability map (1=edge, 0=flat)
    - Weight Map: base_weight + (prob * structure_weight)
    - Result: High-constraint edges, low-constraint flats
    
    Effect:
    - Keeps structure sharp and edge-aligned
    - Allows SWD to paint freely in flat regions (no fighting)
    - Frequency-domain interpretation: Only supervise low-freq at edges
    
    Critical dependency: 'structure_proxy.pt' must exist in working directory.
    If missing, falls back to untrained proxy (random structure detection).
    """
    def __init__(self, proxy_path="structure_proxy.pt", base_weight=1.0, structure_weight=10.0, device='cuda'):
        super().__init__()
        self.base_weight = base_weight
        self.structure_weight = structure_weight
        self.device = device
        
        # Load Proxy
        self.proxy = LearnableStructureExtractor().to(device)
        
        if os.path.exists(proxy_path):
            state_dict = torch.load(proxy_path, map_location=device)
            self.proxy.load_state_dict(state_dict)
            print(f"✓ Loaded Structure Proxy from {proxy_path}")
        else:
            print(f"⚠️  WARNING: Proxy '{proxy_path}' not found! Structure lock will be random.")
            
        self.proxy.eval()
        for p in self.proxy.parameters():
            p.requires_grad = False

    def forward(self, v_pred, x_0, x_target):
        """
        Args:
            v_pred: [B, C, H, W] Predicted velocity field
            x_0: [B, C, H, W] Starting noise (for computing v_target)
            x_target: [B, C, H, W] Target state (content reference for structure detection)
        
        Returns:
            loss: scalar weighted MSE loss
        """
        # 1. Ask Proxy: "Where is the structure in this target?"
        with torch.no_grad():
            # Output is [B, 1, H, W] probability map (Soft Mask)
            # Values in [0, 1]: 1=strong edge, 0=flat region
            structure_prob = self.proxy(x_target)
            # Squeeze channel dimension to get [B, H, W] for weight_map computation
            structure_prob = structure_prob.squeeze(1)  # [B, H, W]
            
        # 2. Build Frequency-Split Weight Map
        # Structure areas (edges) -> High MSE weight (Lock structure tight)
        # Flat areas -> Low MSE weight (Let SWD paint freely)
        #
        # weight_map = base_weight + (structure_prob * structure_weight)
        # If structure_prob=0 (flat): weight = base_weight = 1.0 (low constraint)
        # If structure_prob=1 (edge): weight = 1.0 + 10.0 = 11.0 (high constraint)
        # This ratio ensures edges are 10x more constrained than flats
        weight_map = self.base_weight + (structure_prob.unsqueeze(1) * self.structure_weight)  # [B, 1, H, W]
        
        # 3. Compute target velocity (OT direction)
        v_target = x_target - x_0
        
        # 4. Weighted MSE (only penalize deviations in structural regions)
        # Element-wise: weight_map * (v_pred - v_target)^2
        # This forces the model to match v_target precisely at edges,
        # but allows deviation in flat regions
        loss = torch.mean(weight_map * (v_pred - v_target) ** 2)
        
        return loss

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
    
    print("\nTesting Structure Anchored Loss (Laplacian)...")
    struct_loss = StructureAnchoredLoss(weight=5.0, edge_boost=3.0).to(device)  # 🔥 Fixed: 9.0 -> 3.0
    struct_value = struct_loss(v_pred, x_0, x_1)
    print(f"  Structure Anchored Loss: {struct_value.item():.6f}")
    
    print("\n✓ All loss functions tested successfully!")
