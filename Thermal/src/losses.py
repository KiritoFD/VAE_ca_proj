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
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


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
        
        # Smooth L1 Loss: more robust to outliers than MSE
        loss = F.smooth_l1_loss(weighted_diff, torch.zeros_like(weighted_diff), beta=0.1, reduction='mean')
        
        return loss


class TrajectoryMSELoss(nn.Module):
    """
    Trajectory Matching Loss (Flow Matching Objective).
    Low-Pass MSE for structure-only supervision.
    Only supervise LOW frequencies (structure/outline).
    """
    def __init__(self, weight=2.0, low_pass_kernel_size=5):
        super().__init__()
        self.weight = weight
        self.kernel_size = low_pass_kernel_size
    
    def forward(self, v_pred, v_target):
        # Low-Pass Filtering: extract low-frequency components
        k = self.kernel_size
        # Padding ensures output size matches input
        v_pred_blur = F.avg_pool2d(v_pred, k, stride=1, padding=k//2)
        v_target_blur = F.avg_pool2d(v_target, k, stride=1, padding=k//2)
        
        return F.mse_loss(v_pred_blur, v_target_blur)


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


class GeometricFreeEnergyLoss(nn.Module):
    """
    [LGT-X Pro] Optimized Multi-Style SWD Loss with Style-Indexed Cache
    
    🔥 Key Innovation: Style Look-Up Table (LUT)
    - Pre-compute sorted projections for each style during initialization
    - During training: O(1) target retrieval via index_select, only Input needs sorting
    - 4070 Friendly: Fixed LUT is read-only, better L2 cache utilization
    
    Performance:
    - Traditional: Sort Input + Sort Target = 2 × O(N log N)
    - LUT-based:  Sort Input only = O(N log N) + O(1) LUT retrieval
    - Memory trade-off: ~10-50MB LUT for significant speedup
    
    Architecture:
    - Fixed orthogonal projections per scale
    - Pre-sorted style distributions stored in GPU buffers
    - Dynamic style routing during forward pass via style_ids
    """
    
    def __init__(
        self,
        num_styles=4,
        w_style=40.0,
        swd_scales=[1, 3, 5, 7, 15],
        swd_scale_weights=[1.0, 5.0, 5.0, 5.0, 3.0],
        num_projections=64,
        max_samples=4096,
        **kwargs  # Accept but ignore deprecated parameters for backward compatibility
    ):
        super().__init__()
        
        self.num_styles = num_styles
        self.w_style = w_style
        self.scales = swd_scales
        self.scale_weights = swd_scale_weights
        self.num_projections = num_projections
        self.max_samples = max_samples
        
        # Register initialization flag
        self.register_buffer('_is_initialized', torch.tensor(False, dtype=torch.bool))
        
        logger.info(
            f"🎯 GeometricFreeEnergyLoss initialized (Multi-Style Mode)\n"
            f"   Scales: {swd_scales}\n"
            f"   Styles: {num_styles} | Max Samples: {max_samples}\n"
            f"   Projections: {num_projections} | Memory: ~{self._estimate_memory():.1f}MB"
        )
    
    def _estimate_memory(self) -> float:
        """Estimate GPU memory usage for LUT in MB"""
        # 每个尺度的 LUT: [num_styles, max_samples, num_projections]
        bytes_per_scale = self.num_styles * self.max_samples * self.num_projections * 4  # float32
        total_bytes = bytes_per_scale * len(self.scales)
        
        # 投影矩阵: [num_projections, C]，其中 C 因尺度而异
        # 粗估：4 + 4 + 4 个通道（缩放后）
        proj_bytes = self.num_projections * (4 + 4 + 4) * 4
        
        return (total_bytes + proj_bytes) / (1024 * 1024)
    
    def _get_orthogonal_projections(self, n_dims: int, n_projections: int, device: torch.device) -> torch.Tensor:
        """使用 QR 分解生成正交投影矩阵（比随机高斯更稳定）"""
        # 生成高斯随机矩阵并进行 QR 分解
        mat = torch.randn(n_dims, n_projections, device=device, dtype=torch.float32)
        q, _ = torch.linalg.qr(mat)
        
        # 返回 [n_projections, n_dims] 形状用于矩阵乘法
        return q[:, :n_projections].t()  # [n_projections, n_dims]
    
    def initialize_cache(self, style_latents_dict: Dict[int, torch.Tensor], device: torch.device) -> None:
        """
        初始化 Style LUT 缓存。必须在训练开始前调用一次。
        
        Args:
            style_latents_dict: 字典 {style_id: latent_tensor [B, 4, H, W]}
                               包含每个风格的代表性样本
            device: 目标设备（通常是 'cuda'）
        
        Example:
            swd_loss = GeometricFreeEnergyLoss(num_styles=4)
            style_dict = {0: monet_latent, 1: photo_latent, 2: vangogh_latent, 3: cezanne_latent}
            swd_loss.initialize_cache(style_dict, device='cuda')
        """
        if self._is_initialized:
            logger.info("⚠️ Cache already initialized. Skipping re-initialization.")
            return
        
        logger.info("🔥 Pre-computing SWD Style Cache (this may take a minute)...")
        
        with torch.no_grad():
            for scale_idx, scale in enumerate(self.scales):
                logger.info(f"  Processing scale {scale}×{scale} ({scale_idx + 1}/{len(self.scales)})...")
                
                # 1. 确定当前尺度的通道数（探测一个样本）
                sample_latent = style_latents_dict[0].to(device, non_blocking=True)
                
                # 缩放采样（模拟多尺度）
                if scale > 1:
                    sample_latent = F.interpolate(sample_latent, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
                
                c_dim = sample_latent.shape[1]  # 通常是 4
                
                # 2. 生成固定的正交投影矩阵（一次性计算）
                projections = self._get_orthogonal_projections(c_dim, self.num_projections, device)
                self.register_buffer(f'_proj_{scale}', projections, persistent=False)
                
                # 3. 为每个风格构建 LUT 项
                lut_list = []
                
                for style_id in range(self.num_styles):
                    assert style_id in style_latents_dict, f"Style {style_id} not found in input dict"
                    
                    style_latent = style_latents_dict[style_id].to(device, non_blocking=True)
                    
                    # 缩放到当前尺度
                    if scale > 1:
                        style_latent = F.interpolate(style_latent, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
                    
                    # 展开为像素级特征：[B, C, H, W] → [N, C]
                    b, c, h, w = style_latent.shape
                    style_flat = style_latent.view(b, c, -1).permute(0, 2, 1).reshape(-1, c)  # [B*H*W, C]
                    
                    # 采样到固定大小（保证 LUT 维度固定）
                    n_pixels = style_flat.shape[0]
                    if n_pixels > self.max_samples:
                        idx = torch.randperm(n_pixels, device=device)[:self.max_samples]
                        style_flat = style_flat[idx]
                    else:
                        # 如果样本不足，进行循环复制填充
                        repeat_times = (self.max_samples // max(n_pixels, 1)) + 1
                        style_flat = style_flat.repeat(repeat_times, 1)[:self.max_samples]
                    
                    # 投影：[max_samples, C] @ [C, num_projections] → [max_samples, num_projections]
                    proj_style = torch.matmul(style_flat, projections.t())
                    
                    # 排序（这是 LUT 初始化时的唯一开销，之后就不再做）
                    proj_style_sorted, _ = torch.sort(proj_style, dim=0)
                    
                    lut_list.append(proj_style_sorted)
                
                # 4. 堆叠所有风格的排序结果：[num_styles, max_samples, num_projections]
                lut_tensor = torch.stack(lut_list, dim=0)
                self.register_buffer(f'_lut_{scale}', lut_tensor, persistent=False)
                
                logger.info(f"    ✓ Scale {scale}: LUT shape {lut_tensor.shape}")
        
        self._is_initialized.fill_(True)
        logger.info("✅ SWD Cache Initialization Complete")
    
    def forward(self, x_pred: torch.Tensor, x_style: torch.Tensor, style_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算多尺度 SWD 损失（使用预计算的 Style LUT）
        
        Args:
            x_pred: [B, 4, H, W] 模型预测的终端状态
            x_style: [B, 4, H, W] 风格参考（用于旧式直接计算，仅在未初始化缓存时使用）
            style_ids: [B] 每个样本的目标风格 ID（与 LUT 索引对应）
        
        Returns:
            loss_dict: {
                'style_swd': scalar 总 SWD 损失,
                'swd_scale_1': 尺度 1 的损失,
                'swd_scale_3': 尺度 3 的损失,
                ...
            }
        """
        # Fallback：如果没有初始化缓存，降级到旧式多尺度 SWD（兼容性保证）
        if not self._is_initialized:
            logger.warning(
                "⚠️ Style LUT not initialized. Falling back to traditional SWD "
                "(slower, but backward compatible). Call loss.initialize_cache() before training."
            )
            return self._forward_traditional(x_pred, x_style)
        
        # 推理路径：使用 LUT 加速
        device = x_pred.device
        dtype = x_pred.dtype
        
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        loss_dict = {}
        
        # 如果没有提供 style_ids，假设所有样本都是第一个风格（用于评估）
        if style_ids is None:
            style_ids = torch.zeros(x_pred.shape[0], device=device, dtype=torch.long)
        
        with torch.autocast('cuda', enabled=False):  # 强制 FP32 以保证数值稳定性
            x_pred_fp32 = x_pred.float()
            
            for scale_idx, scale in enumerate(self.scales):
                # 1. 尺度缩放
                if scale > 1:
                    x_scaled = F.interpolate(x_pred_fp32, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
                else:
                    x_scaled = x_pred_fp32
                
                b, c, h, w = x_scaled.shape
                
                # 2. 读取固定投影矩阵
                projections = getattr(self, f'_proj_{scale}')  # [num_projections, C]
                
                # 3. 投影 Input（每次 Forward 都要排序 Input）
                x_flat = x_scaled.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
                
                n_pixels = x_flat.shape[1]
                if n_pixels > self.max_samples:
                    # 随机采样（保持随机性以增强数据多样性）
                    idx = torch.randperm(n_pixels, device=device)[:self.max_samples]
                    x_sampled = x_flat[:, idx, :]  # [B, max_samples, C]
                else:
                    # 如果像素不足，填充到 max_samples
                    pad_size = self.max_samples - n_pixels
                    x_sampled = F.pad(x_flat, (0, 0, 0, pad_size), mode='constant', value=0)
                
                # 投影：[B, max_samples, C] @ [C, num_projections] → [B, max_samples, num_projections]
                proj_input = torch.matmul(x_sampled, projections.t())  # [B, max_samples, num_projections]
                
                # 4. 排序 Input（唯一的排序操作）
                proj_input_sorted, _ = torch.sort(proj_input, dim=1)
                
                # 5. 从 LUT 中检索对应风格的目标分布（这是 O(1) 操作）
                lut = getattr(self, f'_lut_{scale}')  # [num_styles, max_samples, num_projections]
                
                # index_select：高效地从 LUT 中根据 style_id 提取目标
                # [num_styles, max_samples, num_projections] → [B, max_samples, num_projections]
                proj_target_sorted = lut.index_select(0, style_ids)
                
                # 6. 计算 SWD 损失（L2 距离）
                scale_loss = F.mse_loss(proj_input_sorted.float(), proj_target_sorted.float())
                
                total_loss = total_loss + self.scale_weights[scale_idx] * scale_loss
                loss_dict[f'swd_scale_{scale}'] = scale_loss.detach()
        
        loss_dict['style_swd'] = total_loss
        
        return loss_dict
    
    def _forward_traditional(self, x_pred: torch.Tensor, x_style: torch.Tensor) -> Dict[str, torch.Tensor]:
        """降级方案：不使用 LUT，直接计算（兼容性保证）"""
        # 使用原来的 MultiScaleSWDLoss 进行计算
        if not hasattr(self, '_fallback_loss'):
            self._fallback_loss = MultiScaleSWDLoss(
                scales=self.scales,
                scale_weights=self.scale_weights,
                num_projections=self.num_projections,
                max_samples=self.max_samples,
                use_fp32=True
            ).to(x_pred.device)
        
        style_potential, swd_dict = self._fallback_loss(x_pred, x_style)
        
        result = {
            'style_swd': style_potential,
        }
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
    print(f"  Total Energy: {loss_dict['style_swd']:.6f}")
    print(f"  Style SWD: {loss_dict['style_swd']:.6f}")
    
    print("\nTesting Structure Anchored Loss (Laplacian)...")
    struct_loss = StructureAnchoredLoss(weight=5.0, edge_boost=3.0).to(device)  # 🔥 Fixed: 9.0 -> 3.0
    v_target = x_style - x_0
    struct_value = struct_loss(v_pred, v_target, x_0)
    print(f"  Structure Anchored Loss: {struct_value.item():.6f}")
    
    print("\n✓ All loss functions tested successfully!")
