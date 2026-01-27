import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def elastic_deform(x: torch.Tensor, alpha: float = 10.0, sigma: int = 3) -> torch.Tensor:
    """
    Differentiable elastic deformation for latent tensors.

    Args:
        x: Tensor of shape [B, C, H, W]
        alpha: Displacement magnitude in pixels
        sigma: Smoothness parameter (controls blur iterations)

    Returns:
        Deformed tensor of the same shape as input.
    """
    if x.ndim != 4:
        raise ValueError(f"elastic_deform expects [B, C, H, W], got {tuple(x.shape)}")

    b, _, h, w = x.shape
    device = x.device

    dx = torch.rand(b, 1, h, w, device=device) * 2 - 1
    dy = torch.rand(b, 1, h, w, device=device) * 2 - 1

    # Approximate Gaussian smoothing via repeated avg pooling
    for _ in range(max(sigma // 1, 1)):
        dx = F.avg_pool2d(dx, kernel_size=3, stride=1, padding=1)
        dy = F.avg_pool2d(dy, kernel_size=3, stride=1, padding=1)

    flow = torch.cat([dx, dy], dim=1) * alpha

    # Base grid in pixel coordinates
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij'
    )
    grid = torch.stack([x_grid, y_grid], dim=-1).float()  # [H, W, 2]
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, H, W, 2]

    # Normalize grid and flow to [-1, 1]
    grid_norm = grid.clone()
    grid_norm[..., 0] = 2.0 * grid_norm[..., 0] / (w - 1) - 1.0
    grid_norm[..., 1] = 2.0 * grid_norm[..., 1] / (h - 1) - 1.0

    flow_norm = flow.permute(0, 2, 3, 1)
    flow_norm[..., 0] = 2.0 * flow_norm[..., 0] / (w - 1)
    flow_norm[..., 1] = 2.0 * flow_norm[..., 1] / (h - 1)

    sample_grid = grid_norm + flow_norm
    return F.grid_sample(x, sample_grid, mode='bilinear', padding_mode='reflection', align_corners=True)


class LatentDataset(Dataset):
    """
    Balanced in-memory latent dataset with optional elastic deformation.

    - Loads all latents into CPU memory.
    - Samples styles uniformly regardless of per-style image count.
    - Can precompute an elastic-deformed view in __getitem__ to leverage dataloader workers.
    """

    def __init__(
        self,
        data_root: str,
        num_styles: int,
        style_subdirs: Optional[List[str]] = None,
        apply_elastic: bool = False,
        elastic_alpha: float = 1.0,
        elastic_sigma: int = 3,
        rescale_raw: bool = True,
        pin_memory: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.num_styles = num_styles
        self.style_subdirs = style_subdirs or [f"style{i}" for i in range(num_styles)]
        self.apply_elastic = apply_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.rescale_raw = rescale_raw

        self.style_indices: Dict[int, List[int]] = {}
        latents_list = []
        styles_list = []
        current_idx = 0

        logger.info("Loading latents (balanced sampling)...")
        for style_id, subdir in enumerate(self.style_subdirs):
            style_path = self.data_root / subdir
            if not style_path.exists():
                logger.warning(f"Style dir not found: {style_path}")
                continue

            latent_files = sorted(style_path.glob("*.pt"))
            self.style_indices[style_id] = list(range(current_idx, current_idx + len(latent_files)))
            logger.info(f"  Style {style_id} ({subdir}): {len(latent_files)} images")

            for lf in latent_files:
                latent = torch.load(lf, map_location='cpu')
                if latent.ndim == 4:
                    latent = latent.squeeze(0)
                latents_list.append(latent.float())
                styles_list.append(style_id)
                current_idx += 1

        if not latents_list:
            raise RuntimeError(f"No latents found under {self.data_root}")

        self.latents_tensor = torch.stack(latents_list)
        self.styles_tensor = torch.tensor(styles_list, dtype=torch.long)

        # Rescale if raw VAE latents detected
        std_original = self.latents_tensor.std().item()
        scaling_factor = 0.18215
        if rescale_raw and std_original < 0.5:
            logger.info(f"⚠️ Raw VAE latents detected (std={std_original:.4f}). Rescaling by {scaling_factor}.")
            self.latents_tensor = self.latents_tensor / scaling_factor

        if pin_memory:
            self.latents_tensor = self.latents_tensor.pin_memory()
            self.styles_tensor = self.styles_tensor.pin_memory()

        max_count = max(len(idxs) for idxs in self.style_indices.values()) if self.style_indices else 0
        self.virtual_length = max_count * num_styles * num_styles

    def __len__(self) -> int:
        return self.virtual_length

    def __getitem__(self, _):
        # Randomly pick a style uniformly
        style_id = torch.randint(0, self.num_styles, (1,), device='cpu').item()

        if style_id in self.style_indices and self.style_indices[style_id]:
            idx = random.choice(self.style_indices[style_id])
        else:
            idx = random.randint(0, len(self.latents_tensor) - 1)
            style_id = int(self.styles_tensor[idx])

        latent = self.latents_tensor[idx]
        sample = {
            'latent': latent,
            'style_id': torch.tensor(style_id, dtype=torch.long),
        }

        if self.apply_elastic:
            latent_aug = elastic_deform(latent.unsqueeze(0), alpha=self.elastic_alpha, sigma=self.elastic_sigma)
            sample['latent_deformed'] = latent_aug.squeeze(0)

        return sample

    def sample_style_batch(self, target_style_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Vectorized style sampling on CPU latents with device transfer.
        """
        b = target_style_ids.shape[0]
        style_latents = torch.empty((b, *self.latents_tensor.shape[1:]), device=device)
        target_cpu = target_style_ids.detach().cpu()

        for style_id, indices in self.style_indices.items():
            mask = target_cpu == style_id
            count = int(mask.sum().item())
            if count == 0:
                continue
            rand_indices = torch.tensor(indices)[torch.randint(len(indices), (count,))]
            selected = self.latents_tensor[rand_indices]
            style_latents[mask.to(device)] = selected.to(device, non_blocking=True)

        return style_latents
