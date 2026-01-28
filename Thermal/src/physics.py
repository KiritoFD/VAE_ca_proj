import torch
from torch.utils.checkpoint import checkpoint


def get_dynamic_epsilon(epoch: int, target_epsilon: float, warmup_epochs: int = 10) -> float:
    """Linearly warm epsilon from 0 to target over warmup_epochs."""
    if warmup_epochs <= 0:
        return target_epsilon
    if epoch < warmup_epochs:
        return target_epsilon * (epoch / warmup_epochs)
    return target_epsilon


def integrate_ode(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    t_start: torch.Tensor,
    style_id: torch.Tensor,
    steps: int,
    use_checkpoint: bool = True,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    training: bool = True,
    clip_range: float = 6.0,
) -> torch.Tensor:
    """Euler integration of dx/dt = v(x, t, style) with numerical safeguards.
    
    Args:
        clip_range: VAE latent space range to clip to (standard VAE uses ~±3σ, we use 6.0 for safety)
    """
    x = x_t.clone()
    t = t_start.clone()
    num_steps = max(steps, 1)

    def step_fn(x_in, t_in, style_in):
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            return model(x_in, t_in, style_in, use_avg_style=False)

    for _ in range(num_steps):
        t_remaining = 1.0 - t
        dt = t_remaining / num_steps
        if use_checkpoint and training:
            v = checkpoint(step_fn, x, t, style_id, use_reentrant=False)
        else:
            v = step_fn(x, t, style_id)
        x = x + v * dt.view(-1, 1, 1, 1)
        
        # 🔥 数值安全阀：限制在VAE的线性响应区间
        # Cross-Attention检索出的艺术特征会导致极强的速度场，必须截断
        # 防止色块/爆炸的最后一公里
        x = torch.clamp(x, -clip_range, clip_range)
        
        t = t + dt

    return x


def invert_latent(
    model: torch.nn.Module,
    latent: torch.Tensor,
    style_id: torch.Tensor,
    num_steps: int = 15,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    clip_range: float = 6.0,
) -> torch.Tensor:
    """Reverse ODE integration from terminal state to noise with numerical safeguards.
    
    Args:
        clip_range: VAE latent space range to clip to (prevents explosion during inversion)
    """
    b = latent.shape[0]
    device = latent.device
    if isinstance(style_id, int):
        style_id = torch.full((b,), style_id, dtype=torch.long, device=device)

    x = latent.clone()
    dt = 1.0 / max(num_steps, 1)

    for step_idx in range(num_steps):
        t_forward = 1.0 - step_idx * dt
        t = torch.full((b,), t_forward, device=device)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            v = model(x, t, style_id, use_avg_style=False)
        x = x - v * dt
        
        # 🔥 数值安全阀：防止反演过程中的爆炸
        x = torch.clamp(x, -clip_range, clip_range)

    return x


def generate_latent(
    model: torch.nn.Module,
    latent: torch.Tensor,
    style_id: torch.Tensor,
    num_steps: int = 15,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    langevin_sigma: float = 0.1,
    langevin_threshold: float = 0.5,
    clip_range: float = 6.0,
) -> torch.Tensor:
    """Forward ODE integration with optional Langevin noise and numerical safeguards.
    
    Args:
        clip_range: VAE latent space range to clip to (prevents color blocks)
    """
    b = latent.shape[0]
    device = latent.device
    if isinstance(style_id, int):
        style_id = torch.full((b,), style_id, dtype=torch.long, device=device)

    x = latent.clone()
    dt = 1.0 / max(num_steps, 1)

    for step_idx in range(num_steps):
        t_current = step_idx * dt
        t = torch.full((b,), t_current, device=device)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            v = model(x, t, style_id, use_avg_style=False)
        x = x + v * dt
        
        # 🔥 数值安全阀：限制在VAE的线性响应区间，消灭色块
        # Cross-Attention检索出的强艺术特征可能导致速度场爆炸
        x = torch.clamp(x, -clip_range, clip_range)
        
        if t_current > langevin_threshold:
            noise = torch.randn_like(x)
            x = x + langevin_sigma * (dt ** 0.5) * noise

    return x
