"""
LGT Thermodynamic Inference with Langevin Dynamics

Key innovation: Langevin-modified ODE integration
- Deterministic drift: v(x, t, style)
- Stochastic diffusion: σ(t) * sqrt(dt) * ε

Temperature schedule:
- Early phase (t < 0.5): σ = 0 (deterministic, recover structure)
- Late phase (t > 0.5): σ = λ (stochastic, generate texture)

Physical interpretation:
- Drift term: Follows energy gradient (geometric potential)
- Diffusion term: Thermal fluctuations for exploring high-frequency details
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import LGTUNet


class LangevinSampler:
    """
    Langevin dynamics sampler for thermodynamic ODE integration.
    
    LGT++ Enhancement: Ternary Guidance
    - Attracts toward target style distribution
    - Repels away from source style artifacts (brush strokes, noise)
    
    Update rule:
        z_{t+1} = z_t + v(z_t, t, style) * dt + σ(t) * sqrt(dt) * ε
    
    where ε ~ N(0, I)
    
    Ternary Guidance:
        v = v_uncond + w_target*(v_target - v_uncond) - w_repel*(v_source - v_uncond)
    """
    
    def __init__(
        self,
        temperature_lambda=0.3,
        temperature_threshold=0.5,
        use_cfg=True,
        cfg_scale=12.0,
        cfg_decay=True,
        use_source_repulsion=False,
        repulsion_strength=3.7
    ):
        """
        Args:
            temperature_lambda: Noise magnitude in late phase
            temperature_threshold: Time threshold for activating noise (default 0.5)
            use_cfg: Use Classifier-Free Guidance
            cfg_scale: CFG strength
            cfg_decay: Decay CFG scale over time
            use_source_repulsion: Enable ternary guidance (LGT++ enhancement)
            repulsion_strength: Source repulsion weight
        """
        self.temperature_lambda = temperature_lambda
        self.temperature_threshold = temperature_threshold
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale
        self.cfg_decay = cfg_decay
        self.use_source_repulsion = use_source_repulsion
        self.repulsion_strength = repulsion_strength
    
    def get_temperature(self, t):
        """
        Temperature schedule σ(t).
        
        Args:
            t: scalar or tensor, time in [0, 1]
        
        Returns:
            σ: temperature coefficient
        """
        # Early phase: deterministic (σ = 0)
        # Late phase: stochastic (σ = λ)
        if isinstance(t, torch.Tensor):
            sigma = torch.where(
                t < self.temperature_threshold,
                torch.zeros_like(t),
                torch.full_like(t, self.temperature_lambda)
            )
        else:
            sigma = 0.0 if t < self.temperature_threshold else self.temperature_lambda
        
        return sigma
    
    def get_cfg_scale(self, step, total_steps):
        """
        Dynamic CFG scale with optional decay.
        
        Args:
            step: current step index
            total_steps: total number of steps
        
        Returns:
            cfg_scale: guidance strength
        """
        if not self.cfg_decay:
            return self.cfg_scale
        
        # Linear decay: start at cfg_scale, end at 1.0
        decay_rate = (self.cfg_scale - 1.0) / total_steps
        return max(1.0, self.cfg_scale - decay_rate * step)
    
    def velocity_with_cfg(self, model, x, t, style_id, step, total_steps, source_style_id=None):
        """
        Compute velocity with Classifier-Free Guidance and optional Ternary Guidance.
        
        Standard CFG:
            v_guided = v_uncond + w * (v_cond - v_uncond)
        
        Rescale CFG (Paper: Common Diffusion Noise Schedules and Sample Steps are Flawed):
            - Prevents oversaturation by rescaling predicted vector magnitude
            - back to target vector magnitude level
        
        Ternary Guidance (LGT++):
            v_guided = v_uncond + w_target*(v_target - v_uncond) - w_repel*(v_source - v_uncond)
        
        Physics:
        - Attractive force toward target distribution
        - Repulsive force away from source artifacts (critical for painting→photo)
        - Rescale prevents guidance-induced color/brightness drift
        
        Args:
            model: LGT model
            x: [B, 4, H, W] current state
            t: [B] or scalar, current time
            style_id: [B] target style IDs
            step: current step index
            total_steps: total steps for CFG decay
            source_style_id: [B] or scalar, source style IDs (for ternary guidance)
        
        Returns:
            v: [B, 4, H, W] guided velocity
        """
        if not self.use_cfg:
            # No CFG, just conditional velocity
            return model(x, t, style_id, use_avg_style=False)
        
        # Conditional velocity (target style)
        v_target = model(x, t, style_id, use_avg_style=False)
        
        # Unconditional velocity (average style)
        v_uncond = model(x, t, style_id, use_avg_style=True)
        
        # Compute CFG scale
        w = self.get_cfg_scale(step, total_steps)
        
        # Standard CFG
        v_pred = v_uncond + w * (v_target - v_uncond)
        
        # ==========================================
        # 🔥 Rescale CFG: Prevent oversaturation
        # ==========================================
        # Compute correction factor to prevent magnitude drift
        
        # 1. Calculate standard deviations (signal magnitude)
        std_target = v_target.std(dim=(1, 2, 3), keepdim=True)
        std_pred = v_pred.std(dim=(1, 2, 3), keepdim=True)
        
        # 2. Rescale to match target magnitude
        rescale_factor = std_target / (std_pred + 1e-8)
        v_pred_rescaled = v_pred * rescale_factor
        
        # 3. Blend rescaled and original (retain some CFG impact)
        # rescale_weight typically 0.7 balances quality and guidance strength
        rescale_weight = 0.7
        v_guided = v_pred_rescaled * rescale_weight + v_pred * (1 - rescale_weight)
        
        # Ternary Guidance (LGT++ enhancement)
        if self.use_source_repulsion and source_style_id is not None:
            # Compute source velocity
            v_source = model(x, t, source_style_id, use_avg_style=False)
            
            # Add repulsive force away from source
            v_guided = v_guided - self.repulsion_strength * (v_source - v_uncond)
        
        return v_guided
    
    def step(self, model, x, t, style_id, dt, step_idx, total_steps, source_style_id=None):
        """
        Single Langevin integration step with optional ternary guidance.
        
        Args:
            model: LGT model
            x: [B, 4, H, W] current state
            t: [B] current time
            style_id: [B] target style IDs
            dt: time step size
            step_idx: current step index
            total_steps: total number of steps
            source_style_id: [B] or scalar, source style IDs (for ternary guidance)
        
        Returns:
            x_next: [B, 4, H, W] next state
        """
        # Drift term: velocity field (with optional ternary guidance)
        v = self.velocity_with_cfg(model, x, t, style_id, step_idx, total_steps, source_style_id)
        
        # Deterministic update
        x_next = x + v * dt
        
        # Diffusion term: thermal fluctuations
        sigma = self.get_temperature(t)
        
        if isinstance(sigma, torch.Tensor):
            # Handle batch of times
            add_noise = (sigma > 0).any()
        else:
            add_noise = sigma > 0
        
        if add_noise:
            noise = torch.randn_like(x)
            if isinstance(sigma, torch.Tensor):
                sigma = sigma.view(-1, 1, 1, 1)
            x_next = x_next + sigma * np.sqrt(dt) * noise
        
        return x_next
    
    @torch.no_grad()
    def sample(
        self,
        model,
        x_init,
        style_id,
        num_steps=20,
        t_start=0.0,
        t_end=1.0,
        return_trajectory=False,
        source_style_id=None
    ):
        """
        Sample trajectory from x_init to terminal state.
        
        Args:
            model: LGT model
            x_init: [B, 4, H, W] initial state
            style_id: [B] or scalar, target style ID
            num_steps: number of integration steps
            t_start: starting time
            t_end: ending time
            return_trajectory: return all intermediate states
            source_style_id: [B] or scalar, source style ID (for ternary guidance)
        
        Returns:
            x_final: [B, 4, H, W] terminal state
            trajectory: list of states (if return_trajectory=True)
        """
        model.eval()
        
        B = x_init.shape[0]
        device = x_init.device
        
        # Handle scalar style_id
        if isinstance(style_id, int):
            style_id = torch.full((B,), style_id, dtype=torch.long, device=device)
        
        if source_style_id is not None and isinstance(source_style_id, int):
            source_style_id = torch.full((B,), source_style_id, dtype=torch.long, device=device)
        
        # Initialize
        x = x_init.clone()
        dt = (t_end - t_start) / num_steps
        
        trajectory = [x.cpu()] if return_trajectory else None
        
        # Integration loop
        for step_idx in range(num_steps):
            t_current = t_start + step_idx * dt
            
            # Create time tensor
            if isinstance(t_current, torch.Tensor):
                t = t_current
            else:
                t = torch.full((B,), t_current, device=device)
            
            # Langevin step with optional ternary guidance
            x = self.step(model, x, t, style_id, dt, step_idx, num_steps, source_style_id)
            
            if return_trajectory:
                trajectory.append(x.cpu())
        
        if return_trajectory:
            return x, trajectory
        else:
            return x


class LGTInference:
    """
    Complete inference pipeline for LGT.
    
    Supports:
    - Structure-preserving style transfer (inversion + generation)
    - Direct generation from noise
    - Interpolation between styles
    """
    
    def __init__(
        self,
        model_path,
        device='cuda',
        temperature_lambda=0.3,
        temperature_threshold=0.5,
        use_cfg=True,
        cfg_scale=5.0,
        num_steps=20,
        use_source_repulsion=False,
        repulsion_strength=0.7
    ):
        """
        Args:
            model_path: Path to checkpoint
            device: Device for inference
            temperature_lambda: Noise magnitude
            temperature_threshold: Time threshold for noise
            use_cfg: Use CFG
            cfg_scale: CFG strength
            num_steps: Number of integration steps
            use_source_repulsion: Enable ternary guidance (LGT++ enhancement)
            repulsion_strength: Source repulsion weight
        """
        self.device = device
        self.num_steps = num_steps
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']
        
        self.model = LGTUNet(
            latent_channels=config['model']['latent_channels'],
            base_channels=config['model']['base_channels'],
            style_dim=config['model']['style_dim'],
            time_dim=config['model']['time_dim'],
            num_styles=config['model']['num_styles'],
            num_encoder_blocks=config['model']['num_encoder_blocks'],
            num_decoder_blocks=config['model']['num_decoder_blocks']
        ).to(device)
        
        # Handle state_dict from compiled models (_orig_mod. prefix)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            # Strip _orig_mod. prefix from compiled model checkpoint
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Compute average style embedding
        self.model.compute_avg_style_embedding()
        
        # Create sampler with ternary guidance support
        self.sampler = LangevinSampler(
            temperature_lambda=temperature_lambda,
            temperature_threshold=temperature_threshold,
            use_cfg=use_cfg,
            cfg_scale=cfg_scale,
            use_source_repulsion=use_source_repulsion,
            repulsion_strength=repulsion_strength
        )
    
    @torch.no_grad()
    def inversion(self, x1, source_style_id, num_steps=None):
        """
        Invert latent to noise (reverse ODE).
        
        Solve: dx/dt = -v(x, 1-t, c_source), t: 0 → 1
        
        Args:
            x1: [B, 4, H, W] observed latent
            source_style_id: source style ID
            num_steps: number of steps (default: self.num_steps)
        
        Returns:
            x0: [B, 4, H, W] inverted noise
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        B = x1.shape[0]
        device = x1.device
        
        # Handle scalar style_id
        if isinstance(source_style_id, int):
            source_style_id = torch.full((B,), source_style_id, dtype=torch.long, device=device)
        
        x = x1.clone()
        dt = 1.0 / num_steps
        
        # Reverse integration (deterministic, no noise)
        for step_idx in range(num_steps):
            t_forward = 1.0 - step_idx * dt
            t = torch.full((B,), t_forward, device=device)
            
            # Velocity at current state (no CFG for inversion)
            v = self.model(x, t, source_style_id, use_avg_style=False)
            
            # Reverse step: dx = -v * dt
            x = x - v * dt
        
        return x
    
    @torch.no_grad()
    def generation(self, x0, target_style_id, num_steps=None, source_style_id=None):
        """
        Generate from noise with target style.
        
        Args:
            x0: [B, 4, H, W] noise
            target_style_id: target style ID
            num_steps: number of steps
            source_style_id: source style ID (for ternary guidance)
        
        Returns:
            x1: [B, 4, H, W] generated latent
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        return self.sampler.sample(
            self.model,
            x0,
            target_style_id,
            num_steps=num_steps,
            t_start=0.0,
            t_end=1.0,
            source_style_id=source_style_id
        )
    
    @torch.no_grad()
    def transfer_style(
        self,
        x_source,
        source_style_id,
        target_style_id,
        num_steps=None,
        return_intermediate=False,
        use_ternary_guidance=None
    ):
        """
        Structure-preserving style transfer with optional ternary guidance.
        
        Pipeline:
        1. Inversion: x_source → x0 (via source style)
        2. Generation: x0 → x_target (via target style, with optional repulsion from source)
        
        LGT++ Ternary Guidance:
        When enabled, generation actively repels from source style artifacts.
        Critical for painting→photo transfers to eliminate brush strokes.
        
        Args:
            x_source: [B, 4, H, W] source latent
            source_style_id: source style ID
            target_style_id: target style ID
            num_steps: number of steps per stage
            return_intermediate: return (x0, x_target)
            use_ternary_guidance: override sampler's use_source_repulsion setting
        
        Returns:
            x_target: [B, 4, H, W] transferred latent
            (x0, x_target): if return_intermediate=True
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        # Stage 1: Inversion
        x0 = self.inversion(x_source, source_style_id, num_steps)
        
        # Stage 2: Generation with optional ternary guidance
        if use_ternary_guidance is None:
            # Use sampler's default setting
            pass_source_id = source_style_id if self.sampler.use_source_repulsion else None
        elif use_ternary_guidance:
            # Force enable ternary guidance
            pass_source_id = source_style_id
        else:
            # Force disable ternary guidance
            pass_source_id = None
        
        x_target = self.generation(x0, target_style_id, num_steps, source_style_id=pass_source_id)
        
        if return_intermediate:
            return x_target, x0
        else:
            return x_target
    
    @torch.no_grad()
    def interpolate_styles(
        self,
        x_source,
        source_style_id,
        style_ids,
        num_steps=None
    ):
        """
        Generate multiple style transfers from one source.
        
        Args:
            x_source: [1, 4, H, W] source latent
            source_style_id: source style ID
            style_ids: list of target style IDs
            num_steps: number of steps
        
        Returns:
            results: list of [1, 4, H, W] transferred latents
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        # Invert once
        x0 = self.inversion(x_source, source_style_id, num_steps)
        
        # Generate for each target style
        results = []
        for target_style_id in style_ids:
            x_target = self.generation(x0, target_style_id, num_steps)
            results.append(x_target)
        
        return results


# Utility functions for VAE encoding/decoding

def load_vae(device='cuda'):
    """Load Stable Diffusion VAE."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    return vae


@torch.no_grad()
def encode_image(vae, image_tensor, device='cuda'):
    """
    Encode image to latent.
    
    Args:
        vae: VAE model
        image_tensor: [B, 3, H, W] in [-1, 1]
        device: device
    
    Returns:
        latent: [B, 4, H//8, W//8]
    """
    image_tensor = image_tensor.to(device, dtype=torch.float16)
    latent = vae.encode(image_tensor).latent_dist.sample()
    latent = latent * vae.config.scaling_factor  # 0.18215
    return latent


@torch.no_grad()
def decode_latent(vae, latent, device='cuda'):
    """
    Decode latent to image.
    
    Args:
        vae: VAE model
        latent: [B, 4, H, W]
        device: device
    
    Returns:
        image: [B, 3, H*8, W*8] in [0, 1]
    """
    latent = latent.to(device, dtype=torch.float16)
    latent = latent / vae.config.scaling_factor
    image = vae.decode(latent).sample
    image = (image + 1.0) / 2.0  # [-1, 1] → [0, 1]
    image = torch.clamp(image, 0.0, 1.0)
    return image


def tensor_to_pil(tensor):
    """
    Convert tensor to PIL image.
    
    Args:
        tensor: [1, 3, H, W] or [3, H, W] in [0, 1]
    
    Returns:
        PIL.Image
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.cpu().float()
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check arguments
    if len(sys.argv) < 4:
        print("Usage: python inference.py <checkpoint> <source_img> <output_path> [target_style_id]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    source_image_path = sys.argv[2]
    output_path = sys.argv[3]
    target_style_id = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading models...")
    vae = load_vae(device)
    lgt = LGTInference(
        checkpoint_path,
        device=device,
        temperature_lambda=0.1,
        use_cfg=True,
        cfg_scale=5.0,
        num_steps=20
    )
    
    print(f"Loading source image: {source_image_path}")
    # Load and preprocess image
    image = Image.open(source_image_path).convert('RGB')
    image = image.resize((256, 256))
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor * 2.0 - 1.0  # [0,1] → [-1,1]
    
    print("Encoding to latent...")
    latent_source = encode_image(vae, image_tensor, device)
    
    print(f"Transferring style to ID {target_style_id}...")
    latent_target = lgt.transfer_style(
        latent_source,
        source_style_id=0,  # Assume source is style 0
        target_style_id=target_style_id,
        num_steps=20
    )
    
    print("Decoding to image...")
    image_output = decode_latent(vae, latent_target, device)
    
    print(f"Saving to: {output_path}")
    output_pil = tensor_to_pil(image_output)
    output_pil.save(output_path)
    
    print("✓ Done!")
