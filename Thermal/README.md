# LGT: Latent Geometric Thermodynamics

**Geometric free energy optimization for style transfer in VAE latent space**

---

## Overview

LGT implements a physics-inspired approach to style transfer based on **geometric thermodynamics**:

- **Content = Rigid Topology**: Preserved via Cosine Self-Similarity Matrix (SSM)
- **Style = Flowing Distribution**: Matched via Patch-Sliced Wasserstein Distance (SWD)  
- **Dynamics = Thermal Fluctuations**: Enabled via Langevin-modified ODE integration

### Core Innovation

Replace MSE velocity matching with **geometric free energy minimization**:

```
E_total(x) = w_style · SWD(x, x_style) + w_content · SSM(x, x_src)
```

Training optimizes this energy landscape directly, allowing the model to discover physically meaningful trajectories through latent space.

---

## Architecture

### Model: LGTUNet

- **Encoder** (32×32 → 8×8): Standard convolutions extract objective structure
- **Bottleneck** (8×8): Semantic skeleton with high-frequency texture removed
- **Decoder** (8×8 → 32×32): **StyleDynamicConv** generates texture distributions
  - HyperNet produces convolution kernels from style embeddings
  - Each style = unique spatial filter bank
  - Depthwise convolutions for memory efficiency (RTX 4070 compatible)

### Loss Functions

#### 1. Patch-Sliced Wasserstein Distance (Style)
```python
# Pipeline: Unfold → Sample → Project → Sort → MSE
x_patches = unfold(x, kernel_size=3)      # [B, N_patches, 36]
x_sampled = random_sample(x_patches, 4096) # Memory constraint
x_proj = x_sampled @ random_theta          # Radon transform
loss_swd = MSE(sort(x_proj), sort(x_style_proj))
```

**Why Patch-SWD?**
- VAE's 4 channels lack texture info → Extract 3×3 patches → 36D feature space
- Samples 4096 points for 8GB VRAM compatibility
- Distribution matching invariant to spatial layout

#### 2. Cosine Self-Similarity Matrix (Content)
```python
# Pipeline: Normalize → Gram Matrix → Frobenius Norm
z = normalize(x, dim=channels)  # Magnitude-invariant
A = z.T @ z                      # Cosine similarity matrix [N×N]
loss_ssm = ||A_pred - A_src||_F^2
```

**Why Cosine-SSM?**
- Locks angular relationships between spatial positions
- Allows style-induced scaling (color shifts, intensity changes)
- Preserves topological structure regardless of distribution transform

---

## Installation

```bash
# Clone repository
git clone <repository_url>
cd VAE_ca_proj/LGT

# Install dependencies
pip install torch torchvision diffusers pillow tqdm numpy

# Install Stable Diffusion VAE (for preprocessing)
pip install diffusers transformers accelerate
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+ (for `torch.compile` support)
- CUDA 11.7+ (recommended)
- 8GB+ VRAM (RTX 4070 tested)

---

## Usage

### 1. Data Preprocessing

**Convert images to VAE latents** (mandatory before training):

```bash
python preprocess_latents.py \
    --raw_data_root /path/to/images \
    --output_root data/latents \
    --target_size 256
```

**Expected directory structure:**
```
raw_data_root/
  style0/
    img001.jpg
    img002.jpg
  style1/
    img003.jpg
    img004.jpg

→ Generates:

data/latents/
  style0/
    img001.pt  # [4, 32, 32] tensor
    img002.pt
  style1/
    img003.pt
    img004.pt
```

**Update `config.json`:**
```json
{
  "data": {
    "data_root": "data/latents",
    "style_subdirs": ["style0", "style1"]
  },
  "model": {
    "num_styles": 2
  }
}
```

---

### 2. Training

```bash
python train.py
```

**Key training parameters** (in `config.json`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Batch size (reduce if OOM) |
| `learning_rate` | 1e-4 | Initial learning rate |
| `num_epochs` | 100 | Total training epochs |
| `w_style` | 1.0 | Style energy weight |
| `w_content` | 1.0 | Content energy weight |
| `ode_integration_steps` | 5 | ODE steps for computing x_1 |
| `label_drop_prob` | 0.1 | CFG training dropout |

**Training process:**
1. Sample noise x₀ and time t
2. Construct intermediate state x_t = (1-t)·x₀ + t·x_src
3. Integrate ODE to terminal state x₁
4. Compute energy E(x₁) and backpropagate

**Monitoring:**
- Logs saved to `checkpoints/logs/training_YYYYMMDD_HHMMSS.csv`
- Checkpoints saved every 5 epochs to `checkpoints/epoch_XXXX.pt`

---

### 3. Inference

#### Python API

```python
from inference import LGTInference, load_vae, encode_image, decode_latent
from PIL import Image
import torch

# Load models
device = 'cuda'
vae = load_vae(device)
lgt = LGTInference(
    model_path='checkpoints/epoch_0100.pt',
    device=device,
    temperature_lambda=0.1,
    use_cfg=True,
    cfg_scale=2.0,
    num_steps=20
)

# Load and encode image
image = Image.open('input.jpg')
image_tensor = preprocess(image)  # [1, 3, 256, 256]
latent_src = encode_image(vae, image_tensor, device)

# Style transfer
latent_out = lgt.transfer_style(
    x_source=latent_src,
    source_style_id=0,
    target_style_id=1,
    num_steps=20
)

# Decode and save
image_out = decode_latent(vae, latent_out, device)
save_image(image_out, 'output.jpg')
```

#### Command Line

```bash
python inference.py \
    checkpoints/epoch_0100.pt \
    input.jpg \
    output.jpg \
    --target_style_id 1
```

---

## Key Differences from OT-CFM

| Aspect | OT-CFM | LGT |
|--------|--------|-----|
| **Loss** | MSE(v_pred, v_target) | E(x₁) = SWD + SSM |
| **Training** | Direct velocity matching | Energy landscape optimization |
| **Inference** | Deterministic ODE | Langevin dynamics (stochastic) |
| **Style Injection** | AdaGN (scale/shift) | StyleDynamicConv (kernel generation) |
| **Physical Model** | Optimal Transport | Thermodynamic free energy |

**Advantages:**
- **Texture Fidelity**: SWD operates on 36D patch space vs 4D latent
- **Structure Preservation**: Cosine-SSM locks topology regardless of distribution
- **High-Frequency Details**: Langevin noise generates fine textures in late phase

**Trade-offs:**
- **Training Cost**: ODE integration adds ~2× compute vs direct supervision
- **Memory**: Energy computation requires storing intermediate states
- **Stochasticity**: Langevin sampling → non-deterministic outputs (controlled by temperature)

---

## Configuration Reference

### Model Parameters

```json
{
  "model": {
    "latent_channels": 4,          // VAE latent channels
    "base_channels": 128,           // Base hidden dimension
    "style_dim": 256,               // Style embedding dimension
    "time_dim": 256,                // Time embedding dimension
    "num_styles": 2,                // Number of style classes
    "num_encoder_blocks": 2,        // Encoder depth
    "num_decoder_blocks": 3         // Decoder depth
  }
}
```

### Loss Parameters

```json
{
  "loss": {
    "w_style": 1.0,                 // Style energy weight
    "w_content": 1.0,               // Content energy weight
    "patch_size": 3,                // SWD patch size
    "num_projections": 64,          // SWD projection dimensions
    "max_samples": 4096,            // SWD sample limit (VRAM)
    "use_velocity_reg": false,      // Optional L2 regularization
    "vel_reg_weight": 0.01
  }
}
```

### Inference Parameters

```json
{
  "inference": {
    "num_steps": 20,                // ODE integration steps
    "temperature_lambda": 0.1,      // Noise magnitude (t > 0.5)
    "temperature_threshold": 0.5,   // When to activate noise
    "use_cfg": true,                // Classifier-Free Guidance
    "cfg_scale": 2.0,               // CFG strength
    "cfg_decay": true               // Decay CFG over time
  }
}
```

---

## Temperature Scheduling

Critical innovation: **Langevin dynamics with phase-dependent noise**

```python
σ(t) = {
    0       if t < 0.5  (Early: recover structure deterministically)
    λ       if t ≥ 0.5  (Late: generate texture stochastically)
}
```

**Physical interpretation:**
- **t < 0.5**: System follows deterministic gradient descent on energy landscape
- **t ≥ 0.5**: Thermal fluctuations explore local minima for high-frequency details

**Adjust via config:**
- `temperature_lambda`: Noise magnitude (higher = more texture variation)
- `temperature_threshold`: Phase transition point (earlier = more stochastic)

---

## Hardware Optimization

**Designed for RTX 4070 Laptop (8GB VRAM):**

1. **Patch-SWD Sampling**: Max 4096 patches (vs full N²)
2. **Depthwise StyleDynamicConv**: groups=channels reduces params by ~4×
3. **FP32 Critical Ops**: Only SWD sorting and SSM matmul in FP32
4. **BFloat16 AMP**: Rest of model in mixed precision
5. **In-Memory Dataset**: Pre-load all latents (channels_last format)
6. **No Gradient Checkpointing**: 32×32 latents small enough for full backprop

**Estimated VRAM:**
- Model: ~1.5GB
- Batch=16: ~3GB activations
- Optimizer states: ~2GB
- **Total: ~6.5GB** (leaves 1.5GB buffer)

---

## Troubleshooting

### OOM (Out of Memory)

```json
// Reduce batch size
"batch_size": 8  // or 4

// Reduce SWD samples
"max_samples": 2048

// Enable gradient checkpointing (add to model.py)
use_gradient_checkpointing: true
```

### Poor Style Transfer

```json
// Increase style weight
"w_style": 2.0

// Increase temperature for more texture
"temperature_lambda": 0.15

// More ODE steps
"ode_integration_steps": 10
```

### Structure Distortion

```json
// Increase content weight
"w_content": 2.0

// Reduce temperature
"temperature_lambda": 0.05

// Lower CFG scale
"cfg_scale": 1.5
```

---

## Citation

If you use this code, please cite the theoretical framework:

```
LGT: Latent Geometric Thermodynamics for Style Transfer
Based on principles from:
- Optimal Transport (Villani, 2009)
- Sliced-Wasserstein Distance (Rabin et al., 2012)
- Neural Style Transfer (Gatys et al., 2016)
- Langevin Dynamics (Neal, 2011)
```

---

## License

MIT License

---

## Acknowledgments

- **Stable Diffusion VAE**: HuggingFace / Stability AI
- **OT-CFM Framework**: Inspired by rectified flow models
- **Design Document**: Based on `Termal-dynamic.md` theoretical verification

---

## Contact

For questions or issues, please open a GitHub issue or contact the development team.

**Hardware Tested:**
- NVIDIA RTX 4070 Laptop (8GB VRAM) ✓
- NVIDIA RTX 3090 (24GB VRAM) ✓
- CPU (slow but functional) ✓
