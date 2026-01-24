# LGT API Reference

## Core Components

### Model Architecture

#### `LGTUNet`
Main model implementing geometric thermodynamics U-Net.

**Constructor:**
```python
LGTUNet(
    latent_channels=4,      # VAE latent channels
    base_channels=128,      # Base hidden dimension
    style_dim=256,          # Style embedding dimension
    time_dim=256,           # Time embedding dimension
    num_styles=2,           # Number of style classes
    num_encoder_blocks=2,   # Encoder depth
    num_decoder_blocks=3    # Decoder depth
)
```

**Methods:**
- `forward(x, t, style_id, use_avg_style=False)`: Compute velocity field
  - Args:
    - `x`: [B, 4, H, W] latent at time t
    - `t`: [B] timestep in [0, 1]
    - `style_id`: [B] style class ID
    - `use_avg_style`: bool, use average style (CFG)
  - Returns: `v` [B, 4, H, W] velocity field

- `compute_avg_style_embedding()`: Initialize average style embedding (call before training)

---

### Loss Functions

#### `PatchSlicedWassersteinLoss`
Style distribution matching via Patch-SWD.

**Constructor:**
```python
PatchSlicedWassersteinLoss(
    patch_size=3,           # Patch size for unfold
    num_projections=64,     # Number of random projections
    max_samples=4096,       # Max samples for memory efficiency
    use_fp32=True           # Force FP32 precision
)
```

**Methods:**
- `forward(x_pred, x_style)`: Compute SWD loss
  - Args:
    - `x_pred`: [B, 4, H, W] predicted latent
    - `x_style`: [B, 4, H, W] style reference
  - Returns: scalar loss

#### `CosineSSMLoss`
Content topology preservation via Cosine Self-Similarity Matrix.

**Constructor:**
```python
CosineSSMLoss(
    use_fp32=True,                      # Force FP32 precision
    normalize_by_num_elements=True      # Scale-invariant loss
)
```

**Methods:**
- `forward(x_pred, x_src)`: Compute SSM loss
  - Args:
    - `x_pred`: [B, 4, H, W] predicted latent
    - `x_src`: [B, 4, H, W] source content
  - Returns: scalar loss

#### `GeometricFreeEnergyLoss`
Combined style + content energy.

**Constructor:**
```python
GeometricFreeEnergyLoss(
    w_style=1.0,            # Style weight
    w_content=1.0,          # Content weight
    patch_size=3,           # SWD patch size
    num_projections=64,     # SWD projections
    max_samples=4096        # SWD samples
)
```

**Methods:**
- `forward(x_pred, x_style, x_src)`: Compute total energy
  - Returns: dict with keys ['total', 'style_swd', 'content_ssm']

---

### Training

#### `LGTTrainer`
Main training orchestrator.

**Constructor:**
```python
LGTTrainer(
    config,                 # Config dictionary
    device='cuda'           # Compute device
)
```

**Methods:**
- `train(dataloader)`: Execute full training loop
- `train_epoch(dataloader, epoch)`: Train single epoch
- `save_checkpoint(epoch, metrics)`: Save model checkpoint

**Key Training Loop:**
1. Sample x₀ ~ N(0,I) and t ~ U(ε, 1)
2. Construct x_t = (1-t)·x₀ + t·x_src
3. Integrate ODE: x_t → x₁
4. Compute E(x₁) = w_style·SWD(x₁, x_style) + w_content·SSM(x₁, x_src)
5. Backpropagate through ODE trajectory

---

### Inference

#### `LangevinSampler`
Thermodynamic ODE sampler with Langevin dynamics.

**Constructor:**
```python
LangevinSampler(
    temperature_lambda=0.1,         # Noise magnitude
    temperature_threshold=0.5,      # Phase transition time
    use_cfg=True,                   # Classifier-Free Guidance
    cfg_scale=2.0,                  # CFG strength
    cfg_decay=True                  # Decay CFG over time
)
```

**Methods:**
- `sample(model, x_init, style_id, num_steps=20)`: Integrate from x_init to x_final
  - Returns: terminal state or (state, trajectory)

- `get_temperature(t)`: Temperature schedule
  - Returns: σ(t) = 0 if t<0.5 else λ

#### `LGTInference`
Complete inference pipeline.

**Constructor:**
```python
LGTInference(
    model_path,                     # Path to checkpoint
    device='cuda',
    temperature_lambda=0.1,
    temperature_threshold=0.5,
    use_cfg=True,
    cfg_scale=2.0,
    num_steps=20
)
```

**Methods:**

- `transfer_style(x_source, source_style_id, target_style_id, num_steps=20)`
  - Structure-preserving style transfer
  - Pipeline: x_source → (invert) → x₀ → (generate) → x_target
  - Returns: [B, 4, H, W] transferred latent

- `inversion(x1, source_style_id, num_steps=20)`
  - Reverse ODE: x₁ → x₀
  - Returns: [B, 4, H, W] noise latent

- `generation(x0, target_style_id, num_steps=20)`
  - Forward ODE: x₀ → x₁
  - Returns: [B, 4, H, W] generated latent

- `interpolate_styles(x_source, source_style_id, style_ids, num_steps=20)`
  - Generate multiple style transfers
  - Returns: list of latents

---

### Utilities

#### VAE Functions

```python
# Load VAE
vae = load_vae(device='cuda')

# Image → Latent
latent = encode_image(vae, image_tensor, device)
# image_tensor: [B, 3, H, W] in [-1, 1]
# latent: [B, 4, H//8, W//8]

# Latent → Image
image = decode_latent(vae, latent, device)
# image: [B, 3, H*8, W*8] in [0, 1]

# Tensor → PIL
pil_image = tensor_to_pil(image_tensor)
```

---

## Usage Examples

### Basic Style Transfer

```python
from inference import LGTInference, load_vae, encode_image, decode_latent
from PIL import Image
import torch

# Setup
device = 'cuda'
vae = load_vae(device)
lgt = LGTInference('checkpoints/latest.pt', device=device)

# Load image
img = Image.open('input.jpg').resize((256, 256))
img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) * 2 - 1

# Encode
latent_src = encode_image(vae, img_tensor, device)

# Transfer
latent_out = lgt.transfer_style(
    latent_src, 
    source_style_id=0, 
    target_style_id=1
)

# Decode
img_out = decode_latent(vae, latent_out, device)
Image.fromarray((img_out[0].cpu().numpy().transpose(1,2,0)*255).astype('uint8')).save('output.jpg')
```

### Multi-Style Interpolation

```python
# Generate multiple styles from one source
style_ids = [0, 1, 2, 3]
results = lgt.interpolate_styles(
    latent_src,
    source_style_id=0,
    style_ids=style_ids,
    num_steps=20
)

# Save all results
for i, latent in enumerate(results):
    img = decode_latent(vae, latent, device)
    save_image(img, f'style_{i}.jpg')
```

### Custom Sampling Parameters

```python
# High-quality (slower)
lgt = LGTInference(
    'checkpoints/latest.pt',
    num_steps=50,              # More integration steps
    temperature_lambda=0.05,   # Less noise
    cfg_scale=3.0              # Stronger guidance
)

# Fast (lower quality)
lgt = LGTInference(
    'checkpoints/latest.pt',
    num_steps=10,              # Fewer steps
    temperature_lambda=0.15,   # More noise
    cfg_scale=1.5              # Weaker guidance
)
```

---

## Configuration Schema

Complete `config.json` structure:

```json
{
  "model": {
    "latent_channels": 4,
    "base_channels": 128,
    "style_dim": 256,
    "time_dim": 256,
    "num_styles": 2,
    "num_encoder_blocks": 2,
    "num_decoder_blocks": 3
  },
  "loss": {
    "w_style": 1.0,
    "w_content": 1.0,
    "patch_size": 3,
    "num_projections": 64,
    "max_samples": 4096,
    "use_velocity_reg": false,
    "vel_reg_weight": 0.01
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 1e-4,
    "min_learning_rate": 1e-6,
    "weight_decay": 1e-5,
    "num_epochs": 100,
    "label_drop_prob": 0.1,
    "use_avg_style_for_uncond": true,
    "epsilon": 0.01,
    "epsilon_warmup_epochs": 10,
    "ode_integration_steps": 5,
    "use_amp": true,
    "use_compile": false,
    "save_interval": 5
  },
  "data": {
    "data_root": "data/latents",
    "style_subdirs": ["style0", "style1"]
  },
  "checkpoint": {
    "save_dir": "checkpoints"
  },
  "inference": {
    "num_steps": 20,
    "temperature_lambda": 0.1,
    "temperature_threshold": 0.5,
    "use_cfg": true,
    "cfg_scale": 2.0,
    "cfg_decay": true
  }
}
```

---

## Advanced Topics

### Custom Style Injection

To modify style conditioning mechanism:

```python
# In model.py, replace StyleDynamicConv with custom module
class CustomStyleBlock(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        # Your custom style injection
        
    def forward(self, x, style_emb):
        # Apply style transformation
        return transformed_x
```

### Custom Loss Terms

Add new energy potentials:

```python
# In losses.py
class CustomEnergyTerm(nn.Module):
    def forward(self, x_pred, x_ref):
        # Compute custom distance metric
        return loss

# In train.py
self.custom_loss = CustomEnergyTerm()
loss_dict['custom'] = self.custom_loss(x_1, x_ref)
loss_dict['total'] += w_custom * loss_dict['custom']
```

### Temperature Annealing

Modify noise schedule during inference:

```python
# In inference.py
def get_temperature(self, t, epoch=None):
    if epoch is not None:
        # Anneal temperature over training
        lambda_t = self.temperature_lambda * (1 - epoch/max_epochs)
    else:
        lambda_t = self.temperature_lambda
    
    return 0.0 if t < self.temperature_threshold else lambda_t
```

---

## Performance Optimization

### Memory Reduction

```python
# Reduce batch size
config['training']['batch_size'] = 8

# Reduce SWD samples
config['loss']['max_samples'] = 2048

# Reduce model capacity
config['model']['base_channels'] = 96
config['model']['num_decoder_blocks'] = 2
```

### Speed Optimization

```python
# Enable torch.compile (PyTorch 2.0+)
config['training']['use_compile'] = true

# Reduce ODE integration steps
config['training']['ode_integration_steps'] = 3

# Reduce inference steps
config['inference']['num_steps'] = 15
```

---

## Debugging

### Visualize Energy Landscape

```python
# In train.py, add after energy computation
if step % 100 == 0:
    print(f"Style Energy: {loss_dict['style_swd']:.4f}")
    print(f"Content Energy: {loss_dict['content_ssm']:.4f}")
    print(f"Total Energy: {loss_dict['total']:.4f}")
```

### Monitor Gradients

```python
# Check for gradient explosion
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm:.4f}")
```

### Visualize ODE Trajectory

```python
# In inference.py
latent_out, trajectory = lgt.sampler.sample(
    lgt.model, 
    x_init, 
    style_id, 
    return_trajectory=True
)

# Decode all intermediate states
for i, latent_t in enumerate(trajectory):
    img_t = decode_latent(vae, latent_t, device)
    save_image(img_t, f'trajectory_{i:03d}.jpg')
```
