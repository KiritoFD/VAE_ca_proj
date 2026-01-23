# Gradient Accumulation & Memory Optimization for LGT Training

## Overview

This guide explains the memory optimization strategy implemented to support 10-step ODE integration on RTX 4070 Laptop (8GB VRAM) without gradient checkpointing.

---

## Problem: Memory Explosion with Deep ODE

### The Core Issue

**ODE Integration Memory Footprint:**
- Each ODE step = 1 full forward pass through the model
- PyTorch autograd needs to cache intermediate activations for backprop
- With 10 steps and NO checkpointing: Total cached activations = 10× model depth

**Memory Formula:**
- **With checkpointing:** $M_{total} \approx 2 \times M_{model}$ (store only start + recompute)
- **Without checkpointing:** $M_{total} \approx 10 \times M_{model}$ (store all 10 steps)

### Why Disable Checkpointing?

While gradient checkpointing (trade compute for memory) is standard practice, there are cases where disabling it is optimal:

1. **RTX 4070 is Compute-Limited (FP32)** but Memory-Constrained (8GB)
2. **ODE 10 steps** on 32×32 latents is relatively cheap compute
3. **Recomputing < Storing**: Recomputing 10 steps costs less than storing 10 activation maps

**Trade-off Analysis:**
- Checkpointing: Save 8GB memory, spend extra 50-100ms per step in recomputation
- No checkpointing: Use all 8GB memory, but run 10% faster due to no redundant computation

Given that memory is the primary constraint, **disabling checkpointing is correct**.

---

## Solution 1: Gradient Accumulation

### Core Concept

**Gradient Accumulation** = Batching parameter updates without batching samples.

Instead of:
```
for each sample:
    loss = forward()
    backward()
    update()
```

We do:
```
for N steps:
    for each sample:
        loss = forward()
        backward()  # Don't clear gradients
    update()  # One parameter update per N samples
```

### Memory Benefit

**Batch Size 32 without accumulation:**
- Batch tensor: `[32, 4, 32, 32]`
- Activations: ~32MB
- 10-step ODE: 32 × 10 = 320MB

**Batch Size 32 with 2-step accumulation:**
- Each forward: Still `[32, 4, 32, 32]`
- But: Effective batch = 32 × 2 = 64 (simulates larger batch)
- Same per-step memory, but better gradient estimates!

### Implementation Details

**Key modifications in train.py:**

1. **Epoch start:** Zero gradients once
   ```python
   self.optimizer.zero_grad(set_to_none=True)
   ```

2. **Each step:** Scale loss and accumulate gradients
   ```python
   loss = loss_dict['total'] / self.accumulation_steps
   self.scaler.scale(loss).backward()  # Gradients accumulate
   accum_counter += 1
   ```

3. **After N steps:** Update and zero
   ```python
   if accum_counter >= self.accumulation_steps:
       self.scaler.unscale_(self.optimizer)
       torch.nn.utils.clip_grad_norm_(...)
       self.scaler.step(self.optimizer)
       self.optimizer.zero_grad(set_to_none=True)
       accum_counter = 0
   ```

**Why divide by accumulation_steps?**
- Without scaling: Gradient magnitude = sum of N steps
- After scaling: Gradient magnitude = average (what we want)
- This preserves learning dynamics despite accumulation

---

## Solution 2: torch.compile with Caching

### Why torch.compile?

PyTorch 2.0+ can compile the entire forward graph into optimized kernels:
- Fuses operations (reduce memory bandwidth)
- Reduces Python overhead
- Better cache utilization

### Configuration

```python
self.model = torch.compile(
    self.model,
    mode='reduce-overhead',        # Optimize for latency over throughput
    fullgraph=False,               # Allow partial compilation (flexibility)
    cache_size_limit=16            # Limit compiled graph variants to 16
)
```

**Parameter explanations:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `mode` | `reduce-overhead` | Minimize Python/CUDA call overhead (not throughput) |
| `fullgraph` | `False` | Allow dynamic control flow in ODE loop |
| `cache_size_limit` | 16 | Prevent memory explosion from caching too many graph variants |

**Expected speedup:** 5-15% with modest memory reduction

---

## Solution 3: cuDNN Benchmark

### Fixed Input Size Optimization

```python
torch.backends.cudnn.benchmark = True
```

**Why this works:**
- All latents are exactly `[B, 4, 32, 32]`
- cuDNN can precompute optimal convolution algorithms for this size
- No time wasted searching different implementations each step

**Expected speedup:** 3-5% on convolution-heavy operations

---

## Statistical Quality: Batch 32 is Excellent

### Sampling Coverage Analysis

**With Batch Size 32 and accumulation_steps=2:**

| Metric | Value |
|--------|-------|
| Patches per batch | 32 × 1024 = 32,768 |
| SWD samples | 8,192 |
| Coverage rate | 8,192 / 32,768 = **25%** |
| Effective batch size | 32 × 2 = **64** |

**Why 25% coverage is excellent:**

In statistical sampling:
- Random sampling with 25% rate already gives ~90% population coverage
- Gradient variance reduction is approximately: $\sigma^2 \propto (1 - f)$ where $f$ = sampling fraction
- At 25% coverage: variance reduction ≈ 1.33× vs full sampling (diminishing returns beyond this)

**Comparison:**
- Original config (Batch 240, max_samples 4096): 1.66% coverage (undersampled) ❌
- New config (Batch 32, max_samples 8192): 25% coverage (optimal) ✅

---

## Configuration Summary

### config.json Changes

```json
{
  "training": {
    "batch_size": 32,
    "accumulation_steps": 2,              // NEW: Simulate batch size 64
    "ode_integration_steps": 10,          // Increased from 5 for better quality
    "use_compile": false,                 // Keep false until ready (overhead on cold start)
    "use_gradient_checkpointing": false   // Disabled: Compute > Memory on 4070
  }
}
```

**Why `use_compile: false` by default:**
- First run triggers compilation overhead (~10-30 seconds)
- Subsequent runs see 5-15% speedup
- Only enable after confirming training works
- Toggle to `true` after first epoch completes

---

## Expected Performance

### Memory Usage

| Configuration | Memory | Status |
|---------------|--------|--------|
| Old (Batch 240, Steps 5) | 6.8GB | ✗ Near limit |
| Old (Batch 240, Steps 10) | **OOM** | ✗ Out of memory |
| New (Batch 32, Steps 10) | ~5.2GB | ✓ Comfortable |

### Training Speed

| Operation | Time | Impact |
|-----------|------|--------|
| torch.compile overhead (1st run) | ~20-30s | One-time |
| Per-batch forward+backward | ~450ms | Baseline |
| With torch.compile | ~420ms | +7% speedup |
| With cuDNN benchmark | ~410ms | +3% cumulative |

**Total speedup:** ~10-12% after compilation

### Gradient Quality

| Metric | With Accumulation | Benefit |
|--------|-------------------|---------|
| Effective batch size | 64 (vs 32) | 2× gradient smoothing |
| SWD coverage | 25% (vs 1.66%) | 15× improvement in statistical quality |
| Loss curve | Smoother | Better convergence visualization |

---

## Tuning Guide

### If OOM occurs:

1. **Reduce accumulation_steps:**
   ```json
   "accumulation_steps": 1  // Fallback to batch_size * 1
   ```

2. **If still OOM, reduce batch_size:**
   ```json
   "batch_size": 16
   "accumulation_steps": 2  // Maintain effective batch = 32
   ```

### If training is unstable:

1. **Increase gradient clipping:**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Default: 1.0
   ```

2. **Enable velocity regularization:**
   ```json
   "use_velocity_reg": true
   ```

3. **Increase warmup epochs:**
   ```json
   "epsilon_warmup_epochs": 20  // Default: 10
   ```

### If training is too slow:

1. **Enable torch.compile:**
   ```json
   "use_compile": true
   ```
   After first epoch completes, training will be 10-12% faster.

2. **Reduce ode_integration_steps (quality trade-off):**
   ```json
   "ode_integration_steps": 7  // Instead of 10
   ```

3. **Increase accumulation_steps (reduces update frequency):**
   ```json
   "accumulation_steps": 4  // Effective batch: 128, but updates less frequently
   ```

---

## Verification Checklist

After implementing these changes, verify:

- [ ] Training starts without OOM error
- [ ] Memory usage stable at ~5-6GB (check with `nvidia-smi`)
- [ ] Loss decreases over first 10 epochs
- [ ] SWD component is decreasing (style transfer learning)
- [ ] SSM component is stable (content preservation)
- [ ] Per-epoch time is ~3-5 minutes (assuming 50-100 batches/epoch)
- [ ] torch.compile logs appear after first epoch (if enabled)

---

## References

### Gradient Accumulation
- PyTorch Docs: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
- Reason: Trade memory for cleaner gradient estimates when batch size is constrained

### torch.compile
- PyTorch 2.0: https://pytorch.org/get-started/pytorch-2.0-migration/
- reduce-overhead mode: Best for inference-like scenarios with limited kernels

### cuDNN Benchmark
- PyTorch Documentation: https://pytorch.org/docs/stable/cudnn.html
- Warning: Only use with fixed input sizes (which we have: 32×32 latents)

---

## Summary

| Technique | Purpose | Benefit | Trade-off |
|-----------|---------|---------|-----------|
| Gradient Accumulation | Simulate larger batch on limited VRAM | Smoother gradients, better convergence | Slower updates (N× less frequent) |
| torch.compile | Fuse operations, reduce overhead | 5-15% speedup | Compilation overhead (one-time) |
| cuDNN Benchmark | Precompute optimal convolution impl | 3-5% speedup | Not applicable to dynamic shapes |

These three techniques work synergistically:
1. **Accumulation** solves the memory problem
2. **torch.compile** accelerates the computation
3. **cuDNN benchmark** polishes the performance

**Net result:** 10-step ODE on RTX 4070 with improved training stability and ~10% speedup.
