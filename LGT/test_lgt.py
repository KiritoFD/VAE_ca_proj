"""
Test script to verify LGT implementation.
Run this to check if all components work correctly.
"""

import torch
import sys
from pathlib import Path

def test_model():
    """Test model forward pass."""
    print("\n" + "="*80)
    print("Testing Model Architecture")
    print("="*80)
    
    from model import LGTUNet, count_parameters
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = LGTUNet(
        latent_channels=4,
        base_channels=128,
        style_dim=256,
        time_dim=256,
        num_styles=2,
        num_encoder_blocks=2,
        num_decoder_blocks=3
    ).to(device)
    
    model.compute_avg_style_embedding()
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 32, 32, device=device)
    t = torch.rand(batch_size, device=device)
    style_id = torch.randint(0, 2, (batch_size,), device=device)
    
    with torch.no_grad():
        v_cond = model(x, t, style_id, use_avg_style=False)
        v_uncond = model(x, t, style_id, use_avg_style=True)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Conditional output shape: {v_cond.shape}")
    print(f"✓ Unconditional output shape: {v_uncond.shape}")
    print("✓ Model test PASSED")
    
    return True


def test_losses():
    """Test loss functions."""
    print("\n" + "="*80)
    print("Testing Loss Functions")
    print("="*80)
    
    from losses import (
        PatchSlicedWassersteinLoss,
        CosineSSMLoss,
        GeometricFreeEnergyLoss
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dummy data
    B, C, H, W = 4, 4, 32, 32
    x_pred = torch.randn(B, C, H, W, device=device)
    x_style = torch.randn(B, C, H, W, device=device)
    x_src = torch.randn(B, C, H, W, device=device)
    
    # Test Patch-SWD
    print("\nTesting Patch-SWD Loss...")
    swd_loss = PatchSlicedWassersteinLoss().to(device)
    swd_value = swd_loss(x_pred, x_style)
    print(f"  ✓ SWD Loss: {swd_value.item():.6f}")
    assert swd_value.item() >= 0, "SWD loss should be non-negative"
    
    # Test Cosine-SSM
    print("\nTesting Cosine-SSM Loss...")
    ssm_loss = CosineSSMLoss().to(device)
    ssm_value = ssm_loss(x_pred, x_src)
    print(f"  ✓ SSM Loss: {ssm_value.item():.6f}")
    assert ssm_value.item() >= 0, "SSM loss should be non-negative"
    
    # Test Geometric Free Energy
    print("\nTesting Geometric Free Energy Loss...")
    energy_loss = GeometricFreeEnergyLoss(w_style=1.0, w_content=1.0).to(device)
    loss_dict = energy_loss(x_pred, x_style, x_src)
    print(f"  ✓ Total Energy: {loss_dict['total'].item():.6f}")
    print(f"  ✓ Style SWD: {loss_dict['style_swd'].item():.6f}")
    print(f"  ✓ Content SSM: {loss_dict['content_ssm'].item():.6f}")
    
    print("✓ Loss functions test PASSED")
    
    return True


def test_inference():
    """Test inference components."""
    print("\n" + "="*80)
    print("Testing Inference Pipeline")
    print("="*80)
    
    from inference import LangevinSampler
    from model import LGTUNet
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = LGTUNet(
        latent_channels=4,
        base_channels=64,  # Smaller for testing
        style_dim=256,
        time_dim=256,
        num_styles=2,
        num_encoder_blocks=1,
        num_decoder_blocks=1
    ).to(device)
    
    model.compute_avg_style_embedding()
    model.eval()
    
    # Create sampler
    sampler = LangevinSampler(
        temperature_lambda=0.1,
        temperature_threshold=0.5,
        use_cfg=True,
        cfg_scale=2.0
    )
    
    # Test sampling
    print("\nTesting Langevin sampling...")
    x_init = torch.randn(1, 4, 32, 32, device=device)
    style_id = 0
    
    with torch.no_grad():
        x_final = sampler.sample(
            model,
            x_init,
            style_id,
            num_steps=5  # Few steps for testing
        )
    
    print(f"  ✓ Initial shape: {x_init.shape}")
    print(f"  ✓ Final shape: {x_final.shape}")
    assert x_final.shape == x_init.shape, "Output shape mismatch"
    
    # Test temperature schedule
    print("\nTesting temperature schedule...")
    sigma_early = sampler.get_temperature(0.3)
    sigma_late = sampler.get_temperature(0.7)
    print(f"  ✓ Temperature at t=0.3: {sigma_early}")
    print(f"  ✓ Temperature at t=0.7: {sigma_late}")
    assert sigma_early == 0.0, "Early temperature should be 0"
    assert sigma_late == 0.1, "Late temperature should be λ"
    
    print("✓ Inference test PASSED")
    
    return True


def test_data_loading():
    """Test data preprocessing utilities."""
    print("\n" + "="*80)
    print("Testing Data Utilities")
    print("="*80)
    
    import numpy as np
    from PIL import Image
    from preprocess_latents import preprocess_image
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(dummy_img)
    
    # Save temporarily
    temp_path = Path("temp_test_image.jpg")
    img.save(temp_path)
    
    try:
        # Test preprocessing
        img_tensor = preprocess_image(temp_path, target_size=256)
        
        print(f"✓ Preprocessed image shape: {img_tensor.shape}")
        print(f"✓ Value range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
        
        assert img_tensor.shape == (3, 256, 256), "Wrong output shape"
        assert img_tensor.min() >= -1.0 and img_tensor.max() <= 1.0, "Wrong value range"
        
        print("✓ Data utilities test PASSED")
        
        return True
    
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("LGT Implementation Test Suite")
    print("="*80)
    
    tests = [
        ("Model Architecture", test_model),
        ("Loss Functions", test_losses),
        ("Inference Pipeline", test_inference),
        ("Data Utilities", test_data_loading),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result, None))
        except Exception as e:
            print(f"\n✗ {name} FAILED with error:")
            print(f"  {type(e).__name__}: {e}")
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! LGT implementation is ready.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
