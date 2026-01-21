"""
快速测试脚本：验证模型架构和数据流
"""

import torch
import json
from model import create_model, IsoNext


def test_model_architecture():
    """测试模型架构"""
    print("="*80)
    print("Testing Model Architecture")
    print("="*80)
    
    # 加载配置
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 创建模型
    model = create_model(config)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model created successfully")
    print(f"  - Total parameters: {total_params / 1e6:.2f}M")
    print(f"  - Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    batch_size = 4
    latent_size = 64  # 512 / 8
    
    x = torch.randn(batch_size, 4, latent_size, latent_size)
    t = torch.rand(batch_size)
    style_id = torch.randint(0, config['model']['num_styles'], (batch_size,))
    
    print(f"\nTesting forward pass...")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Time shape: {t.shape}")
    print(f"  - Style ID shape: {style_id.shape}")
    
    with torch.no_grad():
        output = model(x, t, style_id)
    
    print(f"  - Output shape: {output.shape}")
    
    assert output.shape == x.shape, "Output shape mismatch!"
    print(f"\n✓ Forward pass successful")
    
    # 测试 CFG
    print(f"\nTesting Classifier-Free Guidance...")
    
    # Handle both new and old model APIs
    if hasattr(model, 'get_null_style_id'):
        null_id = model.get_null_style_id(x.device)
    elif hasattr(model, 'null_style_id'):
        null_id = model.null_style_id
    else:
        # Fallback: use num_styles as null token
        null_id = torch.tensor([config['model']['num_styles']], device=x.device)
    
    print(f"  - Null token ID: {null_id.item()}")
    
    with torch.no_grad():
        v_cond = model(x, t, style_id)
        v_uncond = model(x, t, null_id.expand(batch_size))
    
    print(f"  - Conditional output: {v_cond.shape}")
    print(f"  - Unconditional output: {v_uncond.shape}")
    print(f"\n✓ CFG test successful")
    
    return model


def test_memory_usage():
    """测试显存占用"""
    print("\n" + "="*80)
    print("Testing Memory Usage")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    device = torch.device('cuda')
    
    # 加载配置
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 创建模型
    model = create_model(config).to(device, memory_format=torch.channels_last)
    
    # 记录模型显存
    model_memory = torch.cuda.memory_allocated() / 1e9
    print(f"\n✓ Model loaded to GPU")
    print(f"  - Model memory: {model_memory:.3f} GB")
    
    # 测试不同 batch size
    batch_sizes = [16, 32, 64, 128]
    latent_size = 64
    
    print(f"\nTesting batch sizes (latent_size={latent_size}):")
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            
            # 创建输入
            x = torch.randn(batch_size, 4, latent_size, latent_size, 
                          device=device, memory_format=torch.channels_last)
            t = torch.rand(batch_size, device=device)
            style_id = torch.randint(0, 2, (batch_size,), device=device)
            
            # 前向传播
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = model(x, t, style_id)
                loss = output.mean()
            
            # 反向传播
            loss.backward()
            
            # 记录显存
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            current_memory = torch.cuda.memory_allocated() / 1e9
            
            print(f"  - Batch {batch_size:3d}: Peak {peak_memory:.3f} GB, Current {current_memory:.3f} GB ✓")
            
            # 清理
            del x, t, style_id, output, loss
            model.zero_grad()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  - Batch {batch_size:3d}: OOM ✗")
                break
            else:
                raise e
    
    # 清理
    del model
    torch.cuda.empty_cache()


def test_otcfm_loss():
    """测试 OT-CFM 损失计算"""
    print("\n" + "="*80)
    print("Testing OT-CFM Loss Computation")
    print("="*80)
    
    # 加载配置
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 创建模型
    model = create_model(config)
    model.eval()
    
    # 模拟数据
    batch_size = 8
    x1 = torch.randn(batch_size, 4, 64, 64)  # 真实数据
    style_id = torch.randint(0, 2, (batch_size,))
    
    print(f"\nSimulating OT-CFM training step...")
    print(f"  - x1 (data) shape: {x1.shape}")
    print(f"  - style_id shape: {style_id.shape}")
    
    # 采样 x0 (结构基底)
    x0 = torch.randn_like(x1)
    print(f"  - x0 (noise) shape: {x0.shape}")
    
    # 采样时间
    t = torch.rand(batch_size)
    print(f"  - t (time) shape: {t.shape}, range: [{t.min():.3f}, {t.max():.3f}]")
    
    # 构造路径
    t_expanded = t[:, None, None, None]
    x_t = (1 - t_expanded) * x0 + t_expanded * x1
    print(f"  - x_t (path) shape: {x_t.shape}")
    
    # 目标速度场
    u_t = x1 - x0
    print(f"  - u_t (target) shape: {u_t.shape}")
    
    # 模型预测
    with torch.no_grad():
        v_pred = model(x_t, t, style_id)
    print(f"  - v_pred (prediction) shape: {v_pred.shape}")
    
    # 计算损失
    loss = torch.nn.functional.mse_loss(v_pred, u_t)
    print(f"  - Loss: {loss.item():.6f}")
    
    print(f"\n✓ OT-CFM loss computation successful")


def test_channels_last():
    """测试 channels_last 内存格式"""
    print("\n" + "="*80)
    print("Testing Channels Last Memory Format")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping channels_last test")
        return
    
    device = torch.device('cuda')
    
    # 加载配置
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 创建模型
    model_default = create_model(config).to(device)
    model_channels_last = create_model(config).to(device, memory_format=torch.channels_last)
    
    # 测试输入
    batch_size = 32
    x_default = torch.randn(batch_size, 4, 64, 64, device=device)
    x_channels_last = torch.randn(batch_size, 4, 64, 64, 
                                   device=device, memory_format=torch.channels_last)
    t = torch.rand(batch_size, device=device)
    style_id = torch.randint(0, 2, (batch_size,), device=device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model_default(x_default, t, style_id)
            _ = model_channels_last(x_channels_last, t, style_id)
    
    torch.cuda.synchronize()
    
    # Benchmark default
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model_default(x_default, t, style_id)
    torch.cuda.synchronize()
    time_default = time.time() - start
    
    # Benchmark channels_last
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model_channels_last(x_channels_last, t, style_id)
    torch.cuda.synchronize()
    time_channels_last = time.time() - start
    
    print(f"\nBenchmark results (100 iterations):")
    print(f"  - Default format: {time_default:.3f}s ({time_default/100*1000:.2f}ms per iter)")
    print(f"  - Channels last:  {time_channels_last:.3f}s ({time_channels_last/100*1000:.2f}ms per iter)")
    print(f"  - Speedup: {time_default/time_channels_last:.2f}x")
    
    if time_channels_last < time_default:
        print(f"\n✓ Channels last is faster!")
    else:
        print(f"\n⚠ Channels last is slower (might be due to GPU specifics)")


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print(" OT-CFM Model Test Suite")
    print("="*80 + "\n")
    
    try:
        # 测试 1: 模型架构
        model = test_model_architecture()
        
        # 测试 2: OT-CFM 损失
        test_otcfm_loss()
        
        # 测试 3: 显存占用 (仅 GPU)
        if torch.cuda.is_available():
            test_memory_usage()
            test_channels_last()
        else:
            print("\n⚠ CUDA not available, skipping GPU tests")
        
        print("\n" + "="*80)
        print("✓ All tests passed!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
