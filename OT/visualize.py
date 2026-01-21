"""
训练可视化和监控工具
用于实时监控训练进度、损失曲线和生成样本
"""

import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image


class TrainingVisualizer:
    """训练可视化器"""
    
    def __init__(self, log_dir="training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.losses = []
        self.epochs = []
    
    def log_loss(self, epoch, loss):
        """记录损失"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        # 保存到文件
        log_file = self.log_dir / "losses.txt"
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{loss}\n")
    
    def plot_losses(self, save_path=None):
        """绘制损失曲线"""
        if len(self.losses) == 0:
            print("No losses to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.losses, 'b-', linewidth=2, label='Training Loss')
        
        # 添加平滑曲线
        if len(self.losses) > 10:
            window_size = min(10, len(self.losses) // 10)
            smoothed = np.convolve(self.losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_epochs = self.epochs[:len(smoothed)]
            plt.plot(smoothed_epochs, smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.log_dir / "loss_curve.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"✓ Loss curve saved to {save_path}")
    
    def load_losses(self):
        """从文件加载损失历史"""
        log_file = self.log_dir / "losses.txt"
        if not log_file.exists():
            return
        
        self.epochs = []
        self.losses = []
        
        with open(log_file, 'r') as f:
            for line in f:
                if ',' in line:
                    epoch, loss = line.strip().split(',')
                    self.epochs.append(int(epoch))
                    self.losses.append(float(loss))
        
        print(f"✓ Loaded {len(self.losses)} loss records from {log_file}")


def analyze_checkpoint(checkpoint_path):
    """分析 checkpoint 信息"""
    print(f"Analyzing checkpoint: {checkpoint_path}")
    print("="*80)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 基本信息
    print(f"\nBasic Info:")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # 模型信息
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"  - Total parameters: {total_params / 1e6:.2f}M")
        print(f"  - Number of layers: {len(state_dict)}")
    
    # 优化器信息
    if 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        print(f"\nOptimizer State:")
        print(f"  - Learning rate: {opt_state['param_groups'][0]['lr']:.2e}")
        print(f"  - Weight decay: {opt_state['param_groups'][0].get('weight_decay', 0):.2e}")
    
    # 配置信息
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\nModel Config:")
        for key, value in config['model'].items():
            print(f"  - {key}: {value}")
    
    print("="*80)


def visualize_latents(latent_dir, num_samples=16, save_path="latent_visualization.png"):
    """可视化 latent 分布"""
    latent_dir = Path(latent_dir)
    
    # 收集所有 latent 文件
    latent_files = list(latent_dir.rglob("*.pt"))
    
    if len(latent_files) == 0:
        print(f"No latent files found in {latent_dir}")
        return
    
    # 随机采样
    np.random.seed(42)
    sample_files = np.random.choice(latent_files, min(num_samples, len(latent_files)), replace=False)
    
    # 加载并可视化
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, fpath in enumerate(sample_files):
        latent = torch.load(fpath)  # [4, H, W]
        
        # 可视化第一个通道
        img = latent[0].numpy()
        
        axes[idx].imshow(img, cmap='viridis')
        axes[idx].axis('off')
        axes[idx].set_title(f"{fpath.parent.name}/{fpath.stem}", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Latent visualization saved to {save_path}")
    
    # 统计信息
    all_latents = []
    for fpath in latent_files[:100]:  # 采样100个计算统计
        latent = torch.load(fpath)
        all_latents.append(latent)
    
    all_latents = torch.stack(all_latents)
    
    print(f"\nLatent Statistics (n={len(all_latents)}):")
    print(f"  - Shape: {all_latents[0].shape}")
    print(f"  - Mean: {all_latents.mean():.4f}")
    print(f"  - Std: {all_latents.std():.4f}")
    print(f"  - Min: {all_latents.min():.4f}")
    print(f"  - Max: {all_latents.max():.4f}")


def compare_checkpoints(ckpt_paths, save_path="checkpoint_comparison.png"):
    """比较多个 checkpoint 的性能"""
    results = []
    
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt.get('epoch', 0)
        
        # 这里可以添加更多指标
        results.append({
            'path': Path(ckpt_path).name,
            'epoch': epoch,
        })
    
    # 绘制比较图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = [r['epoch'] for r in results]
    names = [r['path'] for r in results]
    
    ax.bar(names, epochs)
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('Epoch')
    ax.set_title('Checkpoint Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"✓ Checkpoint comparison saved to {save_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Training Visualization Tools')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['plot_losses', 'analyze_ckpt', 'visualize_latents', 'compare_ckpts'],
                       help='Action to perform')
    parser.add_argument('--log_dir', type=str, default='training_logs', help='Log directory')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
    parser.add_argument('--checkpoints', type=str, nargs='+', help='Multiple checkpoint paths')
    parser.add_argument('--latent_dir', type=str, help='Latent directory')
    parser.add_argument('--output', type=str, help='Output path')
    
    args = parser.parse_args()
    
    if args.action == 'plot_losses':
        viz = TrainingVisualizer(args.log_dir)
        viz.load_losses()
        viz.plot_losses(args.output)
    
    elif args.action == 'analyze_ckpt':
        if args.checkpoint is None:
            print("Error: --checkpoint required")
            return
        analyze_checkpoint(args.checkpoint)
    
    elif args.action == 'visualize_latents':
        if args.latent_dir is None:
            print("Error: --latent_dir required")
            return
        visualize_latents(args.latent_dir, save_path=args.output or "latent_visualization.png")
    
    elif args.action == 'compare_ckpts':
        if args.checkpoints is None:
            print("Error: --checkpoints required")
            return
        compare_checkpoints(args.checkpoints, save_path=args.output or "checkpoint_comparison.png")


if __name__ == "__main__":
    # 示例用法
    print("Training Visualization Tools")
    print("="*80)
    print("\nUsage examples:")
    print("  1. Plot loss curve:")
    print("     python visualize.py --action plot_losses --log_dir training_logs")
    print("\n  2. Analyze checkpoint:")
    print("     python visualize.py --action analyze_ckpt --checkpoint checkpoints/stage1_epoch50.pt")
    print("\n  3. Visualize latents:")
    print("     python visualize.py --action visualize_latents --latent_dir data_root")
    print("\n  4. Compare checkpoints:")
    print("     python visualize.py --action compare_ckpts --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt")
    print("="*80)
    
    main()
