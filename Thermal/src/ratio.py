import torch
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 引入你的数据集类
from train import InMemoryLatentDataset

def setup_plot_style():
    """配置科研级绘图风格"""
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    # 使用高对比度色板，确保 4 种风格一眼能分清
    sns.set_palette("husl", 4)

def get_random_batch(dataset, style_id, batch_size=128, device='cuda'):
    """高效获取指定风格的随机 Batch"""
    # 直接操作 GPU Tensor 索引，避免 CPU-GPU 同步开销
    indices = (dataset.styles_tensor == style_id).nonzero(as_tuple=True)[0]
    if len(indices) == 0: return None
    
    # 如果数据不够，允许重复采样 (Replacement)
    cnt = len(indices)
    if cnt < batch_size:
        rand_idx = indices[torch.randint(cnt, (batch_size,), device=device)]
    else:
        rand_idx = indices[torch.randint(cnt, (batch_size,), device=device)]
        
    return dataset.latents_tensor[rand_idx]

def project_and_sort(batch, patch_size, theta, device='cuda'):
    """
    SWD 核心机理的可视化实现：
    1. Unfold (提取 Patch)
    2. Normalize (去均值/去亮度，只留纹理)
    3. Project (投影到随机方向)
    4. Sort (计算 CDF)
    """
    with torch.no_grad():
        B, C, H, W = batch.shape
        
        # --- 1. Unfold (提取局部特征) ---
        if patch_size == 1:
            # 1x1 Patch 特殊优化：不需要 unfold，直接 reshape
            # [B, C, H, W] -> [B*H*W, C]
            flat = batch.permute(0, 2, 3, 1).reshape(-1, C)
        else:
            # [B, C, H, W] -> [B, C*K*K, N_patches]
            patches = F.unfold(batch.float(), kernel_size=patch_size, padding=patch_size//2)
            # [B, N, Feat]
            patches = patches.transpose(1, 2)
            
            # --- 2. Mean Removal (关键步骤) ---
            # 对于 Patch > 1，我们减去 Patch 均值。
            # 物理意义：我们不关心"这个Patch有多亮"，只关心"这个Patch内部的纹理对比度"。
            # 这让 SWD 专注于纹理结构，而不是被亮度带偏。
            patches = patches - patches.mean(dim=2, keepdim=True)
            flat = patches.reshape(-1, patches.shape[2])

        # --- 3. Project (Radon Transform 采样) ---
        # 投影到共享的随机向量 theta 上
        # flat: [N_total, FeatDim], theta: [FeatDim, 1] -> proj: [N_total, 1]
        proj = flat @ theta 

        # --- 4. Sort (计算经验分布函数 CDF) ---
        sorted_proj, _ = torch.sort(proj.view(-1))
        
        # 下采样以方便绘图 (保留 2000 个点足够画出平滑曲线)
        if len(sorted_proj) > 2000:
            idx = torch.linspace(0, len(sorted_proj)-1, 2000, device=device).long()
            sorted_proj = sorted_proj[idx]
            
        return sorted_proj.cpu().numpy()

def plot_multiscale_mechanism(dataset, style_names, save_path, device='cuda'):
    """
    绘制 4 合 1 的多尺度分析图
    """
    print("Generating Multi-Scale SWD Mechanism Plot...")
    
    # 定义我们要分析的 4 个物理尺度
    scales_config = [
        (1, "Patch 1x1 (Color/Tone)", "Focus: Pixel Colors"),
        (3, "Patch 3x3 (Micro-Texture)", "Focus: Brush Strokes / Noise"),
        (5, "Patch 5x5 (Mid-Structure)", "Focus: Patterns"),
        (7, "Patch 7x7 (Macro-Geometry)", "Focus: Shapes / Warping")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    latent_channels = 4 # SD VAE Latent Channels
    
    for ax, (p_size, title, subtitle) in zip(axes, scales_config):
        print(f"  Analyzing {title}...")
        
        # 1. 生成【共享】投影向量
        # 控制变量法：所有风格必须投影到同一个随机方向，比较才有意义
        feature_dim = latent_channels * p_size * p_size
        theta = torch.randn(feature_dim, 1, device=device)
        theta = theta / theta.norm()
        
        # 2. 遍历所有风格
        for i, name in enumerate(style_names):
            # 显存优化：Patch 越大，Unfold 膨胀越厉害，Batch Size 必须减小
            # Patch 7x7 会把数据膨胀 49 倍！
            bs = 256 if p_size <= 3 else (128 if p_size == 5 else 64)
            
            batch = get_random_batch(dataset, i, batch_size=bs, device=device)
            if batch is not None:
                y = project_and_sort(batch, p_size, theta, device)
                # X 轴归一化为 0~1 (Quantile)
                x = np.linspace(0, 1, len(y))
                
                # 绘制 CDF 曲线
                ax.plot(x, y, label=name, linewidth=2.5, alpha=0.85)
        
        # 图表美化
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Quantile (Probability)", fontsize=10)
        ax.set_ylabel("Projected Value", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 在图内添加说明
        ax.text(0.05, 0.95, subtitle, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 只在第一个子图显示图例，避免遮挡
        if p_size == 1:
            ax.legend(loc='lower right', frameon=True, fontsize=12)

    plt.suptitle(f"LGT Multi-Scale Texture Separation Analysis\n(Why different patch sizes matter)", fontsize=18, y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    save_file = save_path / "swd_multiscale_analysis.png"
    plt.savefig(save_file, dpi=300)
    print(f"✓ Analysis saved to {save_file}")
    plt.close()

def main():
    config_path = 'config.json'
    if not Path(config_path).exists():
        print("Config not found.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载数据集
    dataset = InMemoryLatentDataset(
        data_root=config['data']['data_root'],
        num_styles=config['model']['num_styles'],
        style_subdirs=config['data']['style_subdirs'],
        device=device
    )
    
    # 简单的 Cache 补丁 (如果类定义没有 Cache)
    if not hasattr(dataset, 'style_indices_cache'):
        pass # Helper 函数直接用 tensor 索引，不需要 Python dict cache

    output_dir = Path("analysis_plots")
    output_dir.mkdir(exist_ok=True)
    setup_plot_style()
    
    plot_multiscale_mechanism(
        dataset, 
        config['data']['style_subdirs'], 
        output_dir, 
        device
    )

if __name__ == "__main__":
    main()