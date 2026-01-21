"""
LGT Theory Verification Script
验证潜空间几何热力学 (Latent Geometric Thermodynamics) 的核心假设
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms

def sliced_wasserstein_distance(x, y, num_projections=128):
    """
    计算切片 Wasserstein 距离 (衡量分布差异 / 风格差异)
    x, y: [B, N, C]
    """
    if x.dim() == 2: x = x.unsqueeze(0)
    if y.dim() == 2: y = y.unsqueeze(0)
    
    B, N, C = x.shape
    device = x.device
    
    # 随机投影向量
    projections = torch.randn(C, num_projections, device=device)
    projections = F.normalize(projections, dim=0)
    
    # 投影到 1D
    proj_x = torch.matmul(x, projections)
    proj_y = torch.matmul(y, projections)
    
    # 排序 (Wasserstein 的闭式解仅在一维成立)
    proj_x_sorted, _ = torch.sort(proj_x, dim=1)
    proj_y_sorted, _ = torch.sort(proj_y, dim=1)
    
    # 计算 L2 距离
    return F.mse_loss(proj_x_sorted, proj_y_sorted).item()

def self_similarity_matrix(x):
    """
    计算自相似矩阵 (衡量拓扑结构 / 内容指纹)
    """
    if x.dim() == 2: x = x.unsqueeze(0)
    # 归一化 (消除风格/强度的影响)
    x_norm = F.normalize(x, dim=2)
    # Cosine Similarity Gram Matrix
    gram = torch.bmm(x_norm, x_norm.transpose(1, 2))
    return gram

def ssm_distance(x, y):
    gram_x = self_similarity_matrix(x)
    gram_y = self_similarity_matrix(y)
    return F.mse_loss(gram_x, gram_y).item()

# ==============================================================================
# Phase 1: Simulation Test (数学工具有效性验证)
# ==============================================================================
def run_simulation_test():
    print("="*60)
    print("Phase 1: Simulation Test (Mathematical Validation)")
    print("="*60)
    
    dim = 64
    num_points = 1000
    
    # --- 验证 A: 风格即分布 (SWD Sensitivity) ---
    print("\n[Hypothesis A] SWD should detect Covariance (Texture) changes.")
    
    # 风格 1: 各向同性高斯 (Standard Gaussian)
    style_A = torch.randn(1, num_points, dim)
    
    # 风格 1': 同分布的另一次采样
    style_A_prime = torch.randn(1, num_points, dim)
    
    # 风格 2: 具有强相关性的高斯 (Correlated / Elliptical) - 模拟特定纹理
    scale_vec = torch.ones(dim) * 0.1
    scale_vec[0] = 10.0 # 拉长一个维度
    style_B = torch.randn(1, num_points, dim) * scale_vec
    # 随机旋转混合维度
    rot = torch.qr(torch.randn(dim, dim))[0]
    style_B = torch.mm(style_B.squeeze(0), rot).unsqueeze(0)
    
    d_same = sliced_wasserstein_distance(style_A, style_A_prime)
    d_diff = sliced_wasserstein_distance(style_A, style_B)
    
    print(f"  SWD (Same Style):      {d_same:.6f}")
    print(f"  SWD (Different Style): {d_diff:.6f}")
    print(f"  Ratio (Diff/Same):     {d_diff/max(d_same, 1e-9):.1f}x")
    
    if d_diff > d_same * 10:
        print("  [PASS] SWD effectively captures distribution structure.")
    else:
        print("  [FAIL] SWD failed to distinguish styles.")

    # --- 验证 B: 内容即拓扑 (SSM Robustness) ---
    print("\n[Hypothesis B] SSM should be invariant to Style but sensitive to Structure.")
    
    # 内容 1: 有结构的流形 (如圆环)
    theta = torch.linspace(0, 2*np.pi, num_points).unsqueeze(1)
    circle = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
    content_src = torch.cat([circle, torch.randn(num_points, dim-2)*0.01], dim=1).unsqueeze(0)
    
    # 变换 1: 模拟风格迁移 (全局旋转 + 缩放)
    # 这改变了所有点的坐标值，但没改变它们之间的相对夹角关系
    rot_mat = torch.qr(torch.randn(dim, dim))[0]
    scale_val = 2.5
    content_stylized = torch.mm(content_src.squeeze(0), rot_mat).unsqueeze(0) * scale_val
    
    # 变换 2: 模拟内容破坏 (打乱顺序)
    # 统计分布完全没变，但拓扑结构断裂
    idx = torch.randperm(num_points)
    content_broken = content_src[:, idx, :]
    
    d_ssm_style = ssm_distance(content_src, content_stylized)
    d_ssm_break = ssm_distance(content_src, content_broken)
    
    print(f"  SSM (Style Change):    {d_ssm_style:.6f} (Should be low)")
    print(f"  SSM (Content Break):   {d_ssm_break:.6f} (Should be high)")
    print(f"  Ratio (Break/Style):   {d_ssm_break/max(d_ssm_style, 1e-9):.1f}x")
    
    if d_ssm_break > d_ssm_style * 10:
        print("  [PASS] SSM locks topology while allowing style changes.")
    else:
        print("  [FAIL] SSM metric is not behaving as expected.")

# ==============================================================================
# Phase 2: Real-World Test (真实 VAE 验证)
# ==============================================================================
def load_vae():
    try:
        from diffusers import AutoencoderKL
        print("\nLoading SD VAE...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return vae.to(device)
    except Exception as e:
        print(f"⚠ Diffusers not found or model load failed: {e}")
        print("Skipping Phase 2.")
        return None

def get_latent_features(vae, img_path, device):
    """读取图片 -> VAE Latent -> Patch Features"""
    if not os.path.exists(img_path): 
        return None
    
    try:
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        t_img = transforms.ToTensor()(img).unsqueeze(0).to(device) * 2 - 1
        
        with torch.no_grad():
            latent = vae.encode(t_img).latent_dist.sample() * 0.18215
            
        # Unfold to Patches (3x3)
        # [1, 4, 64, 64] -> [1, 36, N_patches]
        patches = torch.nn.functional.unfold(latent, kernel_size=3, padding=1)
        features = patches.permute(0, 2, 1) # [1, N, 36]
        return features
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def run_real_world_test(style_dirs):
    print("\n" + "="*60)
    print("Phase 2: Real-World Test (VAE Latent Space)")
    print("="*60)
    
    vae = load_vae()
    if vae is None: 
        return

    device = next(vae.parameters()).device
    
    # 按风格加载特征
    style_features = {}
    for style_name, img_paths in style_dirs.items():
        print(f"\n[{style_name}] Loading images...")
        feats = []
        for p in img_paths[:5]:  # 每个风格取最多5张
            f = get_latent_features(vae, p, device)
            if f is not None:
                feats.append(f)
                print(f"  [OK] {os.path.basename(p)}")
        
        if feats:
            style_features[style_name] = torch.cat(feats, dim=1)  # [1, N_total, 36]
            print(f"  Total patches: {style_features[style_name].shape[1]}")
    
    if len(style_features) < 2:
        print("[WARN] Need at least 2 style groups to compare.")
        print("Skipping real-world metric calculation.")
        return

    style_names = list(style_features.keys())
    
    # --- 风格距离验证 (SWD) ---
    print("\n" + "-"*60)
    print("--- Style Distance Verification (SWD) ---")
    print("-"*60)
    
    # 同风格内部距离 (应该小)
    swd_intra = []
    for style_name in style_names:
        feat = style_features[style_name]
        # 分成前后两部分
        mid = feat.shape[1] // 2
        d = sliced_wasserstein_distance(feat[:, :mid, :], feat[:, mid:, :])
        swd_intra.append(d)
        print(f"SWD (Within {style_name}):     {d:.6f}")
    
    # 不同风格之间距离 (应该大)
    if len(style_names) == 2:
        swd_inter = sliced_wasserstein_distance(
            style_features[style_names[0]], 
            style_features[style_names[1]]
        )
        print(f"SWD (Between {style_names[0]} & {style_names[1]}): {swd_inter:.6f}")
        
        avg_intra = np.mean(swd_intra)
        ratio = swd_inter / max(avg_intra, 1e-9)
        print(f"\nRatio (Inter/Intra): {ratio:.2f}x")
        
        if ratio > 1.5:
            print("[PASS] Real images show clear Style Separation!")
        else:
            print("[WARN] Style separation weaker than expected (ratio: {ratio:.2f}x)")

    # --- 内容稳定性验证 (SSM) ---
    print("\n" + "-"*60)
    print("--- Content Stability Verification (SSM) ---")
    print("-"*60)
    
    f0 = style_features[style_names[0]]
    
    # 模拟风格微扰 (Channel Scaling)
    f0_shifted = f0 * (torch.rand(1, 1, f0.shape[2]).to(device) + 0.5)
    # 模拟结构破坏 (Patch Shuffle)
    idx = torch.randperm(f0.shape[1])
    f0_broken = f0[:, idx, :]
    
    d_ssm_shift = ssm_distance(f0, f0_shifted)
    d_ssm_break = ssm_distance(f0, f0_broken)
    
    print(f"SSM (Style Perturbation): {d_ssm_shift:.6f} (Should be low)")
    print(f"SSM (Structure Break):    {d_ssm_break:.6f} (Should be high)")
    
    ratio_ssm = d_ssm_break / max(d_ssm_shift, 1e-9)
    print(f"Ratio (Break/Shift): {ratio_ssm:.2f}x")
    
    if d_ssm_break > d_ssm_shift * 5:
        print("[PASS] Real Latent supports Topological Content Locking!")
    else:
        print("[WARN] SSM behavior weaker than expected.")

def find_test_images():
    """从指定的子目录查找真实风格图片"""
    raw_data_root = Path("/mnt/f/monet/monet2photo")
    
    # 尝试查找 trainA 和 trainB 子目录
    style_dirs = {}
    for style_name in ["trainA", "trainB"]:
        style_path = raw_data_root / style_name
        print(f"\n  Checking path: {style_path}")
        print(f"  Path exists: {style_path.exists()}")
        
        if not style_path.exists():
            continue
        
        try:
            # 列出所有文件
            all_files = list(style_path.iterdir())
            print(f"  Total items in directory: {len(all_files)}")
            
            # 过滤图片文件
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
            images = [str(f) for f in all_files if f.is_file() and f.suffix in valid_exts]
            
            if images:
                style_dirs[style_name] = sorted(images)
                print(f"  [OK] Found {len(images)} images in {style_name}")
            else:
                print(f"  [WARN] No image files found in {style_name}")
                # 打印前几个文件名用于调试
                print(f"  Sample files: {[f.name for f in all_files[:3]]}")
        
        except Exception as e:
            print(f"  [ERROR] Error scanning {style_name}: {e}")
    
    return style_dirs

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LGT (Latent Geometric Thermodynamics) Theory Verification")
    print("="*80)
    
    # 1. 运行模拟验证
    run_simulation_test()
    
    # 2. 加载真实图片并验证
    print("\n[INFO] Scanning for test images in /mnt/f/monet/monet2photo/")
    style_dirs = find_test_images()
    
    if len(style_dirs) > 0:
        print(f"\n[OK] Found {len(style_dirs)} style groups:")
        for style_name, paths in style_dirs.items():
            print(f"   - {style_name}: {len(paths)} images")
        print("\nAttempting real-world validation...")
        run_real_world_test(style_dirs)
    else:
        print("\n[WARN] No test images found at /mnt/f/monet/monet2photo/trainA or trainB")
    
    print("\n" + "="*80)
    print("Verification Complete")
    print("="*80)
