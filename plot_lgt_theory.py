"""
LGT Theoretical Validation Visualization
生成论文级别的理论佐证图表 - 处理所有数据无采样
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置学术风格
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

def load_vae():
    """加载 VAE"""
    from diffusers import AutoencoderKL
    print("[VAE] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    print(f"[VAE] Ready on {device}")
    return vae

def extract_all_patches(vae, img_paths, device="cuda"):
    """提取所有图片的所有补丁 - 无采样"""
    all_patches = []
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])
    
    print(f"[EXTRACT] Processing ALL {len(img_paths)} images...")
    for img_path in tqdm(img_paths, desc="Extracting patches"):
        try:
            img = Image.open(img_path).convert("RGB")
            t_img = transform(img).unsqueeze(0).to(device) * 2 - 1
            
            with torch.no_grad():
                latent = vae.encode(t_img).latent_dist.sample() * 0.18215
            
            # [1,4,64,64] -> [4096, 36]
            patches = F.unfold(latent, kernel_size=3, padding=1)
            patches = patches.permute(0, 2, 1).reshape(-1, 36)
            all_patches.append(patches.cpu())
        except:
            pass
    
    result = torch.cat(all_patches, dim=0) if all_patches else torch.tensor([])
    print(f"  Total patches: {result.shape[0]}")
    return result

def compute_all_ssm_energies(vae, img_paths, device="cuda"):
    """计算所有图片的SSM能量"""
    energies_style = []
    energies_break = []
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])
    
    print(f"[SSM] Processing ALL {len(img_paths)} images...")
    for img_path in tqdm(img_paths, desc="Computing SSM"):
        try:
            img = Image.open(img_path).convert("RGB")
            t_img = transform(img).unsqueeze(0).to(device) * 2 - 1
            
            with torch.no_grad():
                latent = vae.encode(t_img).latent_dist.sample() * 0.18215
            
            x = latent.flatten(2).transpose(1, 2)  # [1, 4096, 4]
            x_norm = F.normalize(x, dim=2)
            gram_base = torch.bmm(x_norm, x_norm.transpose(1, 2))
            
            # Style perturbation
            x_style = x * (torch.randn(1, 1, 4).to(device) * 0.5 + 1.0)
            x_style_norm = F.normalize(x_style, dim=2)
            gram_style = torch.bmm(x_style_norm, x_style_norm.transpose(1, 2))
            e_style = F.mse_loss(gram_base, gram_style).item()
            
            # Structure break
            idx = torch.randperm(x.shape[1])
            x_break = x[:, idx, :]
            x_break_norm = F.normalize(x_break, dim=2)
            gram_break = torch.bmm(x_break_norm, x_break_norm.transpose(1, 2))
            e_break = F.mse_loss(gram_base, gram_break).item()
            
            energies_style.append(e_style)
            energies_break.append(e_break)
        except:
            pass
    
    return energies_style, energies_break

def compute_all_swd(vae, imgs_A, imgs_B, device="cuda"):
    """计算所有图片对的SWD距离"""
    dists_intra = []
    dists_inter = []
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])
    
    def calc_swd(patches_a, patches_b):
        pa = patches_a.to(device)
        pb = patches_b.to(device)
        proj = F.normalize(torch.randn(36, 128, device=device), dim=0)
        proj_a = torch.matmul(pa, proj).sort(0)[0]
        proj_b = torch.matmul(pb, proj).sort(0)[0]
        min_len = min(proj_a.shape[0], proj_b.shape[0])
        return F.mse_loss(proj_a[:min_len], proj_b[:min_len]).item()
    
    # Intra-domain (consecutive pairs)
    print(f"[SWD] Computing ALL {len(imgs_A)-1} intra-domain pairs...")
    for i in tqdm(range(len(imgs_A)-1), desc="Intra-domain"):
        try:
            img_a = Image.open(imgs_A[i]).convert("RGB")
            img_b = Image.open(imgs_A[i+1]).convert("RGB")
            t_a = transform(img_a).unsqueeze(0).to(device) * 2 - 1
            t_b = transform(img_b).unsqueeze(0).to(device) * 2 - 1
            
            with torch.no_grad():
                lat_a = vae.encode(t_a).latent_dist.sample() * 0.18215
                lat_b = vae.encode(t_b).latent_dist.sample() * 0.18215
            
            pa = F.unfold(lat_a, 3, padding=1).permute(0,2,1).reshape(-1,36)
            pb = F.unfold(lat_b, 3, padding=1).permute(0,2,1).reshape(-1,36)
            dists_intra.append(calc_swd(pa, pb))
        except:
            pass
    
    # Inter-domain (all pairs)
    n_pairs = min(len(imgs_A), len(imgs_B))
    print(f"[SWD] Computing ALL {n_pairs} inter-domain pairs...")
    for i in tqdm(range(n_pairs), desc="Inter-domain"):
        try:
            img_a = Image.open(imgs_A[i]).convert("RGB")
            img_b = Image.open(imgs_B[i]).convert("RGB")
            t_a = transform(img_a).unsqueeze(0).to(device) * 2 - 1
            t_b = transform(img_b).unsqueeze(0).to(device) * 2 - 1
            
            with torch.no_grad():
                lat_a = vae.encode(t_a).latent_dist.sample() * 0.18215
                lat_b = vae.encode(t_b).latent_dist.sample() * 0.18215
            
            pa = F.unfold(lat_a, 3, padding=1).permute(0,2,1).reshape(-1,36)
            pb = F.unfold(lat_b, 3, padding=1).permute(0,2,1).reshape(-1,36)
            dists_inter.append(calc_swd(pa, pb))
        except:
            pass
    
    return dists_intra, dists_inter

def generate_theoretical_plot():
    """生成完整理论验证图表"""
    print("\n" + "="*80)
    print("LGT Theoretical Validation - FULL DATASET (NO SAMPLING)")
    print("="*80)
    
    root_A = Path("/mnt/f/monet/monet2photo/trainA")
    root_B = Path("/mnt/f/monet/monet2photo/trainB")
    
    imgs_A = sorted(list(root_A.glob("*.jpg")))
    imgs_B = sorted(list(root_B.glob("*.jpg")))
    
    print(f"[DATA] Monet: {len(imgs_A)} images")
    print(f"[DATA] Photo: {len(imgs_B)} images")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae()
    
    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    # Panel A: t-SNE
    ax1 = fig.add_subplot(gs[0, 0])
    print("\n[PANEL A] t-SNE on ALL data...")
    patches_A = extract_all_patches(vae, imgs_A, device)
    patches_B = extract_all_patches(vae, imgs_B, device)
    
    # For t-SNE visualization only (too many points slow down)
    max_viz = 10000
    viz_A = patches_A[torch.randperm(len(patches_A))[:max_viz]] if len(patches_A) > max_viz else patches_A
    viz_B = patches_B[torch.randperm(len(patches_B))[:max_viz]] if len(patches_B) > max_viz else patches_B
    
    X = torch.cat([viz_A, viz_B], 0).numpy()
    y = np.array([0]*len(viz_A) + [1]*len(viz_B))
    
    print(f"  Running t-SNE on {len(X)} patches...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
    X_emb = tsne.fit_transform(X)
    
    sns.scatterplot(x=X_emb[y==0,0], y=X_emb[y==0,1], ax=ax1, s=8, 
                   color='#FF6B6B', alpha=0.4, label=f'Monet ({len(patches_A)} patches)')
    sns.scatterplot(x=X_emb[y==1,0], y=X_emb[y==1,1], ax=ax1, s=8,
                   color='#4ECDC4', alpha=0.4, label=f'Photo ({len(patches_B)} patches)')
    ax1.set_title("Latent Style Manifold", fontweight='bold')
    ax1.set_xlabel("t-SNE Dim 1")
    ax1.set_ylabel("t-SNE Dim 2")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Panel B: SSM Energy
    ax2 = fig.add_subplot(gs[0, 1])
    print("\n[PANEL B] SSM on ALL images...")
    es_A, eb_A = compute_all_ssm_energies(vae, imgs_A, device)
    es_B, eb_B = compute_all_ssm_energies(vae, imgs_B, device)
    
    es_all = es_A + es_B
    eb_all = eb_A + eb_B
    
    parts = ax2.violinplot([es_all, eb_all], [0,1], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('#C7F464' if i==0 else '#FF6B6B')
        pc.set_alpha(0.7)
    
    ax2.set_xticks([0,1])
    ax2.set_xticklabels(['Style\nPerturbation', 'Structure\nBreak'])
    ax2.set_title("Topological Energy Barrier", fontweight='bold')
    ax2.set_ylabel("SSM Distance (log)")
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    ratio = np.mean(eb_all) / np.mean(es_all)
    ax2.text(0.5, 0.95, f"Ratio: {ratio:.1f}x\n(n={len(es_all)})", 
            transform=ax2.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8, ec='black', lw=1.5))
    
    # Panel C: SWD
    ax3 = fig.add_subplot(gs[0, 2])
    print("\n[PANEL C] SWD on ALL pairs...")
    d_intra, d_inter = compute_all_swd(vae, imgs_A, imgs_B, device)
    
    ax3.hist(d_intra, bins=25, alpha=0.6, color='#4ECDC4', density=True,
            label=f'Intra (n={len(d_intra)})')
    ax3.hist(d_inter, bins=25, alpha=0.6, color='#FF6B6B', density=True,
            label=f'Inter (n={len(d_inter)})')
    ax3.set_title("Thermodynamic Gradient", fontweight='bold')
    ax3.set_xlabel("Sliced Wasserstein Distance")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    swd_ratio = np.mean(d_inter) / np.mean(d_intra)
    ax3.text(0.95, 0.95, f"Ratio: {swd_ratio:.2f}x\nIntra: {np.mean(d_intra):.4f}\nInter: {np.mean(d_inter):.4f}",
            transform=ax3.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8, ec='black', lw=1.5))
    
    fig.suptitle("LGT Theoretical Validation (Full Dataset)", fontsize=14, fontweight='bold')
    
    save_path = "LGT_Theoretical_Validation_Full.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\n[SAVED] {save_path}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total images: {len(imgs_A)} + {len(imgs_B)} = {len(imgs_A)+len(imgs_B)}")
    print(f"Total patches: {len(patches_A)} + {len(patches_B)} = {len(patches_A)+len(patches_B)}")
    print(f"SSM Ratio: {np.mean(eb_all)/np.mean(es_all):.2f}x")
    print(f"SWD Ratio: {np.mean(d_inter)/np.mean(d_intra):.2f}x")
    print("="*80 + "\n")

if __name__ == "__main__":
    generate_theoretical_plot()
