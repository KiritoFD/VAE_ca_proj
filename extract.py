import torch
import torch.nn.functional as F
import os
import glob
import numpy as np
from tqdm import tqdm
import gc

# ================= ⚙️ 配置区 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = r"G:\GitHub\VAE_ca_proj\wikiart_latents"
SAVE_DIR = "./output/svd_bases"
TARGET_STYLES = ["Impressionism", "Ukiyo_e", "Cubism", "Art_Nouveau"]

# 物理参数
PATCH_SIZE = 32
LATENT_DIM = 4  
VECTOR_DIM = LATENT_DIM * PATCH_SIZE * PATCH_SIZE # 4 * 32 * 32 = 4096
NUM_COMPONENTS = 4096 # 保留的主成分数量 (不超过 4096)

# 步长：决定采样密度。
# 32的Patch，设为8表示有75%的重叠，保证数据利用率极高。
# 如果设为32则是不重叠采样。建议 4-16。
STRIDE = 8 

# 显存控制：攒够多少个 Patch 进行一次 GPU 计算
# 10000 * 4096 * 4bytes ≈ 160MB 显存，非常安全
GPU_BATCH_SIZE = 80000 
# ============================================

class TorchIncrementalPCA:
    """
    纯 PyTorch 实现的 GPU 增量 PCA
    支持无限数据流，显存占用恒定
    """
    def __init__(self, n_components, device="cuda"):
        self.n_components = n_components
        self.device = device
        # 协方差矩阵累积器 (X^T @ X) [D, D]
        self.cov_sum = None 
        # 均值累积器 (sum(X)) [D]
        self.mean_sum = None
        # 样本计数
        self.n_samples = 0
        
    def partial_fit(self, batch_data):
        """
        batch_data: [N, D] tensor on GPU
        """
        N, D = batch_data.shape
        
        # 初始化累积器 (懒加载，确定维度)
        if self.cov_sum is None:
            self.cov_sum = torch.zeros((D, D), device=self.device, dtype=torch.float32)
            self.mean_sum = torch.zeros((D,), device=self.device, dtype=torch.float32)
        
        # 1. 累积和 (用于计算全局均值)
        self.mean_sum += batch_data.sum(dim=0)
        
        # 2. 累积外积 (X^T X)
        # 这一步是计算瓶颈，GPU 加速效果最明显
        self.cov_sum += torch.matmul(batch_data.T, batch_data)
        
        self.n_samples += N

    def finalize(self):
        """ 处理完所有数据后，执行最终分解 """
        if self.n_samples < 2: return None

        print(f"   ⚙️ 正在执行特征分解 (Cov Matrix Size: {self.cov_sum.shape})...")
        
        # 1. 计算全局均值
        mean = self.mean_sum / self.n_samples
        
        # 2. 构造中心化协方差矩阵
        # 公式推导: Cov = (E[XX^T] - E[X]E[X]^T) * N / (N-1)
        # sum((x-u)(x-u)^T) = sum(xx^T) - N*u*u^T
        cov_matrix = self.cov_sum - self.n_samples * torch.outer(mean, mean)
        cov_matrix = cov_matrix / (self.n_samples - 1)
        
        # 3. 特征分解 (Symeig) - 数值稳定且快
        # eigh 适用于对称矩阵
        S, U = torch.linalg.eigh(cov_matrix)
        
        # 4. 排序 (eigh 返回的是升序，我们要降序)
        S = S.flip(0) # 特征值
        U = U.flip(1) # 特征向量
        
        # 5. 截断
        components = U[:, :self.n_components]
        explained_variance = S[:self.n_components]
        
        ratio = explained_variance.sum() / S.sum()
        
        return {
            "basis": components, # [4096, K]
            "mean": mean,        # [4096]
            "singular_values": torch.sqrt(explained_variance * (self.n_samples - 1)),
            "ratio": ratio
        }

def extract_svd_basis(style_name):
    print(f"\n📐 [全量模式] 正在处理风格流形: {style_name}")
    style_dir = os.path.join(DATA_ROOT, style_name)
    
    # 获取所有文件
    files = glob.glob(os.path.join(style_dir, "*.pt"))
    if not files:
        print(f"❌ 无数据: {style_dir}")
        return None
    
    print(f"   📂 发现文件: {len(files)} 个")

    # 初始化增量计算器
    pca = TorchIncrementalPCA(n_components=NUM_COMPONENTS, device=DEVICE)
    
    # 缓冲区
    buffer_list = []
    buffer_count = 0
    
    # 进度条
    pbar = tqdm(files, desc="Streaming Patches", unit="file")
    
    for f_path in pbar:
        try:
            # 1. 加载 Latent
            z = torch.load(f_path, map_location="cpu").float() # [4, 64, 64]
            if z.dim() == 3: z = z.unsqueeze(0)
            
            # 2. Unfold 切片 (全覆盖，无随机采样)
            # 使用 unfold 提取所有可能的 Patch
            patches = F.unfold(z, kernel_size=PATCH_SIZE, padding=0, stride=STRIDE)
            # [1, 4096, N_patches] -> [N_patches, 4096]
            patches = patches.permute(0, 2, 1).reshape(-1, VECTOR_DIM)
            
            # 加入缓冲区
            buffer_list.append(patches)
            buffer_count += patches.shape[0]
            
            # 3. 缓冲区满了？送入 GPU 计算
            if buffer_count >= GPU_BATCH_SIZE:
                # 拼接
                X_batch = torch.cat(buffer_list, dim=0).to(DEVICE)
                
                # 增量拟合
                pca.partial_fit(X_batch)
                
                # 清理
                buffer_list = []
                buffer_count = 0
                
                # 更新进度条信息
                pbar.set_postfix({"Total Patches": pca.n_samples})

        except Exception as e:
            # print(f"Error reading {f_path}: {e}")
            continue

    # 4. 处理剩余的缓冲区数据
    if buffer_list:
        X_batch = torch.cat(buffer_list, dim=0).to(DEVICE)
        pca.partial_fit(X_batch)
        del X_batch

    if pca.n_samples == 0:
        print("❌ 未提取到任何有效 Patch")
        return None

    print(f"   🧮 数据流结束。总处理 Patch 数: {pca.n_samples}")
    print(f"   ⚙️ 正在计算最终 SVD...")
    
    result = pca.finalize()
    
    print(f"   ✅ 完成。前 {NUM_COMPONENTS} 个基元解释了 {result['ratio']:.2%} 的方差。")

    return {
        "basis": result["basis"].cpu(),
        "mean": result["mean"].cpu(),
        "singular_values": result["singular_values"].cpu()
    }

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 检查维度
    if NUM_COMPONENTS > VECTOR_DIM:
        print(f"⚠️ 请求的主成分数 ({NUM_COMPONENTS}) 大于特征维度 ({VECTOR_DIM})，已修正。")
        NUM_COMPONENTS = VECTOR_DIM

    for style in TARGET_STYLES:
        with torch.no_grad():
            data = extract_svd_basis(style)
            if data:
                save_path = os.path.join(SAVE_DIR, f"{style}.pt")
                torch.save(data, save_path)
                print(f"   💾 基元已保存至: {save_path}")