### 基于流形速度场放大与频谱约束的损失函数设计


#### MSE太低维导致的，高频信息学不到


1. 问题定义：近恒等映射下的梯度消失在流匹配（Flow Matching）框架下，当源域分布 $p_0(x)$ 与目标域分布 $p_1(x)$ 在欧几里得空间距离较近（即 $\|x_1 - x_0\| \to \epsilon$）时，标准均方误差（MSE）损失函数定义为：

$$
\mathcal{L}_{MSE} = \mathbb{E}_{t, x_t} \| v_\theta(x_t, t) - (x_1 - x_0) \|^2
$$





由于目标速度向量

$(x_1 - x_0)$

 的模长极小，神经网络易陷入平凡解 

$v_\theta \approx 0$

（即恒等映射）。此时损失函数的梯度 



$\nabla_\theta \mathcal{L}$

 趋近于零，导致模型无法有效学习到从油画（Monet）到照片（Photo）的风格迁移流形路径。2. 速度场放大 (Velocity Field Amplification)为了解决小模长向量导致的优化停滞，引入标量放大因子 

$\alpha > 1$

 对目标速度场进行重整化。修正后的目标速度定义为：

$$
v_{target}' = \alpha \cdot (x_1 - x_0), \quad \text{where } \alpha = \begin{cases} 1.0 & \text{if } s_{id} = t_{id} \\ \lambda & \text{if } s_{id} \neq t_{id} \end{cases}
$$

其中 

$\lambda$ 为超参数（本文实验取 $\lambda=3.0$）。从优化角度分析，损失函数变为 $\mathcal{L}' = \| v_\theta - \alpha \Delta x \|^2$。对预测速度 $v_\theta$ 求导，其误差信号被放大：
$$
\frac{\partial \mathcal{L}'}{\partial v_\theta} = 2(v_\theta - \alpha \Delta x)
$$

这强制模型输出非零的大模长向量，从而在流形空间中产生显著位移，克服局部极小值。3. 频谱一致性约束 (Spectral Consistency Constraint)由于 MSE 损失在像素空间具有各向同性，模型倾向于拟合低频分量（颜色、轮廓），而忽略高频分量（纹理、噪声）。为解决此谱偏差（Spectral Bias）问题，引入基于帕塞瓦尔定理的频谱损失。定理 3.1：帕塞瓦尔定理 (Parseval's Theorem)设 $f(x)$ 为平方可积函数，$\mathcal{F}[f](\xi)$ 为其傅里叶变换。根据帕塞瓦尔定理，函数在时域（或空域）的能量等于其在频域的能量，即：
$$
\int_{-\infty}^{\infty} |f(x)|^2 dx = \int_{-\infty}^{\infty} |\mathcal{F}[f](\xi)|^2 d\xi
$$

对于离散信号（图像），设 $x[n]$ 为长度为 $N$ 的序列，其离散傅里叶变换（DFT）为 $X[k]$，定理形式为：
$$
\sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2
$$

这表明傅里叶变换是希尔伯特空间 $L^2$ 上的等距同构算子（Unitary Operator）。损失函数构建：基于定理 3.1，虽然理论上最小化空域 MSE 等价于最小化频域 MSE，但在神经网络的非凸优化中，显式地引入频域约束可以改变损失曲面的曲率，使优化器更关注高频系数的拟合。定义频谱损失 $\mathcal{L}_{spec}$ 如下：
$$
\mathcal{L}_{spec} = \| |\mathcal{F}(v_{pred})| - |\mathcal{F}(v_{target}')| \|^2_2
$$

其中 $|\cdot|$ 表示复数的模（幅度谱），$\mathcal{F}$ 为二维快速傅里叶变换（FFT）。此项约束了生成速度场与目标速度场在频率分布上的一致性，确保风格迁移过程中的高频纹理特征得以保留。代码实现修正基于上述定义，以下是完全对应的代码修改。请替换 train.py 中的相关部分。Python# =========================================================

# 辅助函数：定义在类外部或作为静态方法

# =========================================================

def compute_spectral_loss(v_pred, v_gt):
    """
    基于帕塞瓦尔定理计算频域幅度谱损失。
    Args:
        v_pred: 预测速度场 [B, C, H, W]
        v_gt:   目标速度场 [B, C, H, W]
    Returns:
        loss:   标量张量
    """
    # 强制转换为 float32 进行 FFT，避免 BFloat16 精度不足导致数值不稳定
    v_pred_fp32 = v_pred.float()
    v_gt_fp32 = v_gt.float()

    # 执行二维实数 FFT
    # 输出维度: [B, C, H, W//2 + 1]，数据类型为 Complex64
    fft_pred = torch.fft.rfft2(v_pred_fp32, norm='ortho')
    fft_gt = torch.fft.rfft2(v_gt_fp32, norm='ortho')

    # 计算幅度谱 (Modulus of complex numbers)
    amp_pred = torch.abs(fft_pred)
    amp_gt = torch.abs(fft_gt)

    # 计算频域 MSE
    return F.mse_loss(amp_pred, amp_gt)

# =========================================================

# 训练循环内部逻辑 (替换 run_stage1 中的 Loss 计算部分)

# =========================================================

# ... (前文 construct_target_lsfm 代码不变) ...

opt.zero_grad(set_to_none=True)
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    # 1. 构建基础目标速度向量
    v_gt_raw = target - x_c

    # 2. 速度场放大 (Velocity Amplification)
    # 若 s_id != t_id (风格迁移)，则 alpha = 3.0；否则 alpha = 1.0
    is_transfer = (s_id != t_id).view(-1, 1, 1, 1).float()
    alpha = 1.0 + (is_transfer * 2.0)
    v_gt_amplified = v_gt_raw * alpha

    # 3. 模型前向传播
    t = torch.rand(x_c.size(0), device=self.device)
    x_t = (1 - t.view(-1,1,1,1)) * x_c + t.view(-1,1,1,1) * target
    v_pred = model(x_t, x_c, t, t_id)

    # 4. 损失函数计算
    # L_mse: 确保流形路径的主方向正确，且具备足够动能
    loss_mse = F.mse_loss(v_pred, v_gt_amplified)

    # L_spec: 基于帕塞瓦尔定理，确保高频纹理特征对齐
    loss_spec = compute_spectral_loss(v_pred, v_gt_amplified)

    # L_dir: 余弦相似度约束，解耦方向与模长优化
    v_pred_flat = v_pred.flatten(1)
    v_gt_flat = v_gt_amplified.flatten(1)
    cos_sim = F.cosine_similarity(v_pred_flat, v_gt_flat, dim=1, eps=1e-8)
    loss_dir = (1 - cos_sim).mean()

    # 总损失加权 (根据 256x256 分辨率的经验值)
    loss = loss_mse + 0.1 * loss_spec + 0.2 * loss_dir

loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
opt.step()

# ...
