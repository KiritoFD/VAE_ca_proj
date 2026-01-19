# SA-Flow: Latent Space Style Transfer via Optimal Transport Flow Matching

SA-Flow（Style-Aware Flow）是一个面向 **非配对数据（Unpaired Data）** 的高效风格迁移框架。本项目利用 Flow Matching 在 **VAE 潜空间** 构建确定性传输路径，并采用 **Two-Stage** 策略：

- **Stage 1：** 用 Optimal Transport（OT）在 batch 内做动态匹配，实现 **分布层面** 的对齐；
- **Stage 2（Reflow）：** 用 Stage 1 的 Teacher 生成并固化风格化结果，构建 **伪配对数据（Pseudo-paired Data）**，将不确定的分布映射转化为确定的 **有监督回归**，从而稳定映射关系并支持少步推理。

通过结合 **Optimal Transport（最优传输）** 与 **Reflow（重流/整流）** 策略，本算法能够学习从“内容分布”到“风格分布”的 ODE 轨迹，实现高质量、极速（少步数）的风格转换。

---

## 目录（可选）

- [1. 任务定义：VAE 潜空间中的分布映射](#1-任务定义vae-潜空间中的分布映射)
  - [1.1 为什么选择潜空间（Latent Space）？](#11-为什么选择潜空间latent-space)
  - [1.2 为什么选择 Flow Matching？](#12-为什么选择-flow-matching)
- [2. Stage 1：基于 OT 的分布对齐（Distribution Alignment）](#2-stage-1基于-ot-的分布对齐distribution-alignment)
  - [2.1 概率流与向量场回归](#21-概率流与向量场回归)
  - [2.2 最优传输对齐（Optimal Transport Alignment）](#22-最优传输对齐optimal-transport-alignment)
- [3. Reflow / Stage 2：构建伪配对数据并进行确定性回归](#3-reflow--stage-2构建伪配对数据并进行确定性回归)
  - [3.1 Reflow：构建伪配对数据（Pseudo-pairs）](#31-reflow构建伪配对数据pseudo-pairs)
  - [3.2 Stage 2：确定性配对回归与少步推理](#32-stage-2确定性配对回归与少步推理)
- [4. 架构设计逻辑（Model Architecture）](#4-架构设计逻辑model-architecture)
  - [4.1 动态结构门控（Time-Aware Spatial Modulation）](#41-动态结构门控time-aware-spatial-modulation)
  - [4.2 风格注入（AdaGN）](#42-风格注入adagn)

---

端侧部署[KiritoFD/MNN_android_vae_saf](https://github.com/KiritoFD/MNN_android_vae_saf)

用MNN完成了简单的端侧部署

<div align="center">
  <img src="./android_infra.jpg" alt="photo->monet" width="60%" />
  <img src="image/readme/1768810230461.jpg" alt="1768810230461" width="39%" />
</div>

与基模对比，效果和速度有显著优势（下为Mnn-chat上用官方模型市场的SD1.5推理结果）![](image/readme/1768810285282.jpg)

## 1. 任务定义：VAE 潜空间中的分布映射

### 1.1 为什么选择潜空间（Latent Space）？

传统的风格迁移直接在像素空间（\(H \times W \times 3\)）操作，计算开销巨大且难以捕捉高层语义。本项目利用预训练的 AutoencoderKL（VAE）将图像压缩至低维潜空间 \(Z\)。

在非配对风格迁移中，我们只有两组集合/分布：

- 内容集合：\(x_c \sim p_{\text{content}}\)
- 风格集合：\(x_s \sim p_{\text{style}}\)

不存在“一张内容图对应一张标准风格化答案”的真实配对监督。

- 目标：学习映射 \(f(x_c, \text{StyleID}) \to \hat{x}\)，使 \(\hat{x}\) 具备目标风格纹理，同时保留 \(x_c\) 的空间结构。

### 1.2 为什么选择 Flow Matching？

标准扩散模型（Diffusion）定义的目标分布 \(p_1\) 通常是高斯噪声 \( \mathcal{N}(0, I) \)。这意味着生成过程必须从纯噪声开始，难以精确控制内容的保留。

Flow Matching 允许我们自定义源分布 \(p_0\) 和目标分布 \(p_1\)。在本项目中：

$$
p_0 = p_{\text{content}}, \quad p_1 = p_{\text{style}}
$$

目标是学习一个能够将 \(x_c\) 平滑“推”向 \(x_s\) 的向量场。

---

## 2. Stage 1：基于 OT 的分布对齐（Distribution Alignment）

### 2.1 概率流与向量场回归

在潜空间中定义随时间 \(t \in [0, 1]\) 变化的插值路径 \(\psi_t\)。对一对样本 \((x_c, x_s)\)，构造直线路径：

$$
\psi_t(x_c, x_s) = (1-t)x_c + t x_s
$$

其时间导数（真实速度场）为常数：

$$
u_t(x \mid x_c, x_s) = \frac{d}{dt}\psi_t = x_s - x_c
$$

神经网络 \(v_\theta(x, t, c)\) 需要拟合该速度场，训练损失为均方误差：

$$
\mathcal{L}_{\text{CFM}}(\theta)
= \mathbb{E}_{t, x_c, x_s}\left[\|v_\theta(\psi_t(x_c, x_s), t, c) - (x_s - x_c)\|^2\right]
$$

对应实现（`train.py`）：

```python
# ...existing code...
v_pred = model(x_t, x_c, t, s_id)   # 网络预测
v_target = x_target - x_c           # 物理真实速度
loss = torch.mean((v_pred - v_target) ** 2)
# ...existing code...
```

### 2.2 最优传输对齐（Optimal Transport Alignment）

如果在训练 batch 中随机配对 \(x_c\) 与 \(x_s\)（Independent Coupling），会导致潜空间轨迹相互交叉（Crossing Paths）。轨迹交叉对应非平滑向量场，使 ODE 求解更困难，生成质量下降。

**解决方案：** 每个训练步，在 batch 内计算 \(x_c\) 与 \(x_s\) 的代价，并用匈牙利算法求解最小总代价匹配 \(\pi\)（或等价的置换/索引）：

$$
\pi^\* = \arg\min_{\pi} \sum_{i=1}^{B} \left\|x_c^{(i)} - x_s^{(\pi(i))}\right\|^2
$$

实现逻辑（`train.py -> compute_ot_matching`）：使用 `linear_sum_assignment` 动态计算最优索引，并在输入网络前对 \(x_s\) 重排，以降低轨迹交叉。

**局限性（关键）：** Stage 1 主要保证 **分布层面的统计对齐**。对单张内容图而言，其“对应的风格样本”可能会随 batch 组合/epoch 改变，存在 **One-to-Many / 映射不确定** 的问题；这也是 Stage 2（Reflow）要解决的核心动机。

---

## 3. Reflow / Stage 2：构建伪配对数据并进行确定性回归

### 3.1 Reflow：构建伪配对数据（Pseudo-pairs）

在非配对设定下无法获得真实的 \((x_{\text{content}}, x_{\text{style}})\) 监督对，因此使用 Stage 1 训练好的模型作为 **Teacher**，为每个内容样本生成并“固定”一个风格化结果，从而人为构造伪 Ground Truth。

对每个内容潜变量 \(x_c\)，指定目标风格 ID，通过 ODE 求解得到确定输出 \(z\)：

$$
z = \text{ODE\_Solver}(x_c, \text{StyleID}; \theta_{\text{Stage1}})
$$

随后将 \((x_c, z)\) 保存为训练数据。此时 \(z\) 对应“该 Teacher 在该风格条件下的固定答案”，把原本不确定的分布映射问题转化为确定的监督学习问题。

### 3.2 Stage 2：确定性配对回归与少步推理

Stage 2 不再依赖 OT 或随机配对，而是直接在伪配对数据 \((x_c, z)\) 上训练，使模型专注学习确定映射 \(x_c \to z\)。

沿用直线路径插值：

$$
\psi_t(x_c, z) = (1-t)x_c + t z
$$

流匹配回归目标变为：

$$
\mathcal{L}_{\text{S2}}
= \mathbb{E}_{t, x_c}\left[\left\|v_\theta(\psi_t(x_c, z), t, \text{StyleID}) - (z - x_c)\right\|^2\right]
$$

由于 \(z\) 由 \(x_c\) 沿 Teacher 流场演化得到，\((x_c, z)\) 天然“对齐”，该训练会显著稳定映射关系，并在实践中使轨迹更接近直线，从而支持 **单步或少步** 的近似求解（例如欧拉大步长）。

---

## 4. 架构设计逻辑（Model Architecture）

模型 `SAFlow.py` 专为条件流匹配设计，核心在于如何处理时间 \(t\)、内容 \(x_c\) 与风格 \(s\) 的相互作用。

### 4.1 动态结构门控（Time-Aware Spatial Modulation）

**问题：** \(t=0\) 输入是内容图，网络应保留其空间结构；随 \(t \to 1\)，网络应更多关注风格纹理生成。

**形式化：**

$$
h_{\text{out}} = h_{\text{in}} + \text{Conv}(x_{\text{content}})\cdot \sigma(\text{MLP}(t))
$$

其中 Sigmoid 门控 \(\sigma(\cdot)\) 允许网络根据时间 \(t\) 自适应调节内容结构（Structure）的注入强度。

### 4.2 风格注入（AdaGN）

风格迁移本质上是对特征统计量的调整。采用自适应组归一化（AdaGN）：

$$
\text{AdaGN}(x, s, t) = \frac{x - \mu}{\sigma} \cdot (1 + \gamma(s,t)) + \beta(s,t)
$$

将 Style Embedding 映射为仿射参数 \(\gamma, \beta\)，使潜变量特征分布向目标风格域对齐（可视作对齐均值与方差，是 Gram Matrix 匹配的泛化形式）。
