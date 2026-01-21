# OT-CFM 风格迁移系统使用指南

## 📋 系统概述

这是一个基于**最优传输条件流匹配 (OT-CFM)** 和**等距流形映射 (Isotropic Manifold Mapping)** 的高效风格迁移系统，专为 **RTX 4070 Laptop (8GB VRAM)** 优化。

### 核心特性

- ✅ **无需 U-Net**: 采用等距 ConvNeXt 架构，全网无下采样
- ✅ **无需对抗训练**: 纯 MSE 损失，基于 OT-CFM 数学原理
- ✅ **结构守恒**: 利用 ODE 可逆性保证结构一致性
- ✅ **快速推理**: 10-20 步 ODE 求解即可生成高质量结果
- ✅ **底层优化**: torch.compile + BF16 + channels_last + TF32

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate
pip install pillow numpy tqdm
```

### 2. 数据准备

#### 数据结构

将您的数据按以下结构组织：

```
raw_data_root/
├── style_0/           # 风格 0 (例如：照片)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── style_1/           # 风格 1 (例如：莫奈画)
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

#### 预处理为 Latent

```bash
python preprocess_latents.py --config config.json --target_size 512
```

这将把所有图片编码为 VAE Latent 并保存到 `data_root` 目录：

```
data_root/
├── style_0/
│   ├── image1.pt
│   ├── image2.pt
│   └── ...
└── style_1/
    ├── image1.pt
    ├── image2.pt
    └── ...
```

**⚠️ 重要**: 训练前必须完成预处理，严禁在训练循环中运行 VAE Encoder！

---

## 🎯 训练

### 配置文件

编辑 `config.json` 以适配您的数据：

```json
{
  "model": {
    "latent_channels": 4,      // VAE latent 通道数
    "hidden_dim": 384,          // 模型隐藏层维度 (384/512)
    "num_layers": 12,           // Isotropic Block 数量 (12-18)
    "num_styles": 2,            // 风格类别数
    "kernel_size": 7            // Depthwise Conv 核大小
  },
  "training": {
    "batch_size": 64,           // 批量大小 (根据显存调整)
    "learning_rate": 1e-4,
    "stage1_epochs": 200,       // 训练轮数
    "use_amp": true,            // 混合精度训练 (BF16)
    "label_drop_prob": 0.15,    // CFG label dropping 概率
    "num_workers": 4
  },
  "data": {
    "raw_data_root": "/path/to/raw/images",
    "data_root": "/path/to/processed/latents",
    "num_classes": 2
  },
  "checkpoint": {
    "save_dir": "checkpoints"
  }
}
```

### 开始训练

```bash
python train.py
```

训练过程会：
1. 自动应用 `torch.compile` 优化
2. 启用 BF16 混合精度
3. 使用 TF32 加速矩阵乘法
4. 每 10 个 epoch 保存一次 checkpoint

### 显存优化建议

- **8GB VRAM**: `batch_size=64`, `hidden_dim=384`, `num_layers=12`
- **12GB VRAM**: `batch_size=128`, `hidden_dim=512`, `num_layers=16`
- **16GB+ VRAM**: `batch_size=256`, `hidden_dim=512`, `num_layers=18`

---

## 🎨 推理

### 基本用法

```bash
python inference.py \
  --checkpoint checkpoints/stage1_epoch200.pt \
  --input test_image.jpg \
  --source_style 0 \
  --target_style 1 \
  --output output.png \
  --num_steps 20 \
  --cfg_scale 2.0
```

### 参数说明

- `--checkpoint`: 模型 checkpoint 路径
- `--input`: 输入图片或 latent (.pt 文件)
- `--source_style`: 源风格 ID (0, 1, ...)
- `--target_style`: 目标风格 ID
- `--output`: 输出图片路径
- `--num_steps`: ODE 积分步数 (10-20 推荐)
- `--cfg_scale`: 分类器自由引导强度 (1.0-3.0)

### 推理流程

系统采用**结构守恒回路**：

1. **Inversion (结构析出)**: 将输入图片逆向积分到结构空间
   ```
   x1 (source) -> x0 (structure)
   ```

2. **Generation (风格重绘)**: 从结构空间正向积分到目标风格
   ```
   x0 (structure) -> x1' (target)
   ```

这保证了结构在数学上的守恒性。

---

## 📊 数学原理

### OT-CFM 损失函数

```python
# 采样路径
x0 = torch.randn_like(x1)           # 结构基底 (高斯噪声)
t = torch.rand(batch_size)          # 时间 t ~ Uniform(0, 1)
x_t = (1 - t) * x0 + t * x1         # 线性插值路径

# 目标速度场
u_t = x1 - x0                       # 恒定速度场

# 模型预测
v_pred = model(x_t, t, style_id)

# MSE 损失
loss = ||v_pred - u_t||^2
```

### AdaGN (自适应组归一化)

```python
# 分离结构和风格
x_norm = GroupNorm(x)               # 归一化 (保留结构)

# 风格注入
scale, shift = MLP(style_embedding)
x_styled = scale * x_norm + shift   # 仿射变换 (改变风格)
```

---

## 🔧 高级配置

### 自定义模型架构

在 `config.json` 中调整：

```json
{
  "model": {
    "hidden_dim": 512,        // 更大模型
    "num_layers": 18,         // 更深网络
    "style_dim": 512,         // 风格嵌入维度
    "time_dim": 512           // 时间嵌入维度
  }
}
```

### CFG 调优

- **CFG Scale = 1.0**: 无引导，风格较弱
- **CFG Scale = 2.0**: 标准引导 (推荐)
- **CFG Scale = 3.0+**: 强引导，风格极强但可能失真

### ODE 求解器选择

```bash
# Euler 方法 (快速)
python inference.py --method euler --num_steps 10

# RK4 方法 (高精度)
python inference.py --method rk4 --num_steps 20
```

---

## 📈 性能优化清单

### 训练优化

- [x] **预计算 Latents**: 消除 VAE Encoder 瓶颈
- [x] **channels_last**: 提升卷积吞吐量 ~20%
- [x] **BFloat16 AMP**: 减少显存 ~50%，加速 ~2x
- [x] **torch.compile**: 减少 Python 开销
- [x] **TF32**: 自动启用 Tensor Cores
- [x] **pin_memory**: 加速 CPU->GPU 传输

### 推理优化

- [x] **Euler Solver**: 10-20 步快速求解
- [x] **Latent Clamping**: 防止数值爆炸
- [x] **@torch.no_grad()**: 禁用梯度计算

---

## 🐛 常见问题

### Q: 显存不足 (OOM)

**解决方案**:
1. 减小 `batch_size`
2. 减小 `hidden_dim`
3. 减少 `num_layers`
4. 确保启用 `use_amp: true`

### Q: 训练损失不下降

**检查**:
1. 是否正确预处理了数据？
2. `label_drop_prob` 是否设置合理 (0.1-0.2)？
3. 学习率是否过大？尝试 `1e-5`

### Q: 生成结果风格不明显

**调整**:
1. 增加 `cfg_scale` (2.0 -> 3.0)
2. 增加训练轮数
3. 检查训练数据是否包含足够的风格多样性

### Q: 结构变化过大

**原因**: ODE 求解步数过少或数值不稳定

**解决**:
1. 增加 `num_steps` (10 -> 20)
2. 使用 RK4 求解器
3. 启用 `latent_clamp`

---

## 📁 文件说明

```
.
├── config.json                 # 配置文件
├── model.py                    # 模型架构 (IsoNext, AdaGN)
├── train.py                    # 训练脚本
├── inference.py                # 推理脚本 (Inversion + Generation)
├── preprocess_latents.py       # 数据预处理工具
├── checkpoints/                # 模型权重
│   ├── stage1_epoch10.pt
│   ├── stage1_epoch20.pt
│   └── ...
└── data_root/                  # 预处理的 Latent 数据
    ├── style_0/
    └── style_1/
```

---

## 📖 引用

本实现基于以下论文和技术：

1. **Flow Matching for Generative Modeling** (Lipman et al., 2022)
2. **Optimal Transport Flow Matching** (Tong et al., 2023)
3. **ConvNeXt V2** (Woo et al., 2023)
4. **Classifier-Free Guidance** (Ho & Salimans, 2021)

---

## 📧 技术支持

遇到问题？请检查：
1. PyTorch 版本 >= 2.0
2. CUDA 版本兼容
3. 数据预处理正确完成
4. 配置文件路径正确

---

**Happy Styling! 🎨**
