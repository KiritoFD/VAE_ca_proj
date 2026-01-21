# OT-CFM 风格迁移系统

基于**最优传输条件流匹配 (Optimal Transport Conditional Flow Matching)** 和**等距流形映射 (Isotropic Manifold Mapping)** 的高效风格迁移系统。

## 🌟 核心特性

- ✅ **纯数学驱动**: 基于 ODE 可逆性，无需对抗训练
- ✅ **轻量高效**: 专为 8GB VRAM 优化，支持 RTX 4070 Laptop
- ✅ **结构守恒**: Inversion + Generation 双向流程保证结构一致性
- ✅ **快速推理**: 10-20 步 ODE 求解即可完成风格迁移
- ✅ **底层优化**: torch.compile + BF16 + TF32 + channels_last

## 📦 文件结构

```
.
├── model.py                    # 模型架构 (IsoNext, AdaGN, TimestepEmb)
├── train.py                    # 训练脚本 (OT-CFM Loss)
├── inference.py                # 推理脚本 (Inversion + Generation)
├── preprocess_latents.py       # 数据预处理 (VAE Encoding)
├── test_model.py               # 模型测试脚本
├── config.json                 # 配置文件
├── USAGE_GUIDE.md             # 详细使用指南
└── README.md                   # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate pillow numpy tqdm
```

### 2. 测试模型

```bash
python test_model.py
```

这将验证：
- 模型架构正确性
- OT-CFM 损失计算
- 显存占用情况
- channels_last 加速效果

### 3. 准备数据

将数据组织为：
```
raw_data/
├── style_0/
│   └── *.jpg
└── style_1/
    └── *.jpg
```

然后运行预处理：
```bash
python preprocess_latents.py --config config.json
```

### 4. 训练

```bash
python train.py
```

模型将保存在 `checkpoints/` 目录。

### 5. 推理

```bash
python inference.py \
  --checkpoint checkpoints/stage1_epoch200.pt \
  --input test.jpg \
  --source_style 0 \
  --target_style 1 \
  --output result.png \
  --num_steps 20
```

## 📐 数学原理

### OT-CFM 流匹配

将风格迁移建模为在概率流形上寻找最优路径：

$$p_t(x) = (1-t) \cdot \mathcal{N}(0, I) + t \cdot p_{\text{data}}(x)$$

目标：学习速度场 $v_\theta(x_t, t, c)$ 使得：

$$\frac{dx}{dt} = v_\theta(x_t, t, c)$$

损失函数极其简洁：

$$\mathcal{L} = \mathbb{E}_{x_0, x_1, t} \left[ \| v_\theta(x_t, t, c) - (x_1 - x_0) \|^2 \right]$$

### 结构守恒回路

利用 ODE 可逆性实现结构守恒：

1. **Inversion (结构析出)**
   ```
   x₁ --[反向ODE]--> x₀
   (源图片)           (结构坐标)
   ```

2. **Generation (风格重绘)**
   ```
   x₀ --[正向ODE]--> x₁'
   (结构坐标)         (目标风格)
   ```

### AdaGN (自适应组归一化)

核心风格注入机制：

```python
x_norm = GroupNorm(x)                    # 保留结构
scale, shift = MLP(style_embedding)      # 预测仿射参数
x_styled = scale * x_norm + shift        # 注入风格
```

## 🎯 模型架构

```
IsoNext (等距 ConvNeXt)
├── Input Projection: [4, H, W] -> [D, H, W]
├── Isotropic Blocks (×12-18)
│   ├── Depthwise Conv 7×7
│   ├── AdaGN (风格注入)
│   ├── Pointwise Conv (升维)
│   ├── GELU
│   ├── Pointwise Conv (降维)
│   └── Residual Connection
└── Output Projection: [D, H, W] -> [4, H, W]
```

**关键特性**:
- 全等距架构，无下采样
- 大核卷积 (7×7) 捕捉全局结构
- AdaGN 实现风格流动性
- 残差连接保证梯度流动

## ⚙️ 配置说明

编辑 `config.json`:

```json
{
  "model": {
    "hidden_dim": 384,        // 隐藏层维度 (384/512)
    "num_layers": 12,         // 网络深度 (12-18)
    "num_styles": 2           // 风格类别数
  },
  "training": {
    "batch_size": 64,         // 批量大小
    "learning_rate": 1e-4,
    "stage1_epochs": 200,     // 训练轮数
    "label_drop_prob": 0.15   // CFG dropping 概率
  }
}
```

## 📊 性能指标

**显存占用** (Batch Size 64, Hidden Dim 384):
- 模型: ~350MB
- 训练峰值: ~6.5GB
- 推理峰值: ~2GB

**训练速度** (RTX 4070):
- ~0.5 sec/batch (BF16 + compile)
- ~2000 batches/epoch
- ~16 min/epoch

**推理速度**:
- 10 步 Euler: ~0.3s
- 20 步 RK4: ~0.8s

## 🔬 底层优化技术

1. **预计算 Latents**: 消除 VAE Encoder 瓶颈
2. **BFloat16 AMP**: 减少显存 ~50%，加速 ~2x
3. **torch.compile**: 减少 Python 解释器开销
4. **channels_last**: 提升卷积吞吐量 ~20%
5. **TF32**: 自动启用 Tensor Cores
6. **Gradient Checkpointing**: (可选) 进一步节省显存

## 🎨 使用场景

- 照片 → 油画风格
- 素描 → 彩色作品
- 现实 → 卡通风格
- 任意风格域迁移

## 📚 参考文献

1. **Flow Matching for Generative Modeling**  
   Lipman et al., ICLR 2023

2. **Improving and Generalizing Flow-Based Generative Models with Optimal Transport**  
   Tong et al., TMLR 2023

3. **ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders**  
   Woo et al., CVPR 2023

4. **Classifier-Free Diffusion Guidance**  
   Ho & Salimans, NeurIPS 2021 Workshop

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系

遇到问题？请查看 [USAGE_GUIDE.md](USAGE_GUIDE.md) 获取详细说明。

---

**Built with ❤️ and Mathematics**
