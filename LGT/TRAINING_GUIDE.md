# LGT Training with Evaluation and Resume Support

## New Features

### 1. 自动推理评估 (Automatic Inference Evaluation)

每个`eval_interval`周期，训练器会：
- 从`test_image_dir`中每个风格子目录读一张图
- 对每张源图，转换到**所有其他风格**
- 保存结果到`checkpoints/inference/epoch_XXXX/`

**配置示例：**
```json
{
  "training": {
    "eval_interval": 5,
    "test_image_dir": "/mnt/f/monet2photo/test"
  }
}
```

**输出结构：**
```
checkpoints/
  inference/
    epoch_0005/
      styleA_to_styleB.jpg
      styleA_to_styleC.jpg
      styleB_to_styleA.jpg
      ...
    epoch_0010/
      styleA_to_styleB.jpg
      ...
```

---

### 2. 定期保存检查点 (Checkpoint Saving)

训练过程中每`save_interval`周期自动保存检查点，包含：
- 模型权重
- 优化器状态
- 学习率调度器状态
- AMP缩放器状态
- 训练配置和指标

**配置示例：**
```json
{
  "training": {
    "save_interval": 5
  }
}
```

**生成的检查点：**
```
checkpoints/
  epoch_0005.pt
  epoch_0010.pt
  ...
  latest.pt  # 最新检查点
  logs/
    training_20260121_143021.csv
```

---

### 3. 断点续传支持 (Resume from Checkpoint)

支持从任意检查点恢复训练，保留完整的训练状态。

#### 方法1：在config中指定

```json
{
  "training": {
    "resume_checkpoint": "checkpoints/epoch_0050.pt"
  }
}
```

然后运行：
```bash
python train.py
```

#### 方法2：命令行参数

```bash
python train.py --resume checkpoints/epoch_0050.pt
```

#### 方法3：从最新检查点恢复

```bash
python train.py --resume checkpoints/latest.pt
```

---

## 使用流程

### 完整训练流程

```bash
# 1. 首次训练（从头开始）
python train.py

# [训练进行中，第50个epoch时中断...]

# 2. 断点续传
python train.py --resume checkpoints/epoch_0050.pt

# [继续从epoch 51开始训练...]
```

**训练日志输出示例：**
```
Loading checkpoint: checkpoints/epoch_0050.pt
✓ Resumed from epoch 50
  Next epoch will be: 51

================================================================================
Starting LGT Training
================================================================================
Epoch 51/100 | Loss: 0.2345 | SWD: 0.1123 | SSM: 0.1222 | LR: 8.95e-05

================================================================================
Running Inference Evaluation (Epoch 55)
================================================================================
  Test image for styleA: photo_001.jpg
  Test image for styleB: monet_001.jpg
Processing source: styleA
  → styleA to styleB
    ✓ Saved: styleA_to_styleB.jpg
Processing source: styleB
  → styleB to styleA
    ✓ Saved: styleB_to_styleA.jpg

================================================================================
✓ Inference completed. Results saved to: checkpoints/inference/epoch_0055
================================================================================
```

---

## 配置参数详解

在`config.json`中的新配置项：

```json
{
  "training": {
    "save_interval": 5,              // 每N个epoch保存一次checkpoint
    "eval_interval": 5,              // 每N个epoch进行一次推理评估
    "test_image_dir": "/path/to/test", // 测试图像目录
    "resume_checkpoint": null        // 恢复检查点路径（可选）
  }
}
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_interval` | int | 5 | 每N个epoch保存检查点 |
| `eval_interval` | int | 5 | 每N个epoch进行推理评估 |
| `test_image_dir` | str | `/mnt/f/monet2photo/test` | 测试图像根目录 |
| `resume_checkpoint` | str/null | null | 恢复的检查点路径 |

---

## 测试图像目录结构

`test_image_dir`应该按风格组织：

```
/mnt/f/monet2photo/test/
  trainA/
    photo_001.jpg
    photo_002.jpg
    ...
  trainB/
    monet_001.jpg
    monet_002.jpg
    ...
```

**训练器会自动：**
1. 遍历每个子目录（对应一个风格）
2. 从每个子目录选择第一张图作为源图
3. 将该图转换到所有其他风格
4. 保存结果到对应的epoch目录

---

## 高级用法

### 案例1：训练-推理-调参循环

```bash
# 训练50个epoch
python train.py --config config_v1.json

# 检查推理结果
# 查看 checkpoints/inference/epoch_0050/ 中的结果

# 调整参数并继续训练
# 编辑 config.json 调整 w_style, w_content 等

python train.py --resume checkpoints/epoch_0050.pt
```

### 案例2：快速评估模型

```bash
# 训练到某个epoch后暂停
# Ctrl+C 停止训练

# 立即评估最新模型（不修改检查点）
# 在config中设置：
#   "eval_interval": 1
# 临时改为每1个epoch评估

python train.py --resume checkpoints/epoch_0050.pt
```

### 案例3：生产环保存和备份

```bash
# 定期备份最佳检查点
cp checkpoints/latest.pt backups/best_epoch_${epoch}.pt

# 在不同配置上训练
python train.py --config config_high_quality.json

# 如果新训练不理想，快速回到旧版本
python train.py --resume backups/best_epoch_0050.pt
```

---

## 故障排查

### 问题1：恢复检查点后提示epoch不匹配

**错误信息：**
```
RuntimeError: Expected all tensors to be on the same device
```

**解决方案：**
确保检查点和当前device一致：
```bash
python train.py --resume checkpoints/epoch_0050.pt
```

### 问题2：恢复后学习率异常

**症状：** 损失值跳跃或增加

**原因：** 学习率调度器状态未正确恢复

**解决方案：** 检查checkpoint是否完整
```python
import torch
ckpt = torch.load('checkpoints/epoch_0050.pt')
print(ckpt.keys())  # 应该包含: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, scaler_state_dict
```

### 问题3：推理评估速度慢

**原因：** ODE积分步数过多

**优化方案：** 在推理时使用更少的步数
```python
# 在 train.py 中 _invert_latent 和 _generate_latent 调用时修改 num_steps
num_steps=10  # 从15降到10
```

---

## 监控训练进度

### 1. 实时日志

```bash
# 监控训练日志
tail -f checkpoints/logs/training_*.csv

# 每秒更新
watch -n 1 'tail -20 checkpoints/logs/training_*.csv'
```

### 2. 推理结果对比

```bash
# 对比不同epoch的推理结果
ls -lh checkpoints/inference/epoch_*/
```

### 3. Python脚本监控

```python
import pandas as pd
from pathlib import Path

# 读取训练日志
log_file = Path('checkpoints/logs/training_*.csv')
df = pd.read_csv(log_file)

# 查看最后10行
print(df.tail(10))

# 统计最佳loss
print(f"最小loss: {df['loss_total'].min():.6f} (epoch {df.loc[df['loss_total'].idxmin(), 'epoch']})")
```

---

## 常见配置组合

### 快速调试
```json
{
  "training": {
    "batch_size": 8,
    "num_epochs": 10,
    "save_interval": 2,
    "eval_interval": 2,
    "ode_integration_steps": 3
  }
}
```

### 高质量训练
```json
{
  "training": {
    "batch_size": 16,
    "num_epochs": 200,
    "save_interval": 10,
    "eval_interval": 10,
    "ode_integration_steps": 10
  }
}
```

### 轻量级推理
```json
{
  "training": {
    "batch_size": 32,
    "num_epochs": 50,
    "save_interval": 1,
    "eval_interval": 1,
    "ode_integration_steps": 3
  }
}
```

---

## 最佳实践

### ✓ 推荐做法

1. **定期保存**
   - 设置`save_interval=5`定期保存
   - 保留`latest.pt`用于快速恢复

2. **评估驱动开发**
   - 用`eval_interval`自动生成样本
   - 根据结果调整超参数

3. **增量式改进**
   - 从最好的checkpoint恢复
   - 调整参数后继续训练
   - 比较前后结果

4. **版本管理**
   ```bash
   cp checkpoints/latest.pt backups/v1.0_epoch_0050.pt
   ```

### ✗ 避免做法

1. **不要删除checkpoint** - 无法恢复训练
2. **不要修改config后恢复** - 可能导致不兼容
3. **不要在同一目录训练多个模型** - checkpoint会覆盖
4. **不要忽视evaluation结果** - 这是模型质量的直接反映

---

## 下一步

- 查看 [README.md](README.md) 了解整体架构
- 查看 [QUICKSTART.md](QUICKSTART.md) 快速开始
- 查看 [API.md](API.md) 详细API文档
