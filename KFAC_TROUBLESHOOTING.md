# K-FAC优化器故障排除指南

## 问题描述

训练时遇到错误：
```
RuntimeError: syevd_cpu: U(157993,157993) is zero, singular U.
```

这是K-FAC (Kronecker-Factored Approximate Curvature) 优化器的数值稳定性问题。

## 问题原因

K-FAC优化器需要计算协方差矩阵的特征值分解。当矩阵变得奇异（不可逆）时，特征值分解会失败。

常见原因：
1. 梯度消失或爆炸
2. 协方差矩阵条件数过大
3. 数值精度问题
4. 训练早期统计信息不足

## 解决方案

### 方案1: 使用改进的K-FAC优化器（推荐）

我们提供了一个增强版的K-FAC优化器，具有以下改进：

- ✓ 自动SGD后备机制
- ✓ 更强的正则化
- ✓ 条件数检查
- ✓ 更保守的特征值过滤

**使用方法：**
```bash
python robust_train.py
```

这个脚本会：
1. 自动备份原始K-FAC优化器
2. 使用改进版本进行训练
3. 遇到数值问题时自动切换到SGD

### 方案2: 调整K-FAC参数

如果想继续使用原始K-FAC，可以尝试调整参数：

```python
# 在 acktr/algo/acktr_pipeline.py 中修改
KFACOptimizer(
    model,
    lr=0.25,
    momentum=0.9,
    stat_decay=0.99,
    kl_clip=0.001,
    damping=1e-2,  # 增加这个值，如 1e-1 或 1e-0
    weight_decay=0,
    fast_cnn=False,
    Ts=1,
    Tf=10
)
```

关键参数：
- `damping`: 增加到 0.1 或 1.0 可以提高稳定性
- `stat_decay`: 降低到 0.95 可以减少历史统计的影响
- `Tf`: 增加到 20 或 50 可以减少特征值分解频率

### 方案3: 使用A2C优化器

A2C使用标准的RMSprop优化器，更稳定但可能收敛较慢：

```bash
python main.py --mode train --item-seq rs --uncertainty-enabled --save_model --num_processes 1 --algorithm a2c
```

### 方案4: 减少并行进程

多进程训练可能加剧数值问题：

```bash
# 已经在使用 --num_processes 1
# 如果还有问题，确保环境变量设置正确
```

## 训练脚本对比

| 脚本 | 优化器 | 稳定性 | 速度 | 推荐场景 |
|------|--------|--------|------|----------|
| `robust_train.py` | K-FAC (改进) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 推荐首选 |
| `single_process_train.py` | K-FAC (原始) | ⭐⭐ | ⭐⭐⭐⭐ | 原始实现 |
| `ultra_simple_train.py` | K-FAC (原始) | ⭐⭐ | ⭐⭐⭐⭐ | 简化版本 |
| A2C模式 | RMSprop | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 最稳定但较慢 |

## 监控训练健康度

训练时注意以下指标：

1. **梯度范数**: 应该保持在合理范围（0.1-10）
2. **损失值**: 不应该出现NaN或Inf
3. **奖励**: 应该逐渐增加
4. **数值问题计数**: 改进版K-FAC会报告切换到SGD的次数

## 如果问题持续

如果所有方案都失败：

1. 检查环境配置
   ```bash
   python test_environment.py
   ```

2. 验证数据集
   ```bash
   python -c "import torch; print(torch.load('dataset/cut_2.pt'))"
   ```

3. 降低学习率
   ```python
   # 在main.py中修改
   lr=0.1  # 从0.25降低
   ```

4. 使用更简单的网络结构
   - 减少卷积层数量
   - 减少隐藏单元数量

## 技术细节

### K-FAC算法简介

K-FAC是一种二阶优化方法，通过近似Fisher信息矩阵来加速训练。它需要：

1. 计算激活值的协方差矩阵 (A)
2. 计算梯度的协方差矩阵 (G)
3. 对两个矩阵进行特征值分解
4. 使用Kronecker积近似Fisher矩阵

### 为什么会出现奇异矩阵

- 协方差矩阵秩不足
- 某些特征值接近零
- 数值舍入误差累积
- 训练早期统计不稳定

### 改进版K-FAC的改进

```python
# 1. 增加正则化
matrix = matrix + damping * I

# 2. 更保守的特征值过滤
eigenvalues[eigenvalues < 1e-4] = 0  # 从1e-6提高到1e-4

# 3. 条件数检查
condition_number = max_eigenvalue / min_eigenvalue
if condition_number > 1e6:
    use_sgd_fallback()

# 4. 异常捕获和后备
try:
    kfac_step()
except RuntimeError:
    sgd_step()
```

## 参考资料

- [K-FAC论文](https://arxiv.org/abs/1503.05671)
- [ACKTR论文](https://arxiv.org/abs/1708.05144)
- PyTorch优化器文档

## 获取帮助

如果问题仍未解决，请提供：
1. 完整错误信息
2. 使用的训练脚本
3. Python和PyTorch版本
4. 训练参数配置
