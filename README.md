# Online-3D-BPP-DRL: High-Reliability Packing Simulation & Verification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)

本仓库是对 **AEI (Advanced Engineering Informatics, 2023)** 顶刊论文 *Towards reliable robot packing system based on deep reinforcement learning* 的深度复现。

## 🔬 研发背景与定位

本项目不仅完整复现了 AEI 论文中的 Online 3D-BPP 算法逻辑，更作为**本人国家发明专利（[这里可填专利名或申请号]）**的关键验证平台。

**专利验证说明**：
本项目通过在复现算法中引入自定义的“多属性约束模块”，成功验证了专利中提出的“复杂场景下兼顾物理稳定性与空间利用率”的核心创新点。该实验平台为专利的创新性提供了量化数据支撑，证明了所提方案在提高机器人装箱可靠性方面的有效性。

---

## 🌟 核心功能

* **顶刊算法深度复现**：实现了基于 DRL 的在线三维装箱，包含 Candidate Map 生成机制与 DRL 决策网络。
* **物理稳定性增强模块**：在原算法基础上，针对工业场景下的抓取与摆放稳定性，加入了重心评估与支撑力学约束过滤。
* **多约束验证平台**：支持自定义不同的物理属性约束（如重量分布、表面摩擦等），用于测试不同策略下的装箱鲁棒性。

---

## 🏗️ 算法实现逻辑



1.  **State Representation**：基于容器高度图（Height Map）与待装箱项的尺寸信息。
2.  **Action Space**：利用 Candidate Map 筛选潜在放置位置，显著降低搜索维度。
3.  **Reward Function**：在原始的体积利用率奖励基础上，通过专利相关的稳定性指标对奖励函数进行了重塑（Reshaping）。

---

## 🚀 环境与运行

### 环境配置+运行实验
```bash
pip install -r requirements.txt
python unified_test.py