---
title: "机器学习第三讲：支撑向量机(Support Vector Machine)"
published: 2025-10-11
pinned: false
description: "A personal reflection and review of Lecture 3 -- Support Vector Machine -- of Machine Learning."
tags: ["ml", "svm", "reflection"]
category: ml
author: "hAk0"
draft: true
---

## 一、引言：为什么需要SVM？

### 1. 线性分类器的困境

- 对于线性可分数据，存在无穷多个分离超平面。
- 朴素选择（如感知机）不考虑**泛化能力**。
- 直观目标：选择**最鲁棒**（对噪声容忍度最高）的分类器。

### 2. 最大间隔思想（Maximum Margin）

- **间隔（Margin）**：分类超平面到最近样本点的距离的两倍。
- **支撑向量（Support Vectors）**：落在间隔边界上的样本点。
- **核心直觉**：
  - 间隔越大，分类器对噪声越鲁棒；
  - 间隔最大的分类器具有最佳泛化性能。

---

## 二、硬间隔SVM（Hard-Margin SVM）

### 1. 问题建模

- **目标**：最大化间隔  
  $$\max_{\mathbf{w}, b} \frac{2}{\|\mathbf{w}\|}$$
- **约束**：所有样本被正确分类  
  $$y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1,\quad \forall i = 1,\dots,n$$

### 2. 等价凸优化问题

- 将最大化间隔转化为最小化 $\|\mathbf{w}\|^2$（便于求解）：
  $$
  \begin{aligned}
    \min_{\mathbf{w}, b}\ & \frac{1}{2} \|\mathbf{w}\|^2 \\
    \text{s.t. }\ & y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1,\quad \forall i
  \end{aligned}
  $$
- 此为**凸二次规划问题**（Convex QP），但仅适用于**线性可分**情形。

> **注意**：硬间隔SVM在实际中几乎不用，因为现实数据通常含噪声或不可分。

---

## 三、软间隔SVM（Soft-Margin SVM）

### 1. 引入松弛变量（Slack Variables）

- 允许部分样本违反间隔约束，引入松弛变量 $\xi_i \geq 0$：
  $$
  y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i,\quad \xi_i \geq 0
  $$
- $\xi_i$ 的含义：
  - $\xi_i = 0$：样本在正确一侧且在间隔外；
  - $0 < \xi_i < 1$：样本在间隔内但分类正确；
  - $\xi_i > 1$：样本被错误分类。

### 2. 优化目标

- 在最大化间隔与最小化误分类之间权衡：
  $$
  \begin{aligned}
    \min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \\
    \text{s.t. } y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i,\quad \xi_i \geq 0
  \end{aligned}
  $$
- 超参数 $C > 0$ 控制**间隔大小**与**分类误差**的权衡：
  - $C$ 大 → 强惩罚误分类 → 小间隔、低偏差、高方差；
  - $C$ 小 → 容忍更多误分类 → 大间隔、高偏差、低方差。

---

## 四、SVM的对偶问题（Dual Problem）

### 1. 拉格朗日函数

- 引入拉格朗日乘子 $\alpha_i \geq 0$（对应主约束）和 $\mu_i \geq 0$（对应 $\xi_i \geq 0$）：
  $$
  \mathcal{L} = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i - \sum_i \alpha_i [y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 + \xi_i] - \sum_i \mu_i \xi_i
  $$

### 2. KKT条件与对偶形式

- 对 $\mathbf{w}, b, \xi_i$ 求偏导并令为0，得：
  $$
  \mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i,\quad \sum_{i=1}^n \alpha_i y_i = 0,\quad \alpha_i = C - \mu_i \leq C
  $$
- 代入后得到**对偶问题**：
  $$
  \begin{aligned}
  \max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j \\
  \text{s.t. } \sum_{i=1}^n \alpha_i y_i = 0,\quad 0 \leq \alpha_i \leq C
  \end{aligned}
  $$

### 3. 支撑向量与稀疏性

- **KKT互补松弛条件**：$\alpha_i [y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 + \xi_i] = 0$
- 因此：
  - 若 $\alpha_i > 0$，则样本 $i$ 是**支撑向量**；
  - 决策函数仅依赖支撑向量：
    $$
    f(\mathbf{x}) = \text{sign}\left( \sum_{i:\alpha_i>0} \alpha_i y_i \mathbf{x}_i^\top \mathbf{x} + b \right)
    $$
- 解具有**稀疏性**：多数 $\alpha_i = 0$。

### 4. 求解算法

- **序列最小优化（SMO）**：高效求解对偶QP问题；
- **核技巧兼容**：对偶形式天然支持核函数。

---

## 五、核方法（Kernel Method）

### 1. 动机：处理非线性问题

- 通过特征映射 $\phi: \mathcal{X} \to \mathcal{H}$ 将原始空间映射到高维（甚至无限维）特征空间。
- 在 $\mathcal{H}$ 中构造线性分类器，等价于在 $\mathcal{X}$ 中构造非线性分类器。

### 2. 核函数定义

- **核函数**：$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$
- **优势**：无需显式计算 $\phi(\mathbf{x})$，只需计算核函数（“核技巧”）。

### 3. 常用核函数

| 核函数 | 表达式 | 说明 |
|--------|--------|------|
| 线性核 | $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^\top \mathbf{x}_j$ | 等价于原始SVM |
| 多项式核 | $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^\top \mathbf{x}_j + c)^d$ | 捕捉特征交互 |
| RBF（高斯）核 | $K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)$ | 最常用，适合任意光滑边界 |
| Sigmoid核 | $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(a \mathbf{x}_i^\top \mathbf{x}_j + b)$ | 类似神经网络 |

> **带宽选择**：RBF核中 $\sigma$ 可用“中位数技巧”：$\sigma = \text{median}(\|\mathbf{x}_i - \mathbf{x}_j\|)$

### 4. Mercer定理

- 函数 $K$ 是合法核函数 $\iff$ 对任意数据集，核矩阵 $\mathbf{K}_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ 是**半正定**的。

---

## 六、SVM的原始问题求解：随机梯度下降（SGD）

### 1. 合页损失（Hinge Loss）

- 软间隔SVM等价于以下**正则化风险最小化**问题：
  $$
  \min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + \frac{C}{n} \sum_{i=1}^n \max(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))
  $$
- 其中 $\ell(z) = \max(0, 1 - z)$ 称为**合页损失**。

### 2. SGD算法

- 每轮随机采样一个样本 $(\mathbf{x}_i, y_i)$：
  - 若 $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$，梯度为 $\mathbf{w}$；
  - 否则，梯度为 $\mathbf{w} - C y_i \mathbf{x}_i$。
- 更新规则：
  $$
  \mathbf{w} \leftarrow \mathbf{w} - \eta \left( \mathbf{w} - C y_i \mathbf{x}_i \cdot \mathbb{I}[y_i f(\mathbf{x}_i) < 1] \right)
  $$
- **优势**：可扩展至大规模数据，避免求解QP。

---

## 七、多分类SVM

### 1. 一对多（One-vs-Rest）

- 训练 $K$ 个二分类器，第 $k$ 个区分类别 $k$ 与其他类。
- **问题**：各分类器独立训练，输出不可比。

### 2. 一对多联合训练（Crammer & Singer）

- 联合优化所有分类器：
  $$
  \begin{aligned}
  \min_{\mathbf{W}, \boldsymbol{\xi}} \frac{1}{2} \sum_{k=1}^K \|\mathbf{w}_k\|^2 + C \sum_{i=1}^n \xi_i \\
  \text{s.t. } \mathbf{w}_{y_i}^\top \mathbf{x}_i \geq \mathbf{w}_k^\top \mathbf{x}_i + 1 - \xi_i,\quad \forall k \neq y_i
  \end{aligned}
  $$
- **缺点**：不能输出概率，不如Softmax自然。

---

## 八、SVM变体与扩展

### 1. 支持向量回归（SVR）

- 使用 $\epsilon$-不敏感损失：
  $$
  \ell(y, f(\mathbf{x})) = \max(0, |y - f(\mathbf{x})| - \epsilon)
  $$
- 目标：拟合一个“管”，管内无损失。

### 2. 半监督SVM（TSVM）

- 利用未标记数据，假设决策边界应穿过低密度区域。
- 通过自训练（self-training）迭代标注高置信度未标记样本。

### 3. 一类SVM（One-Class SVM）

- 用于**异常检测**：在特征空间中寻找包含大多数正常样本的最小超球面或最大间隔超平面。

---

## 九、实践建议

### 1. 何时使用SVM？

- 中小规模数据集（$n < 10^5$）；
- 特征维度适中；
- 需要强理论保证和良好泛化性能。

### 2. 调参指南

- **核选择**：默认尝试RBF核；
- **超参数**：通过交叉验证调优 $C$ 和 $\sigma$（RBF）或 $d$（多项式）；
- **预处理**：必须对特征进行标准化（尤其使用RBF核时）。

### 3. 优缺点总结

| **优点** | **缺点** |
|---------|---------|
| 泛化能力强（最大间隔） | 大规模数据训练慢（对偶问题 $O(n^2 \sim n^3)$） |
| 核方法灵活处理非线性 | 无法直接输出概率（需 Platt Scaling） |
| 稀疏解（仅依赖支撑向量） | 对噪声和异常值敏感（尤其 $C$ 大时） |
| 理论基础坚实 | 多分类需特殊处理 |

---

## 十、总结

支撑向量机通过**最大间隔原则**和**核技巧**，在理论与实践之间取得了优雅平衡。尽管在深度学习时代其地位有所下降，但在中小规模结构化数据任务中，SVM仍是强大而可靠的基线模型。

> **关键思想回顾**：  
>
> - 最大间隔 → 泛化能力  
> - 对偶问题 + 核技巧 → 非线性分类  
> - 合页损失 + SGD → 可扩展性

---
