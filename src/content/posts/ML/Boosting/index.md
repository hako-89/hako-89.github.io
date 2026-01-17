---
title: "Boosting"
published: 2025-11-05
pinned: false
description: "本讲深入探讨了集成学习的概念，重点介绍了Boosting算法家族。涵盖了从AdaBoost的原理、误差界限证明，到梯度提升（Gradient Boosting）的通用框架，最后详细讲解了工程上广泛应用的XGBoost和LightGBM的优化细节。"
image: ""
tags: ["Machine Learning", "Boosting", "AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM"]
category: "Machine Learning Course Notes"
author: "Assistant"
draft: false
---

## 一、集成学习 (Ensemble Learning)

### 1. 概述与分类

集成学习通过训练多个学习器并将它们组合起来使用，旨在提升整体性能。

**分类：**

| 特性 | 并行集成 (Parallel Ensemble) | 序列集成 (Sequential Ensemble) |
| :--- | :--- | :--- |
| **构建方式** | 每个模型独立构建 | 模型按顺序构建 |
| **代表算法** | Bagging, Random Forest | Boosting, XGBoost |
| **核心思想** | 结合多个**强模型**以避免**过拟合** (Overfitting) | 结合多个**弱模型**以避免**欠拟合** (Underfitting) |
| **策略** | 独立同分布，降低方差 | 添加新模型以修正前序模型的错误 |

### 2. 为什么要集成 (Why Ensemble)

**定理**：令 $\mathcal{H}$ 为取值为 $\{-1, +1\}$ 且 VC 维为 $d$ 的函数族，则对于任意 $\delta > 0$，以至少 $1-\delta$ 的概率，对于所有 $h \in \mathcal{H}$：
$$
\mathcal{E}(h) \le \hat{\mathcal{E}}_n(h) + \sqrt{\frac{2d \log \frac{en}{d}}{n}} + \sqrt{\frac{\log(1/\delta)}{2n}}
$$
基于学习理论，复杂模型具有较大的泛化误差。通过结合多个不同模型，我们可以逼近真实的标记函数。

**误差控制与 Hoeffding 不等式**：
假设训练 $T$ 个二分类器 $\{h_t(x)\}_{t=1}^T$，组合方式为投票 $H(x) = \text{sign}(\sum_{t=1}^T h_t(x))$。
如果每个基分类器的错误率独立且 $P(h_t(x) \neq y) = \epsilon$，则集成后的错误率随 $T$ 指数级下降：
$$
P(H(x) \neq y) \le 2 \exp(-\frac{1}{2}T(1-2\epsilon)^2)
$$
这要求基学习器之间是**独立**的，即它们应当是**多样化 (Diverse)** 的。

---

## 二、Boosting 基础

### 1. 核心概念

- **弱学习器 (Weak Learner)**：一个分类概念类是弱 PAC 可学习的，如果存在算法能生成一个错误率略优于随机猜测（即错误率 $\le \frac{1}{2} - \gamma$，$\gamma > 0$）的假设。
- **主要思想**：
    1. 寻找简单且相对准确的基分类器通常不难。
    2. 组合许多弱分类器可能创建一个强学习器。
    3. Boosting 按顺序添加新模型，新模型专注于之前模型表现不佳的地方。

### 2. 顺序学习框架

最终的二分类器形式为：
$$
H(x) = \text{sign}(\alpha_T(x)), \quad \text{where } \alpha_T(x) = \sum_{t=1}^T \beta_t h_t(x)
$$
Boosting 采用顺序优化，每次只优化一个模型 $h_t$ 和其权重 $\beta_t$：
$$
\{\beta_t, h_t\} = \operatorname*{argmin}_{\beta, h} \frac{1}{n} \sum_{i=1}^n \ell(y_i, \alpha_{t-1}(x_i) + \beta h(x_i))
$$

---

## 三、自适应提升 (AdaBoost)

### 1. 算法详解

AdaBoost 使用**指数损失 (Exponential Loss)**：$\ell(y, f(x)) = e^{-y f(x)}$。

**AdaBoost 算法流程**：
1. 初始化样本权重 $D_1(i) = 1/m$。
2. 对于 $t = 1$ 到 $T$：
    - 训练基分类器 $h_t$，使其加权误差 $\epsilon_t = \Pr_{i \sim D_t}[h_t(x_i) \neq y_i]$ 最小。
    - 计算权重 $\alpha_t = \frac{1}{2} \ln \frac{1-\epsilon_t}{\epsilon_t}$。
    - 计算归一化因子 $Z_t = 2\sqrt{\epsilon_t(1-\epsilon_t)}$。
    - 更新样本权重：
        $$
        D_{t+1}(i) = \frac{D_t(i) \exp(-\alpha_t y_i h_t(x_i))}{Z_t}
        $$
3. 输出 $H(x) = \text{sign}(\sum_{t=1}^T \alpha_t h_t(x))$。

### 2. 坐标下降解释 (Coordinate Descent Explanation)

AdaBoost 等价于对指数损失目标函数 $G(\bar{\alpha})$ 进行**坐标下降 (Coordinate Descent)** 优化。
- 目标函数：$G(\bar{\alpha}) = \frac{1}{n} \sum_{i=1}^n \exp(-y_i \sum_{t=1}^T \bar{\alpha}_{t, k} h_k(x_i))$
- **方向**：在每一轮，选择下降方向最陡的基分类器 $h_t$。
- **步长**：步长 $s$ 正好对应 AdaBoost 中的权重 $\alpha_t = \frac{1}{2} \ln \frac{1-\epsilon_t}{\epsilon_t}$。

### 3. 经验误差界 (Empirical Error Bound)

**定理**：AdaBoost 输出的分类器 $H$ 的经验误差满足：
$$
\hat{\mathcal{E}}(H) \le \exp \left( -2 \sum_{t=1}^T (\frac{1}{2} - \epsilon_t)^2 \right)
$$
如果对于所有 $t$，$\gamma_t = \frac{1}{2} - \epsilon_t \ge \gamma$，则误差以 $\exp(-2\gamma^2 T)$ 的速度指数衰减。

### 4. 泛化误差与间隔 (Margin)

- **VC 维界限**：随着 $T$ 增加，模型复杂度增加，理论上可能导致过拟合，但实证观察表明 AdaBoost 往往不发生过拟合。
- **间隔 (Margin)**：$y H(x)$。间隔越大，预测越自信。
- **解释**：Schapire 等人证明 AdaBoost 能够增大训练样本的间隔。即使训练误差已降为 0，继续训练仍能推动间隔增大，从而降低泛化误差界。
- **Rademacher 复杂度界**：基于间隔损失 (Margin Loss) $\Phi_\rho$ 和 Rademacher 复杂度的泛化误差界与 $T$ 无关（在一定条件下）。

---

## 四、梯度提升 (Gradient Boosting)

### 1. 动机与框架

AdaBoost 局限于二分类和指数损失。**梯度提升 (Gradient Boosting)** 是一种适用于任意可微损失函数的通用方法。

**核心思想**：
我们希望找到 $\beta_t h_t$ 来最小化目标函数 $G(\alpha_t) = \sum_{i=1}^n \ell(y_i, \alpha_{t-1}(x_i) + \beta_t h_t(x_i))$。
这可以类比于梯度下降：
$$
\alpha_t \approx \alpha_{t-1} - s \nabla_\alpha \sum \ell(y_i, \alpha)
$$
因此，第 $t$ 个基学习器 $h_t(x_i)$ 应当拟合**负梯度**（即伪残差）：
$$
h_t(x_i) \approx - \frac{\partial \ell(y_i, \alpha_{t-1}(x_i))}{\partial \alpha_{t-1}(x_i)}
$$

### 2. 前向分步加法模型 (Forward Stagewise Additive Modeling)

1. 初始化 $f_0(x) = 0$。
2. 对于 $t=1$ 到 $T$：
    - 计算负梯度方向（伪残差）。
    - 拟合基学习器 $h_t$ 逼近负梯度。
    - 选择步长 $\beta_t$。
    - 更新模型：$f_t(x) = f_{t-1}(x) + \beta_t h_t(x)$。

### 3. 具体实例

- **L2 Boosting**：使用平方损失 $\ell(y, f(x)) = (y - f(x))^2$。
  - 负梯度即为残差：$2(y_i - f_{t-1}(x_i))$。
  - 每一步是对残差进行回归。
- **Binomial Boosting**：使用 Logistic Loss $\ell(y, f) = \log(1 + e^{-yf})$。
  - 负梯度为 $\frac{y_i}{1 + e^{y_i f_{t-1}(x_i)}}$。

---

## 五、XGBoost (Extreme Gradient Boosting)

### 1. 系统与算法改进

XGBoost 是一个可扩展的树集成学习系统，主要改进包括：
- **系统优化**：核外计算 (Out-of-core computing)、并行化 (Parallelization)、缓存优化、分布式计算。
- **算法改进**：稀疏感知算法、加权分位数略图 (Weighted approximate quantile sketch)。

### 2. 正则化学习目标 (Regularized Objective)

目标函数包含训练损失和正则化项：
$$
\text{Obj} = \sum_{i=1}^n \ell(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)
$$
**树的复杂度正则化**：
$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \|w\|^2
$$
其中 $T$ 是叶子节点数量，$w$ 是叶子节点的权重向量。

### 3. 泰勒展开近似 (Taylor Expansion)

不同于传统 GBM 仅使用一阶导数，XGBoost 对目标函数进行**二阶泰勒展开**：
$$
\text{Obj}^{(t)} \approx \sum_{i=1}^n [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t) + \text{const}
$$
其中：
- $g_i = \partial_{\hat{y}^{(t-1)}} \ell(y_i, \hat{y}^{(t-1)})$ (一阶梯度)
- $h_i = \partial^2_{\hat{y}^{(t-1)}} \ell(y_i, \hat{y}^{(t-1)})$ (二阶梯度/Hessian)

### 4. 结构分数 (Structure Score)

对于固定的树结构 $q(x)$，叶子节点 $j$ 的最优权重 $w_j^*$ 和对应的最小目标函数值（结构分数）为：
$$
w_j^* = - \frac{G_j}{H_j + \lambda}, \quad \text{Obj}^* = - \frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j + \lambda} + \gamma T
$$
其中 $G_j = \sum_{i \in I_j} g_i$，$H_j = \sum_{i \in I_j} h_i$ 是叶子节点 $j$ 中样本的梯度统计量。

### 5. 贪心树学习 (Greedy Tree Learning)

- **分裂增益 (Gain)**：
    $$
    \text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
    $$
    该公式衡量了分裂后的结构分数提升减去引入新叶子的复杂度代价 $\gamma$。
- **策略**：从深度 0 开始，对每个叶节点尝试添加分裂，线性扫描排序后的实例以决定最佳分裂点。
- **剪枝**：支持预停止 (Pre-stopping) 和后剪枝 (Post-pruning)。

---

## 六、LightGBM

- **特点**：更快的速度，更少的内存，更高的准确率。
- **Leaf-wise 生长**：采用 Best-first 策略，选择具有最大 delta loss 的叶子进行生长。虽然可能导致过拟合，但通过限制 `max_depth` 可以控制。
- **直方图算法 (Histogram Sampler)**：将连续特征离散化为直方图，加速分裂寻找。
- **类别特征优化**：不需要 One-hot 编码，直接支持最优分裂。

---

## 本章要点

1. **集成优于单一模型**：通过组合多个弱学习器（Boosting）或强学习器（Parallel Ensemble），可以显著降低偏差或方差。
2. **AdaBoost**：通过调整样本权重关注错分样本，本质上是指数损失函数的坐标下降优化。
3. **梯度提升 (GBM)**：通过拟合负梯度（一阶导数）来通用化 Boosting，适用于各种损失函数。
4. **XGBoost 的优势**：引入二阶导数（Hessian）和显式的正则化项（树结构复杂度），并结合系统级优化，使其在大规模数据上高效且准确。
5. **Boosting 发展历程**：从理论证明 (Weak Learner) -> 算法实现 (AdaBoost) -> 通用框架 (GBM) -> 高效系统 (XGBoost/LightGBM)。