---
title: "机器学习第一讲：绪论(introduction)"
published: 2025-09-22
pinned: false
description: "A personal reflection and review of Lecture 1 -- Introduction -- of Machine Learning."
image: "./cover.png"
tags: ["ml", "intro", "reflection"]
category: ml
author: "hAk0"
draft: false
---

## 一、什么是机器学习？

### 1. 定义

- **Arthur Samuel (1959)**：  
  机器学习是研究如何让计算机无需显式编程，就能从数据中学习并自动提升性能的学科。

- **Tom Mitchell (1997) 形式化定义**：  
  若一个计算机程序在任务 $T$ 上的性能 $P$，随经验 $E$ 的增加而提升，则称该程序从经验 $E$ 中学习了任务 $T$。  
  数学表达：  
  $$P(y|x) \text{ 在任务 } T \text{ 上通过经验 } E \text{ 得到提升}$$

---

### 2. 核心地位

- 位于计算机科学与统计学的交叉点
- 是人工智能和数据科学的核心技术
- 三大支柱：数据、算法、计算力

---

## 二、机器学习的核心框架

### 1. 学习的组成要素

| 要素         | 符号          | 描述                                                                 |
|--------------|---------------|----------------------------------------------------------------------|
| 输入空间     | $\mathcal{X}$ | 特征向量 $\mathbf{x} \in \mathbb{R}^d$                               |
| 输出空间     | $\mathcal{Y}$ | 标签 $y$（分类：$\{0,1\}$，回归：$\mathbb{R}$）                     |
| 目标函数     | $f: \mathcal{X} \to \mathcal{Y}$ | 理想映射（未知，又称"最优分类器"）           |
| 训练数据     | $(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_n, y_n)$ | 已知样本对                         |
| 假设空间     | $\mathcal{H}$ | 候选函数集合 $h: \mathcal{X} \to \mathcal{Y}$（参数化为 $h(\mathbf{x};\theta)$） |
| 学习算法     | $A$           | 从 $\mathcal{H}$ 中选择最优 $h$ 的策略                               |

**最终目标**：找到 $h^* \approx f$，其在测试集上的误差称为**贝叶斯误差**（Bayes Error）。

### 2. 损失函数（Loss Function）

- **回归任务**：  
  $$L(y, h(\mathbf{x})) = (y - h(\mathbf{x}))^2$$
- **分类任务**：  
  $$L(y, h(\mathbf{x})) = \mathbb{I}(y \neq h(\mathbf{x}))$$
- **训练目标**：  
  $$\min_{h \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n L(h(\mathbf{x}_i), y_i)$$

- Virtually every machine learning algorithm has this form, just specify
  - What is the **hypothesis function**?
  - What is the **loss function**?
  - How do we solve the **training problem**?

### 3. 学习算法

- **优化目标**：最小化经验风险（Empirical Risk）  
  $$R_{\text{emp}}(h) = \frac{1}{n} \sum_{i=1}^n L(h(\mathbf{x}_i), y_i)$$
- **核心挑战**：  
  - 过拟合（Overfitting）：模型复杂度过高，学习训练数据噪声  
  - 欠拟合（Underfitting）：模型复杂度过低，无法捕捉数据规律

---

## 三、学习的基本原则

### 1. 奥卡姆剃刀原则（Occam's Razor）

- **核心思想**：在所有能解释数据的模型中，选择最简单的模型
- **数学表达**：  
  $$\text{Complexity}(h) + \lambda \cdot R_{\text{emp}}(h)$$
  其中 $\lambda$ 控制复杂度与误差的权衡
- **实践意义**：避免过拟合，提升泛化能力

### 2. 无免费午餐定理（No Free Lunch Theorem）

- **核心结论**：  
  没有普遍最优的学习算法，所有算法在所有问题上的平均性能相同
- **数学表达**：  
  $$\sum_{f} P(h|D) = \text{constant}$$
  其中 $f$ 是所有可能的目标函数
- **实践启示**：  
  - 需针对具体问题选择算法  
  - 先验知识（领域知识）至关重要

---

## 四、模型选择与评估

### 1. 交叉验证（Cross-Validation）

- **K折交叉验证**：  
  1. 将数据集分为 $K$ 个子集  
  2. 轮流使用 $K-1$ 个子集训练，1个子集验证  
  3. 计算平均验证误差作为模型性能估计  
  $$\text{CV Error} = \frac{1}{K} \sum_{i=1}^K \text{Error}_i$$

### 2. 偏差-方差分解（Bias-Variance Tradeoff）

- **期望预测误差**：  
  $$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2$$
  - **偏差**：模型预测与真实值的偏离程度  
  - **方差**：模型对训练数据扰动的敏感度  
  - **噪声**：数据内在随机性（不可约误差）

### 3. 学习曲线分析

- **诊断工具**：  
  - 横轴：训练样本数量  
  - 纵轴：训练误差/验证误差  
- **典型模式**：  
  - 高偏差：训练误差和验证误差均高且接近  
  - 高方差：训练误差远低于验证误差  

---

## 五、KNN算法详解

### 1. 算法思想

- **核心假设**：相似样本具有相似标签（"物以类聚"）
- **归纳偏好（inductive bias）**：
  - Similar points have similar labels.
  - All dimensions are created equal.
- **数学表达**：  
  $$h(\mathbf{x}) = \text{majority}\{ y_i \mid \mathbf{x}_i \in N_k(\mathbf{x}) \}$$
  其中 $N_k(\mathbf{x})$ 是 $\mathbf{x}$ 的 $k$ 个最近邻

### 2. 关键组件

- **Feature Normalization**:
为使不同量纲的特征对模型贡献公平，**必须在训练前对原始特征做标准化 / 归一化**。常用策略：

| 方法 | 公式 | 说明 |
|---|---|---|
| Min-Max 缩放 | $x' = \dfrac{x - x_{\min}}{x_{\max}-x_{\min}}$ | 压缩到 [0,1]；对异常值敏感 |
| Z-score 标准化 | $x' = \dfrac{x - \mu}{\sigma}$ | 零均值、单位方差；最常用 |
| 均值归一化 | $x' = \dfrac{x - \mu}{x_{\max}-x_{\min}}$ | 分母用极差，减少异常值影响 |
| L2 归一化 | $x' = x / \|\mathbf{x}\|_2$ | 向量长度缩为 1，常用于余弦相似度场景 |

> **注意**：统计量（$\mu,\sigma,x_{\min},x_{\max}$）**只在训练集计算**，再统一应用到验证 / 测试集，避免数据泄露。

- **距离度量**：
k-NN、聚类、RBF 核等依赖距离或相似度，请选择与业务/数据特点匹配的量度：

| 名称 | 公式 | 特点 |
|---|---|---|
| 欧氏距离 (Euclidean) | $d_E(\mathbf{x}_i,\mathbf{x}_j)=\sqrt{\sum_{l=1}^d(x_{il}-x_{jl})^2}$ | 连续稠密特征、各向同性空间 |
| 曼哈顿距离 (Manhattan) | $d_M(\mathbf{x}_i,\mathbf{x}_j)=\sum_{l=1}^d(x_{il}-x_{jl})$ | 稀疏或网格状数据，抗异常值 |
| 余弦相似度 (Cosine) | $s_C(\mathbf{x}_i,\mathbf{x}_j)=\dfrac{\mathbf{x}_i\cdot\mathbf{x}_j}{\|\mathbf{x}_i\|\|\mathbf{x}_j\|}$ | 忽略幅值，关注方向；文本 / TF-IDF 常用 |
| 切比雪夫距离 | $d_C(\mathbf{x}_i,\mathbf{x}_j)=\max_l(x_{il}-x_{jl})$ | 棋盘/网格移动步数 |
| 闵氏距离 (p-范数) | $d_p=\Bigl(\sum_l(x_{il}-x_{jl})^p\Bigr)^{1/p}$ | p=1,2 为上面特例；p→∞→切比雪夫 |
| Mahalanobis Distance | $d_M(\mathbf{x}_i,\mathbf{x}_j)=\sqrt{(\mathbf{x}_i-\mathbf{x}_j)^T𝑴(\mathbf{x}_i-\mathbf{x}_j)}$ | 𝑴 can also be learned from data → metric learning |

> **技巧**：高维数据可先降维（PCA、t-SNE）再算距离，减轻维度灾难。

- **$k$值选择**：  

k 是超参数，直接控制模型复杂度：

| k 大小 | 决策边界 | 偏差-方差权衡 | 常见风险 |
|---|---|---|---|
| **小**（k=1~3） | 锯齿状、高灵活 | 低偏差 / 高方差 | 过拟合、对噪声敏感 |
| **中**（交叉验证最优） | 平滑适度 | 偏差≈方差 | 测试误差最低 |
| **大**（k→N） | 极度平滑 | 高偏差 / 低方差 | 欠拟合，丢失局部结构 |

**调参流程**  

1. 在训练集上用 **交叉验证（Grid-Search）** 评估不同 k 的准确率 / F1  
2. 绘制 **k vs. 验证误差** 曲线，选误差最小且不过大 k（兼顾效率）  
3. 若类别不平衡，可改用 **距离加权投票**（权重∝1/d）缓解 k 过大带来的多数类偏向

> **经验**：
>
> - 样本量 N 很大时，k 可适当放大（噪声被平均）
> - 特征归一化后再调 k，否则距离被大尺度特征主导，k 失去物理意义

### 3. 算法变种

- **加权KNN**：  
  $$h(\mathbf{x}) = \text{weighted majority}\left\{ w_i y_i \mid w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)} \right\}$$
- **KD树优化**：  
  构建空间划分树，将最近邻搜索复杂度从 $O(n)$ 降至 $O(\log n)$
  要求数据量$n$远大于维度$d$
  - For naïve KNN:
    - 𝑂(1) for training.
    - 𝑂(𝑛𝑑) to find 𝑘 closest examples.
  - For KNN with k-d Tree:
    - 𝑂(𝑑𝑛 log 𝑛) for training (build k-d tree).
    - 𝑂($2^d$ log 𝑛) on average when query
- **Nearest Centroid Classifier**：
  假设：每个类满足高斯先验分布

---

## 六、维度灾难（Curse of Dimensionality）

### 1. 现象描述

随着维度 $d$ 增加，数据在高维空间中变得极其稀疏。

### 2. 数学解释

- **体积覆盖问题**：  
  在 $[0,1]^d$ 空间中，要覆盖比例 $s$ 的体积，边长需满足：  
  $$\text{边长} = s^{1/d}$$
  **示例**：当 $s=0.1, d=10$ 时，边长 $=0.1^{0.1} \approx 0.8$，意味着"邻居"不再邻近。
- **球内均匀问题**：
  - Consider random variables $𝑋_1$,$𝑋_2$,...,$𝑋_n$ drawn i.i.d. in dimension 𝑑, with uniform distribution on the unit ball.
  - Median distance from the origin to the closest data point
  - 𝑛 = 500,𝑑 = 10 → med = 0.52.
  - **Which means that most data points are closer to the edge of the ball than to the center.**

### 3. 严重后果

- **距离失效**：高维空间中所有点对距离趋于一致  
  $$\lim_{d \to \infty} \frac{\text{max distance} - \text{min distance}}{\text{min distance}} \to 0$$
- **样本需求爆炸**：  
  保持密度不变所需样本数随维度指数增长：$n \propto s^{-d}$
- **KNN失效**：最近邻概念在高维空间失去意义

### 4. 应对策略

- **降维技术**：PCA、t-SNE、自编码器  
- **流形假设**：假设数据位于高维空间中的低维流形上  
- **特征选择**：移除无关特征，保留判别性特征  

---

## 七、KNN算法总结

### 适用场景

- 特征维度低（$d < 20$）
- 训练样本量大（$n > 10^3$）
- 目标函数复杂，参数化模型难以拟合
- 类别数多（多分类问题）

### 优缺点对比

| **优点**                       | **缺点**                     |
|------------------------------ |------------------------------|
| 可逼近任意复杂函数            | 推理速度慢（需加速技术）     |
| 不丢失信息（存储原始数据）      | 高维下效果差（维度灾难）     |
| 支持海量数据和类别            | 易受无关特征干扰（特征工程很重要）        |
| 天然支持多分类                | 内存消耗大（存储全部样本）   |
| 无需训练过程（惰性学习）        | 对数据尺度敏感（需标准化）   |

### 实践建议

1. **数据预处理**：  
   - 标准化：$x' = \frac{x - \mu}{\sigma}$  
   - 归一化：$x' = \frac{x - \min}{\max - \min}$  
2. **特征工程**：  
   - 使用PCA或特征选择降低维度  
   - 移除冗余/无关特征  
3. **加速技术**：  
   - n变小——索引；d变小——压缩
   - KD树、Ball树等空间索引结构  
   - 近似最近邻搜索（ANN）算法  

---
