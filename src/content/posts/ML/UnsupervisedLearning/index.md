---
title: "Unsupervised Learning"
published: 2025-11-16
pinned: false
description: "无监督学习概述，内容涵盖聚类、降维及其核心算法。"
image: ""
tags: ["Machine Learning", "Unsupervised Learning", "Clustering", "Dimensionality Reduction", "K-means", "PCA", "NMF"]
category: "Machine Learning"
author: hako"
draft: false
---

## 一、无监督学习 (Unsupervised Learning)

### 1. 人类的学习能力

- 人类即使在没有先验知识的情况下，也能分辨和**区分**物体（如上下姿势、光照方向、左右姿势）。
- 人类具备**总结**的能力，例如能总结出“这些图像是同一个物体的多个视角”。
- 机器学习的起点是让机器模仿这些人类的学习能力。

### 2. 无监督学习的必要性

- 标签数据非常宝贵，但有时收集标签的成本**高昂**，难以承受。
- 之前介绍的所有模型都需要标签数据，这是一个主要**瓶颈**。
- **无监督学习**研究如何在没有标签的数据下进行学习。
- 无监督学习被称为人工智能的**暗物质**（或暗能量）。

### 3. 不同学习范式的信息量对比

| 学习范式 | 机器预测内容 | 信息量 (每样本) |
| :--- | :--- | :--- |
| **纯强化学习 (Pure Reinforcement Learning)** | 偶尔给出的标量奖励 | 少量比特 |
| **有监督学习 (Supervised Learning)** | 每个输入对应的类别或少量数字 | 10 - 10,000 比特 |
| **无监督/预测学习 (Unsupervised/Predictive Learning)** | 其输入的任何部分（对任何观察到的部分），如预测视频中的未来帧 | 数百万比特 |

---

## 二、聚类 (Clustering)

### 1. 聚类的基本概念

- **聚类**是一个无监督学习问题。
- **输入**: $n$ 个无标签数据点 $\{x_{i}\}_{i=1}^{n}$；分区数量 $k$（这是一个**超参数**，难以选择）。
- **目标**: 将样本分组为 $k$ 个分区。
- **好的聚类**标准:
  - **高**簇内相似度 (High within-cluster similarity)
  - **低**簇间相似度 (Low inter-cluster similarity)

### 2. 相似度/距离的度量

- 最常用的度量是在特征空间中的**距离**。
- 数据集 $\left\{x_{i}\right\}_{i=1}^{n}\subset\mathbb{R}^{d}$，在 $\mathbb{R}^{d}$ 空间中的距离可以有以下形式：
  - **欧几里得距离 (Euclidean Distance)** (在某些情况下表现良好):

    $$
    D(x,z)=||x-z||_{2}=\sqrt[2]{(\Sigma_{j=1}^{d}(x_{j}-z_{j})^{2})}
    $$

  - **闵可夫斯基距离 (Minkowski Distance)**:

    $$
    D(x,z)=||x-z||_{p}=\sqrt[p]{(\Sigma_{j=1}^{d}(x_{j}-z_{j})^{p})}
    $$

  - **核距离 (Kernel Distance)** (在某些情况下是必需的):

    $$
    D(x,z)=||\Phi(x)-\Phi(z)||=\sqrt{k(x,x)+k(z,z)-2k(x,z)}
    $$

### 3. K-means 算法

#### 1) K-means 目标函数与算法流程

- **假设**: 每个簇中都有一个位于中心的**原型** $\mu_j$。
- 设 $h(x_{i})$ 输出 $x_{i}$ 最近的中心。
- **目标函数 (Objective Function)**:
  
  $$
  min_{\mu_{1},...\mu_{k}}\sum_{i=1}^{n}||\mu_{h(x_{i})}-x_{i}||_{2}^{2}
  $$
  
- K-means 目标是**非凸**的，且没有**封闭解** (Closed Form Solution)。
- 采用**迭代**的近似方法。
- **K-means 算法步骤**:
  1. **初始化**: 随机选择 $k$ 个中心 $\mu_{j} \leftarrow \text{Random}(x_i), j = 1, \dots, k$。
  2. **重复**直到收敛 (目标函数将递减):
     a. **计算簇分配 (标签)**: $y_{i}=h(x_{i})=\operatorname{argmin}_{j}||\mu_{j}-x_{i}||_{2}^{2}, i=1,\dots,n$。
     b. **计算均值 (更新中心)**: $\mu_{j}\leftarrow \text{Mean}(\{x_{i}|y_{i}=j\}), j=1,\dots,k$。

#### 2) K-means 收敛性与初始化改进 (K-means++)

- **收敛性**: K-means 可能收敛到**次优的局部最优解** (suboptimal local optima)。它依赖于不同的起始点 (Random init)。
- **K-means++ 算法**: 使用一种启发式方法寻找更好的初始化。
  - **核心思想**: 顺序选择 $\mu_j$，并以概率 $p_i$ 采样，其中 $p_i$ 与到所有已有质心的最小平方距离**成比例**，即选择**更远的点**作为质心，使中心点更加分散 (make centers more separate)。
  - **K-means++ 初始化步骤**:
    1. 初始化: $\mu_{1} \leftarrow \text{Random } x_{i=1}^n$。
    2. 对于 $j = 2, \dots, k$:
       - 选择新簇中心: $\mu_{j} \leftarrow \text{Random } x_{i=1}^n$，其中概率 $p_i \propto \min_{l<j} ||\mu_{l} - x_{i}||^2$。
    3. 完成初始化后，正常运行 K-means。

#### 3) K 值选择与应用

- **K 值选择**: 没有标准的选取 $k$ 的方法。
  - **观察损失函数**: $k$ 越大损失越低，因此通常观察损失函数曲线的**拐点** (Elbow method)。
  - **添加正则项**: 添加正则化项来惩罚 $k$，例如 **AIC** (Akaike Information Criterion) 和 **BIC** (Bayesian Information Criterion)。
    - **AIC**: $\text{AIC} = 2k - 2 \ln L_K$。
    - **BIC**: $\text{BIC} = k \ln n - 2 \ln L_K$。
    - 其中 $L_K$ 是最大似然 (max likelihood)。
- **K-means 应用**:
  - 可以使用簇标签作为编码来**压缩数据**。
  - 应用于**向量量化 (Vector Quantization, VQ)** 和 **词袋模型 (Bag of Word, BOW)**。
  - **示例**: 在 MNIST 数据集上，用 $k=50$ 训练 K-means，簇中心 (Mean images) 可以被可视化为**想象中的图像**。

### 4. 量化 (Quantization) 与近似最近邻搜索 (Approximate NNS)

#### 1) 最近邻搜索 (NNS)

- **目标**: 给定数据库 $\mathcal{X} = \{x_1, \dots, x_n\}$ 和查询 $q \in \mathbb{R}^d$，返回最近邻 $NN(q) = \min_{x \in \mathcal{X}} \text{dist}(x, q)$。
- **精确 NNS 复杂度**: $O(nd)$。对于大数据集 ($n \rightarrow \infty$) 来说是**难以处理**的 (Intractable)。
- 需要向量搜索的**近似方法**。

#### 2) 近似最近邻搜索的方法

- **减少距离计算次数**: $O(n'd)$, 其中 $n' \ll n$。
  - **方法**: 索引 (Indexing: 倒排索引、搜索树、邻域图)。
- **减少每次距离计算的成本**: $O(nd')$, 其中 $d' \ll d$。
  - **方法**: 哈希 (Hashing: **局部敏感哈希 (LSH)**, **谱哈希 (SH)**); **量化 (Quantization)**: **向量量化 (Vector Quantization)**, **乘积量化 (Product Quantization, PQ)**。
- **量化**是向量搜索中的**主力** (Workhorse)。

### 5. 谱聚类 (Spectral Clustering)

#### 1) 最小割 (MinCut) 问题

- **定义**: 将图划分成两个集合 $A$ 和 $B$，使得连接 $A$ 中顶点到 $B$ 中顶点的边权重**最小**。
- **公式**:
  
  $$
  \text{cut}(A, B) = \sum_{i \in A, j \in B} W_{ij}
  $$
  
- **问题**: 最小割容易解决，但往往将**离群点** (outliers) 分离成独立的簇，得到的划分不理想。

#### 2) 图拉普拉斯算子 (Graph Laplacian)

- **相似度矩阵** $\mathbf{W}$: $\mathbf{W}(i, j) = W(x_i, x_j) = W_{ij}$。
- **度矩阵** $\mathbf{D}$: $\mathbf{D}(i, i) = \sum_j W_{ij}$。
- **非归一化图拉普拉斯算子 (Unnormalized Graph Laplacian)**:
  
  $$
  \mathbf{L} = \mathbf{D} - \mathbf{W}
  $$
  
- **归一化图拉普拉斯算子 (Normalized Graph Laplacian)**:
  - **对称归一化 (Symmetric Normalized)**:

    $$
    \mathbf{L}_{\text{sym}} = \mathbf{D}^{-1/2} \mathbf{L} \mathbf{D}^{-1/2} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2}
    $$

  - **随机游走归一化 (Random Walk Normalized)**:

    $$
    \mathbf{L}_{\text{rw}} = \mathbf{D}^{-1} \mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{W}
    $$

- **作用**: 图拉普拉斯算子使我们能够**高效地求解归一化割** (Normalized Cut)。

#### 3) 归一化割 (Normalized Cut, Ncut)

- **归一化割准则**:
  
  $$
  Ncut(A, B) = \frac{\text{cut}(A, B)}{\text{vol}(A)} + \frac{\text{cut}(A, B)}{\text{vol}(B)}
  $$
  
  其中 $\text{vol}(A) = \sum_{i \in A} d_i$。
- **等价的 Rayleigh 商**:
  - 最小化 Ncut 可转化为求解 **Rayleigh 商**的最小值问题:

    $$
    \min_{\mathbf{f}} Ncut(A, B) = \frac{\mathbf{f}^T \mathbf{L} \mathbf{f}}{\mathbf{f}^T \mathbf{D} \mathbf{f}}
    $$

  - 这是一个**广义特征值问题** (generalized eigen-problem) 的解。

    $$
    \mathbf{L}\mathbf{f} = \lambda \mathbf{D}\mathbf{f} \quad (\text{或 } \mathbf{L}\mathbf{f} = \lambda (\mathbf{D} + \gamma \mathbf{I}) \mathbf{f})
    $$

- **谱分割 (Spectral Segmentation)**: 利用图拉普拉斯矩阵的第 2、3、4 **特征向量** (Eigenvector) 进行分割。

---

## 三、降维 (Dimensionality Reduction)

### 1. 维度灾难 (Curse of Dimensionality)

- 随着相关维度 $d$ 的增加，感兴趣的配置数量可能会**指数级增长**。
- 要区分 $d$ 维上每轴 $v$ 个值，需要 $O(v^d)$ 区域/样本。
- 导致**统计挑战 (statistical challenge)**：高维空间中的配置数量远大于训练样本数量，**典型网格单元中没有训练样本**。

### 2. 流形学习 (Manifold Learning)

- **流形 (Manifold)** 是机器学习中许多思想的基础概念。
- **定义**: 流形是一个**连通区域**，在局部上，它表现为一个**欧几里得空间**。
- **高维数据**通常**集中在一个低维流形附近**。
- [图：数据集中在扭曲的线状一维流形附近]
- **应用示例**: Isomap应用于人脸图像，发现其由左右姿势、上下姿势和光照方向这三个**自由度**决定。应用于手写数字“2”，发现其主要特征是底部循环和顶部拱形。

### 3. 主成分分析 (Principal Component Analysis, PCA)

- PCA 是最著名的**线性降维 (Linear Dimensionality Reduction)** 技术。
- **目标**: 找到一个 **$k$ 维子空间 ($k<d$)**，将原始 $d$ 维数据投影到该子空间上，使得投影后的数据**方差最大**，或**重构误差最小**。
- **目标函数 (Minimum Reconstruction Error)**: 寻找投影矩阵 $\boldsymbol{W} \in \mathbb{R}^{d \times k}$，使得重构误差最小：

$$
\min_{\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}} ||\boldsymbol{X} - \boldsymbol{W}\boldsymbol{W}^{\top}\boldsymbol{X}||_{F}^{2}
$$

- **最优解**: 投影矩阵 $\boldsymbol{W}$ 由数据协方差矩阵 $\boldsymbol{\Sigma} = \frac{1}{n} \boldsymbol{X}^{\top} \boldsymbol{X}$ 的**前 $k$ 个最大特征值**对应的**特征向量**构成。
- **步骤**:
  1. **数据预处理**: 对原始数据 $\boldsymbol{X}$ 进行中心化（减去均值 $\bar{\boldsymbol{x}}$）。
  2. **计算协方差矩阵**: 计算 $\boldsymbol{\Sigma} = \frac{1}{n} \boldsymbol{X}^{\top} \boldsymbol{X}$。
  3. **特征值分解**: 对 $\boldsymbol{\Sigma}$ 进行特征值分解，得到特征值 $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$ 和对应的特征向量 $\boldsymbol{v}_1, \boldsymbol{v}_2, \dots, \boldsymbol{v}_d$。
  4. **选择主成分**: 选取前 $k$ 个最大的特征值对应的特征向量 $\boldsymbol{v}_1, \dots, \boldsymbol{v}_k$ 构成投影矩阵 $\boldsymbol{W} = [\boldsymbol{v}_1, \dots, \boldsymbol{v}_k]$。
  5. **降维**: 投影数据 $\boldsymbol{Z} = \boldsymbol{X}\boldsymbol{W}$。

### 4. 非负矩阵分解 (Nonnegative Matrix Factorization, NMF)

- NMF 将非负数据矩阵 $\boldsymbol{X}$ 分解为两个非负矩阵 $\boldsymbol{L}$ (字典/基) 和 $\boldsymbol{R}$ (编码/系数)。

$$
\boldsymbol{X} \approx \boldsymbol{L}\boldsymbol{R}
$$

- NMF 因子对应于人脸的**部分 (parts of faces)**。
- NMF 字典和编码产生**可解释的、基于部分的表示**。

#### 1) NMF 算法

- 梯度下降 (GD) 通常很慢；随机梯度下降不适用。
- 关键方法是**交替最小化 (Alternating Minimization)**，也称为**块坐标下降 (Block Coordinate Descent, CD)**。

1. 选择起始点 $\boldsymbol{L}^{(0)}$ 和 $\boldsymbol{R}^{(0)}$。
2. 重复直到收敛:
   - 固定 $\boldsymbol{R}$，优化 $\boldsymbol{L}$。
   - 固定 $\boldsymbol{L}$，优化 $\boldsymbol{R}$。

> 步骤 2.1 和 2.2 比完整问题容易得多。

### 5. 稀疏编码与字典学习 (Sparse Coding and Dictionary Learning)

- **稀疏编码 (Sparse Coding)** 旨在找到一个字典 $\boldsymbol{D}$ 和一个稀疏编码 $\boldsymbol{\alpha}$ 来表示数据 $\boldsymbol{x}$，即 $\boldsymbol{x} \approx \boldsymbol{D}\boldsymbol{\alpha}$。
- **在线字典学习 (Online Dictionary Learning)** 算法步骤:
  - 对于 $t = 1, \dots, T$:
    1. 抽取新样本 $\boldsymbol{x}_t$。
    2. 使用**近端梯度下降 (Proximal Gradient Descent, PGD)** 找到稀疏编码 $\boldsymbol{\alpha}_t$:

    $$
    \boldsymbol{\alpha}_{t} = \operatorname{argmin}_{\boldsymbol{\alpha} \in \mathbb{R}^p} \frac{1}{2} ||\boldsymbol{x}_{t} - \boldsymbol{D}_{t-1}\boldsymbol{\alpha}||_{2}^{2} + \lambda ||\boldsymbol{\alpha}||_{1}
    $$

    3. 使用**块坐标方法**更新字典 $\boldsymbol{D}_t$:

    $$
    \boldsymbol{D}_t = \operatorname{argmin}_{\boldsymbol{D} \in \mathcal{D}} \frac{1}{t} \sum_{i=1}^t \frac{1}{2} ||\boldsymbol{x}_i - \boldsymbol{D}\boldsymbol{\alpha}_i||_2^2 + \lambda ||\boldsymbol{\alpha}_i||_1
    $$

- **PGD** 用于 $\boldsymbol{L}_1$ 正则化优化问题 $\min_{\boldsymbol{x}} f(\boldsymbol{x}) + \lambda ||\boldsymbol{x}||_1$。

#### 1) 自学习 (Self-taught Learning)

- 稀疏编码基（例如从手写数字中学到的“笔画”）可以作为特征用于其他任务（例如手写英文字符分类）。
- 稀疏编码特征**单独使用**不一定比原始特征好，但与原始特征**结合使用**时性能会显著提高。

#### 2) 稀疏性原则 (Sparsity Principle)

- **历史**: Wrinch and Jeffrey’s simplicity principle (1921), Lasso (1994–1996), Olshausen and Field’s dictionary learning (1996)。
- **简单性**是可解释性和模型选择的关键。
- **观点**: Simplicity is not enough. Various forms and robustness and stability are also needed. [Yu and Kumbier, 2019].
  
---
