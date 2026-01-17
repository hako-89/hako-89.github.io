---
title: "Linear Model"
published: 2025-09-29
pinned: false
description: "线性模型理论基础，包含梯度优化、正则化、损失函数的统计视角分析等。"
image: "./cover.png"
tags: [Machine Learning, Linear Model, Linear Regression, Regularization, Optimization, Logistic Regression, Softmax Regression]
category: "Machine Learning"
author: "hako"
draft: false
---

## 一、线性回归 (Linear Regression)

### 1. 学习的组成部分 (Components of Learning)

机器学习的目的是找到一个近似于未知目标函数 $f$ 的最终假设 $h$。

| 概念 | 定义/符号 | 描述 |
| :--- | :--- | :--- |
| **未知目标函数** | $f:x\rightarrow y$ | 理想的映射函数，也被称为 **Optimal Classifier (最优分类器)**。 |
| **训练样本** | $(x_{1},y_{1}),...,(x_{n},y_{n})$ | 用于学习的数据集。 |
| **假设空间** | $H$ | 学习算法 $A$ 从中选择假设函数的函数集。 |
| **最终假设** | $h\approx f$ | 学习算法输出的假设函数。 |
| **最优错误率** | **Bayes Error (贝叶斯误差)** | 目标函数 $f$ 在测试数据上的错误率。 |

### 2. 假设空间 (Hypothesis Space)

- 假设空间 $H$ 是一个将 $x\rightarrow y$ 的函数集合，它是我们从中选择预测函数的集合。
- 通常通过参数 $\theta$ 进行参数化，即 $h_{\theta}(x)$。
- 我们希望假设函数具有以下**正则性 (Regularity)**：
  - **Continuity (连续性)**
  - **Smoothness (光滑性)**
  - **Simplicity (简单性)**
- 示例：用于分类的所有**线性超平面**。

### 3. 损失函数 (Loss Function)

- **损失函数** $l:y\times\mathcal{Y}\rightarrow\mathbb{R}_{+}$ 衡量预测 $h(x)$ 与实际输出 $y$ 之间的差异。
- 常见损失函数：
  - **回归 (Regression)**：**平方损失** (Squared Loss, $\text{L2}$ loss)：$l(y,h(x))=(y-h(x))^{2}$。
  - **分类 (Classification)**：$l(y,h(x))=1[y\ne h(x)]$。
- **训练问题**：找到最佳假设 $\hat{\epsilon}(h)$
  
  $$
  \hat{\epsilon}(h)=min_{\theta}\sum_{i=1}^{n}l(h_{\theta}(x_{i}),y_{i})
  $$
  
### 4. 线性回归模型 (Linear Regression Model)

- **问题示例**：预测夏季用电高峰需求 (**回归问题**)。
- **数据探索 (EDA)** 表明线性模型可能适用。
- **假设空间**：
  - 特征向量 $x\in\mathbb{R}^{d}$，响应 $y\in\mathbb{R}$。
  - 将常数维度 $+1$ 添加到特征向量 $x$，等效于在线性模型中添加**截距 (intercept)**：

    $$
    x=\begin{pmatrix}+1\\ x_{1}\\ ...\\ x_{d}\end{pmatrix}\in\mathbb{R}^{d+1}
    $$

  - 假设 $h$ 的预测：

    $$
    h(x)=w_{0}+\sum_{j=1}^{d}w_{j}x_{j}=\sum_{j=0}^{d}w_{j}x_{j}=w\cdot x
    $$

    其中 $w$ 为**参数向量** (法向量)。
- **损失函数**：使用**平方损失 ($\text{L2}$ loss)** 计算训练集上的误差：
  
  $$
  \hat{\epsilon}(h)=\sum_{i=1}^{n}(h(x_{i})-y_{i})^{2}
  $$
  
### 5. 优化 (Optimization)

#### 5.1. 解析解 (Analytical Solution)

- 训练误差的矩阵形式：$\hat{\mathcal{L}}(w) = \|Xw - Y\|_{2}^{2}$
  - $X$ 是 $n \times (d+1)$ 的**数据矩阵 (设计矩阵)**。
  - $Y$ 是 $n \times 1$ 的**标签矩阵**。
- 对 $\hat{\mathcal{L}}(w)$ 求梯度并置零，得到最优参数 $w^*$：
  1. 梯度：$\nabla_{w}\hat{\mathcal{L}}(w) = 2X^{T} (Xw - Y) = 0$
  2. **正规方程 (Normal Equation)**：$X^{T}Xw = X^{T}Y$
  3. 最优解：
  $$
  w^* = (X^{T}X)^{-1}X^{T}Y
  $$
  
- **计算复杂度**：$O(d^3 + d^2n)$。对于高维大数据，成本不可承受。

#### 5.2. 梯度下降 (Gradient Descent, GD)

- 对于一般**可微分**的损失函数，使用梯度下降。
- **梯度 (Gradient)** $\nabla = \nabla_{w}\mathcal{L}(w)$ 指向函数变化最快的方向。
- 迭代步骤 (适用于**凸函数**)：
  1. $w^{(t+1)} \leftarrow w^{(t)} - R\nabla_{w}\hat{\mathcal{L}}(w)|_{w^{(t)}}$
  - $R$ 是**学习率 (Learning Rate)**，必须仔细选择。
- **线性回归的梯度**：$\nabla_{w}\hat{\mathcal{L}}(w) = 2X^{T}(Xw - Y)$
- **计算复杂度**：$O(d n T)$ (T 为迭代次数)。**无需计算逆矩阵**。

#### 5.3. 随机梯度下降 (Stochastic Gradient Descent, SGD)

- 适用于**高维大数据**。
- **迭代步骤** (在每次迭代 $t$ 中)：
  1. **随机采样**一个大小为 $B \ll n$ 的 **minibatch** (小批量) $\left\{(x_{i}, y_{i})\right\}_{i=1}^{B}$。
  2. 计算小批量上的损失梯度：$\Delta_t = \nabla_{w}\mathcal{L}_{t}(w^{(t)})$。
  3. 更新参数：$w^{(t+1)} = w^{(t)} - R\Delta_t$。

### 6. 统计视角：最大似然估计 (Maximum Likelihood Estimation, MLE)

#### 6.1. 似然 (Likelihood)

- 对于参数模型 $\left\{\mathfrak{p}(x; \theta)|\theta \in \Theta\right\}$ 和 i.i.d. 样本 $\mathcal{D} = \{x_{1}, \ldots, x_{n}\}$：
  - **似然函数 (Likelihood)**：$\mathfrak{L}(\hat{\theta}; \mathcal{D}) \triangleq \prod_{i=1}^{n} \mathfrak{p}(x_{i}; \hat{\theta})$
  - **对数似然 (Log-Likelihood)**：$\log \mathfrak{L}(\hat{\theta}; \mathcal{D}) \triangleq \sum_{i=1}^{n} \log \mathfrak{p}(x_{i}; \hat{\theta})$ (数值稳定且易于处理)

#### 6.2. 最大似然估计 (MLE)

- **最大似然估计** $\hat{\theta}_{ML}$：
  
  $$
  \hat{\theta}_{ML} \in \arg \max_{\theta \in \Theta} \log \mathfrak{L}(\mathcal{D}; \hat{\theta})
  $$
  
- **判别模型 (Discriminative Model)** 的 MLE 关注条件概率 $\mathfrak{p}(y|x; \theta)$：
  
  $$
  \hat{\theta} = \arg \max_{\theta} \sum_{i=1}^{n} \log \mathfrak{p}(y_{i}|x_{i}; \theta)
  $$
  
#### 6.3. 线性回归与高斯噪声 (Gaussian Noise)

- 假设：观测值 $y$ 服从**高斯分布 (Gaussian distribution)**
  - $y = w^{T}x + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, \sigma^{2})$ (高斯噪声)。
  - 条件分布：$y_{i}|w, x_{i} \sim \mathcal{N}(w^{T}x_{i}, \sigma^{2})$。
- MLE 的目标函数：
  
  $$
  w^* = \arg \max_{w} \sum_{i=1}^{n} \log \mathfrak{p}(y_{i}|x_{i}; w)
  $$
  
- **推导结论**：在**高斯噪声假设**下，参数 $w$ 的最大似然估计等价于使用**平方损失**的线性回归：
  
  $$
  w^* = \arg \min_{w} \sum_{i=1}^{n}(y_{i}-w^{T}x_{i})^{2}
  $$
  
---

## 二、非线性化 (Nonlinearization)

### 1. 基础函数 (Basis Function)

- 线性模型假设 $y=w^{T}x$ 是线性的。
- 为了处理**非线性关系**，我们可以将线性模型应用在一个更高维度的特征空间 $\mathcal{X}' = \Phi(\mathcal{X})$ 中。
- **假设**：$h(x) = w^{T}\Phi(x)$。
- **特征映射**：$x' = \Phi(x) = (z_{1}(x), \ldots, z_{m}(x))$。
- $\{z_{j}\}_{j=1}^{m}$ 称为**基础函数 (Basis Functions)**。

| 基础函数类型 | 示例 $\Phi(x)$ |
| :--- | :--- |
| **多项式基础函数** (Polynomial) | 1-D: $(1, x_1, x_1^2, x_1^3, \ldots)$ |
| **径向基础函数** (Radial Basis Function, RBF) | $z_{j}(x) = \exp(-\frac{\|x - \mu_{j}\|_{2}^{2}}{2\sigma^{2}})$ |

- **风险**：更复杂的特征可能导致训练集效果更好，但测试集效果更差，即**过拟合 (Overfitting)**。

---

## 三、正则化 (Regularization)

### 1. $\text{L2}$ 正则化 (Ridge Regression)

- 用于处理**过拟合**，通过**控制参数 $w$ 的范数**来控制模型复杂度。
- **$\text{L2}$ 正则化的线性回归**被称为 **岭回归 (Ridge Regression)**。
- **目标函数** (加入 $\text{L2}$-范数惩罚项 $\lambda \|w\|_{2}^{2}$)：
  
  $$
  w^* = \arg \min_{w} \sum_{i=1}^{n}(w^{T}x_{i}-y_{i})^{2} + \lambda \|w\|_{2}^{2}
  $$
  
- **作用**：促使模型参数**权重衰减 (Weight decay)**，使参数值接近于零。

### 2. $\text{L1}$ 正则化 (Lasso Regression)

- **$\text{L1}$ 正则化的线性回归**被称为 **套索回归 (Lasso Regression)**。
- **目标函数** (加入 $\text{L1}$-范数惩罚项 $\lambda \|w\|_{1}$)：
  
  $$
  w^* = \arg \min_{w} \sum_{i=1}^{n}(w^{T}x_{i}-y_{i})^{2} + \lambda \|w\|_{1}
  $$
  
- **作用**：促进解的**稀疏性 (Sparsity)**，能够实现**特征选择 (Feature selection)**，将不重要特征的权重直接降为零。

### 3. $\text{L1}$ vs. $\text{L2}$ 正则化路径比较

| 特征 | $\text{L2}$ 正则化 (Ridge) | $\text{L1}$ 正则化 (Lasso) |
| :--- | :--- | :--- |
| **范数** | $\text{L2}$ 范数 (平方项) | $\text{L1}$ 范数 (绝对值之和) |
| **几何约束** | 球体 (圆) | 多面体 (菱形/方形) |
| **对参数的影响** | 权重衰减 (Weight decay)，使参数值变小 | 特征选择 (Feature selection)，使部分参数恰好为零 (稀疏性) |

### 4. 优化非光滑问题 (Non-smooth Optimization)

- $\text{L1}$ 正则化损失函数**非光滑 (non-smooth)**。
- 对于**凸函数 (Convex function)**，**所有局部最小值都是全局最小值**。
- **次梯度下降 (Subgradient Descent)**：适用于一般非光滑问题。
  - **次梯度 (Subgradient)** 是函数在不可导点处的导数的推广。
- **近端梯度下降 (Proximal Gradient Descent)**：
  - 适用于 $\min_{w} f(w) + \lambda \|w\|_{1}$ 形式的问题，其中 $f(w)$ 可微。
  - 更新步骤涉及最小化二次近似项加上 $\text{L1}$ 惩罚项。

### 5. 统计视角：最大后验估计 (Maximum a Posteriori Estimation, MAP)

#### 5.1. 贝叶斯规则 (Bayes Rule)

$$
\mathfrak{p}(\theta|\mathcal{D}) = \frac{\mathfrak{p}(\mathcal{D}|\theta)\mathfrak{p}(\theta)}{\mathfrak{p}(\mathcal{D})} \quad \text{或} \quad \mathfrak{p}(\theta|\mathcal{D}) \propto \mathfrak{p}(\mathcal{D}|\theta)\mathfrak{p}(\theta)
$$

| 概念 | 符号 | 描述 |
| :--- | :--- | :--- |
| **后验 (Posterior)** | $\mathfrak{p}(\theta\|\mathcal{D})$ | 观察到数据 $\mathcal{D}$ 后，参数 $\theta$ 的概率分布。 |
| **似然 (Likelihood)** | $\mathfrak{p}(\mathcal{D}\|\theta)$ | 给定参数 $\theta$，观察到数据 $\mathcal{D}$ 的概率。 |
| **先验 (Prior)** | $\mathfrak{p}(\theta)$ | 在观察到数据之前，参数 $\theta$ 的预先知识。 |

#### 5.2. 最大后验估计 (MAP)

- **MAP 估计** $\hat{\theta}_{MAP}$：
  
  $$
  \hat{\theta}_{MAP} = \arg \max_{\theta} \log \mathfrak{p}(\theta|\mathcal{D}) = \arg \max_{\theta} \{ \log \mathfrak{p}(\mathcal{D}|\theta) + \log \mathfrak{p}(\theta) \}
  $$
  
- **MLE** 等价于 MAP 估计中使用了**均匀先验 (Uniform Prior)**。
- 引入**先验** $\mathfrak{p}(\theta)$ 起到了**正则化**的作用。

#### 5.3. 正则化线性回归与先验 (Prior)

| 正则化类型 | 对应统计学先验 | 作用 | MAP 目标函数 (与 $\min$ 等价) |
| :--- | :--- | :--- | :--- |
| $\text{L2}$ (Ridge) | **高斯先验 (Gaussian Prior)**：$\mathfrak{p}(w) = \mathcal{N}(0, \sigma_{\theta}^2 I)$ | 权重衰减 (Weight decay) | $\sum_{i=1}^{n}(y_{i}-w^{T}x_{i})^{2} + \lambda \|w\|_{2}^{2}$ |
| $\text{L1}$ (Lasso) | **拉普拉斯先验 (Laplacian Prior)** | 稀疏性促进 (Sparsity-promoting) | $\sum_{i=1}^{n}(y_{i}-w^{T}x_{i})^{2} + \lambda \|w\|_{1}$ |

---

## 四、线性分类 (Linear Classification)

### 1. Logistic 回归 (Logistic Regression, LR)

- **问题**：二分类问题 (Discrete Label Space $\mathcal{Y} = \{0, 1\}$)。
- **假设函数**：使用 **Sigmoid 函数** 将线性输出映射到概率。
  
  $$
  h_{w}(x) = g(w^{T}x) = \frac{1}{1 + \exp(-w^{T}x)}
  $$
  
  $$
  \mathfrak{p}(y=1|x; w) = h_{w}(x), \quad \mathfrak{p}(y=0|x; w) = 1 - h_{w}(x)
  $$
  
- **损失函数**：**交叉熵损失 (Cross-Entropy Loss)**，等价于**负对数似然 (Negative Log-Likelihood)**。
  
  $$
  \ell(h(x_{i}), y_{i}) = \begin{cases} -\log h_{w}(x_{i}) & \text{if } y_{i} = 1 \\ -\log (1-h_{w}(x_{i})) & \text{if } y_{i} = 0 \end{cases}
  $$
  
  - 该损失函数是**凸的 (Convex)**。
- **正则化**：当数据**线性可分 (Linearly Separable)** 时，权重 $w$ 可能趋于无穷大 (过拟合)，因此需要 $\text{L1}$ 或 $\text{L2}$ 正则化来**防止权重发散**。
- **优化**：没有解析解，使用 **GD, SGD, Newton method** 等迭代优化算法。

### 2. Softmax 回归 (Softmax Regression)

- **问题**：**多分类问题** (Multiclass Classification)。
- **Softmax 函数**：将多个线性预测器的输出规范化为一个**概率向量**。
  
  $$
  \mathfrak{p}(y=k|x; W) = \frac{\exp(w_{k}^{T}x)}{\sum_{L=1}^{m} \exp(w_{L}^{T}x)}
  $$
  
  - $\sum \mathfrak{p} = 1$，且每个 $\mathfrak{p} > 0$。
  - 预测类别为 $\arg \max_{k} \mathfrak{p}(y=k|x)$。
- **损失函数**：最小化**负对数似然** (Negative Log-Likelihood)，即 **Cross-Entropy Loss**。
  
  $$
  \mathcal{L}_{NLL}(W; \mathcal{D}) = \min_{w} -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{m} \mathbb{I}(y_{i}=k) \log \frac{\exp(w_{k}^{T}x_{i})}{\sum_{L=1}^{m} \exp(w_{L}^{T}x_{i})}
  $$
  
  - 该损失函数是**凸的 (Convex)**。
- **优化**：使用 **GD, SGD**，同样需要**正则化**以防止参数发散。

---
