---
title: "Support Vector Machine"
published: 2025-10-06
pinned: false
description: "Support Vector Machine (SVM) 核心内容：包含硬间隔、软间隔、优化方法及核方法等。"
image: "./cover.png"
tags: [Machine Learning, SVM, Support Vector Machine, Kernel Method]
category: "Machine Learning"
author: "hako"
draft: false
---

## 一、**Support Vector Machine**

### 1. **线性分类** (Linear Classification)

- **实例空间**: $\mathcal{X}=\mathbb{R}^{d}$
- **离散标签空间**: $y=\{-1,1\}$ (约定: 将 $\{0,1\}$ 更改为 $\{-1,+1\}$)
- **线性假设**: $h_{w}=w\cdot x+b$ (约定: 明确写出截距 $b$)
- **分类标准** (01-损失):
  - **正确分类**: $y(w\cdot x+b)>0$
  - **错误分类**: $y(w\cdot x+b)<0$
  - **决策边界**: $w\cdot x+b=0$

> **问题**: 如何选择最佳的线性分类器（超平面）?

### 2. **最大间隔分类器** (Maximum Margin Classifier)

- 在线性可分设置下，能产生最低测试误差的分类器，是与训练数据有**最大间隔** (largest margin) 的分类器。
- 它是对**含噪数据** (noisy data) **最鲁棒的分类器** (most robust classifier)。

#### 1) **间隔** (Margin)

- **定义**: 间隔是到两个类别最近点的距离的两倍。
- 间隔表示分类器能容忍的**最大噪声**。
- **目标**: 找到具有最大间隔的线性分类器。
- **约束优化问题** (初步表达):
  
  $$
  \max_{w,b}\text{margin}(w,b)
  $$
  
  $$
  \text{s. t.} \quad y_{i}(w\cdot x_{i}+b)\ge1, \quad 1\le i\le n
  $$
  
#### 2) **点到超平面的距离** (Point-Hyperplane Distance)

- **定理**: 从点 $x_{0}$ 到法向量为 $w$、截距为 $b$ 的超平面的距离为 $\frac{|w\cdot x_{0}+b|}{||w||}.$

#### 3) **硬间隔支持向量机** (Hard-Margin Support Vector Machine)

- **间隔的量化**: 最接近分类器的点位于 $w\cdot x+b=\pm1$ 两条线上。
- **间隔** $\gamma$: $\gamma=\frac{1}{||w||_{2}}+\frac{|-1|}{||w||_{2}}=\frac{2}{||w||_{2}}$
- **硬间隔SVM (Hard-margin SVM)**：
  - **最大化间隔的非凸问题**:

    $$
    \max_{w,b}\frac{2}{||w||_{2}}
    $$

    $$
    \text{s. t.} \quad y_{i}(w\cdot x_{i}+b)\ge1, \quad 1\le i\le n
    $$

  - **等价的最小化凸问题**:

    $$
    \min_{w,b}\frac{1}{2}||w||_{2}^{2}
    $$

    $$
    \text{s. t.} \quad y_{i}(w\cdot x_{i}+b)\ge1, \quad 1\le i\le n
    $$

    > 注：$\frac{1}{2}$ 是为计算方便而添加的。$||w||_{2}^{2}$ 是 $w$ 的凸函数。
- 仅适用于**线性可分** (Linearly Separable) 的情况，**实践中很少使用**。

#### 4) **软间隔支持向量机** (Soft-Margin Support Vector Machine)

- 针对**线性不可分** (Linearly Non-Separable) 或存在**离群点** (Outliers) 的情况。
- **思想**: 允许一些点位于间隔的错误一侧，但数量应很小。
- 引入**松弛变量** (Slack variables) $\xi_i \ge 0$.
- **软间隔SVM (Primal Problem) (QP 形式)**:
  
  $$
  \min_{w,b,\xi} \frac{1}{2}||w||_{2}^{2} + C\sum_{i=1}^{n}\xi_i
  $$
  
  $$
  \text{s. t.} \quad y_{i}(w\cdot x_{i}+b)\ge1-\xi_i, \quad \xi_{i}\ge0, \quad 1\le i\le n
  $$
  
#### 5) **软间隔SVM：多分类** (Multiclass Classification)

- **One-vs-Rest (独立训练)**: 独立训练 $T$ 个分类器。
  - **问题**: 预测值可能不可比，导致**歧义** (Ambiguity)。
- **联合训练** (Joint Training):
  
  $$
  \min_{w_1,\ldots,w_T,b,\xi} \frac{1}{2}\sum_{k=1}^{T} ||w_k||_{2}^{2} + C\sum_{i=1}^{n}\xi_i
  $$
  
  $$
  \text{s. t.} \quad \forall j \ne y_i, \quad w_{y_i}\cdot x_i + b_{y_i} \ge w_j\cdot x_i + b_j + 1 - \xi_i, \quad \xi_{i}\ge0, \quad i\in [n]
  $$
  
---

## 二、**优化** (Optimization)

### 1. **约束优化** (Constrained Optimization)

- 优化问题通常分为**原问题** (Primal Problem) 和**对偶问题** (Dual Problem)。
- SVM 的硬间隔和软间隔问题都是带有**不等式约束**的优化问题。
  - 硬间隔 SVM (原问题): $\min_{w,b}\frac{1}{2}||w||_{2}^{2}$, s. t. $y_{i}(w\cdot x_{i}+b)\ge1$
  - 软间隔 SVM (原问题): $\min_{w,b,\xi} \frac{1}{2}||w||_{2}^{2} + C\sum_{i=1}^{n}\xi_i$, s. t. $y_{i}(w\cdot x_{i}+b)\ge1-\xi_i, \xi_{i}\ge0$

#### 1) **一般优化问题形式**

考虑以下一般形式的优化问题：

$$
\min_{x \in \mathbb{R}^{d}} f(x)
$$

$$
\text{s. t.} \quad g_{i}(x) \le 0, \quad i=1, \ldots, m
$$

$$
\quad \quad h_{j}(x) = 0, \quad j=1, \ldots, p
$$

- $f(x)$ 是目标函数。
- $g_{i}(x) \le 0$ 是**不等式约束** (inequality constraints)。
- $h_{j}(x) = 0$ 是**等式约束** (equality constraints)。

#### 2) **拉格朗日函数** (Lagrangian Function)

- 引入拉格朗日乘子 (Lagrange multipliers) $\alpha_i \ge 0$ (对应不等式约束) 和 $\beta_j$ (对应等式约束)。
- **定义**: **拉格朗日函数** $\mathcal{L}(x, \alpha, \beta)$ 将约束项合并到目标函数中：
  
  $$
  \mathcal{L}(x, \alpha, \beta) = f(x) + \sum_{i=1}^{m} \alpha_{i} g_{i}(x) + \sum_{j=1}^{p} \beta_{j} h_{j}(x)
  $$
  
#### 3) **原问题与对偶问题**

- **原问题** (Primal Problem): 求解 $x$ 使得 $\mathcal{L}(x, \alpha, \beta)$ 在 $\alpha, \beta$ 固定时最小化，然后对结果关于 $\alpha, \beta$ 最大化，从而恢复约束。
  
  $$
  \min_{x} \max_{\alpha \ge 0, \beta} \mathcal{L}(x, \alpha, \beta)
  $$
  
- **对偶问题** (Dual Problem): 求解 $\alpha, \beta$ 使得 $\mathcal{L}(x, \alpha, \beta)$ 在 $x$ 固定时最大化，然后对结果关于 $x$ 最小化。
  
  $$
  \max_{\alpha \ge 0, \beta} \min_{x} \mathcal{L}(x, \alpha, \beta)
  $$
  
- **对偶目标函数** (Lagrangian dual function):
  
  $$
  g(\alpha, \beta) \triangleq \min_{x} \mathcal{L}(x, \alpha, \beta)
  $$
  
#### 4) **弱对偶性与强对偶性**

- **弱对偶性** (Weak Duality): 对偶问题的最优值总是**小于或等于**原问题的最优值。
  
  $$
  \max_{\alpha \ge 0, \beta} g(\alpha, \beta) \le \min_{x} f(x)
  $$
  
- **对偶问题总是凹的** (concave)。
  - **定理**: 仿射函数 (affine functions) 的逐点下确界 (pointwise infimum) 是凹函数。
- **强对偶性** (Strong Duality): 如果原问题是**凸的** (Convex)，且满足 **Slater 条件** (或其他约束资格条件)，则原问题最优值等于对偶问题最优值。
  
  $$
  \max_{\alpha \ge 0, \beta} g(\alpha, \beta) = \min_{x} f(x)
  $$
  
- 对于 SVM，由于原问题是凸的且满足约束条件，因此**强对偶性成立**。这意味着我们可以通过求解通常更容易的**对偶问题**来获得原问题的最优解。

### 2. **对偶问题：拉格朗日法** (Dual Problem: Lagrangian Method)

#### 1) **软间隔SVM的拉格朗日函数**

对于软间隔SVM的原问题:

$$
\min_{w,b,\xi} \frac{1}{2}||w||_{2}^{2} + C\sum_{i=1}^{n}\xi_i
$$

$$
\text{s. t.} \quad 1 - \xi_i - y_{i}(w\cdot x_{i}+b) \le 0
$$

$$
\quad \quad -\xi_{i} \le 0
$$

引入拉格朗日乘子 $\alpha_i \ge 0$ 和 $\mu_i \ge 0$，得到拉格朗日函数:

$$
\mathcal{L}(w, b, \xi, \alpha, \mu) = \frac{1}{2}||w||_{2}^{2} + C\sum_{i=1}^{n}\xi_i + \sum_{i=1}^{n} \alpha_i(1 - \xi_i - y_{i}(w\cdot x_i+b)) - \sum_{i=1}^{n} \mu_i\xi_i
$$

#### 2) **对偶目标函数求解**

求解对偶目标函数 $g(\alpha, \mu) = \min_{w, b, \xi} \mathcal{L}(w, b, \xi, \alpha, \mu)$，通过对 $w, b, \xi_i$ 求偏导并设为零：

- $\frac{\partial \mathcal{L}}{\partial w} = 0 \Rightarrow w = \sum_{i=1}^{n} \alpha_i y_{i}x_i$
- $\frac{\partial \mathcal{L}}{\partial b} = 0 \Rightarrow \sum_{i=1}^{n} \alpha_i y_{i} = 0$
- $\frac{\partial \mathcal{L}}{\partial \xi_i} = 0 \Rightarrow C - \alpha_i - \mu_i = 0 \Rightarrow C = \alpha_i + \mu_i$

#### 3) **软间隔SVM的最终对偶问题**

将上述三个条件代入 $\mathcal{L}$ 函数并消除 $\mu_i$ (由于 $\mu_i \ge 0$ 且 $C = \alpha_i + \mu_i$，得到 $\alpha_i \le C$)，得到最终的对偶问题:

$$
\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i\alpha_j y_{i}y_j x_i \cdot x_j
$$

$$
\text{s. t.} \quad \sum_{i=1}^{n} \alpha_i y_{i} = 0, \quad 0 \le \alpha_i \le C, \quad 1 \le i \le n
$$

- **求解**: 这是一个**二次规划** (Quadratic Program, QP) 问题，可使用 **SMO** (Sequential Minimal Optimization) 算法高效求解。
- **支持向量** (Support Vectors): 只有 $\alpha_i > 0$ 的样本点 $x_i$ 才会对 $w$ 的计算做出贡献，这些点被称为支持向量。

#### 4) **KKT 条件** (Karush-Kuhn-Tucker Conditions)

最优解 $\hat{w}, \hat{b}, \hat{\xi}, \hat{\alpha}, \hat{\mu}$ 必须满足 KKT 条件，其中包括**互补松弛条件** (Complementary Slackness):

$$
\hat{\alpha}_{i} [1 - \hat{\xi}_{i} - y_{i}(\hat{w}\cdot x_{i}+\hat{b})] = 0, \quad 1 \le i \le n
$$

$$
\hat{\mu}_{i} \hat{\xi}_{i} = 0, \quad 1 \le i \le n
$$

这些条件有助于识别支持向量，并用于求解截距 $b$。

### 3. **原问题：随机梯度下降** (Primal Problem: Stochastic Gradient Descent)

#### 1) **软间隔SVM的无约束形式**

软间隔SVM的原问题等价于最小化正则化风险 (Regularized Risk Minimization, RRM):

$$
\min_{w,b} \frac{1}{2}||w||_{2}^{2} + C\sum_{i=1}^{n}\max(0, 1 - y_{i}(w\cdot x_{i}+b))
$$

其中 $\max(0, 1 - y_{i}(w\cdot x_{i}+b))$ 是 **Hinge loss** (铰链损失)。

#### 2) **SGD 算法**

由于 Hinge loss 是凸函数但**不可微** (Non-differentiable)（在 $1 - y_{i}(w\cdot x_{i}+b)=0$ 处），我们使用**随机子梯度下降** (Stochastic Subgradient Descent)。

- **Hinge Loss 的子梯度** $\frac{\partial \ell}{\partial (w\cdot x+b)}$:
  
  $$
  \frac{\partial \ell}{\partial (w\cdot x+b)} = \begin{cases} 0 & \text{if } y(w\cdot x+b) \ge 1 \\ -y & \text{otherwise} \end{cases}
  $$
  
- **SGD 步骤**:
  1. **初始化** $w^0, b^0$。
  2. **在每次迭代** ($t \le T$) 中:
      - 随机均匀选择一个样本 $(x_i, y_i)$。
      - **计算子梯度** $g_t$ (对目标函数 $\frac{1}{2}||w||_{2}^{2} + C \ell$ 的子梯度):

        $$
        g_t = \begin{cases} w^{t} & \text{if } y_{i}(w^{t}\cdot x_{i}+b) \ge 1 \\ w^{t} - C y_{i}x_{i} & \text{otherwise} \end{cases}
        $$

      - **更新**: $w^{t+1} \leftarrow w^{t} - \eta g_t$ ($\eta$ 为学习率 $\alpha$)
  3. **输出** (平均解): $\bar{w} = \frac{1}{T}\sum_{t=1}^{T} w^{t}$

- **收敛率**: 对于凸优化，收敛率通常为 $\mathcal{O}(\frac{1}{T})$。

#### 3) **优势**

- SGD 避免了求解复杂的二次规划问题，特别是在处理**大规模数据**时具有速度优势。

#### 4) **支持向量回归** (Support Vector Regression, SVR)

- 使用 **$\epsilon$-不敏感损失** ($\epsilon$-insensitive loss): $\max(|h_{w}(x) - y| - \epsilon, 0)$
- **优化问题**:
  
  $$
  \min_{w,b} \frac{1}{2}||w||_{2}^{2} + C\sum_{i=1}^{n}\max(0, |w\cdot x_{i}+b - y_{i}| - \epsilon)
  $$
  
- 可使用 SGD 有效求解。

#### 5) **转导支持向量机** (Transductive SVM, TSVM)

- 用于**半监督学习** (Semi-supervised learning)，结合有标签数据 $\mathcal{L}$ 和无标签数据 $\mathcal{U}$。
- **优化问题** (使用特征映射 $\phi$):
  
  $$
  \min_{w,b,\xi} \frac{1}{2}||w||_{2}^{2} + C_{\mathcal{L}}\sum_{i=1}^{\ell}\xi_i + C_{\mathcal{U}}\sum_{i=\ell+1}^{\ell+u}\xi_i
  $$
  
  $$
  \text{s. t.} \quad y_{i}(w\cdot \phi(x_i)+b)\ge1-\xi_i, \quad \xi_{i}\ge0, \quad 1\le i\le \ell
  $$
  
  $$
  |w\cdot \phi(x_i)+b|\ge1-\xi_i, \quad \xi_{i}\ge0, \quad \ell+1\le i\le \ell+u
  $$
  
---

## 三、**核方法** (Kernel Method)

### 1. **表示定理** (Representer Theorem)

- **核心思想**: 对于一类特定的正则化风险最小化问题，其最优解 $w^*$ 可以表示为训练数据特征映射的**线性组合**。这为我们引出核函数提供了理论基础。

#### 1) **定理** (正则化风险最小化解的表示)

- 考虑以下形式的正则化风险最小化问题：
  
  $$
  \min_{w} f(w) = \mathcal{A}(w \cdot \phi(x_1), \ldots, w \cdot \phi(x_n)) + \Omega(||w||)
  $$
  
  - $\phi: \mathcal{X} \to \mathcal{H}$ 是将输入空间 $\mathcal{X}$ 映射到希尔伯特空间 (Hilbert space) $\mathcal{H}$ 的**特征映射**。
  - $\mathcal{A}: \mathbb{R}^n \to \mathbb{R}$ 是一个任意函数 (通常与损失函数相关)。
  - $\Omega: [0, \infty) \to \mathbb{R}$ 是一个**非递减函数** (nondecreasing function) (通常是正则项，如 $\Omega(||w||) = \frac{1}{2}||w||^2$)。
- **结论**: 则存在向量 $\alpha = (\alpha_1, \ldots, \alpha_n) \in \mathbb{R}^n$ 使得
  
  $$
  w^*=\sum_{i=1}^{n} \alpha_i \phi(x_i)
  $$
  
  是 $\min_{w} f(w)$ 的一个最优解。
- **推论**: **最优解** $w^*$ 位于训练样本的特征映射 $\{\phi(x_1), \ldots, \phi(x_n)\}$ 所张成的空间 $\text{span}(\{\phi(x_1), \ldots, \phi(x_n)\})$ 内。

### 2. **核支持向量机** (Kernel SVM)

#### 1) **核化** (Kernelization)

- 基于表示定理，将 $w = \sum_{i=1}^{n} \alpha_i \phi(x_i)$ 代入到软间隔 SVM 的**对偶问题**中。
- **核函数** (Kernel Function) $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 的定义为：
  
  $$
  k(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)
  $$
  
  它计算了两个数据点在特征空间 $\mathcal{H}$ 中的内积，而无需显式计算 $\phi(x)$。
- **内积替换**:
  - $w \cdot \phi(x_j) = \sum_{i=1}^{n} \alpha_i \phi(x_i) \cdot \phi(x_j) = \sum_{i=1}^{n} \alpha_i k(x_i, x_j)$
  - 范数替换: $||w||^2 = w \cdot w = \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i\alpha_j \phi(x_i) \cdot \phi(x_j) = \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i\alpha_j k(x_i, x_j)$

#### 2) **软间隔SVM的对偶问题（核函数形式）**

- 通过将原始对偶问题中的 $x_i \cdot x_j$ 替换为核函数 $k(x_i, x_j)$，得到：
  
  $$
  \max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i\alpha_j y_{i}y_j k(x_i, x_j)
  $$
  
  $$
  \text{s. t.} \quad \sum_{i=1}^{n} \alpha_i y_{i} = 0, \quad 0 \le \alpha_i \le C, \quad 1 \le i \le n
  $$
  
- 优化问题仅依赖于核函数矩阵 $\mathbf{K} \in \mathbb{R}^{n \times n}$，其中 $\mathbf{K}_{ij} = k(x_i, x_j)$。

#### 3) **决策函数 (Testing)**

- 求解 $\alpha$ 后，新的数据点 $x$ 的分类决策函数为：
  
  $$
  h(x) = \text{sgn}(w \cdot \phi(x) + b) = \text{sgn}\left(\sum_{i=1}^{n} \alpha_i y_{i}k(x_i, x) + b\right)
  $$
  
- **优势**:
  1. 可以在**无限维**的特征空间中进行线性分类。
  2. 在训练和测试过程中，**避免了显式计算**高维特征 $\phi(x)$，只需计算**核函数** $k(x_i, x)$，从而利用**核技巧** (Kernel Trick) 解决了维度灾难。

#### 4) **常用核函数** (Common Kernels)

- **线性核** (Linear Kernel): $k(x_i, x_j) = x_i \cdot x_j$ (退化为原始线性 SVM)。
- **多项式核** (Polynomial Kernel): $k(x_i, x_j) = (\gamma x_i \cdot x_j + r)^p$
- **径向基函数核/高斯核** (Radial Basis Function Kernel, **RBF Kernel**):
  
  $$
  k(X_1, X_2) = \exp(-\gamma ||X_1 - X_2||^2)
  $$
  
  其中 $\gamma = 1 / (2\sigma^2)$。RBF 核诱导的特征向量 $\phi(X)$ 是**无限维**的 (在 $\ell_2$ 空间中)。

#### 5) **结构化数据的核函数** (Kernel Functions on Structured Data)

- 核函数可以编码关于**结构化数据**的**先验知识**，如序列、图、文本等。
- **图核** (Graph kernel): 衡量图结构数据的相似性。
  
  $$
  k(x_1, x_2) \propto \exp(-\frac{1}{\sigma^2} d(x_1, x_2)^2)
  $$
  
  其中 $d$ 是输入空间 (如图空间) 中的相似性度量。
- **字符串核** (String kernel): 衡量两个字符串的相似性，例如通过计算它们共享子序列的程度。
  
  $$
  k(X_1, X_2) = \sum_{u \in \Sigma^*} \lambda_u \psi_u(X_1)\psi_u(X_2)
  $$
  
  - $\psi_u(X)$ 是子字符串 $u$ 在字符串 $X$ 中出现的次数。
  - $\Sigma^*$ 是所有有限长度字符串的集合。

### 3. **其他核方法SVM变体**

#### 1) **One-Class SVM ($\nu$-SVM)**

- **目的**: 异常检测或新颖性检测。在特征空间 $\phi$ 中，找到一个超平面将所有数据点与**原点**分开，并最大化超平面到原点的距离 $\rho$。
- **优化问题**:
  
  $$
  \min_{w, \xi, \rho} \frac{1}{2}||w||_{2}^{2} + \frac{1}{\nu n}\sum_{i=1}^{n}\xi_i - \rho
  $$
  
  $$
  \text{s. t.} \quad w \cdot \phi(x_i) \ge \rho - \xi_i, \quad \xi_{i}\ge0, \quad 1\le i\le n
  $$
  
- **参数 $\nu$**: 约束异常值比例的**上界**。

#### 2) **支持向量数据描述** (Support Vector Data Description, SVDD)

- **目的**: 在特征空间 $\phi$ 中，找到一个**最小体积**的**球体边界**，包含尽可能多的数据点。
- **决策函数**: $h(x) = \text{sgn}(R^2 - ||\phi(x) - a||^2)$ ($a$ 为球心，$R$ 为半径)。
- **优化问题**: 最小化球体半径 $R^2$。
  
  $$
  \min_{R, a, \xi} R^2 + C\sum_{i=1}^{n}\xi_i
  $$
  
  $$
  \text{s. t.} \quad ||\phi(x_i) - a||^2 \le R^2 + \xi_i, \quad \xi_{i}\ge0, \quad 1\le i\le n
  $$
  
---
