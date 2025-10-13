---
title: "机器学习第二讲：线性模型(Linear Models)"
published: 2025-09-29
updated: 2025-10-13
pinned: false
description: A personal reflection and review of Lecture 2 -- Linear Models -- of Machine Learning.
image: cover.png
tags: ["ml", "lm", "reflection"]
category: ml
author: "hAk0"
draft: false
---

## 知识大纲

- **学习框架总览**
  - 机器学习五要素
  - 模型选择策略
- **线性回归**
  - 假设空间与损失函数
  - 解析解与梯度优化
  - 统计解释（MLE）
- **非线性化方法**
  - 基函数扩展
  - 局部加权回归
- **正则化技术**
  - L2（岭回归）与 L1（Lasso）
  - 贝叶斯视角（MAP）
- **线性分类模型**
  - Logistic 回归
  - Softmax 回归（多分类）

---

## 一、学习框架总览

### 1. 机器学习五要素

| 组成 | 符号 | 说明 |
|---|---|---|
| 未知目标函数 | $f: \mathcal{X}\to\mathcal{Y}$ | 理论上最优的映射 |
| 训练集 | $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n$ | 独立同分布样本 |
| 假设空间 | $\mathcal{H}=\{h_\theta\}$ | 备选模型集合（本文 $\mathcal{H}$ 为线性/广义线性） |
| 学习算法 | $\mathcal{A}$ | 在 $\mathcal{H}$ 中搜索最优 $h$ 的过程 |
| 最终假设 | $h\approx f$ | 使期望误差（Bayes error）最小 |

> **What Model to Choose?**
>
> - **Exploratory Data Analysis (EDA)** to ease model selection

---

## 二、线性回归

### 1. 假设空间与损失函数

#### 1）假设形式

$$
h_w(x)=w^\top x,\quad x\in\mathbb{R}^{d+1},\ w\in\mathbb{R}^{d+1}\ (\text{已增广常数维度})
$$

#### 2）损失函数

- **平方损失（L2-loss）- regression**
  $$
  \ell\bigl(h_w(x),y\bigr)=\bigl(w^\top x-y\bigr)^2
  $$
- **0/1损失 - classification**
  $$
  \ell\bigl(h_w(x),y\bigr)=1[y≠h_w(x)]
  $$

#### 3）训练目标

$$
\hat w=\arg\min_w\sum_{i=1}^{n}\bigl(w^\top x_i-y_i\bigr)^2
$$

### 2. 解析解（Normal Equation）

记设计矩阵 $X\in\mathbb{R}^{n\times (d+1)}$，标签向量 $y\in\mathbb{R}^{n}$  
$$
\hat w=(X^\top X)^{-1}X^\top y
$$

- **复杂度**：$\mathcal{O}(d^2(d+n))$，高维/大数据不可用。

### 3. 梯度下降 (GD & SGD)

#### 1）梯度计算

对线性回归平方损失  
$$
\hat\varepsilon(w)=\sum_{i=1}^{n}(w^\top x_i-y_i)^2
$$
求梯度得  
$$
\nabla_w\hat\varepsilon(w)=2X^\top(Xw-y)
$$

#### 2）更新规则

- **全量梯度下降 (GD)**  
  $$
  w^{t+1}=w^t-\eta\cdot 2X^\top(Xw^t-y)
  $$
- **随机梯度下降 (SGD)**  
  随机选取一个（或 mini-batch）样本 $i$，  
  $$
  w^{t+1}=w^t-2\eta (x_i^\top w^t-y_i)x_i
  $$
  > 有利于逃离**波动较大的鞍点**

#### 3）学习率 $\eta$ 的选择

| 方案 | 说明 |
|---|---|
| 固定常数 | 简单但需手动调优 |
| 衰减策略 | $\eta_t=\eta_0/(1+\alpha t)$ 或阶梯式，保证收敛 |
| 自适应方法 | Adam、AdaGrad、RMSprop 等，无需精细调参 |

#### 4）收敛速率

- **凸函数**下，GD 理论收敛速率 $O(1/T)$；SGD 为 $O(1/\sqrt{T})$。
- 实际中，SGD 一次 epoch 即可遍历全部数据，通常更快获得满意解。

#### 5）计算复杂度

| 算法 | 每步复杂度 | 备注 |
|---|---|---|
| GD | $\mathcal{O}(dn)$ | 需全量矩阵乘法 $X^\top X w$ |
| mini-batch SGD | $\mathcal{O}(d\cdot\mathcal{B})$ | $\mathcal{B}$ 常取 $2^k$ (32~256)，适合 GPU 并行 |
| 单样本 SGD | $\mathcal{O}(d)$ | 噪声大，但更新最廉价 |

> **经验**：大数据高维场景，优先使用 **mini-batch SGD + 衰减学习率 + 特征归一化**；若 $d$ 中等且内存充裕，可用 **GD** 或 **正规方程**一步求解

### 4. 统计视角：最大似然估计 (MLE)

> Why should we use squared loss?

#### 1）概率假设

我们假设线性回归模型的输出受到高斯噪声干扰：

$$
y = w^\top x + \varepsilon,\quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
\quad \Rightarrow \quad
y \mid x \sim \mathcal{N}(w^\top x, \sigma^2)
$$

#### 2）对数似然

给定独立同分布（i.i.d.）的数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$，其联合似然为：

$$
p(\mathcal{D}; w) = \prod_{i=1}^n p(y_i \mid x_i; w)
= \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - w^\top x_i)^2}{2\sigma^2} \right)
$$

取对数得到对数似然函数：

$$
\log p(\mathcal{D}; w)
= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n}(y_i - w^\top x_i)^2
$$

#### 3）结论

由于第一项与参数 $w$ 无关，最大化对数似然等价于最小化平方误差项：

$$
\max_w \log p(\mathcal{D}; w)
\quad \Longleftrightarrow \quad
\min_w \sum_{i=1}^{n}(y_i - w^\top x_i)^2
$$

因此，**在高斯噪声假设下，使用平方损失（L2 损失）等价于最大似然估计**。

> **注**：若将噪声分布替换为长尾分布（如 t-分布），则对数似然对应的是对异常值更鲁棒的损失函数（如 Huber 损失或 Student-t 负对数似然），从而得到**鲁棒回归**模型。这说明损失函数的选择本质上依赖于对数据生成过程的统计假设。

---

## 三、非线性化方法

> 线性模型虽然简洁高效，但在面对非线性关系的数据时表现受限。如何在保持模型线性结构的同时引入非线性表达能力？

### 1. 动机：处理非线性关系

在许多实际问题中（如全年用电峰值预测），输入特征与输出之间并非简单的线性关系。例如，夏季高温和冬季低温都可能导致用电高峰，呈现出“U 型”关系。此时，直接使用线性回归会导致**欠拟合（underfitting）**。

### 2. 基函数变换（Basis Function Expansion）

核心思想：将原始输入 $x \in \mathbb{R}^d$ 通过非线性映射 $\Phi: \mathbb{R}^d \to \mathbb{R}^D$ 转换到高维特征空间，在该空间中仍使用线性模型：

$$
h(x) = w^\top \Phi(x) = \sum_{j=1}^D w_j \phi_j(x)
$$

其中 $\{\phi_j(\cdot)\}_{j=1}^D$ 称为**基函数（basis functions）**。

#### 1）常见基函数类型

- **多项式基函数（Polynomial Basis）**  
  例如对一维输入 $x$，3 阶多项式基为：
  $$
  \Phi(x) = [1, x, x^2, x^3]^\top
  $$

- **径向基函数（Radial Basis Functions, RBF）**  
  $$
  \phi_j(x) = \exp\left( -\frac{\|x - \mu_j\|^2}{2\sigma^2} \right)
  $$

### 3. 模型能力与过拟合风险

- 更复杂的基函数（如高阶多项式）能更好地拟合训练数据，但可能导致**过拟合（overfitting）**，尤其在数据量有限时。
- 高阶模型往往产生**极大或极小的系数**，导致对新数据泛化能力差。

> **关键权衡**：模型复杂度 vs. 泛化性能。后续的正则化技术（如 L2 正则）可有效缓解此问题。

### 4. 局部加权线性回归

另一种非线性策略是对每个预测点 $x$ 赋予训练样本不同的权重（如高斯核权重）：
$$
\hat{w} = \arg\min_w \sum_{i=1}^n \alpha_i(x) \left( y_i - w^\top x_i \right)^2, \quad
\alpha_i(x) = \exp\left( -\frac{\|x_i - x\|^2}{2\sigma^2} \right)
$$
该方法保留所有训练数据用于预测，属于**非参数方法**。

> **总结**：通过基函数将输入非线性映射到高维空间，线性模型即可拟合复杂非线性关系。但需警惕模型复杂度过高引发的过拟合，需结合正则化等手段控制泛化误差。

---

## 四、正则化技术

> 高阶非线性模型虽能完美拟合训练数据，却容易过拟合（overfitting）。如何控制模型复杂度、提升泛化能力？

### 1. 动机：过拟合与模型复杂度

- 在非线性回归（如高阶多项式）中，模型可能“记住”训练噪声，导致测试误差显著上升。
- 过拟合常表现为：**系数幅值极大**（如 $w_9 = 125201.43$），模型对输入扰动极度敏感。
- **目标**：在拟合能力与泛化能力之间取得平衡。

### 2. L2 正则化（岭回归，Ridge Regression）

> Weight decay

在平方损失基础上加入 L2 范数惩罚项：

$$
\hat{w}_{\text{ridge}} = \arg\min_{w \in \mathbb{R}^{d}} \left\{ \sum_{i=1}^n (y_i - w^\top x_i)^2 + \lambda \|w\|_2^2 \right\}
$$

- $\|w\|_2^2 = \sum_{j=1}^d w_j^2$，$\lambda \geq 0$ 为超参数。
- **解析解**（Normal Equation with Regularization）：
  $$
  w = (X^\top X + \lambda I)^{-1} X^\top y
  $$
  其中 $I$ 为单位矩阵，保证 $X^\top X + \lambda I$ **始终可逆**（即使 $X^\top X$ 奇异）。
- **效果**：压缩系数幅度，使模型更平滑（Lipschitz 常数减小），提升稳定性。

### 3. L1 正则化（Lasso 回归）

> Feature selection

使用 L1 范数作为正则项：

$$
\hat{w}_{\text{lasso}} = \arg\min_{w \in \mathbb{R}^{d}} \left\{ \sum_{i=1}^n (y_i - w^\top x_i)^2 + \lambda \|w\|_1 \right\}
$$

- $\|w\|_1 = \sum_{j=1}^d |w_j|$。
- **关键性质**：诱导**稀疏解**（部分 $w_j = 0$），实现**自动特征选择**。
- **几何解释**：L1 单位球在坐标轴上有“尖角”，更易与损失等高线在轴上相交。
- **优化挑战**：目标函数不可微，需使用**次梯度下降（Subgradient Descent）** 或 **近端梯度法（Proximal Gradient Descent）**。

### 4. 统计视角：最大后验估计（MAP）

正则化可从贝叶斯推断角度理解：

- **L2 正则 ⇔ 高斯先验**：假设 $w \sim \mathcal{N}(0, \tau^2 I)$，则 MAP 估计等价于 Ridge 回归。
- **L1 正则 ⇔ 拉普拉斯先验**：假设 $p(w) \propto \exp(-\|w\|_1 / b)$，则 MAP 估计等价于 Lasso。

> **结论**：正则项 = 负对数先验；MLE（无先验）是 MAP 的特例（均匀先验）。

### 5. 正则化的作用机制

- **控制模型复杂度**：通过约束参数范数，限制假设空间大小。
- **提升泛化能力**：避免对训练噪声过度拟合。
- **数值稳定性**：L2 正则改善矩阵条件数，利于数值求解。

> **实践建议**：  
>
> - 若关注预测精度且特征均重要 → 用 **L2**。  
> - 若特征冗余、希望自动筛选 → 用 **L1**。  
> - 超参数 $\lambda$ 通常通过交叉验证选择。

---

## 五、线性分类模型

### 1. 从回归到二分类

- 标签 $y\in\{0,1\}$，希望模型输出为概率值，即 $h_w(x)\in[0,1]$。
- 直接使用线性回归 + 阈值（如 $h_w(x)\ge 0.5$ 判为 1）存在以下问题：
  - 输出无界（可能 $<0$ 或 $>1$）；
  - 平方损失在分类任务中非凸，优化困难；
  - 对远离决策边界的错误样本惩罚不足。
- 引入 **Sigmoid 激活函数** 将线性输出映射到概率空间：
  $$
  \sigma(a)=\frac{1}{1+e^{-a}},\quad p(y=1\mid x;w)=\sigma(w^\top x)
  $$
- 决策规则：若 $p(y=1\mid x) > 0.5$，预测为 1；否则为 0。

### 2. 交叉熵损失 (Cross-Entropy)

- 平凡误差损失函数不合适
- 基于 **伯努利分布假设** 的最大似然估计（MLE）导出损失函数。
- 负对数似然（即交叉熵损失）为：
  $$
  \ell_{\text{CE}}(w; x, y) = -y\log\sigma(w^\top x) - (1-y)\log\big[1-\sigma(w^\top x)\big]
  $$
- 优点：
  - **凸函数**：保证梯度下降可收敛到全局最优（在无正则化时）；
  - **对错分样本惩罚更强**：当 $y=1$ 但 $\sigma(w^\top x)\to 0$ 时，损失趋于无穷；
  - 与概率解释自然契合，支持软预测。

### 3. Logistic 回归 + 正则化

- 优化目标（带正则项）：
  $$
  \min_w \; -\sum_{i=1}^{n} \Big[ y_i\log\sigma(w^\top x_i) + (1-y_i)\log(1-\sigma(w^\top x_i)) \Big] + \lambda\Omega(w)
  $$
  其中 $\Omega(w)$ 常取：
  - **L2 正则（Ridge）**：$\|w\|_2^2$，控制权重大小，提升泛化，防止在**线性可分数据上权重发散至无穷**；
  - **L1 正则（Lasso）**：$\|w\|_1$，诱导稀疏解，实现特征选择。
- **无解析解**，需采用数值优化方法：
  - 梯度下降（GD）
  - 随机梯度下降（SGD）
  - 牛顿法、拟牛顿法（如 L-BFGS）

> 特别注意：当数据**线性可分**时，若无正则化，最大似然解将导致 $\|w\|\to\infty$，模型过拟合且泛化差。

### 4. 多分类：Softmax 回归

- **任务设定**：多类分类问题，标签 $y \in \{1, 2, \dots, C\}$，共 $C$ 个类别。
- **模型结构**：
  - 为每个类别 $c$ 引入一个线性参数向量 $w_c \in \mathbb{R}^d$；
  - 参数整体表示为矩阵 $W = [w_1, w_2, \dots, w_C] \in \mathbb{R}^{d \times C}$；
  - 对输入 $x$，计算每个类别的线性得分 $w_c^\top x$。
- **Softmax 概率映射**：
  $$
  p(y = c \mid x; W) = \frac{\exp(w_c^\top x)}{\sum_{r=1}^{C} \exp(w_r^\top x)}
  $$
- **损失函数（多类交叉熵）**：
  $$
  \ell(W) = -\sum_{i=1}^{n} \sum_{c=1}^{C} \mathbf{1}\{y_i = c\} \log p(y = c \mid x_i; W)
  $$
- **优化与正则化**：
  - 无解析解，常用优化方法包括：GD、SGD、L-BFGS；
  - **需引入正则化（如 L2）**：当数据线性可分时，最大似然解会导致 $\|w_c\| \to \infty$；
  - 正则化形式通常为 $\lambda \|W\|_F^2$（Frobenius 范数）。
- **预测规则**：
  $$
  \hat{y} = \arg\max_{c \in \{1,\dots,C\}} p(y = c \mid x) = \arg\max_{c} w_c^\top x
  $$

---
