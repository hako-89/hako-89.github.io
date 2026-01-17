---
title: "Learning Theory"
published: 2025-10-13
pinned: false
description: "机器学习理论核心概览：从PAC学习框架到有限/无限假设空间的泛化界，涵盖Rademacher复杂度、VC维泛化分析。"
image: ""
tags: ["Machine Learning", "Learning Theory", "PAC", "VC Dimension", "Rademacher Complexity"]
category: "Machine Learning"
author: "hako"
draft: false
---

## 一、PAC 学习：统计基础与框架 (PAC Learning: Foundations & Framework)

PAC（Probably Approximately Correct，概率近似正确）学习理论是机器学习理论分析的基础框架。为了建立这一框架，我们需要首先确立机器学习的统计学视角和核心组件。

### 1. 机器学习的统计视角与组件 (Statistical View & Components)

在统计机器学习中，我们关注如何通过有限的样本来近似未知的目标函数，并保证在未见样本上的表现。

#### 1) 学习的核心组件

- **未知目标函数 (Unknown Target Function)**: $f: \mathcal{X} \rightarrow \mathcal{Y}$，这是我们希望学习到的理想映射。
- **训练样本 (Training Examples)**: $\mathcal{D}_{n}=(x_{1},y_{1}),...,(x_{n},y_{n})$，用于训练的数据集。
- **假设空间 (Hypothesis Space)**: $\mathcal{H}$，算法从中搜索候选函数的集合。
- **经验风险最小化 (ERM)**: 学习算法 $\mathcal{A}$ 通常通过最小化经验误差来选择假设，可能包含正则化项：
  $$
  \min_{h \in \mathcal{H}} \sum_{i=1}^{n} l(h(x_{i}),y_{i}) + \Omega(h)
  $$
- **最终假设 (Final Hypothesis)**: $h_{\mathcal{D}_{n}} \approx f$，算法输出的最终模型。

#### 2) I.I.D. 与 OOD

统计机器学习依赖于一个基础假设：

- **I.I.D. (独立同分布)**: 假设存在一个潜在的数据生成分布 $D_{\mathcal{X} \times \mathcal{Y}}$，所有的训练样本和测试样本 $(x, y)$ 都是从该分布中**独立且同分布 (Independent and Identically Distributed)** 生成的。
- **OOD (Out-of-Distribution)**: 如果测试数据的分布 $D'$ 与训练数据的分布 $D$ 不同 ($D \neq D'$)，模型通常无法有效工作。这被称为分布外 (OOD) 问题。

#### 3) 泛化 (Generalization) vs. 记忆 (Memorization)

- **学习 $\neq$ 拟合**: 我们的目标是**泛化**到新的、未见过的样本，而不仅仅是**记忆**训练数据。
- **复杂度的权衡**:
  - 过于复杂的规则（如复杂的分类边界）可能是糟糕的预测器（过拟合）。
  - **权衡 (Trade-off)**: 需要在假设空间的复杂度与样本量之间寻找平衡，避免欠拟合 (Underfitting) 和过拟合 (Overfitting)。
- **误差类型**:
  - **期望误差 (Expected Error)**: 在真实分布 $D$ 上的误差，$\mathcal{E}(h) = \mathbb{E}_{(x,y)\sim D} l(h(x),y)$。这是我们最终想最小化的目标，但无法直接计算。
  - **经验误差 (Empirical Error)**: 在训练集 $\mathcal{D}_n$ 上的误差，$\hat{\mathcal{E}}_{\mathcal{D}_{n}}(h) = \frac{1}{n} \sum_{i=1}^{n} l(h(x_{i}),y_{i})$。它是期望误差的无偏估计。

### 2. 偏差-方差分解 (Bias-Variance Decomposition)

为了理解泛化误差的来源，我们可以将回归问题（L2损失）的期望误差进行分解。

- **分解公式**: 对于给定输入 $x$，期望误差可以分解为三部分：
  $$
  \mathcal{E}_{L2}(x) = \text{Noise} + \text{Bias}^2 + \text{Variance}
  $$
  具体形式为：
  $$
  \mathcal{E}_{L2}(x) = Var(y|x) + (E_{\mathcal{D}}[h_{\mathcal{D}}(x)] - f^*(x))^2 + E_{\mathcal{D}}[(h_{\mathcal{D}}(x) - E_{\mathcal{D}}[h_{\mathcal{D}}(x)])^2]
  $$

| 误差项 | 符号表示 | 含义 |
| :--- | :--- | :--- |
| **贝叶斯误差 (Bayes Error)** | $Var(y\|x)$ | 数据本身的噪声，是不可约减的误差 (Irreducible Error)。 |
| **偏差 (Bias)** | $E_{\mathcal{D}}[h_{\mathcal{D}}(x)] - f^*(x)$ | 模型的期望预测与真实函数 $f^*(x)$ 之间的差异。高偏差通常意味着欠拟合。 |
| **方差 (Variance)** | $Var_{\mathcal{D}}[h_{\mathcal{D}}(x)]$ | 模型预测随训练数据集 $\mathcal{D}$ 变化而产生的波动。高方差通常意味着过拟合。 |

> **近似误差 vs. 估计误差**：
> 另一个视角的分解是：
>
> - **近似误差 (Approximation Error)**: $\mathcal{E}(h^*) - \mathcal{E}^*(f)$。由假设空间 $\mathcal{H}$ 的选择决定（非随机），衡量假设空间的表达能力。
> - **估计误差 (Estimation Error)**: $\mathcal{E}(h) - \mathcal{E}(h^*)$。由在有限样本上训练决定（随机），衡量学习到的假设与假设空间中最佳假设的差距。**泛化界 (Generalization Bound)** 主要关注限制这一项。

### 3. PAC 学习框架 (PAC Learning Framework)

基于上述统计视角，PAC 框架提供了一个定量的标准来评估学习算法的能力。

#### 1) 核心定义

一个假设空间 $ \mathcal{H} $ 是 **PAC 可学习的 (PAC-learnable)**，如果存在一个学习算法 $\mathcal{A}$ 和一个多项式函数 $poly()$，使得对于任意的精度参数 $\epsilon > 0$ 和置信度参数 $\delta > 0$，对于任意数据分布 $\mathcal{D}$ 和任意目标概念 $f \in \mathcal{H}$，只要样本量 $n$ 满足：
$$
n \ge poly(1/\epsilon, 1/\delta, size(\mathcal{H}), size(f))
$$
则算法输出的假设 $h_{\mathcal{D}_n}$ 满足以下概率界：
$$
P_{\mathcal{D}_n \sim \mathcal{D}^n} (\mathcal{E}(h_{\mathcal{D}_n}) - \min_{h \in \mathcal{H}}\mathcal{E}(h) \ge \epsilon) \le \delta
$$

#### 2) 含义解析

- **Probably (可能)**: 以至少 $1-\delta$ 的大概率成功。
- **Approximately Correct (近似正确)**: 学习到的假设的误差与最优假设的误差之差不超过 $\epsilon$。
- **样本复杂度 (Sample Complexity)**: 满足上述条件所需的最小样本量 $n$。

#### 3) 泛化界的目标

由于估计误差 $\hat{\mathcal{E}}(h) - \mathcal{E}(h)$ 是随机变量，PAC 分析的核心目标是寻找一个上界 $\epsilon$，使得我们能以高概率保证：
$$
\mathcal{E}(h) \le \hat{\mathcal{E}}(h) + \epsilon
$$
这意味着训练误差加上一个复杂度项（$\epsilon$）可以作为测试误差的上限。

---

## 二、概率论基础：集中不等式 (Probability Basics: Concentration Inequalities)

在PAC学习中，我们需要保证经验误差 $\hat{\mathcal{E}}(h)$ 能够很好地近似期望误差 $\mathcal{E}(h)$。集中不等式提供了随机变量偏离其期望值的概率上界，即寻找 $P(|X - \mathbb{E}[X]| \ge \epsilon) \le \delta$ 形式的界。

### 1. 基础不等式 (Basic Inequalities)

这些不等式对分布的假设较少，但提供的界通常较松。

#### 1) 马尔可夫不等式 (Markov's Inequality)

最基础的不等式，仅要求随机变量非负。

- **定理**: 如果 $X$ 是非负随机变量，则对于任意 $\epsilon > 0$：
  $$
  P(X \ge \epsilon) \le \frac{\mathbb{E}[X]}{\epsilon}
  $$
- **特点**: 仅依赖于均值。

#### 2) 切比雪夫不等式 (Chebyshev's Inequality)

利用方差来约束偏离程度。

- **定理**: 对于任意随机变量 $X$（方差有限），对于任意 $\epsilon > 0$：
  $$
  P(|X - \mathbb{E}[X]| \ge \epsilon) \le \frac{Var(X)}{\epsilon^2}
  $$
- **推导**: 基于 $P(Y \ge \epsilon) \le \mathbb{E}[Y]/\epsilon$ 的单调性，令 $Y = (X - \mathbb{E}[X])^2$ 得到。

#### 3) 大数定律 (Law of Large Numbers)

- **弱大数定律**: 基于切比雪夫不等式，当样本量 $n \to \infty$ 时，样本均值依概率收敛于期望。
- **局限性**: 在机器学习中，样本量 $n$ 是有限的，我们不能依赖渐近分析 ($n \to \infty$)，而需要**有限样本的概率界 (Finite-sample probabilistic bound)**。

### 2. 指数级界 (Exponential Bounds)

为了获得更紧致的界（随样本量 $n$ 增加呈指数级衰减），我们需要利用矩生成函数。

#### 1) 切诺夫界 (Chernoff Bound)

通过引入参数 $\lambda$ 并利用 $e^x$ 的凸性，可以得到比多项式级更紧的界。

- **形式**: $P(X \ge \epsilon) = P(e^{\lambda X} \ge e^{\lambda \epsilon}) \le \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda \epsilon}}$。
- **应用**: 对于独立同分布 (i.i.d.) 随机变量的和，利用 $\mathbb{E}[e^{\lambda \sum X_i}] = \prod \mathbb{E}[e^{\lambda X_i}]$ 可以极大地简化计算。

#### 2) 霍夫丁引理 (Hoeffding's Lemma)

这是推导霍夫丁不等式的关键引理，针对有界随机变量。

- **定理**: 设 $X$ 是均值为 0 的有界随机变量，即 $X \in [a, b]$ 且 $\mathbb{E}[X] = 0$。则对于任意 $\lambda > 0$：
  $$
  \mathbb{E}[e^{\lambda X}] \le e^{\frac{\lambda^2 (b-a)^2}{8}}
  $$
- **意义**: 刻画了有界随机变量的矩生成函数的上界。

#### 3) 霍夫丁不等式 (Hoeffding's Inequality)

这是机器学习理论中最常用的不等式之一，适用于有界独立随机变量的和。

- **定理**: 设 $X_1, ..., X_n$ 为独立随机变量，且 $X_i \in [a_i, b_i]$ 。令 $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$，则对于任意 $\epsilon > 0$：
  $$
  P(|\bar{X} - \mathbb{E}[\bar{X}]| \ge \epsilon) \le 2 \exp \left( - \frac{2n^2\epsilon^2}{\sum_{i=1}^n (b_i - a_i)^2} \right)
  $$
- **应用**: 在泛化误差分析中，若损失函数 $l \in [0, 1]$（即 $b_i - a_i = 1$），则有：
  $$
  P(|\hat{\mathcal{E}}(h) - \mathcal{E}(h)| \ge \epsilon) \le 2 \exp(-2n\epsilon^2)
  $$
  这直接给出了泛化误差的指数级收敛界。

#### 4) 麦克迪亚米德不等式 (McDiarmid's Inequality)

霍夫丁不等式的推广，适用于一般函数，只要该函数对输入的变化是“稳定”的。

- **条件**: 设 $f: \mathcal{X}^n \to \mathbb{R}$ 满足**有界差分性质**，即改变输入中的任意一个 $x_i$，函数值的变化被 $c_i$ 有界：
  $$
  \sup_{x_1,...,x_n, x_i'} |f(x_1, ..., x_i, ..., x_n) - f(x_1, ..., x_i', ..., x_n)| \le c_i
  $$
- **定理**: 对于独立随机变量 $X_1, ..., X_n$：
  $$
  P(f(X_1, ..., X_n) - \mathbb{E}[f] \ge \epsilon) \le \exp \left( - \frac{2\epsilon^2}{\sum_{i=1}^n c_i^2} \right)
  $$
- **关系**: 当 $f$ 为求和平均函数时，麦克迪亚米德不等式退化为霍夫丁不等式。

---

## 三、有限假设空间的泛化界 (Generalization Bound for Finite Hypothesis Space)

当假设空间 $\mathcal{H}$ 是一个有限集合，即 $|\mathcal{H}| < \infty$ 时，我们可以利用概率论中的**霍夫丁不等式** (Hoeffding's Inequality) 和**联合界** (Union Bound) 来推导泛化界。

### 1. **单一假设的泛化界 (Bound for a Single Hypothesis)**

#### 1) 霍夫丁不等式应用

我们首先考虑假设空间 $\mathcal{H}$ 中**任意固定**的假设 $h$。
设损失函数 $l(h(x), y)$ 的值域在 $[0, 1]$ 内，则对于固定的 $h$，其经验误差 $\hat{\mathcal{E}}(h)$ 是真实误差 $\mathcal{E}(h)$ 的一个平均估计。

根据霍夫丁不等式，对于任意 $\delta' > 0$：
$$
P(|\hat{\mathcal{E}}_{\mathcal{D}_n}(h) - \mathcal{E}(h)| \ge \epsilon) \le 2 \exp(-2n\epsilon^2)
$$
等价地，以至少 $1-\delta'$ 的概率，我们有：
$$
\mathcal{E}(h) \le \hat{\mathcal{E}}_{\mathcal{D}_n}(h) + \sqrt{\frac{\log(2/\delta')}{2n}}
$$
这里的 $\epsilon$ 称为**单点泛化界 (Single-point Generalization Bound)**。

### 2. **多个假设的泛化界 (Bound for Multiple Hypotheses)**

在实际学习过程中，我们通过 ERM 算法选择的是最小化**经验误差**的假设 $h_{ERM}$。我们不能只关注一个固定的 $h$，而必须保证对于**整个**假设空间 $\mathcal{H}$ 中的**所有**假设 $h$，$|\hat{\mathcal{E}}(h) - \mathcal{E}(h)|$ 都能被界定。

#### 1) 联合界原理 (Union Bound)

联合界 (或 Bonferroni 校正) 用于处理多个事件的并集概率。

- **定理**: 对于 $M$ 个事件 $A_1, ..., A_M$，它们中至少一个发生的概率满足：
  $$
  P(A_1 \cup A_2 \cup ... \cup A_M) \le \sum_{i=1}^M P(A_i)
  $$
- **应用**:
  令 $A_h$ 为“假设 $h$ 的经验误差和真实误差偏差过大”的事件，即 $A_h = \{|\hat{\mathcal{E}}(h) - \mathcal{E}(h)| \ge \epsilon\}$。
  我们关心的是 $\mathcal{H}$ 中**至少存在一个** $h$ 使得该事件发生：$P(\bigcup_{h \in \mathcal{H}} A_h)$。
  $$
  P\left(\sup_{h \in \mathcal{H}} |\hat{\mathcal{E}}(h) - \mathcal{E}(h)| \ge \epsilon\right) = P\left(\bigcup_{h \in \mathcal{H}} A_h\right) \le \sum_{h \in \mathcal{H}} P(A_h)
  $$

#### 2) 有限假设空间的泛化界定理

结合霍夫丁不等式和联合界，对于 $\mathcal{H}$ 中所有的 $|\mathcal{H}|$ 个假设，我们得到：
$$
\sum_{h \in \mathcal{H}} P(|\hat{\mathcal{E}}_{\mathcal{D}_n}(h) - \mathcal{E}(h)| \ge \epsilon) \le \sum_{h \in \mathcal{H}} 2 \exp(-2n\epsilon^2) = 2 |\mathcal{H}| \exp(-2n\epsilon^2)
$$
令 $\delta = 2 |\mathcal{H}| \exp(-2n\epsilon^2)$，解出 $\epsilon$，得到**有限假设空间的泛化界**：

- **定理**: 设 $\mathcal{H}$ 为有限假设空间 ($|\mathcal{H}| < \infty$)，损失函数值域为 $[0, 1]$。对于任意 $\delta > 0$，以至少 $1-\delta$ 的概率，对于**所有** $h \in \mathcal{H}$：
  $$
  \mathcal{E}(h) \le \hat{\mathcal{E}}_{\mathcal{D}_n}(h) + \sqrt{\frac{\log |\mathcal{H}| + \log (2/\delta)}{2n}}
  $$

### 3. **界限的分析与解释**

#### 1) 结论

- **误差界** $\epsilon$ 主要取决于**假设空间的复杂度** ($\log |\mathcal{H}|$) 和**样本数量** ($n$)。
- 随着 $|\mathcal{H}|$ 增大， $\log |\mathcal{H}|$ 增大，$\epsilon$ 增大，即泛化界变松弛，这与更大的假设空间更容易过拟合的直觉一致。

#### 2) 奥卡姆剃刀 (Occam's Razor) 原理

- $\log_2 |\mathcal{H}|$ 可以视为编码假设空间所需的比特数，是模型复杂度的度量。
- **原理**: 在保证经验误差 $\hat{\mathcal{E}}(h)$ 较低的前提下，应该选择复杂度**最小**的假设空间 $\mathcal{H}$。越简单的模型，其泛化能力越强，因为其 $\log |\mathcal{H}|$ 越小，$\epsilon$ 越小。

#### 3) 离散化技巧 (Discretization Trick)

- 即使一个假设空间（如参数为实数的线性分类器）在数学上是无限的，但在实际计算机中，由于浮点数的精度有限（如64位），模型参数的表示是**离散**且**有限**的。
- 例如，一个具有 $d$ 个浮点参数的模型，其可表示的假设数量为 $|\mathcal{H}| \le (2^{64})^d$，这是一个有限但巨大的数字。
- **意义**: 在某些情况下，有限假设空间的界可以近似地用于分析参数连续的模型。

---

## 四、无限假设空间的泛化界 (Generalization Bound for Infinite Hypothesis Space)

当假设空间 $|\mathcal{H}| = \infty$ 时（例如，具有连续参数的线性分类器），有限假设空间的界不再适用。我们需要一种新的方法来衡量假设空间**拟合数据**的能力，而不是假设的数量。

### 1. Rademacher 复杂度 (Rademacher Complexity)

Rademacher 复杂度衡量假设空间 $\mathcal{H}$ **拟合随机噪声**的能力，它提供了比 VC 维更紧致的泛化界。

#### 1) 经验 Rademacher 复杂度 (Empirical Rademacher Complexity)

- **定义**: 设 $S = \{z_1, ..., z_n\}$ 为 $n$ 个样本点组成的集合，$\sigma_1, ..., \sigma_n$ 为**Rademacher 变量**（以 $0.5$ 的概率取 $+1$ 或 $-1$）。函数族 $\mathcal{G}$ 相对于样本 $S$ 的经验 Rademacher 复杂度定义为：
  $$
  \hat{\mathfrak{R}}_S(\mathcal{G}) = \mathbb{E}_{\sigma} \left[ \sup_{g \in \mathcal{G}} \frac{1}{n} \sum_{i=1}^n \sigma_i g(z_i) \right]
  $$
- **解释**: $\sum_{i=1}^n \sigma_i g(z_i)$ 衡量了函数 $g$ 对随机标签 $\sigma_i$ 的拟合程度。$\sup_{g \in \mathcal{G}}$ 找到了 $\mathcal{G}$ 中最能拟合噪声的函数。 $\mathbb{E}_{\sigma}$ 对噪声的随机性取期望。$\hat{\mathfrak{R}}_S(\mathcal{G})$ 值越大，表示函数族 $\mathcal{G}$ 越复杂，越容易过拟合。

#### 2) Rademacher 复杂度 (Rademacher Complexity)

- **定义**: 函数族 $\mathcal{G}$ 的 Rademacher 复杂度是经验 Rademacher 复杂度在数据分布 $D$ 上的期望：
  $$
  \mathfrak{R}_n(\mathcal{G}) = \mathbb{E}_{S \sim D^n} [\hat{\mathfrak{R}}_S(\mathcal{G})]
  $$
- $\mathfrak{R}_{n+1}(\mathcal{G}) \le \mathfrak{R}_n(\mathcal{G})$
- 以至少 $1-\delta$ 的概率:
  $$
  \mathfrak{R}_{n}(\mathcal{G}) \le \hat{\mathfrak{R}}_n(\mathcal{G}) + \sqrt{\frac{\log(1/\delta)}{2n}}
  $$

#### 3) 基于 Rademacher 复杂度的泛化界 (Generalization Bound)

- **定理**: 对于值域在 $[0, 1]$ 的函数族 $\mathcal{G}$，以至少 $1-\delta$ 的概率，对于所有 $g \in \mathcal{G}$：
  $$
  \mathbb{E}[g] \le \frac{1}{n} \sum_{i=1}^n g(z_i) + 2 \mathfrak{R}_n(\mathcal{G}) + \sqrt{\frac{\log(1/\delta)}{2n}}
  $$
- 基于经验 Rademacher 复杂度的界:
  $$
  \mathbb{E}[g] \le \frac{1}{n} \sum_{i=1}^n g(z_i) + 2 \hat{\mathfrak{R}}_n(\mathcal{G}) + 3\sqrt{\frac{\log(2/\delta)}{2n}}
  $$
- **与二分类假设空间的关系**: 设 $\mathcal{H}$ 是一个取值在 $\{-1, +1\}$ 的二分类假设族，$\mathcal{G}$ 是 $\mathcal{H}$ 对应的 $0$-1 损失函数族：
  $$
  \mathcal{G} = \{ (x, y) \mapsto \mathbb{I}[h(x) \neq y] : h \in \mathcal{H} \}
  $$
  则 $\mathcal{G}$ 的 Rademacher 复杂度与 $\mathcal{H}$ 的复杂度满足以下关系：
  $$
  \mathfrak{R}_n(\mathcal{G}) = \frac{1}{2} \mathfrak{R}_n(\mathcal{H})
  $$

### 2. 增长函数 (Growth Function)

由于计算 $\mathbb{E}_{\sigma} [\cdot]$ 涉及到 $2^n$ 次 $\sup$ 运算（通常是 **NP-难**问题），我们需要通过组合度量来进一步界定 $\mathfrak{R}_n(\mathcal{H})$。

#### 1) 概念与定义

- **增长函数 $\Pi_{\mathcal{H}}(n)$**: 假设空间 $\mathcal{H}$ 对 $n$ 个点 $\{x_1, ..., x_n\}$ 能产生的**最大二分 (Dichotomies)** 数量。
  $$
  \Pi_{\mathcal{H}}(n) = \max_{x_1,...,x_n} | \{ (h(x_1), ..., h(x_n)) : h \in \mathcal{H} \} |
  $$
- $\Pi_{\mathcal{H}}(n) \le 2^n$ 或 $\Pi_{\mathcal{H}}(n) \le |\mathcal{H}|$。

#### 2) Massart 引理 (Massart's Lemma)

Massart 引理提供了连接有限集合 Rademacher 复杂度与其大小的界限。

- **定理**: 设 $\mathcal{A} \subseteq \mathbb{R}^n$ 是一个有限集，其范数 $R = \max_{a \in \mathcal{A}} \|a\|_2$。则经验 Rademacher 复杂度有界：
  $$
  \mathbb{E}_{\sigma} \left[ \frac{1}{n} \sup_{a \in \mathcal{A}} \sum_{i=1}^n \sigma_i a_i \right] \le \frac{R}{n} \sqrt{2 \log |\mathcal{A}|}
  $$

#### 3) Rademacher 复杂度与增长函数的关系

应用 Massart 引理，可以将 **增长函数** 与 **Rademacher 复杂度** 关联起来：

- **定理**: 设 $\mathcal{H}$ 是一个取值在 $\{-1, +1\}$ 的函数族，则：
  $$
  \mathfrak{R}_n(\mathcal{H}) \le \sqrt{\frac{2 \log \Pi_{\mathcal{H}}(n)}{n}}
  $$

#### 4) 基于增长函数的泛化界

- **定理**: 设 $\mathcal{H}$ 是一个取值在 $\{-1, +1\}$ 的函数族。对于任意 $\delta > 0$，以至少 $1-\delta$ 的概率，对于所有 $h \in \mathcal{H}$：
  $$
  \mathcal{E}(h) \le \hat{\mathcal{E}}_{\mathcal{D}_n}(h) + \sqrt{\frac{2 \log \Pi_{\mathcal{H}}(n)}{n}} + \sqrt{\frac{\log(1/\delta)}{2n}}
  $$

### 3. VC 维 (Vapnik-Chervonenkis Dimension)

VC 维是衡量**无分布**二分类假设空间复杂度的度量，它提供了一个有限的组合参数来替代 $\Pi_{\mathcal{H}}(n)$。

#### 1) 打散 (Shattering)

- **定义**: 如果假设空间 $\mathcal{H}$ 能够对 $n$ 个点 $x_1, ..., x_n$ 实现**所有** $2^n$ 种可能的标签组合，则称 $\mathcal{H}$ **打散 (Shatters)** 了该 $n$ 点集。

#### 2) VC 维定义

- **定义**: 假设空间 $\mathcal{H}$ 的 **VC 维** $d = VCdim(\mathcal{H})$ 是 $\mathcal{H}$ **能够打散的最大样本集的大小**。
- 如果 $\mathcal{H}$ 可以打散任意大小的集合（如具有 $\omega$ 频率的**正弦函数** $\text{sign}(\sin(\omega x))$），则 $VCdim(\mathcal{H}) = +\infty$。
- **典型例子**:

| 模型 | VC 维 $d$ |
| :--- | :--- |
| $\mathbb{R}^1$ 上的区间分类器 | 2 |
| $\mathbb{R}^d$ 上的线性分类器 (超平面) | $d+1$ |
| 任意 $d$ 个参数的 $p$ 阶多项式 | $d+1$ |
| 正弦函数 $h(x) = \text{sign}(\sin(ax+b))$ | $\infty$ |

#### 3) Sauer 引理 (Sauer's Lemma)

Sauer 引理说明了如果 VC 维是有限的 $d$，那么 $\Pi_{\mathcal{H}}(n)$ 的增长率是多项式级的。

- **定理**: 设 $\mathcal{H}$ 的 $VCdim(\mathcal{H}) = d$，则对于所有 $n \ge d$：
  $$
  \Pi_{\mathcal{H}}(n) \le \sum_{i=0}^d \binom{n}{i} \le \left( \frac{en}{d} \right)^d = O(n^d)
  $$
- **意义**: 证明了即使 $|\mathcal{H}| = \infty$，但只要 $d < \infty$，其有效假设数量 $\Pi_{\mathcal{H}}(n)$ 仍被多项式函数界定，从而保证了可学习性。

#### 4) 基于 VC 维的泛化界 (VC Dimension Bound)

- **定理**: 设 $\mathcal{H}$ 是一个 $VCdim(\mathcal{H}) = d$ 的二分类假设空间，损失函数值域为 $[0, 1]$。对于任意 $\delta > 0$，以至少 $1-\delta$ 的概率，对于所有 $h \in \mathcal{H}$：
  $$
  \mathcal{E}(h) \le \hat{\mathcal{E}}_{\mathcal{D}_{n}}(h) +  \sqrt{\frac{2d \log(en/d)}{n}} + \sqrt{\frac{\log(1/\delta)}{2n}} 
  $$

> **注意**: VC 维界在 $d \gg n$ 时，即维度远大于样本量时，将变得**不informative**（信息量不足）。

---

## 六、线性模型的泛化界

### 1. 高维问题

对于 $\mathbb{R}^d$ 中的线性模型，VC维是 $d+1$。如果特征维度 $d \gg n$（样本数），基于 VC 维的界将变得松散且无信息量。

### 2. 基于范数的界 (Rademacher Bound)

为了解决高维问题，可以限制参数的范数而不是维度。

- 假设空间: $\mathcal{H} = \{ x \mapsto w \cdot x : \|w\|_2 \le \Lambda \}$，且数据有界 $\|x\|_2 \le R$。
- **Rademacher 复杂度界**:
  $$
  \hat{\mathfrak{R}}_S(\mathcal{H}) \le \frac{R \Lambda}{\sqrt{n}}
  $$
- **泛化界**:
  $$
  \mathcal{E}(h) \le \hat{\mathcal{E}}_{\mathcal{D}_n}(h) + \frac{R \Lambda}{\sqrt{n}} + 3\sqrt{\frac{\log(2/\delta)}{2n}}
  $$
- **核方法 (Kernel Machine)**: 如果使用核函数 $K$，$R$ 对应于核空间中的数据半径，$R^2 = \sup_x K(x,x)$。

---

## 七、总结对比

| 方法 | 假设空间依赖 | 数据量依赖 | 数据依赖 | 分布依赖 | 特点 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **有限假设集** | $\log \|\mathcal{H}\|$ | $1/\sqrt{n}$ | × | × | 简单但受限于有限集合 |
| **VC 维** | $d$ (VC-dim) | $\sqrt{\frac{2d \log(en/d)}{n}}$ | × | × | 通用，组合性质，易于计算，仅适用于二分类 |
| **增长函数** |  $\Pi_{\mathcal{H}}(n)$ | $\sqrt{\frac{2 \log \Pi_{\mathcal{H}}(n)}{n}}$ | × | × | 通用，组合性质，易于计算，仅适用于二分类 |
| **经验 Rademacher** | $\hat{\mathfrak{R}}_n(\mathcal{H})$ | ✔ | ✔ | × | 紧致 (Tight)，依赖数据 |
| **Rademacher** | $\mathfrak{R}_n(\mathcal{H})$ | ✔ | × | ✔ | 紧致 (Tight)，依赖数据分布，适用于各种损失函数 |

> **注**: 在实践中，通常从 Rademacher 复杂度界出发，因为它能提供依赖于数据分布的更紧致的界。
