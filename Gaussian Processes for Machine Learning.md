
---
title: Gaussian Processes for Machine Learning
layout: default
permalink: /gaussian-processes-for-machine-learning/
---



### Noise Models

| Noise Model | Equation | Effect |
|-------------|----------|--------|
| **Additive** | $$ y = f(x) + \varepsilon,\quad \varepsilon \sim \mathcal{N}(0,\sigma^2) $$ | Constant-variance, independent noise (homoscedastic) |
| **Multiplicative** | $$ y = f(x)\bigl(1 + \varepsilon\bigr) $$ | Variance scales with signal magnitude |
| **Heteroscedastic** | $$ y = f(x) + \varepsilon(x),\quad \varepsilon(x) \sim \mathcal{N}\bigl(0,\sigma^2(x)\bigr) $$ | Noise variance depends on input \(x\) |
| **Correlated (GP-style)** | $$ \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \Sigma) $$ | Observations have nonzero covariance; needed for GP/regression with correlated errors |






### Covariance Function and GP Properties

#### 1. Covariance Function ⇒ Function Properties

Given  
$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$  

#### Definition of the Gaussian Process

A Gaussian Process (GP) is a collection of random variables, any finite subset of which has a joint Gaussian distribution. The notation
$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$
means that for any finite set of points $x_1, \ldots, x_n$, the random vector $\mathbf{f} = [f(x_1), \ldots, f(x_n)]^\top$ is distributed as a multivariate normal:
$$
\mathbf{f} \sim \mathcal{N}(\mathbf{m}, \mathbf{K}),
$$
where:
- $m(x)$ is the mean function: $m(x) = \mathbb{E}[f(x)]$
- $k(x, x')$ is the covariance (kernel) function: $k(x, x') = \operatorname{Cov}(f(x), f(x'))$
- $\mathbf{m} = [m(x_1), \ldots, m(x_n)]^\top$
- $\mathbf{K}$ is the $n \times n$ covariance matrix with entries $K_{ij} = k(x_i, x_j)$

#### Joint Gaussian Distribution Definition

For a vector $\mathbf{f} \in \mathbb{R}^n$, the multivariate normal (Gaussian) distribution is defined as:
$$
\mathbf{f} \sim \mathcal{N}(\mathbf{m}, \mathbf{K})
$$
with density
$$
p(\mathbf{f}) = \frac{1}{(2\pi)^{n/2}|\mathbf{K}|^{1/2}} 
\exp\!\left(-\frac{1}{2}(\mathbf{f}-\mathbf{m})^\top \mathbf{K}^{-1}(\mathbf{f}-\mathbf{m})\right)
$$

| Symbol            | Definition                                                                                  |
|-------------------|--------------------------------------------------------------------------------------------|
| $\mathbf{f}$      | $n$-dimensional vector: $\mathbf{f} = [f(x_1), \ldots, f(x_n)]^\top$                       |
| $\mathbf{m}$      | Mean vector: $\mathbf{m} = [m(x_1), \ldots, m(x_n)]^\top$                                  |
| $\mathbf{K}$      | Covariance matrix: $K_{ij} = k(x_i, x_j)$                                                  |
| $n$               | Number of points ($x_1, \ldots, x_n$)                                                      |
| $|\mathbf{K}|$    | Determinant of the covariance matrix $\mathbf{K}$                                          |
| $p(\mathbf{f})$   | Probability density function for $\mathbf{f}$ under $\mathcal{N}(\mathbf{m}, \mathbf{K})$  |


| Property                 | Determined by $k(x,x')$           | Example Kernel               |
|--------------------------|----------------------------------|-----------------------------|
| Smoothness / differentiability | How fast covariance decays with distance | Squared exponential → infinitely smooth |
| Periodicity              | Whether covariance repeats        | Periodic kernel             |
| Stationarity             | Depends only on $x - x'$          | RBF, Matérn                 |
| Non-stationarity         | Depends on $x$ and $x'$ separately | Linear, neural network kernel |
| Amplitude / variance     | Diagonal term $k(x,x)$             | Scaling factor $\sigma^2$   |
| Correlation length / structure | Shape of decay                  | Lengthscale $\ell$          |

#### 2. Why Many Kernels Are Possible

The only mathematical constraint is that $k$ must be positive semi-definite (PSD):  
$$
\forall \{x_i\}, \quad K_{ij} = k(x_i, x_j) \Rightarrow \mathbf{K} \succeq 0
$$  
This allows infinitely many valid covariance structures — any PSD kernel corresponds to a valid GP prior.

You can also combine kernels:  
$$
k_{\text{sum}} = k_1 + k_2, \quad k_{\text{prod}} = k_1 \times k_2
$$  
which yields new induced properties (e.g., periodic + smooth).

#### 2.1 Positive Semi-Definiteness Explained

The notation  
$$
\mathbf{K} \succeq 0
$$  
means that the kernel (covariance) matrix $\mathbf{K}$ is **positive semi-definite (PSD)**.

**Step-by-step explanation:**

1. **Definition of $\mathbf{K}$:**  
   $$
   \mathbf{K} = \begin{bmatrix}
   k(x_1, x_1) & k(x_1, x_2) & \cdots & k(x_1, x_n) \\
   k(x_2, x_1) & k(x_2, x_2) & \cdots & k(x_2, x_n) \\
   \vdots & \vdots & \ddots & \vdots \\
   k(x_n, x_1) & k(x_n, x_2) & \cdots & k(x_n, x_n)
   \end{bmatrix}
   $$

2. **PSD condition:**  
   For any vector $v \in \mathbb{R}^n$,  
   $$
   v^\top \mathbf{K} v \geq 0.
   $$

3. **Intuition:**  
   Positive semi-definiteness ensures that all variances computed from $\mathbf{K}$ are non-negative. This is necessary because variances cannot be negative.

4. **Why it matters for Gaussian Processes:**  
   Since $f(x) \sim \mathcal{N}(0, \mathbf{K})$ is a multivariate Gaussian distribution with covariance $\mathbf{K}$, the PSD property guarantees that this distribution is valid (i.e., $\mathbf{K}$ is a valid covariance matrix).

| Symbol       | Meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| $\mathbf{K}$      | Covariance matrix with entries $K_{ij} = k(x_i, x_j)$                  |
| $\succeq 0$       | Indicates positive semi-definiteness (PSD)                             |
| $v^\top \mathbf{K} v \geq 0$ | For all $v \in \mathbb{R}^n$, quadratic form is non-negative      |
| PSD implication    | Ensures $\mathbf{K}$ is a valid covariance matrix for a Gaussian      |

#### 3. Intuition

- The kernel tells how similar $f(x)$ and $f(x')$ are expected to be.
- Choosing $k = \text{smooth}$ ⇒ functions vary slowly.
- Choosing $k = \text{periodic}$ ⇒ functions oscillate.
- Choosing $k = \text{linear}$ ⇒ functions are straight lines.

Thus, different covariance functions induce different function spaces that the GP “believes” are plausible before seeing data.

### Covariance

| Concept | Formula | Sample Formula | Meaning |
|----------|----------|----------------|----------|
| Variance | $\operatorname{Var}(X) = \mathbb{E}[(X - \mu_X)^2]$ | $\widehat{\operatorname{Var}}(X) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2$ | Measures how a single variable spreads around its mean |
| Covariance | $\operatorname{Cov}(X,Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$ | $\widehat{\operatorname{Cov}}(X,Y) = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$ | Measures how two variables move together |
| Correlation | $\operatorname{Corr}(X,Y) = \frac{\operatorname{Cov}(X,Y)}{\sigma_X \sigma_Y}$ | $\widehat{\operatorname{Corr}}(X,Y) = \frac{\widehat{\operatorname{Cov}}(X,Y)}{s_X s_Y}$ | Standardized measure of linear relationship between two variables, always in $[-1, 1]$ |

### Expectation 

- $\mathbb{E}[\cdot]$ = expectation: a property of the *true* underlying probability distribution  
- $\mu$ = the (unknown) true mean, often equal to $\mathbb{E}[X]$  
- $\bar{x}$ = sample average from observed data (an estimator of $\mu$)

Relationship (law of large numbers):

$$
\bar{x} \xrightarrow[n \to \infty]{} \mathbb{E}[X]
$$

### Multivariate Covariance Matrix

For $k$ random variables collected into a vector:

$$
\mathbf{X} =
\begin{bmatrix}
X_1 \\ X_2 \\ \vdots \\ X_k
\end{bmatrix}, \quad
\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}] =
\begin{bmatrix}
\mu_1 \\ \mu_2 \\ \vdots \\ \mu_k
\end{bmatrix}.
$$

The covariance matrix is:

$$
\Sigma = \mathbb{E}\!\left[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top\right].
$$

This expands to:

$$
\Sigma =
\begin{bmatrix}
\operatorname{Var}(X_1) & \operatorname{Cov}(X_1,X_2) & \cdots & \operatorname{Cov}(X_1,X_k) \\
\operatorname{Cov}(X_2,X_1) & \operatorname{Var}(X_2) & \cdots & \operatorname{Cov}(X_2,X_k) \\
\vdots & \vdots & \ddots & \vdots \\
\operatorname{Cov}(X_k,X_1) & \operatorname{Cov}(X_k,X_2) & \cdots & \operatorname{Var}(X_k)
\end{bmatrix}.
$$

**Properties:**
- Symmetric: $\Sigma^\top = \Sigma$  
- Positive semi-definite: $v^\top \Sigma v \ge 0 \; \forall v$

**Sample covariance matrix** (for data $A \in \mathbb{R}^{n \times k}$):

$$
\widehat{\Sigma} = \frac{1}{n-1}(A - \mathbf{1}\bar{A})^\top (A - \mathbf{1}\bar{A})
$$

Each element:

$$
\widehat{\Sigma}_{ij} = \frac{1}{n-1}\sum_{t=1}^n (a_{ti} - \bar{a}_i)(a_{tj} - \bar{a}_j)
$$

**Summary Table:**

| Object | Dimension | Definition | Interpretation |
|---------|------------|-------------|----------------|
| $\operatorname{Var}(X_i)$ | scalar | $\mathbb{E}[(X_i - \mu_i)^2]$ | Spread of variable $i$ |
| $\operatorname{Cov}(X_i, X_j)$ | scalar | $\mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]$ | Joint variation of $i$ and $j$ |
| $\Sigma$ | $k \times k$ | $\mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top]$ | Joint variability of all $k$ variables |

**Geometric interpretation:**
- Defines the elliptical shape of a multivariate distribution.
- For Gaussian $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$, contours satisfy:
  $$
  (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) = c.
  $$
- Eigenvectors → principal directions of spread.
- Eigenvalues → magnitude of spread along those directions.


### Why divide by $n - k$? (Degrees of Freedom)

When we estimate parameters from data, each estimated parameter imposes a constraint that reduces the number of *free* observations contributing to variability.

In general, the unbiased estimator of variance or covariance divides by:

$$
n - k
$$

where:
- $n$ = number of observations  
- $k$ = number of independent parameters or constraints estimated from data  

---

#### Example 1: Sample Mean (k = 1)

Using the sample mean $\bar{x}$ introduces one constraint:

$$
\sum_{i=1}^n (x_i - \bar{x}) = 0
$$

That removes 1 degree of freedom, giving denominator $n - 1$.

$$
\widehat{\sigma}^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2
$$

---

#### Example 2: Linear Regression (k = p + 1)

For a regression model:

$$
y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \varepsilon_i
$$

There are $p + 1$ estimated parameters (intercept + slopes).  
Residuals:

$$
r_i = y_i - \hat{y}_i
$$

The unbiased estimate of the residual variance uses denominator $n - (p + 1)$:

$$
\widehat{\sigma}^2 = \frac{1}{n - (p+1)} \sum_{i=1}^n r_i^2
$$

---

#### Example 3: General Matrix Form

Let $Y \in \mathbb{R}^{n \times 1}$, $X \in \mathbb{R}^{n \times k}$, and fitted parameters $\hat{\beta} = (X^\top X)^{-1}X^\top Y$.

Then:

$$
\widehat{\sigma}^2 = \frac{1}{n - k}(Y - X\hat{\beta})^\top (Y - X\hat{\beta})
$$

Here, $k$ parameters have been estimated from $n$ data points, leaving $n - k$ effective degrees of freedom.

---

#### Geometric Interpretation

Each estimated parameter removes one dimension from the space of residuals.  
The variance is averaged over the remaining $n - k$ directions:

$$
\text{df} = n - k
$$

For the sample mean, $k = 1$;  
for regression, $k = p + 1$;  
for general models, $k$ equals the number of fitted parameters.

---

**Summary Table**

| Scenario | Parameters Estimated ($k$) | Denominator | Interpretation |
|-----------|---------------------------|--------------|----------------|
| Sample mean | 1 | $n-1$ | Mean estimated → 1 df lost |
| Simple regression | 2 | $n-2$ | Intercept + slope |
| Multiple regression | $p+1$ | $n-(p+1)$ | One df per parameter |
| General model | $k$ | $n-k$ | Subtract fitted parameters |

This generalizes the $n-1$ rule: divide by the number of remaining degrees of freedom after fitting parameters.


#### Vector Convention in Linear Algebra and Machine Learning

In linear algebra and machine learning, vectors are assumed to be **column vectors** by default. This means:

- A column vector $\mathbf{x} \in \mathbb{R}^d$ is represented as a $d \times 1$ matrix.
- Its transpose, $\mathbf{x}^\top \in \mathbb{R}^{1 \times d}$, is a row vector.
- This convention allows matrix multiplication such as $\mathbf{y} = A\mathbf{x}$, where $A$ is a matrix, and the inner product $\mathbf{x}^\top \mathbf{w}$ to yield correctly dimensioned results.
- Some fields, such as statistics or econometrics, sometimes use row vectors as the default, but they adjust shapes accordingly to maintain consistent operations.

| Symbol             | Shape     | Description                            |
|--------------------|-----------|------------------------------------|
| $\mathbf{x}$       | $d \times 1$ | Column vector (features)            |
| $\mathbf{x}^\top$  | $1 \times d$ | Row vector (transpose of $\mathbf{x}$) |
| $\mathbf{w}$       | $d \times 1$ | Column vector (weights)             |
| $\mathbf{x}^\top \mathbf{w}$ | $1 \times 1$ | Scalar (dot product / weighted sum) |

Thus, by default, vectors are treated as columns, and the transpose symbol ensures the correct orientation for inner products and linear transformations.
