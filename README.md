# Fundamentals of Machine Learning Theory

This repo contains mathematical derivations of key concepts in machine learning theory I worked on during my studies. All work is authored by Darien Nouri.

## Files and Topics

### 1. `multi_class_grad_MLE_hinge_loss.pdf`
- **Model Selection**
  - Criteria for choosing the best model based on validation error
  - Reporting generalization error
  
- **Gradient of Multi-Class Classifier**
  - Derivation of the gradient of the cross-entropy loss function with respect to the parameter matrix
  
- **Maximum Likelihood Estimate of a Gaussian Model**
  - Expression of the log-likelihood as a function of mean and variance
  - Partial derivatives with respect to mean and variance, and solving for maximum likelihood estimates
  
- **Hinge Loss Gradients**
  - Piece-wise representation and gradient of the hinge loss function

### 2. `perceptron_logistic_regression_grad.pdf`
- **Empirical vs. Expected Cost**
  - Discussion on empirical cost function and differential weighting of data points
  
- **Perceptron Learning Algorithm**
  - Proof of key properties and weight update rule correctness for the Perceptron algorithm
  
- **Gradient of Logistic Regression**
  - Derivation of the gradient for the logistic regression loss function

### 3. `poisson_gradients_pdf_properties.pdf`
- **Poisson Distribution**
  - Derivation and properties of the Poisson distribution
  
- **Gradient Computation**
  - Calculation of gradients in different contexts including partial derivatives and chain rule applications
  
- **Integration and PDF Properties**
  - Analysis of functions for PDF properties
  - Computation of expected value and variance for given distributions
  - Probability computations involving joint density functions
  - Central Limit Theorem applications


# Sample Derivations

## T1: Model Selection

1. Which $i$ and $t$ should we pick as the best model and why?

   We should pick the model $M_t^i$ with the lowest validation error $\mathcal{L}_{val,t}^i$ with an $i$ based on the lower validation error of either logistic regression $(i = 1)$ or SVMs $(i=0)$. We should pick $t$ where the validation error was the lowest for the selected model.

2. How should we report the generalization error of the model?

   We should report the generalization error of the model based on its performance on the test set $D_{test}$ using the parameter configuration that was selected based on validation set performance.


<br/>
<br/>


## T2: Gradient of Multi-Class Classifier

Derivation of the gradient of the cross-entropy loss function $L_w$ with respect to the parameter matrix $w$.

Given, $$\mathcal{L}_w(x, y) = - \sum_{j=1}^{K} y_j \cdot \log p_j \qquad p_j=\sigma(w^Tx)_j = \frac{e^{w_j^Tx}}{\sum_{i=1}^{K} e^{w_i^Tx}}$$

$\log(p_j)$ w.r.t. $p_j$: $$\frac{\partial}{\partial p_j} \log(p_j) = \frac{1}{p_j}$$

$$
\begin{align*} 
    \frac{\partial \mathcal{L}_w(x, y)}{\partial w} &= \frac{\partial}{\partial w} - \sum_{j=1}^K y_j \log p_j 
    = - \sum_{j=1}^K y_j \frac{\partial \log p_j}{\partial w} \\
\end{align*}
$$

Using the chain rule we can find the partial derivatives that compose the gradient such that:

$$
\begin{align*} 
    \frac{\partial \mathcal{L}_w(x, y)}{\partial w} &= - \sum_{j=1}^K y_j \frac{\partial \log p_j}{\partial p_j} \cdot \frac{\partial p_j}{\partial w_{k}^T} \cdot \frac{\partial w_{k}^T}{\partial w}
\end{align*}
$$

First we will find the partial derivative of $\mathcal{L}_w(x, y)$ w.r.t $P_j$:

$$
\begin{align*} 
    \frac{\partial \mathcal{L}_w(x, y)}{p_j} 
    &= \frac{\partial}{\partial p_j} - \sum_{j=1}^K y_i \log p_j 
	= - \sum_{j=1}^K y_i \cdot \log p_j 
    = - \sum_{j=1}^K \frac{y_j}{p_j}
\end{align*}
$$

Next we'll differentiate $p_j$ w.r.t $w_k^Tx$. We must consider two cases:

**Case 1:** $j=k$, to find how the softmax probability of class $j$ changes w.r.t its own score.

$$
\begin{align*}
    \frac{\partial p_j}{\partial w_k^Tx}  &= \frac{\partial }{\partial w_k^Tx}  \frac{e^{w_j^Tx}}{\sum_{i=1}^{K} e^{w_i^Tx}}
    = \frac{e^{w_j^Tx} \sum_{i=1}^{K} e^{w_i^Tx} - e^{w_j^Tx}e^{w_j^Tx}}{(\sum_{i=1}^{K} e^{w_i^Tx})^2}\\\\
    &= \frac{e^{w_j^Tx} \left(\sum_{i=1}^{K} e^{w_i^Tx} - e^{w_j^Tx}\right)}{(\sum_{i=1}^{K} e^{w_i^Tx})^2}
    = \frac{e^{w_j^Tx}}{\sum_{i=1}^{K} e^{w_i^Tx}} \cdot \left( 1 - \frac{e^{w_j^Tx}}{\sum_{i=1}^{K} e^{w_i^Tx}} \right)\\\\
    &= p_j (1 - p_j)
\end{align*}
$$

**Case 2:** $j \neq k$, to find how the softmax probability of class $j$ changes w.r.t. the score of a different class $k$.

$$
\begin{align*}
    \frac{\partial p_j}{\partial w_k^Tx}  &= \frac{\partial }{\partial w_k^Tx}  \frac{e^{w_j^Tx}}{\sum_{i=1}^{K} e^{w_i^Tx}}
    = \frac{0 - e^{w_j^Tx}e^{w_k^Tx}}{(\sum_{i=1}^{K} e^{w_i^Tx})^2}\\\\
    &= - \frac{e^{w_j^Tx}}{\sum_{i=1}^{K} e^{w_i^Tx}} \cdot \frac{e^{w_k^Tx}}{\sum_{i=1}^{K} e^{w_i^Tx}}
    = - p_j p_k
\end{align*}
$$

Now, let's differentiate $ w_k^Tx $ with respect to $ w $:

$$
\begin{align*}
\frac{\partial \mathcal{L}w(x, y)}{\partial w_{k}^T}
&= - y_k \frac{\partial \log p_k}{\partial p_k} (1-p_k) + \sum_{j \neq k} \frac{y_j}{p_j} (-p_j p_k) \\
&= - y_k (1-p_k) + \sum_{j \neq k} y_j p_k \\
&= - y_k (1-p_k) + p_k \sum_{j \neq k} y_j \\
&= - y_k (1-p_k) + p_k (1-p_k) 
= p_k - y_k
\end{align*}
$$

Finally, we find the gradient of $ \mathcal{L}_w(x, y) $ with respect to $ W $. Luckily for us that only mean multiplying what we have with the partial derivative of the sarimax input w.r.t $w$ which is simply the input vector $x$.

$$
\begin{align*}
\frac{\partial \mathcal{L}w(x, y)}{\partial w}
&= - \sum_{j=1}^k y_j \frac{\partial \log p_j}{\partial p_j} \cdot \frac{\partial p_j}{\partial w_{k}^T} \cdot \frac{\partial w_{k}^T}{\partial w} \\\\
&= x(p_k - y_k)
\end{align*}
$$

<br>
<br/>

## T3: Maximum Likelihood Estimate of a Gaussian Model

The pdf of a Gaussian Distribution with mean $\mu$ and variance $\sigma^2$ is given by:

$$
    P(x|\mu,\sigma) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
$$

The likihood of observing the dataset D with mean $\mu$ and variance $\sigma^2$:

$$
    \mathcal{L}(D|\mu,\sigma) = \prod_{i=1}^{n} \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
$$

1. Expression of log-likelihood $\mathcal{L}_{\mu, \sigma}(D)$ as a function of $\mu$ and $\sigma$.

   By taking the logrithm on both sides and expanding:

   $$
   \begin{align*}
       \log \mathcal{L}(D|\mu,\sigma) &= \log \left[  \prod_{i=1}^{n} \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}} \right] \\
       &= \sum_{i=1}^{n} \log \left( \frac{1}{\sigma \sqrt{2\pi}} \right) - \frac{(x - \mu)^2}{2\sigma^2}\\
       &= \sum_{i=1}^{n} -\log(\sigma) -\frac{1}{2}\log(2\pi) - \frac{(x - \mu)^2}{2\sigma^2}\\
       &= -n\log(\sigma) - \frac{n}{2} \log(2\pi) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i-\mu)^2
   \end{align*}
   $$

2. Partial derivative of $\mathcal{L}(D)$ with respect to $\mu$, and equating to zero.

   $$
   \begin{align*}
       \frac{\partial\mathcal{L}(D|\mu,\sigma)}{\partial\mu} 
       &= \frac{\partial}{\partial\mu} \left[-nlog(\sigma) - \frac{n}{2}\log(2\pi) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i-\mu)^2\right]\\
       &= - \frac{1}{2\sigma^2} \sum_{i=1}^{n} \frac{\partial}{\partial\mu} (x_i-\mu)^2\\
       &= \frac{1}{\sigma^2} \sum_{i=1}^{n} (x_i-\mu)\\
       \text{Equate to zero:}\\
       0 &= \frac{1}{\sigma^2} \sum_{i=1}^{n} (x_i-\mu)\\
       n\mu &= \sum_{i=1}^{n}x_i\\
       \mu &= \frac{1}{n} \sum_{i=1}^{n}x_i
   \end{align*}
   $$

3. Partial derivative of $\mathcal{L}(D)$ with respect to $\sigma$, and equating to zero.

   $$
   \begin{align*}
       \frac{\sigma\mathcal{L}(D|\mu,\sigma)}{\sigma\mu} 
       &= \frac{\sigma}{\sigma\mu} \left[-nlog(\sigma) - \frac{n}{2}\log(2\pi) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i-\mu)^2\right]\\
       \frac{n}{\sigma} &= \frac{1}{\sigma^3} \sum_{i=1}^{n} (x_i-\mu)^2\\
       \sigma^2 &= \frac{1}{n} \sum_{i=1}^{n} (x_i-\mu)^2\\
       \sigma &= \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i-\mu)^2}
   \end{align*}
   $$

<br>

## T4: Hinge Loss Gradients

Since Hinge loss includes a non-differentiable point at $1 - y \cdot f_\theta(x) = 1$, it is not continuously differentiable at all points. However, the linear segments that compose Hinge loss are differentiable within their intervals. If we have the function

$$
L_{\text{Hinge}}(x, y, \theta) = \max\left[0, 1 - y \cdot f_{\theta}(x)\right]
$$

We can reconstruct the function into its piece-wise representation

$$
L_{\text{Hinge}}(x, y, \theta) =
\begin{cases} 
0 & \text{if } y \cdot f_\theta(x) \geq 1, \\
1 - y \cdot f_\theta(x) & \text{if } y \cdot f_\theta(x) < 1.
\end{cases}
$$

Such that the gradient of the loss w.r.t $\theta$ is

$$
\nabla_\theta L_{\text{Hinge}}(x, y, \theta) =
\begin{cases} 
0 & \text{if } y \cdot f_\theta(x) \geq 1, \\
-y \cdot \nabla_\theta f_\theta(x) & \text{if } y \cdot f_\theta(x) < 1.
\end{cases}
$$

In the first case, where the loss is 0, when the model's prediction is correct and beyond the margin, defined by $1-y \cdot f_0(x)$, the parameters are not updated because the example is correctly classified.

In the second case, the loss is the function of $\theta$ through $f_\theta(x)$, which is differentiable w.r.t to $\theta$ such that its gradient exists.

Therefore, even though there exists a non-differentiable point, namely at $1 - y \cdot f_\theta(x) = 0$, the use of gradient-based optimization is not a problem because:

- When an example is correctly classified the loss and gradient are both zero, and no update is made.
- When an example is misclassified the loss is linear and the gradient can be normally calculated.
