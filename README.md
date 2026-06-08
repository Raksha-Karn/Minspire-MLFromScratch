# Machine Learning from Scratch

## Day 1 Notebook - Optimization Algorithms from Scratch

In this notebook, I implemented **Gradient Descent**, **Stochastic Gradient Descent (SGD)**, **Momentum-Based Descent**, **RMSProp**, and the **Adam Optimizer** completely from scratch. 

The goal of this project was to dive beneath the surface of modern machine learning frameworks to understand the mathematical physics driving model training, how to effectively tune learning rates ($\alpha$), and how to select the optimal optimizer for various loss landscapes.

---

## 💡 Key Insights & Takeaways

### 1. The Geometry of Learning Rates & Curvature
Through implementing these algorithms, I discovered a profound mathematical link between a function's curvature and its optimal learning rate:
* The ideal learning rate is fundamentally bounded by the inverse of the function's second derivative ($1 / f''(x)$).
* In a perfect 1D or 2D world, replacing a guessed learning rate with the inverse of the second derivative (**Newton's Method**) allows the optimizer to calculate the exact structural curvature of the bowl, auto-scale its step size, and land at the absolute local minimum in **exactly one single step**.

### 2. The Computational Reality of Deep Learning
While second derivatives provide the perfect, hyperparameter-free step scale, they are not viable in modern deep learning practice:
* For a Large Language Model (like Gemini) dealing with billions or trillions of parameters, the second derivative becomes a massive grid of every parameter pair called the **Hessian Matrix**.
* Calculating and inverting a trillion-by-trillion matrix on every single training step would instantly paralyze the world's fastest supercomputers.

### 3. Why Adam Rules the AI Industry
Because calculating the true second derivative is too computationally expensive, **Adam acts as a brilliant mathematical cheat code**. 
* By maintaining an exponential moving average of squared gradients in the denominator (`v_hat`), Adam dynamically calculates a lightning-fast, cheap *approximation* of the second derivative's scaling properties.
* Combined with directional momentum (`m_hat`) in the numerator to smooth out noisy trajectories, Adam delivers the ultimate balance of **stability, noise filtering, and computational efficiency**, making it the undisputed default optimizer for modern deep learning.