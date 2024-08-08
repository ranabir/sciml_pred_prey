# Predator Prey Lotka-Volterra Model
Looking at Predator Prey Model and applying Neural ODE, UDE and PINNs to fit the model
The Lotka-Volterra model, also known as the predator-prey model, describes the dynamics of biological systems in which two species interact, one as a predator and the other as prey. The model is represented by a pair of first-order, nonlinear differential equations:

### Summary

The Lotka-Volterra model captures the cyclical nature of predator-prey interactions:

$$
\begin{cases}
\frac{dx}{dt} = \alpha x - \beta xy \\
\frac{dy}{dt} = \delta xy - \gamma y
\end{cases}
$$

where:
- $x$ represents the number of prey (e.g., rabbits).
- $y$ represents the number of predators (e.g., foxes).
- $\alpha$ is the natural growth rate of prey in the absence of predators.
- $\beta$ is the rate at which predators destroy prey.
- $\gamma$ is the natural death rate of predators in the absence of prey.
- $\delta$ is the rate at which predators increase by consuming prey.

### Plot:
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/9832d203-5f0f-44d5-a560-c43f2fecb11c">

### Dynamics:
1. When prey is abundant, predators have plenty to eat, leading to an increase in the predator population.
2. As the predator population grows, it consumes more prey, leading to a decrease in the prey population.
3. With fewer prey available, the predator population starts to decline due to starvation.
4. As the predator population decreases, the prey population begins to recover, starting the cycle anew.

   


