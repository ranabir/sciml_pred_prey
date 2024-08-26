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

### Neural ODE 
Using true data from the above and then fitting Neural ODE to fit the data
Using Adam Optimizer to optimize

### Forecast: 
Purpose: The forecasting part is intended to use the trained neural ODE model to predict future states of the system beyond the time range it was originally trained on.
Steps:
- Define an extended time span for prediction.
- Specify the time points for which predictions are needed.
- Create a neural ODE object configured for forecasting.
- Run the neural ODE to generate forecasts using the optimized model parameters.
- Visualize both the training and forecasted data to compare model predictions against actual data and observe expected future behavior.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/fd5cd3c5-12a8-4b72-902f-cd5e0feedaee">

### Code Files:
The Lotka_Volterra_Model.jl files contains synthetic data(noiseless) using Lotka-Volterra model 

   



