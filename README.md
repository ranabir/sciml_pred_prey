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

### Code Files:
1. The Lotka_Volterra_Model.jl files contains synthetic data(noiseless) using Lotka-Volterra model, application of Neural ODE and forecasting using
   - relu activation function
   - 3 layers (including input and output layer)
   - Adam Optimizer with maximum iterations of 1000 and learning rate of 0.001
2. **The Lotka_Volterra_Model_rbf_BFGS.jl** file contains same data and follows same methodology except change in following ones:
   - rbf activation function
   - 4 layers (including input and output layer)
   - Adam Optimizer with maximum iterations of 500 and learning rate of 0.001
   - BFGS Optimizer with maximum iterations of 100 and learning rate of 0.01
3. The **Lotka_Volterra_Model_forecast.jl** file contains the forecasting breakdown case scenarios:
   - 90% training data
   - 70% training data
   - 50% training data
   - 30% training data
   - 10% training data

### Neural ODE 
Using true data from the above and then fitting Neural ODE to fit the data
Using Adam Optimizer to optimize:
<img width="500" alt="image" src="https://github.com/user-attachments/assets/63094ad3-057c-4983-af69-b21b5fb27179"> 



**After using RBF Activation Function and both ADAM and BFGS Optimizer, Final Output:**


<img width="500" alt="lotka_volterra_optimized_fit_rbf_bfgs" src="https://github.com/user-attachments/assets/fa44ccda-468f-4ccc-ab5a-fa63900d757c">


### Forecast: 
Purpose: The forecasting part is intended to use the trained neural ODE model to predict future states of the system beyond the time range it was originally trained on.
Steps:
- Defined an extended time span for prediction.
- Specified the time points for which predictions are needed.
- Created a neural ODE object configured for forecasting.
- Ran the neural ODE to generate forecasts using the optimized model parameters.
- Plotted both the training and forecasted data to compare model predictions against actual data and observe expected future behavior.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/fd5cd3c5-12a8-4b72-902f-cd5e0feedaee">

**After Further Optimization with BFGS and rbf activation function :**

<img width="500" alt="lotka_volterra_forecast_optimized" src="https://github.com/user-attachments/assets/5e47e50b-e1af-4f07-8d08-c68de540f4ea">


## Forecast Break Down Point: 
### Case 1: Training Neural ODE with 90% data and forecasting on remaining 10% 

<img width="500" alt="lotka_volterra_forecast_90_10" src="https://github.com/user-attachments/assets/cddd8d51-d586-4103-92c5-a739498e59cf">

### Case 2: Training Neural ODE with 70% data and forecasting on remaining 30% 

<img width="500" alt="image" src="https://github.com/user-attachments/assets/5a1ffb5f-e064-44ea-ab6e-a7eaecdc2df8">

### Case 3: Training Neural ODE with 50% data and forecasting on remaining 50% 

<img width="500" alt="lotka_volterra_forecast_50_50" src="https://github.com/user-attachments/assets/118c2912-646e-484e-bc36-b678b3ae1a9c">

### Case 4: Training Neural ODE with 40% data and forecasting on remaining 60% 

<img width="500" alt="lotka_volterra_forecast_40_60" src="https://github.com/user-attachments/assets/5e081ec3-06b3-4bbd-b57f-fa81710d371d">

### Case 5: Training Neural ODE with 35% data and forecasting on remaining 65% 

<img width="500" alt="lotka_volterra_forecast_35_65" src="https://github.com/user-attachments/assets/ff2ae49d-6e66-469b-95ea-862663cde2c1">

### Case 5: Revised Plot: Training Neural ODE with 35% data and forecasting on remaining 65% 

<img width="500" alt="lotka_volterra_forecast_35_65_revised1" src="https://github.com/user-attachments/assets/2b8a2e7c-c1fa-4c31-b609-01700c0e12f6">

### Case 6: Training Neural ODE with 30% data and forecasting on remaining 70% 

<img width="500" alt="lotka_volterra_forecast_30_70" src="https://github.com/user-attachments/assets/1ac45e2c-9861-46c0-9ce8-6e3e6ffe6e81">

### Case 7: Training Neural ODE with 10% data and forecasting on remaining 90% 

<img width="500" alt="lotka_volterra_forecast_10_90" src="https://github.com/user-attachments/assets/5d7ff1e3-a42c-4097-9e04-91e469f5b240">









