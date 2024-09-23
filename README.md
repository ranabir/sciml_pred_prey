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
4. The **UDE/Lotka_Volterra_UDE_shallow.jl** file contains optimized model fit code for UDE system

### Neural ODE 
Using true data from the above and then fitting Neural ODE to fit the data
Using Adam Optimizer to optimize:
<img width="500" alt="image" src="https://github.com/user-attachments/assets/63094ad3-057c-4983-af69-b21b5fb27179"> 




**After using RBF Activation Function and both ADAM and BFGS Optimizer, Final Output:**


<img width="500" alt="lotka_volterra_optimized_fit_rbf_bfgs" src="https://github.com/user-attachments/assets/fa44ccda-468f-4ccc-ab5a-fa63900d757c">

## Hyperparameter Tuning Plots

### Logarithmic Loss Variation with Different Activation Functions

<img width="500" alt="Log 10 Loss Variation with Activation Functions" src="https://github.com/user-attachments/assets/2e0859eb-d10c-4cf2-934f-22d813ece501">


### Loss Variation with Different Number of Hidden Units

<img width="500" alt="Loss Variation with Different Number of Hidden Units" src="https://github.com/user-attachments/assets/ca4cf3d7-b61a-4282-89ba-b920652b254d">

### Logarithmic Loss Variation with Different Adam Step Sizes

<img width="500" alt="Log Loss Variation with Adam Step Sizes" src="https://github.com/user-attachments/assets/7f019f41-22b0-4d14-9e71-02efa3deec3b">


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

<img width="500" alt="lotka_volterra_forecast_90_10_revised" src="https://github.com/user-attachments/assets/c2d687a8-2a75-4b94-9a6c-ff676686b843">

### Case 2: Training Neural ODE with 70% data and forecasting on remaining 30% 

<img width="500" alt="image" src="https://github.com/user-attachments/assets/5a1ffb5f-e064-44ea-ab6e-a7eaecdc2df8">

### Case 3: Training Neural ODE with 50% data and forecasting on remaining 50% 

<img width="500" alt="lotka_volterra_forecast_50_50_revised" src="https://github.com/user-attachments/assets/76961685-4ba8-4b74-856c-2ab4b832b789">

### Case 4: Training Neural ODE with 40% data and forecasting on remaining 60% 

<img width="500" alt="lotka_volterra_forecast_40_50_revised" src="https://github.com/user-attachments/assets/d009e436-2181-4290-9ddc-dbbfc460a4b5">

### Case 5: Revised Plot: Training Neural ODE with 35% data and forecasting on remaining 65% 

<img width="500" alt="lotka_volterra_forecast_35_65_revised1" src="https://github.com/user-attachments/assets/2b8a2e7c-c1fa-4c31-b609-01700c0e12f6">

### Case 6: Training Neural ODE with 30% data and forecasting on remaining 70% 

<img width="500" alt="lotka_volterra_forecast_30_70_revised" src="https://github.com/user-attachments/assets/8765e1be-2886-4614-a594-b9faa5a2d11e">

### Case 7: Training Neural ODE with 10% data and forecasting on remaining 90% 

<img width="500" alt="lotka_volterra_forecast_10_90" src="https://github.com/user-attachments/assets/5d7ff1e3-a42c-4097-9e04-91e469f5b240">



## UDE
The generated data for the prey and predator populations over a time span of 10 units is then used to train a UDE model. The UDE incorporates neural networks to learn the unknown interaction terms in the Lotka-Volterra equations. Specifically, we treat the interaction terms $\beta xy$ and $\delta xy$ as functions that can be learned from data using neural networks.
UDE Construction
In this implementation:

The learned system using neural networks is described as:

$$
\begin{cases}
\frac{dx}{dt} = \alpha x - \text{NN}_1(x, y) \\
\frac{dy}{dt} = -\gamma y + \text{NN}_2(x, y)
\end{cases}
$$

Where:
- $\text{NN}_1(x, y)$ is a neural network that approximates the interaction term $\beta xy$.
- $\text{NN}_2(x, y)$ is a neural network that approximates the interaction term $\delta xy$.

Two neural networks (NN1 and NN2) model the interaction terms $\beta xy$ and $\delta xy$ respectively.
These networks are trained to minimize the loss between the generated data and the predicted populations.

### Optimization Strategy
To train the UDE, the following optimization process is used:

Adam Optimizer: The UDE is first trained using the Adam optimizer with a learning rate of 0.001 over 20,000 iterations.
RMSProp Optimizer: The model is then fine-tuned using RMSProp with a learning rate of 0.001 and momentum ρ = 0.9 for 5,000 iterations.

### Optimized Fitted Data: 
<img width="500" alt="shallow_network_fit" src="https://github.com/user-attachments/assets/c0b92ba3-d3a5-4116-ac24-7f99368423cf">


### Neural Network Architecture
Most importantly in UDE it works perfectly with shallow networks 

The neural networks $\text{NN}_1$ and $\text{NN}_2$ are constructed using the `Lux` library in the following architecture:

```julia
NN1 = Lux.Chain(
    Lux.Dense(2, 10, relu),
    Lux.Dense(10, 10, relu),
    Lux.Dense(10, 10, relu),
    Lux.Dense(10, 1)
)

NN2 = Lux.Chain(
    Lux.Dense(2, 10, relu),
    Lux.Dense(10, 10, relu),
    Lux.Dense(10, 10, relu),
    Lux.Dense(10, 1)
)
```

### Results
The final trained UDE model is used for forecasting the population dynamics over a longer time span of 20 units. The results show that the neural network accurately captures the cyclical behavior of the predator-prey dynamics. The predictions are visualized alongside the original data for comparison.

### Visualization
The following plot shows the learned dynamics:

The solid lines represent the generated data for the prey and predator populations.
The dashed lines show the UDE model's predictions for both populations over the forecasting period.
The model accurately predicts the oscillations in prey and predator populations, indicating the successful learning of the interaction dynamics.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/0e742ad7-7c4c-41b2-849c-e3189926c2ed">

### Forecasting 
The UDE model is trained on lesser data (% of original time plan) and forecasted on remaining part to find the forecasting breakdown point 
From the above Neural ODE model we found that the forecast breakdown point was 0.35 % of training data and the forecast was completely wrong 
Whereas in case of UDE for the same 0.35% training data, the model is able to capture the underlying data very well 

### Case: Training Neural ODE with 35% data and forecasting on remaining 65% 
<img width="500" alt="forecast ude 0 35 trial 2" src="https://github.com/user-attachments/assets/797b7131-3ba7-49cc-9988-8a22fc72f634">

In addition to it, the UDE trains well with even shallower neural network of 5 neurons of 3 layers








