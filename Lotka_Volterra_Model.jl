using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using ComponentArrays

# Generate synthetic data using the Lotka-Volterra model
α = 1.5
β = 1.0
γ = 0.5
δ = 2.0

function lotka_volterra!(du, u, p, t)
    x, y = u
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
end

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(lotka_volterra!, u0, tspan)
sol = solve(prob, Tsit5(), saveat=0.1)



# Extract time and data for training
t = sol.t
prey_data = sol[1, :]
predator_data = sol[2, :]

# Visualize the data
plot(t, prey_data, label="Prey (Rabbits)", xlabel="Time", ylabel="Population")
plot!(t, predator_data, label="Predators (Wolves)")



## Neural ODE
using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots

# Neural network architecture
function neural_ode!(du, u, p, t)
    du .= p(u)
end

# Initialize neural network with Lux
nn = Lux.Chain(
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2)
)

# Neural ODE problem definition
rng = Random.default_rng()
p, st = Lux.setup(rng, nn)  # Initialize parameters
prob_neural = ODEProblem(neural_ode!, u0, tspan, p)

# Loss function for training
function predict_neural(p)
    solve(prob_neural, Tsit5(), u0=u0, p=p, saveat=0.1)
end

loss_neural(p) = sum(abs2, sol.u - predict_neural(p).u)

# Use Flux's destructure to flatten parameters
params, re = Flux.destructure(nn)

# Define the optimization problem using flat parameters
optprob = Optimization.OptimizationProblem(x -> loss_neural(re(x)), params)

# Solve the optimization problem
result = Optimization.solve(optprob, OptimizationOptimJL.BFGS())



# Update parameters after optimization
p_opt = Lux.unflatten(result.u, st)


## Forecasting
# Extend the time span for forecasting
tspan_extended = (0.0, 20.0)  # Forecast up to 20 time units
prob_neural_extended = ODEProblem(neural_ode!, u0, tspan_extended, p_opt)
forecasted_sol = solve(prob_neural_extended, Tsit5(), saveat=0.1)

# Plot the results
plot(sol.t, prey_data, label="Prey (Actual)", xlabel="Time", ylabel="Population")
plot!(sol.t, predator_data, label="Predators (Actual)")
plot!(forecasted_sol.t, forecasted_sol[1, :], label="Prey (Forecasted)", linestyle=:dash)
plot!(forecasted_sol.t, forecasted_sol[2, :], label="Predators (Forecasted)", linestyle=:dash)
