using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimisers,OptimizationOptimJL, Random, Plots, ComponentArrays, Optim

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
tsteps = range(tspan[1], tspan[2], length=101)
prob = ODEProblem(lotka_volterra!, u0, tspan)
sol = solve(prob, Tsit5(), saveat=tsteps)



# Extract time and data for training
t = sol.t
prey_data = sol[1, :]
predator_data = sol[2, :]

# Visualize the data
plot(t, prey_data, label="Prey (Rabbits)", xlabel="Time", ylabel="Population")
plot!(t, predator_data, label="Predators (Wolves)")


# Extract the training data
t_train = sol.t
y_train = Array(sol)  # Convert solution to array format

#Define RBF function
function rbf(x)
    return exp.(-x.^2)
end

# Define a neural network using Lux
dudt = Lux.Chain(
    Lux.Dense(2, 100, rbf),
    Lux.Dense(100, 100, rbf),
    Lux.Dense(100, 100, rbf),
    Lux.Dense(100, 2)
)

# Initialize neural network parameters
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, dudt)

# Define the neural ODE problem
n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=tsteps)

# Prediction function using neural ODE
function predict_neuralode(p)
    sol, _ = n_ode(u0, p, st)
    return Array(sol)
end

# Loss function comparing predictions to training data
function loss_neuralode(ps)
    pred = predict_neuralode(ps)
    loss = sum(abs2, y_train .- pred)
    return loss, pred
end

# Callback function for optimization process
function callback(p, l, pred; doplot = true)
    println("Loss: ", l)
    if doplot
        plt = plot(t_train, y_train[1, :], label="Prey Data", linestyle=:solid, linewidth=4)
        scatter!(plt, t_train, pred[1, :], label="Prey Prediction")
        plot!(plt, t_train, y_train[2, :], label="Predator Data", linestyle=:solid, linewidth=4)
        scatter!(plt, t_train, pred[2, :], label="Predator Prediction")
        display(plot(plt))
    end
    return false
end

# Initial parameters for optimization
pinit = ComponentArray(nn_params)

# Optimization setup with Adam optimizer
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

optprob = Optimization.OptimizationProblem(optf, pinit)

# First optimization run using Adam optimizer
result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.001), maxiters = 500, callback = callback)



# Train with BFGS
optprob2 = Optimization.OptimizationProblem(optf, result_neuralode.minimizer)
res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01), callback=callback, maxiters = 100)



# Forecasting with the trained neural ODE model
tspan_forecast = (0.0, 20.0)
n_ode_forecast = NeuralODE(dudt, tspan_forecast, Tsit5(), saveat=range(tspan_forecast[1], tspan_forecast[2], length=201))
forecasted_sol, _ = n_ode_forecast(u0, res2.u, st)
forecasted_sol = Array(forecasted_sol)

# Plot the forecasting results
plot(sol.t, y_train[1, :], label="Prey (Actual)", xlabel="Time", ylabel="Population",linestyle=:solid, linewidth=4)
plot!(sol.t, y_train[2, :], label="Predators (Actual)", linestyle=:solid, linewidth=4)
plot!(range(tspan_forecast[1], tspan_forecast[2], length=201), forecasted_sol[1, :], label="Prey (Forecasted)", linestyle=:dash,linewidth=4)
plot!(range(tspan_forecast[1], tspan_forecast[2], length=201), forecasted_sol[2, :], label="Predators (Forecasted)", linestyle=:dash, linewidth=4)