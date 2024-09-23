using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, OptimizationOptimisers
using ComponentArrays, OptimizationOptimisers

# Generated data using the Lotka-Volterra Predator-Prey Model
α = 1.5
β = 1.0
γ = 0.5
δ = 2.0

u0 = [1.0, 1.0]  # Initial conditions [Prey, Predators]
tspan = (0.0, 10.0)  # Time span
datasize = 101
t = range(tspan[1], tspan[2], length=datasize)

# Known Lotka-Volterra equations
function lotka_volterra!(du, u, p, t)
    x, y = u
    du[1] = α * x - β * x * y  # Prey
    du[2] = -δ * y + γ * x * y  # Predators
end

# ODE problem to generate data
prob = ODEProblem(lotka_volterra!, u0, tspan)
sol = Array(solve(prob, Tsit5(), u0=u0, saveat=t))

Prey_Data = Array(sol)[1, :]
Predator_Data = Array(sol)[2, :]

# Construction of UDE
# Define neural networks to model interactions in the Lotka-Volterra system
rng = Random.default_rng()
Random.seed!(1028)

NN1 = Lux.Chain(
    Lux.Dense(2, 5, relu),
    Lux.Dense(5, 5, relu),
    Lux.Dense(5, 1)
)

NN2 = Lux.Chain(
    Lux.Dense(2, 5, relu),
    Lux.Dense(5, 5, relu),
    Lux.Dense(5, 1)
)

# Initialize neural network parameters
p1, st1 = Lux.setup(rng, NN1)
p2, st2 = Lux.setup(rng, NN2)

p0_vec = (layer_1 = p1, layer_2 = p2)
p0_vec = ComponentArray(p0_vec)

# Define UDE system, where interaction terms are learned by neural networks
function ude_system!(du, u, p, t)
    x, y = u
    NN_beta_xy = abs(NN1([x, y], p.layer_1, st1)[1][1])
    NN_gamma_xy = abs(NN2([x, y], p.layer_2, st2)[1][1])
    
    du[1] = α * x - NN_beta_xy  # Neural network learns β * x * y
    du[2] = -δ * y + NN_gamma_xy  # Neural network learns γ * x * y
end

# UDE problem
prob_ude = ODEProblem{true}(ude_system!, u0, tspan)

# Prediction function using UDE
function predict_ude(θ, t_train_partial)
    sol_ude = Array(solve(prob_ude, Tsit5(), p=θ, saveat=t_train_partial,
                          sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
    return sol_ude
end

# Define loss function based on available Prey and Predator data
function loss_ude(θ, t_train_partial, Prey_Train, Predator_Train)
    sol_pred = predict_ude(θ, t_train_partial)
    if size(sol_pred, 2) != length(Prey_Train)
        error("Dimension mismatch: Predictions have size ", size(sol_pred), " but expected size is (2, ", length(Prey_Train), ")")
    end
    loss = sum(abs2, Prey_Train .- sol_pred[1, :]) + sum(abs2, Predator_Train .- sol_pred[2, :])
    return loss
end

# Define callback function to monitor progress and plot during optimization
function callback(θ, l, pred, iteration, opt_method; doplot=true)
    println("Iteration: ", iteration, ", Optimization Method: ", opt_method, ", Loss: ", l)

    if iteration % 10 == 0 
        if doplot
            plt = plot(t, Prey_Data, label="Prey Data", linestyle=:solid, linewidth=4)
            scatter!(plt, t, pred[1, :], label="Prey Prediction")
            plot!(plt, t, Predator_Data, label="Predator Data", linestyle=:solid, linewidth=4)
            scatter!(plt, t, pred[2, :], label="Predator Prediction")
            display(plt)
        end
    end
    return false  # Continue the optimization
end

# Initialize the previous_loss variable
previous_loss = Ref(Inf)

# RMSProp optimizer setup
η = 0.0001   # Learning rate
ρ = 0.9     # Momentum
ϵ = 1e-8    # Epsilon to avoid division by zero
centred = false  # Whether to use centered RMSProp

# Function to train and forecast with different train ratios
function train_and_forecast_with_ratio(train_ratio)
    n_train = floor(Int, train_ratio * datasize)

    # Training data
    t_train_partial = t[1:n_train]
    Prey_Train = Prey_Data[1:n_train]
    Predator_Train = Predator_Data[1:n_train]

    # Define a new problem using the same initial conditions and UDE, but for the partial training data
    prob_partial = ODEProblem{true}(ude_system!, u0, (tspan[1], t[n_train]))

    # Optimization function for the partial training data
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_ude(x, t_train_partial, Prey_Train, Predator_Train), adtype)
    optprob = Optimization.OptimizationProblem(optf, p0_vec)

    # Adam optimizer setup with 20000 iterations
    iteration = Ref(0)
    η_adam = 0.001  # Adam learning rate
    
    res_adam = Optimization.solve(optprob, OptimizationOptimisers.Adam(η_adam), callback = (p, l) -> begin
        iteration[] += 1
        pred = predict_ude(p, t_train_partial)
        callback(p, l, pred, iteration[], "Adam")
    end, maxiters=20000)

    optprob_rmsprop = Optimization.OptimizationProblem(optf, ComponentArray(res_adam.minimizer))

    # Optimization run using RMSProp optimizer
    iteration[]=0
    
    res_rmsprop = Optimization.solve(optprob_rmsprop, OptimizationOptimisers.RMSProp(η, ρ, ϵ; centred=centred), callback = (p, l) -> begin
        iteration[] += 1
        pred = predict_ude(p, t_train_partial)
        callback(p, l, pred, iteration[], "RMSProp")
    end, maxiters=10000)

    # Forecasting data (remaining)
    last_point_train = predict_ude(res_adam.minimizer, t_train_partial)[:, end]  # Using last point of training as initial condition for forecasting
    t_forecast_partial = t[n_train:end]
    prob_forecast_partial = ODEProblem{true}(ude_system!, last_point_train, (t_train_partial[end], tspan[2]))

    # Forecast for the remaining time span using the trained UDE
    forecasted_sol = Array(solve(prob_forecast_partial, Tsit5(), p=res_rmsprop.minimizer, saveat=t_forecast_partial))

    # Calculate deviation between predicted and actual values
    forecast_deviation_prey = abs.(Prey_Data[n_train:end] - forecasted_sol[1, :])
    forecast_deviation_predator = abs.(Predator_Data[n_train:end] - forecasted_sol[2, :])

    # Plotting the training data and the forecast
    plt = plot(
        t_train_partial, Prey_Train, label="Ground truth training (prey)", xlabel="Time", ylabel="Population", 
        linestyle=:solid, linewidth=4, color=:red , legendfontsize=6
    )
    plot!(plt, t_train_partial, Predator_Train, label="Ground truth training (predator)", linestyle=:solid, linewidth=4, color=:blue, legendfontsize=6)
    plot!(plt, t_forecast_partial, Prey_Data[n_train:end], label="Ground truth forecasting (prey)", linestyle=:solid, linewidth=0.5, color=:red, legendfontsize=6)
    plot!(plt, t_forecast_partial, Predator_Data[n_train:end], label="Ground truth forecasting (predator)", linestyle=:solid, linewidth=0.5, color=:blue, legendfontsize=6)
    plot!(plt, t_forecast_partial, forecasted_sol[1, :], label="Neural ODE prediction + forecasting (prey)", linestyle=:dash, linewidth=4, color=:red, legendfontsize=6)
    plot!(plt, t_forecast_partial, forecasted_sol[2, :], label="Neural ODE prediction + forecasting (predator)", linestyle=:dash, linewidth=4, color=:blue, legendfontsize=6)
    
    # Plotting deviations as shaded areas
    plot!(plt, t_forecast_partial, Prey_Data[n_train:end], ribbon=forecast_deviation_prey, fillalpha=0.1, color=:red ,label=false)
    plot!(plt, t_forecast_partial, Predator_Data[n_train:end], ribbon=forecast_deviation_predator, fillalpha=0.1, color=:blue , label=false)
    
    # Displaying the final plot
    display(plt)
end

# Forecasting and finding Breakdown-Point with `train_and_forecast_with_ratio` for different training ratios:
#train_and_forecast_with_ratio(0.9)
#train_and_forecast_with_ratio(0.7)
#train_and_forecast_with_ratio(0.5)
train_and_forecast_with_ratio(0.35)
#train_and_forecast_with_ratio(0.3)
