using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimisers, OptimizationOptimJL, Random, Plots, ComponentArrays, Optim

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
y_train = Array(sol)  # Convert solution to array format

# Define RBF function
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

# Prediction function using neural ODE
function predict_neuralode(p, u0, tspan_train, saveat_train, st)
    n_ode_train = NeuralODE(dudt, tspan_train, Tsit5(), saveat=saveat_train)
    sol, _ = n_ode_train(u0, p, st)
    return Array(sol)
end

# Loss function comparing predictions to training data
function loss_neuralode(ps, y_train_partial, u0, tspan_train, saveat_train, st)
    pred = predict_neuralode(ps, u0, tspan_train, saveat_train, st)
    loss = sum(abs2, y_train_partial .- pred)
    return loss, pred
end

# Callback function for optimization process
function callback(p, l, pred; doplot = true, iteration = 0, opt_method = "")
    println("Iteration: ", iteration, ", Optimization Method: ", opt_method, ", Loss: ", l)
    if doplot
        plt = plot(t_train, y_train[1, :], label="Prey Data", linestyle=:solid, linewidth=4)
        scatter!(plt, t_train, pred[1, :], label="Prey Prediction")
        plot!(plt, t_train, y_train[2, :], label="Predator Data", linestyle=:solid, linewidth=4)
        scatter!(plt, t_train, pred[2, :], label="Predator Prediction")
        display(plot(plt))
    end
    return false
end

# Function to train and forecast
function train_and_forecast(train_fraction, y_train, t_train, dudt, u0, st)
    n_train = Int(floor(train_fraction * length(t_train)))
    y_train_partial = y_train[:, 1:n_train]
    t_train_partial = t_train[1:n_train]
    tspan_train = (t_train_partial[1], t_train_partial[end])
    saveat_train = t_train_partial

    # Neural ODE with limited training data
    n_ode_partial = NeuralODE(dudt, tspan_train, Tsit5(), saveat=saveat_train)

    # Initial parameters for optimization
    pinit = ComponentArray(nn_params)
    
    # Optimization setup with Adam optimizer
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x, y_train_partial, u0, tspan_train, saveat_train, st), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)
    
    # First optimization run using Adam optimizer
    iteration = 0
    result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.001), maxiters=500, callback = (p, l, pred) -> begin
        iteration += 1
        callback(p, l, pred, iteration=iteration, opt_method="Adam")
    end)
    
    # Train with BFGS
    optprob2 = Optimization.OptimizationProblem(optf, result_neuralode.minimizer)
    iteration = 0
    res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01), callback = (p, l, pred) -> begin
        iteration += 1
        callback(p, l, pred, iteration=iteration, opt_method="BFGS")
    end, maxiters=50)
    
    # Forecasting with the trained neural ODE model
    tspan_forecast = (t_train_partial[1], t_train[end])
    n_ode_forecast = NeuralODE(dudt, tspan_forecast, Tsit5(), saveat=t_train)
    forecasted_sol, _ = n_ode_forecast(u0, res2.u, st)
    forecasted_sol = Array(forecasted_sol)

    # Calculate deviations for the forecast period
    forecast_deviation_prey = abs.(y_train[1, n_train:end] .- forecasted_sol[1, n_train:end])
    forecast_deviation_predator = abs.(y_train[2, n_train:end] .- forecasted_sol[2, n_train:end])

    # Plot the training and forecasting results with deviation
    plt = plot(
        t_train_partial, y_train[1, 1:n_train], label="Prey (Training)", xlabel="Time", ylabel="Population", 
        linestyle=:solid, linewidth=4, color=:blue
    )
    plot!(plt, t_train_partial, y_train[2, 1:n_train], label="Predators (Training)", linestyle=:solid, linewidth=4, color=:red)
    plot!(plt, t_train[n_train:end], forecasted_sol[1, n_train:end], label="Prey (Forecasted)", linestyle=:dash, linewidth=4, color=:blue)
    plot!(plt, t_train[n_train:end], forecasted_sol[2, n_train:end], label="Predators (Forecasted)", linestyle=:dash, linewidth=4, color=:red)
    
    # Plot deviations as shaded areas
    plot!(plt, t_train[n_train:end], y_train[1, n_train:end], ribbon=forecast_deviation_prey, fillalpha=0.3, label="Prey Deviation", color=:blue)
    plot!(plt, t_train[n_train:end], y_train[2, n_train:end], ribbon=forecast_deviation_predator, fillalpha=0.3, label="Predator Deviation", color=:red)
    
    display(plt)
end

# Different training scenarios
#train_and_forecast(0.9, y_train, t, dudt, u0, st)  # Train on 90%, forecast on 10%
train_and_forecast(0.75, y_train, t, dudt, u0, st)  # Train on 75%, forecast on 25%
train_and_forecast(0.5, y_train, t, dudt, u0, st)  # Train on 50%, forecast on 50%
train_and_forecast(0.3, y_train, t, dudt, u0, st)  # Train on 30%, forecast on 70%
train_and_forecast(0.1, y_train, t, dudt, u0, st)  # Train on 30%, forecast on 70%
