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
    result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.001), maxiters=200, callback = (p, l, pred) -> begin
        iteration += 1
        callback(p, l, pred, iteration=iteration, opt_method="Adam")
    end)
    
    # Train with BFGS
    optprob2 = Optimization.OptimizationProblem(optf, result_neuralode.minimizer)
    iteration = 0
    res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01), callback = (p, l, pred) -> begin
        iteration += 1
        callback(p, l, pred, iteration=iteration, opt_method="BFGS")
    end, maxiters=20)
    
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
        t_train_partial, y_train[1, 1:n_train], label="Ground truth training (prey)", xlabel="Time", ylabel="Population", 
        linestyle=:solid, linewidth=4, color=:red , legendfontsize=6
    )
    plot!(plt, t_train_partial, y_train[2, 1:n_train], label="Ground truth training (predator)", linestyle=:solid, linewidth=4, color=:blue, legendfontsize=6)
    plot!(plt, t_train[n_train:end], y_train[1, n_train:end], label="Ground truth forecasting (prey)", linestyle=:solid, linewidth=0.5, color=:red, legendfontsize=6)
    plot!(plt, t_train[n_train:end], y_train[2, n_train:end], label="Ground truth forecasting (predator)", linestyle=:solid, linewidth=0.5, color=:blue, legendfontsize=6)
    plot!(plt, t_train[n_train:end], forecasted_sol[1, n_train:end], label="Neural ODE prediction + forecasting (prey)", linestyle=:dash, linewidth=4, color=:red, legendfontsize=6)
    plot!(plt, t_train[n_train:end], forecasted_sol[2, n_train:end], label="Neural ODE prediction + forecasting (predator)", linestyle=:dash, linewidth=4, color=:blue, legendfontsize=6)
    
    # Plot deviations as shaded areas
    plot!(plt, t_train[n_train:end], y_train[1, n_train:end], ribbon=forecast_deviation_prey, fillalpha=0.1, color=:red ,label=false)
    plot!(plt, t_train[n_train:end], y_train[2, n_train:end], ribbon=forecast_deviation_predator, fillalpha=0.1, color=:blue , label=false)
    
    display(plt)
end

# Different training scenarios
train_and_forecast(0.9, y_train, t, dudt, u0, st)  # Train on 90%, forecast on 10%
train_and_forecast(0.75, y_train, t, dudt, u0, st)  # Train on 75%, forecast on 25%
train_and_forecast(0.5, y_train, t, dudt, u0, st)  # Train on 50%, forecast on 50%
train_and_forecast(0.4, y_train, t, dudt, u0, st)  # Train on 40%, forecast on 60%
train_and_forecast(0.35, y_train, t, dudt, u0, st)  # Train on 35%, forecast on 65%
train_and_forecast(0.3, y_train, t, dudt, u0, st)  # Train on 30%, forecast on 70%
train_and_forecast(0.1, y_train, t, dudt, u0, st)  # Train on 10%, forecast on 90%

################################################################################################
################################## Hyper Parameter Tuning ######################################
################################################################################################


using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimisers, OptimizationOptimJL, Random, Plots, ComponentArrays, Optim
#Loss Variation with different Activation Functions
# Function to define neural network with different activation functions
function create_neural_ode(hidden_units, activation_fn)
    dudt = Lux.Chain(
        Lux.Dense(2, hidden_units, activation_fn),
        Lux.Dense(hidden_units, hidden_units, activation_fn),
        Lux.Dense(hidden_units, hidden_units, activation_fn),
        Lux.Dense(hidden_units, 2)
    )
    return dudt
end

# Function to run training and collect loss data
function train_with_activation(train_fraction, activation_fn, u0, st, hidden_units, step_size)
    n_train = Int(floor(train_fraction * length(t)))
    y_train_partial = y_train[:, 1:n_train]
    t_train_partial = t[1:n_train]
    tspan_train = (t_train_partial[1], t_train_partial[end])
    saveat_train = t_train_partial

    # Initialize neural network with given activation function
    dudt = create_neural_ode(hidden_units, activation_fn)
    nn_params, st = Lux.setup(rng, dudt)
    
    # Initial parameters for optimization
    pinit = ComponentArray(nn_params)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x, y_train_partial, u0, tspan_train, saveat_train, st), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)
    
    # Optimization setup with Adam optimizer
    loss_history = []
    iteration = 0
    result = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), maxiters=200, callback = (p, l, pred) -> begin
        push!(loss_history, l)
        iteration += 1
        false
    end)
    
    return loss_history
end

# Activation functions
activations = Dict("Sigmoid" => σ, "ReLU" => relu, "Tanh" => tanh, "RBF" => rbf)

# Store loss data
loss_data_activations = Dict()

# Run training for each activation function and collect loss data
for (name, act_fn) in activations
    loss_data_activations[name] = train_with_activation(0.9, act_fn, u0, st, 100, 0.001)
end

# Plotting loss variation with different activation functions
plt = plot()
for (name, loss_history) in loss_data_activations
    plot!(plt, 1:length(loss_history), loss_history, label=name)
end
xlabel!("Iteration")
ylabel!("Loss")
title!("Loss Variation with Different Activation Functions")
display(plt)


## Loss Variation with Different Adam Step Sizes

# Function to run training and collect loss data for step sizes
function train_with_step_size(train_fraction, hidden_units, activation_fn, u0, st, step_size)
    n_train = Int(floor(train_fraction * length(t)))
    y_train_partial = y_train[:, 1:n_train]
    t_train_partial = t[1:n_train]
    tspan_train = (t_train_partial[1], t_train_partial[end])
    saveat_train = t_train_partial

    # Initialize neural network with given number of hidden units
    dudt = create_neural_ode(hidden_units, activation_fn)
    nn_params, st = Lux.setup(rng, dudt)
    
    # Initial parameters for optimization
    pinit = ComponentArray(nn_params)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x, y_train_partial, u0, tspan_train, saveat_train, st), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)
    
    # Optimization setup with Adam optimizer
    loss_history = []
    iteration = 0
    result = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), maxiters=200, callback = (p, l, pred) -> begin
        push!(loss_history, l)
        iteration += 1
        false
    end)
    
    return loss_history
end

# Step sizes
step_sizes = [0.1, 0.01, 0.001]

# Store loss data
loss_data_step_sizes = Dict()

# Run training for each step size and collect loss data
for step_size in step_sizes
    loss_data_step_sizes[string(step_size)] = train_with_step_size(0.9, 100, rbf, u0, st, step_size)
end

# Plotting loss variation with different step sizes
plt = plot()
for (name, loss_history) in loss_data_step_sizes
    plot!(plt, 1:length(loss_history), loss_history, label=name)
end
xlabel!("Iteration")
ylabel!("Loss")
title!("Loss Variation with Different Adam Step Sizes")
display(plt)


# Loss Variation with Different Number of Hidden units
# Function to run training and collect loss data for hidden units
function train_with_hidden_units(train_fraction, hidden_units, activation_fn, u0, st, step_size)
    n_train = Int(floor(train_fraction * length(t)))
    y_train_partial = y_train[:, 1:n_train]
    t_train_partial = t[1:n_train]
    tspan_train = (t_train_partial[1], t_train_partial[end])
    saveat_train = t_train_partial

    # Initialize neural network with given number of hidden units
    dudt = create_neural_ode(hidden_units, activation_fn)
    nn_params, st = Lux.setup(rng, dudt)
    
    # Initial parameters for optimization
    pinit = ComponentArray(nn_params)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x, y_train_partial, u0, tspan_train, saveat_train, st), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)
    
    # Optimization setup with Adam optimizer
    loss_history = []
    iteration = 0
    result = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), maxiters=200, callback = (p, l, pred) -> begin
        push!(loss_history, l)
        iteration += 1
        false
    end)
    
    return loss_history
end

# Hidden units
hidden_units_list = [5, 10, 50, 100]

# Store loss data
loss_data_hidden_units = Dict()

# Run training for each number of hidden units and collect loss data
for hidden_units in hidden_units_list
    loss_data_hidden_units[string(hidden_units)] = train_with_hidden_units(0.9, hidden_units, rbf, u0, st, 0.001)
end

# Plotting loss variation with different hidden units
plt = plot()
for (name, loss_history) in loss_data_hidden_units
    plot!(plt, 1:length(loss_history), loss_history, label=name)
end
xlabel!("Iteration")
ylabel!("Loss")
title!("Loss Variation with Different Number of Hidden Units")
display(plt)
