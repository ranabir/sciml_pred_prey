using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, OptimizationOptimisers
using ComponentArrays, OptimizationOptimisers

# Generate data using the Lotka-Volterra Predator-Prey Model

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

# Solve ODE problem to generate data
prob = ODEProblem(lotka_volterra!, u0, tspan)
sol = Array(solve(prob, Tsit5(), u0=u0, saveat=t))

# Visualize generated data
plot(t, sol[1,:], label="Prey (Generated)", xlabel="Time", ylabel="Population")
plot!(t, sol[2,:], label="Predators (Generated)")

Prey_Data = Array(sol)[1, :]
Predator_Data = Array(sol)[2, :]

# Construction of UDE
# Define neural networks to model interactions in the Lotka-Volterra system
rng = Random.default_rng()
Random.seed!(1028)

# Define RBF function
function rbf(x)
    return exp.(-x.^2)
end

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
function predict_ude(θ)
    sol_ude = Array(solve(prob_ude, Tsit5(), p=θ, saveat=t,
                          sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
    return sol_ude
end

# Define loss function based on available Prey and Predator data
function loss_ude(θ)
    sol_pred = predict_ude(θ)
    if size(sol_pred, 2) != length(Prey_Data)
        error("Dimension mismatch: Predictions have size ", size(sol_pred), " but expected size is (2, ", length(Prey_Data), ")")
    end
    loss = sum(abs2, Prey_Data .- sol_pred[1, :]) + sum(abs2, Predator_Data .- sol_pred[2, :])
    return loss
end

# Define callback function to monitor progress and plot during optimization
function callback(θ, l, pred; iteration, doplot=true, opt_method="AdamW")
    println("Iteration: ", iteration, ", Optimization Method: ", opt_method, ", Loss: ", l)

    previous_loss[] = l  # Update the previous loss to the current one

    if iteration % 10 == 0 
        if doplot
            # Clear and replot at each iteration
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

# First optimization using Adam
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_ude(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p0_vec)

# First optimization run using Adam optimizer with adjusted learning rate
iteration = Ref(0)
result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.001), maxiters=20000, allow_f_increases = false, callback = (p, l) -> begin
    iteration[] += 1
    callback(p, l, predict_ude(p), iteration=iteration[], opt_method="Adam")
end)

#RMSProp
optprob2 = Optimization.OptimizationProblem(optf, ComponentArray(result_neuralode.minimizer))

# Define RMSProp optimizer with the specified parameters
η = 0.001   # Learning rate
ρ = 0.9     # Momentum
ϵ = 1e-8    # Epsilon to avoid division by zero
centred = false  # Whether to use centered RMSProp

# Define optimization problem using RMSProp
res_rmsprop = Optimization.solve(optprob2, OptimizationOptimisers.RMSProp(η, ρ, ϵ; centred=centred), callback = (p, l) -> begin
    iteration[] += 1
    callback(p, l, predict_ude(p), iteration=iteration[], opt_method="RMSProp")
end, maxiters=5000)

# Forecasting with the trained UDE model
t_forecast = range(0.0, 20.0, length=201)
prob_forecast = ODEProblem{true}(ude_system!, u0, (0.0, 20.0))
sol_forecast = Array(solve(prob_forecast, Tsit5(), p=res_rmsprop.minimizer, saveat=t_forecast))

# Visualizing the final predictions
plot(legend=:topleft)
plot!(t, Prey_Data, label="Prey Data", color=:blue)
plot!(t, Predator_Data, label="Predator Data", color=:red)
plot!(t_forecast, sol_forecast[1, :], label="Prey (Predicted)", linestyle=:dash, color=:blue)
plot!(t_forecast, sol_forecast[2, :], label="Predator (Predicted)", linestyle=:dash, color=:red)
