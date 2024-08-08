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

print(sol)
# Extract time and data for training
t = sol.t
prey_data = sol[1, :]
predator_data = sol[2, :]

# Visualize the data
plot(t, prey_data, label="Prey (Rabbits)", xlabel="Time", ylabel="Population")
plot!(t, predator_data, label="Predators (Wolves)")