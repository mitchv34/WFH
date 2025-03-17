import Term: install_term_stacktrace
using Term
install_term_stacktrace()

# Include the modules
include("./model_functions.jl")
include("./types.jl")
include("./model_solution.jl")
include("./plots_search_model.jl")

using .Types
using .ModelFunctions
using .ModelSolution
using .ModelPlots

# Set paths
function main(; verbose::Bool=true)
    config_path = "src/structural/search_model/revamped/parameters.yaml"
    
    # Initialize model
    prim, res = initializeModel(config_path)
    
    # Solve the model with a single function call
    solve_model!(prim, res, n_iter_max=5000, tol=1e-6, verbose=verbose)
    
    return prim, res
end

prim, res = main(verbose=true);
model_plots = create_all_plots!("create", prim, res; verbose=true);

model_plots.job_finding_prob


using Parameters
@unpack β, x_grid, ψ_grid, h_grid, production_function, utility_function = prim
@unpack α_policy, w_policy = res
# TODO: Make a version of this function that iterates to find a fixed point
#! The current version of the model implies a closed form solution for the firm value function    


# Create evaluation functions for the production and wage functions this function takes index of 
# the worker skill and the value of the worker skill and returns the value of the production function
# and the wage function when the firm sets α to the optimal value
# Production function
Y = (i_h, i_ψ) -> evaluate(production_function, h_grid.values[i_h], α_policy[i_ψ, i_h],  ψ_grid.values[i_ψ])

Y_ = i_ψ -> Y(50, i_ψ)
plot(ψ_grid.values, Y_.(1:ψ_grid.n), label="Y(20, ψ)", xlabel="ψ", ylabel="Y(20, ψ)", title="Production function for h=20")


Y_.(1:ψ_grid.n) .- w_policy[:, 70, :]

h_grid.values[50]

for i_ψ in 1:ψ_grid.n
    for i_h in 1:h_grid.n
        Y_val = Y(i_h, i_ψ)
        δ_val = res.δ_grid[i_ψ, i_h, :]
        res.J[i_ψ, i_h, :] = (Y_val .- w_policy[i_ψ, i_h,:] ) ./ (1 .- β .* (1 .- δ_val))
    end
end

γ = prim.matching_function.γ
κ = prim.κ

i_h = 70
Ej = sum(res.J[:, i_h, :] .* prim.ψ_grid.pdf, dims=1)[:] 
θ_x = similar(prim.x_grid.values)
θ_x[Ej .> κ] = ((Ej[Ej .> κ] / κ).^γ .- 1).^(1/γ)

p = θ -> θ * (θ^(γ) + 1)^(-1/γ)

plot(p.(θ_x))