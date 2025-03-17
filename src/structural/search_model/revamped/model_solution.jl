#==========================================================================================
Title: model_solution.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-26
Description: Functions for solving the labor market model.
==========================================================================================#
# TODO: Add a description of the module
module ModelSolution

using Parameters, LinearAlgebra, Term, Term.Progress
using ..Types, ..ModelFunctions

export solve_model!, iterateValueMatch, solveSubMarketTightness!, iterateW_U

#==========================================================================================
# Main solution function that coordinates all solution steps
==========================================================================================#
function solve_model!(prim::Primitives, res::Results; 
                    n_iter_max::Int64=5000, 
                    tol::Float64=1e-6,
                    verbose::Bool=true)
    
    if verbose println(@bold @blue "Starting model solution...") end

    # 1. Solve firm value function
    solveValueFirm!(prim, res, n_iter_max=n_iter_max, tol=tol, verbose=verbose)
    
    # 2. Compute market tightness
    solveSubMarketTightness!(prim, res, verbose=verbose)
    
    # 3. Solve worker value functions
    solveValueWorker!(prim, res, n_iter_max=n_iter_max, tol=tol, verbose=verbose)
    
    if verbose println(@bold @green "Model solution completed!") end
end
#==========================================================================================
# Step 1: Solve firm value function
==========================================================================================#
function solveValueFirm!(prim::Primitives, res::Results; n_iter_max::Int64=5000, tol::Float64=1e-6, verbose::Bool=true)
    @unpack β, x_grid, ψ_grid, h_grid, production_function, utility_function = prim
    @unpack α_policy, w_policy = res
    # TODO: Make a version of this function that iterates to find a fixed point
    #! The current version of the model implies a closed form solution for the firm value function    
    
    if verbose println(@bold @blue "Solving the firm value function J(ψ, h, x)") end
    
    # Create evaluation functions for the production and wage functions this function takes index of 
    # the worker skill and the value of the worker skill and returns the value of the production function
    # and the wage function when the firm sets α to the optimal value
    # Production function
    Y = (i_h, i_ψ) -> evaluate(production_function, h_grid.values[i_h], α_policy[i_ψ, i_h],  ψ_grid.values[i_ψ])

    for i_ψ in 1:ψ_grid.n
        for i_h in 1:h_grid.n
            Y_val = Y(i_h, i_ψ)
            δ_val = res.δ_grid[i_ψ, i_h, :]
            res.J[i_ψ, i_h, :] = (Y_val .- w_policy[i_ψ, i_h,:] ) ./ (1 .- β .* (1 .- δ_val))
        end
    end
end

#==========================================================================================
# Step 2: Compute market tightness
==========================================================================================#
function solveSubMarketTightness!(prim::Primitives, res::Results; verbose::Bool=true)
    @unpack κ = prim
    
    if verbose println(@bold @blue "Calculating market tightness θ(h,x)") end
    
    for h_ind in 1:prim.h_grid.n
        for x_ind in 1:prim.x_grid.n
            J_slice = res.J[:, h_ind, x_ind]
            Ej = dot(J_slice, prim.ψ_grid.pdf)
            res.θ[h_ind, x_ind] = invert_vacancy_fill(prim.matching_function, κ, Ej)
        end
    end
end

#==========================================================================================
# Step 3: Solve worker value functions
==========================================================================================#
function solveValueWorker!(prim::Primitives, res::Results; n_iter_max::Int64=5000, tol::Float64=1e-6, verbose::Bool=true)
    @unpack β, x_grid, ψ_grid, h_grid, b, matching_function = prim
    @unpack θ = res

    if verbose println(@bold @blue "Solving worker value functions") end
    
    err = Inf
    n_iter = 0
    
    W_new = similar(res.W)
    U_old = copy(res.U)
    
    # Create a function that evaluates the probability of finding a job
    p = θ -> eval_prob_job_find(matching_function, θ)

    while err > tol && n_iter < n_iter_max
        # Update employed value function
        for (i_ψ, ψ) in enumerate(ψ_grid.values)
            δ = res.δ_grid[i_ψ, :, :]
            W_new[i_ψ, :, :] = (x_grid.values' .+ β .* δ .* U_old ) ./ (1 .- β .* δ)
        end
        
        # Compute expected worker value and update unemployed value
        for (i_h, h) in enumerate(h_grid.values)
            EW = dropdims( sum(W_new[:, i_h, :] .* ψ_grid.pdf, dims=1), dims=1)
            continuation_values = p.( θ[i_h, :] ) .* EW .+ (1 .- p.( θ[i_h, :] ) ) .* U_old[i_h]
            res.U[i_h] = b + β * maximum(continuation_values)
            res.x_policy[i_h] = argmax(continuation_values)
        end
        err = max(norm(W_new - res.W, Inf), norm(res.U - U_old, Inf))
        res.W .= W_new
        U_old .= res.U
        n_iter += 1

        if n_iter % 100 == 0 && verbose
            println(@bold @yellow "Iteration: $n_iter, Error: $(round(err, digits=6))")
        elseif err < tol && verbose
            println(@bold @green "Iteration: $n_iter, Converged!")
        end
    end

    if n_iter >= n_iter_max && verbose
        println(@bold @red "Worker value functions did not converge within $n_iter_max iterations.")
    end
end #

end# module ModelSolution