using Term
using LinearAlgebra
using Interpolations
include("types.jl")

"""
Solve firm's value function
"""
function solve_firm_value!(prim::Primitives, res::Results; 
                        max_iter::Int64=5000, tol::Float64=1e-6)
    println(@bold @blue "Solving firm value function J(ψ,x)")
    
    # Update policies before iteration
    compute_optimal_α!(prim, res)
    
    converged = false
    n_iter = 0
    
    while !converged && n_iter < max_iter
        next_J = update_firm_value(prim, res)
        
        # Check convergence
        err = maximum(abs.(next_J .- res.J))
        converged = err < tol
        
        # Update and report progress
        res.J = copy(next_J)
        report_progress(n_iter, err, converged)
        
        n_iter += 1
    end
end

"""
Solve worker's value functions
"""
function solve_worker_value!(prim::Primitives, res::Results;
                        max_iter::Int64=5000, tol::Float64=1e-6)
    println(@bold @blue "Solving worker value functions")
    
    # Initialize interpolation and temporary storage
    U_interp = setup_interpolation(prim)
    temp_storage = initialize_temp_storage(prim)
    
    converged = false
    n_iter = 0
    
    while !converged && n_iter < max_iter
        # Update employed and unemployed values
        W_new, U_new = update_worker_values(prim, res, U_interp, temp_storage)
        
        # Check convergence
        err = check_convergence(W_new, U_new, res)
        converged = err < tol
        
        # Update values and report progress
        update_results!(res, W_new, U_new)
        report_progress(n_iter, err, converged)
        
        n_iter += 1
    end
end