#==========================================================================================
Title: model_solution.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-26
Description: Functions for solving the labor market model.
==========================================================================================#
# TODO: Add a description of the module
module ModelSolution
using Base.Threads # To check nthreads()

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

    if verbose 
        println(@bold @blue "Solving worker value functions") 
        println(@bold @blue "Running with $(Threads.nthreads()) threads.")
    end

    start_time = time()  # Start timing
    # Create a function that evaluates the probability of finding a job
    p = θ -> eval_prob_job_find(matching_function, θ)
    # --- MPI Parameters ---
    max_iter_mpi = 200      
    tol_mpi = 1e-7          
    m = 10                  

    # --- Pre-allocate ---
    W_eval = similar(res.W) 
    U_eval = similar(res.U)
    W_new_eval = similar(res.W) 
    U_new_eval = similar(res.U)
    new_policy = similar(res.x_policy)

    for iter_mpi in 1:max_iter_mpi
        
        # Store old policy to check for convergence
        current_policy = copy(res.x_policy) 
        
        # --- 1. Policy Evaluation Step (m iterations of VFI with fixed policy) ---
        W_eval .= res.W # Start evaluation from previous iteration's values
        U_eval .= res.U

        for k in 1:m 
            # --- Update W_eval based on U_eval (Parallelized over i_ψ) ---
            Threads.@threads for i_ψ in 1:size(W_eval, 1) # Use 1:size instead of enumerate for @threads
                δ_slice = view(res.δ_grid, i_ψ, :, :) # Use view for efficiency
                W_eval_slice = view(W_eval, i_ψ, :, :)
                W_new_eval_slice = view(W_new_eval, i_ψ, :, :)

                # Each thread calculates its slice. Reads W_eval/U_eval, writes W_new_eval slice. Safe.
                W_new_eval_slice .= x_grid.values' .+ β .* ( (1 .- δ_slice) .* W_eval_slice .+ δ_slice .* U_eval ) 
            end # Implicit barrier: all threads finish W_new_eval before proceeding
            
            # --- Update U_eval based on W_new_eval (Parallelized over i_h, using FIXED policy) ---
            Threads.@threads for i_h in 1:size(U_eval, 1) # Use 1:size instead of enumerate for @threads
                # Get the action index from the CURRENT FIXED policy
                x_idx = current_policy[i_h] 
                
                # Calculate EW only for the specific action x_idx
                EW_slice_eval = view(W_new_eval, :, i_h, x_idx) 
                EW_at_policy = sum(EW_slice_eval .* ψ_grid.pdf) # Sum over ψ dimension

                # Calculate continuation value using the fixed policy action x_idx
                p_at_policy = p(θ[i_h, x_idx]) 
                continuation_value_at_policy = p_at_policy * EW_at_policy + (1 - p_at_policy) * U_eval[i_h] 
                
                # Each thread writes to its own U_new_eval[i_h]. Safe.
                U_new_eval[i_h] = b + β * continuation_value_at_policy
            end # Implicit barrier: all threads finish U_new_eval before proceeding

            # Simple update (could add convergence check within evaluation too)
            W_eval .= W_new_eval 
            U_eval .= U_new_eval
        end # End of k loop (inner evaluation iterations)

        # --- 2. Policy Improvement Step (Parallelized over i_h) ---
        Threads.@threads for i_h in 1:size(new_policy, 1)
            # Calculate expected value of W across all possible next actions 'x'
            EW_slice = view(W_eval, :, i_h, :) 
            EW_all_x = vec(sum(EW_slice .* ψ_grid.pdf, dims=1)) # Sum over ψ, result is vector over x

            # Calculate continuation values for all possible actions x
            continuation_values = p.( θ[i_h, :] ) .* EW_all_x .+ ( 1 .- p.( θ[i_h, :] ) ) .* U_eval[i_h]
            
            # Find the best action index (policy improvement)
            new_policy[i_h] = argmax(continuation_values) 
        end # Implicit barrier: all threads finish policy improvement before proceeding

        # --- Check for Convergence ---
        if new_policy == current_policy
            if verbose
                println(@bold @green "Policy converged after $iter_mpi iterations.")
            end
            res.W .= W_eval # Store final values
            res.U .= U_eval
            res.x_policy .= new_policy
            break
        end
        
        # Update policy for the next iteration
        res.x_policy .= new_policy
        res.W .= W_eval 
        res.U .= U_eval 

        if iter_mpi == max_iter_mpi
            println(@bold @green "MPI reached max iterations without policy convergence.")
        end

    end # End of iter_mpi loop

    if verbose
        elapsed_time = time() - start_time
        println(@bold @green "Worker value functions solved in $(elapsed_time) seconds.")
    end
    #> Potential Further Optimizations:
    # TODO: Pre-calculating p.(θ): If p and θ don't change, p_theta = p.(θ) and one_minus_p_theta = 1 .- p_theta could be computed once outside all loops.
    # TODO: Optimizing sum: Ensure the sum(EW_slice .* ψ_grid.pdf) is efficient. If ψ_grid.pdf can be represented correctly, matrix multiplication might be faster for the EW_all_x calculation (W_eval_reshaped * ψ_grid.pdf or similar), though this requires careful reshaping. Packages like LoopVectorization.jl (@turbo macro) could potentially speed up these loops further, but test carefully as it might conflict with @threads.
    # TODO: Allocation Profiling: After adding threads, run the profiler again (@profile) to see if any unexpected allocations are occurring within the threaded loops.
end
end 

