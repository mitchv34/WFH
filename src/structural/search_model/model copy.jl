using Parameters
using LinearAlgebra
using YAML
using Interpolations
using Term
using CSV
using Tables

include("generate_distribution_psi.jl")

@with_kw struct Primitives   
    # Parameters
    β      ::Float64                # discount factor
    γ      ::Float64                # job finding rate parameter
    κ      ::Float64                # vacancy posting cost
    δ_bar  ::Float64                # job destruction rate
    σ      ::Float64                # risk aversion (workers)
    A::Float64                      # Total factor productivity 
    ψ₀::Float64                     # Minimum efficiency threshold
    c::Float64                      # Disutility scaling factor
    χ::Float64                      # Remote work preference 
    φ::Float64                      # Fraction of current wage transferred upon separation
    b_prob ::Float64                # probability of losing unemployment benefits
    
    # Grids
    ## Remote work productivity grid
    n_ψ::Int64                      # Number of ψ grid points
    ψ_min::Float64                  # minimum ψ
    ψ_max::Float64                  # maximum ψ
    ψ_grid::Array{Float64}          # grid of ψ values
    ψ_pdf ::Array{Float64}          # pdf of ψ values
    ψ_cdf ::Array{Float64}          # cdf of ψ values
    
    ## Utility grid
    n_x::Int64                      # number of x grid points
    x_min::Float64                  # minimum x
    x_max::Float64                  # maximum x
    x_grid::Vector{Float64}         # grid of x values

    b_grid ::Vector{Float64}        # unemployment benefits grid

    # Functions 
    p::Function = θ -> θ*(1 + θ^γ)^(-1/γ)
    Y::Function = (α, ψ) -> A * ((1 - α) + α * (ψ - ψ₀))  # Production [NEW]
    u::Function
end # struct Primitives

function getPrimitives(parameters_path::String, psi_distribution_data_path::String)
    parameters = YAML.load_file(parameters_path)["Primitives"]
    
    # Parameters
    β         = parameters["beta"]
    γ         = parameters["gamma"]
    κ         = parameters["kappa"]
    δ_bar     = parameters["delta_bar"]
    σ         = parameters["sigma"]
    A         = parameters["A"]
    
    χ         = parameters["chi"]
    c         = parameters["c"]
    φ         = parameters["phi"]
    b_prob    = parameters["b_prob"]
    # Grid parameters
    n_ψ       = parameters["n_psi"]
    ψ_min     = parameters["psi_min"] # minimum ψ for normalizing the grid
    ψ_max     = parameters["psi_max"] # maximum ψ for normalizing the grid

    # Read from the CSV file necessary data for obtaining the distribution of ψ
    columns = Tables.columns(CSV.File(psi_distribution_data_path; header=false))
    # Access columns by index (e.g., column1, column2)
    ψ_vals = parse.(Float64, columns.Column4[2:end])
    weights = parse.(Float64, columns.Column5[2:end])
    ψ_grid, ψ_pdf, ψ_cdf = fit_kde_psi(
                                        ψ_vals,
                                        weights; 
                                        num_grid_points=n_ψ, 
                                        engine = "python",
                                        boundary = false
                                    ) 
    # Estimate the value of ψ₀ consistent with the distribution and the parameters
    α_data = parameters["alpha_data"]
    ψ₀ = estimate_ψ₀(ψ_grid, ψ_pdf, A, c, χ, α_data)
    
    # Rescale ψ₀ to the new grid
    ψ₀  = ψ_min .+ ( (ψ_max - ψ_min) .* ( ψ₀ .- minimum(ψ_grid) ) ./ ( maximum(ψ_grid) .- minimum(ψ_grid) )   )
    # Normalize the grid so that it ranges between ψ_min and ψ_max
    ψ_grid    = ψ_min .+ ( (ψ_max - ψ_min) .* ( ψ_grid .- minimum(ψ_grid) ) ./ ( maximum(ψ_grid) .- minimum(ψ_grid) )   )


    n_x       = parameters["n_x"]
    x_min     = parameters["x_min"]
    x_max     = parameters["x_max"]
    x_grid    = range(x_min, x_max, length=n_x)
    
    # b_grid is a range of possible unemployment benefits 
    b_grid    = range(0.0, φ, length=n_x)

    # Functions
    p  = (θ) -> θ .* (1 .+ θ.^γ).^(-1 ./ γ) # Job finding rate
    Y  = (α, ψ) -> A .* ((1 .- α) .+ α .* (ψ .- ψ₀))  # Production function
    
    u  = (w, α) -> w .+ α.^χ  # Utility function

    # Return the updated Primitives struct

    return Primitives(β, γ, κ, δ_bar, σ, A, ψ₀, χ, c, φ, b_prob, n_ψ, ψ_min, ψ_max, ψ_grid, ψ_pdf, ψ_cdf, 
                        n_x, x_min, x_max, x_grid, b_grid, p, Y, u)
end # function getPrimitives

mutable struct Results
    J           ::Array{Float64,2}      # Value function J(ψ, x)
    θ           ::Array{Float64,1}      # Market tightness θ(x)
    α_policy    ::Array{Float64}        # Optimal remote work fraction α(ψ)
    w_policy    ::Array{Float64,2}      # Optimal wage w(ψ, x) = x + c(1 - α(ψ))^χ
    δ_grid      ::Array{Float64, 2}     # job destruction grid
    W           ::Array{Float64, 2}        # Worker value employed W(ψ, x)
    U           ::Array{Float64}        # Unemployed value U(x(b,1))
    x_policy    ::Array{Float64}        # Optimal utility search
    ψ_bottom    ::Float64               # Threshold for hybrid work
    ψ_top       ::Float64               # Threshold for full-time remote work
end # struct Results

function initializeModel(parameters_path::String, psi_distribution_data_path::String)
    prim = getPrimitives(parameters_path, psi_distribution_data_path)
    
    J = zeros(prim.n_ψ, prim.n_x)
    θ = zeros(prim.n_x)
    α_policy = zeros(prim.n_ψ)
    w_policy = zeros(prim.n_ψ, prim.n_x)
    δ_grid = prim.δ_bar .* ones(prim.n_ψ, prim.n_x)
    W = zeros(prim.n_ψ, prim.n_x)
    U = zeros(prim.n_x)
    x_policy = zeros(prim.n_x)
    # Compute critical thresholds
    # ψ_bottom is the threshold below which WFH is not feasible
    ψ_bottom = prim.ψ₀ + (1 - (prim.c * prim.χ ) / prim.A)
    # ψ_top is the threshold above which full-time WFH is optimal
    ψ_top = prim.ψ₀ + 1
    
    return prim, Results(J, θ, α_policy, w_policy, δ_grid, W, U, x_policy, ψ_bottom, ψ_top)
end # function initializeModel

# Optimal WFH function (α*(ψ))
function optimal_wfh(ψ, A, c, χ, ψ₀, γ) 
    # Compute critical thresholds
    ψ_bottom = ψ₀ + (1 - (c * χ ) / A)^(1/γ)
    ψ_top = ψ₀ + 1
    # Compute optimal WFH
    if ψ <= ψ_bottom
        return 0
    elseif ψ > ψ_top
        return 1
    else
        return 1 - ( (c * χ )/(A * (1 - (ψ - ψ₀)^γ )) )^(1/(1-χ))
    end # if
end # function optimal_wfh


function compute_optimal_α!(prim::Primitives, res::Results)
    @unpack χ,c, ψ_grid, ψ₀, A, n_x, n_ψ, x_grid, δ_bar, Y = prim
    @unpack ψ_bottom, ψ_top = res
    
    α_pol = zeros(n_ψ)
    w_pol = zeros(n_ψ, n_x)
    δ_grid = δ_bar .* ones(n_ψ, n_x)
    
    γ = 1 # TODO: I have to remove this from the theorethical model

    for (i, ψ) in enumerate(ψ_grid) # Loop over productivity grid
        if ψ <= ψ_bottom
            α_pol[i] =  0.0
        elseif ψ > ψ_top
            α_pol[i] = 1.0
        else
            α_pol[i] = 1 - ( (c * χ )/(A * (1 - (ψ - ψ₀)^γ )) )^(1/(1-χ))
        end # if
        # @show ψ, ψ_bottom
        # @show α_pol[i]
        for (j, x) in enumerate(x_grid)
            # Compute the optimal wage
            w = x +  c .* ( 1 - α_pol[i])^χ
            # @show w
            # Compute firm's profit
            # Pi =  Y(α_pol[i], ψ) - w
            # if firm profit or wages are negative, this job is not feasible
            # if (Pi < 0) || (w < 0)
            #     δ_grid[i, j] = 1.0
            # end # if
            w_pol[i, j] = w
        end # Loop over utility grid
    end # Loop over productivity grid
    # Update the α_policy in the Results struct
    res.α_policy = copy(α_pol)
    # Update the w_policy in the Results struct
    res.w_policy = copy(w_pol)
    # Update the job destruction grid
    res.δ_grid = copy(δ_grid)    
end # function compute_optimal_α!

function iterateValueMatch(prim::Primitives, res::Results; n_iter_max::Int64=5000, tol::Float64=1e-6)
	@unpack β, x_grid, ψ_grid, n_ψ, n_x, Y = prim
    @unpack δ_grid = res

	println(@bold @blue "Solving the firm value function J(ψ,x)")
	err   = Inf
	n_iter= 0
    # Update the optimal α and w policies and δ_grid
    compute_optimal_α!(prim, res)

	while err > tol && n_iter < n_iter_max
		next_J = zeros(n_ψ, n_x)
        for (i_ψ, ψ) in enumerate(ψ_grid)
            for (i_x, x) in enumerate(x_grid)
                δ = δ_grid[i_ψ, i_x]
                # Get the optimal wage for the current (ψ, x)
                w = res.w_policy[i_ψ, i_x]
                # Update firm value
                α = res.α_policy[i_ψ]
                next_J[i_ψ, i_x] = Y(α, ψ) - w + β*(1 - δ)*res.J[i_ψ, i_x]
            end # for i_x, x in enumerate(x_grid)
        end # for i_ψ, ψ in enumerate(ψ_grid)
		
		err  = maximum(abs.(next_J .- res.J))
		n_iter += 1
		
		if n_iter % 100 == 0
			println(@bold @yellow "Iteration: $n_iter, Error: $(round(err, digits=6))")
		elseif err < tol 
			println(@bold @green "Iteration: $n_iter, Converged!")
		end # if
		
		res.J = copy(next_J)
	end # while
end # function iterateValueMatch

# @unpack β, x_grid, ψ_grid, n_ψ, n_x, Y = prim
# @unpack δ_grid = res

# next_J = zeros(n_ψ, n_x)
# for (i_ψ, ψ) in enumerate(ψ_grid)
# i_ψ = 1
# ψ = ψ_grid[i_ψ]
# δ = δ_grid[i_ψ, :]
# w = res.w_policy[i_ψ, :]
# α = res.α_policy[i_ψ]
# (Y(α, ψ) .- w) ./  (1 .- β .* (1 .- prim.δ_bar))
#     for (i_x, x) in enumerate(x_grid)
#         δ = δ_grid[i_ψ, i_x]
#         # Get the optimal wage for the current (ψ, x)
#         w = res.w_policy[i_ψ, i_x]
#         # Update firm value
#         α = res.α_policy[i_ψ]
#         next_J[i_ψ, i_x] = (Y(α, ψ) - w) / (1 - β*(1 - δ))
#     end # for i_x, x in enumerate(x_grid)
# end # for i_ψ, ψ in enumerate(ψ_grid)


function solveSubMarketTightness!(prim::Primitives, res::Results)
    @unpack κ, γ, n_ψ, n_x, ψ_pdf = prim
    @unpack J = res

    println(@bold @blue "Calculating market tightness θ(x)")

    for i_x in 1:n_x # Loop over utility grid
        # Compute the expected value of J(ψ, x) for each x
        j = J[:, i_x]
        Ej = j' * ψ_pdf
        # If the expected value of J(ψ, x) is greater than κ, set θ(x) to the value that solves the equation
        if Ej > κ
            res.θ[i_x] = ((Ej/κ)^γ - 1)^(1/γ)
        else # Otherwise, set θ(x) to 0 (market is not active)
            res.θ[i_x] = 0.0
        end
    end # Loop over utility grid
end # function solveSubMarketTightness!

# function iterateW_U(prim::Primitives, res::Results; n_iter_max::Int64=5000, tol::Float64=1e-6)
    @unpack β, x_grid, ψ_grid, ψ_pdf, p, u, b_grid, b_prob, φ = prim
    # Parameters for worker-side transitions:
    b_decay = 1 - b_prob       # Unemployment benefit decay factor (0 < b_decay < 1)
    println(@bold @blue "Solving the worker value functions")
    # Initialize employed and unemployed value functions
    W_new = similar(res.W)         # Expected employed value on x_grid (vector length n_x)
    U_old = copy(res.U)            # Unemployed value function on b_grid (vector length n_x)
    U_new = similar(U_old)         # Unemployed value function on b_grid (vector length n_x)
    EW = zeros(length(x_grid))     # Temporary storage for expected employed value on x_grid


    # Temporary storage for employed worker values for each (ψ, x)
    W_employed = zeros(length(ψ_grid), length(x_grid))
    
    err = Inf
    n_iter = 0

    # while err > tol && n_iter < n_iter_max
        # --- 1. Update Employed Worker Value ---
        # For each firm type ψ and submarket x compute:
        # W(x,ψ) = [ x + β δ(x,ψ) U(φ · w(x,ψ)) ] / [1 - β (1 - δ(x,ψ))]
        # Here, res.δ_grid and res.w_policy are defined on the (ψ,x)-grid.
        # We use interpolation on U_old to get U(φ·w) at any value.
        U_interp = LinearInterpolation(collect(b_grid), U_old, extrapolation_bc=Line())
        for (i, ψ) in enumerate(ψ_grid)
        # i = 1
        # ψ = ψ_grid[i]
            for (j, x) in enumerate(x_grid)
                δ_val = res.δ_grid[i, j]
                # δ_val = res.δ_grid[i, :]
                w_val = res.w_policy[i, j]
                # w_val = res.w_policy[i, :]
                U_unemp = U_interp(φ * w_val)
                W_new[i, j] = ( x + β * δ_val * U_unemp ) / ( 1 - β * (1 - δ_val) )
            end
        end

        # Average over ψ (using the density ψ_pdf) to get the expected employed value in submarket x:
        for (j, x) in enumerate(x_grid)
            EW[j] = sum( W_new[:, j] .* ψ_pdf )
        end

        # --- 2. Update Unemployed Worker Value ---
        # For each unemployment benefit level b ∈ b_grid, the worker chooses the best submarket:
        # V(x, b) = p(θ(x)) * W(x) + (1 - p(θ(x))) * U(b_decay · b)
        # Then: U(b) = u(b,1) + β * max_{x in x_grid} V(x, b)
        U_interp = LinearInterpolation(collect(b_grid), U_old, extrapolation_bc=Line())
        for (i, b) in enumerate(b_grid)
            continuation_values = similar(x_grid)
            for (j, x) in enumerate(x_grid)
                continuation_values[j] = p( res.θ[j] ) * EW[j] +
                                           (1 - p( res.θ[j] )) * U_interp( b_decay * b )
            end
            # Update the unemployed value function
            U_new[i] = u(b, 1) + β * maximum(continuation_values)
            # Update the optimal submarket for each b
            res.x_policy[i] = x_grid[argmax(continuation_values)]
        end

        plot( p.( res.θ ) .* EW .+ (1 .- p( res.θ )) .* U_interp( b_decay * b_grid[1] ))
        plot!( p.( res.θ ) .* EW .+ (1 .- p( res.θ )) .* U_interp( b_decay * b_grid[20] ))
        plot!( p.( res.θ ) .* EW .+ (1 .- p( res.θ )) .* U_interp( b_decay * b_grid[50] ))
        plot!( p.( res.θ ) .* EW .+ (1 .- p( res.θ )) .* U_interp( b_decay * b_grid[100] ))
        W_new[:, 1]
        
        # --- 3. Check Convergence and Update ---
        err_W = norm(W_new - res.W, Inf)
        err_U = norm(U_new - U_old, Inf)
        err = max(err_W, err_U)
        
        res.W .= W_new
        res.U .= U_new
        U_old = copy(U_new)
        
        n_iter += 1
        
        if n_iter % 100 == 0
            println(@bold @yellow "Iteration: $n_iter, Error: $(round(err, digits=6))")
        elseif err < tol
            println(@bold @green "Iteration: $n_iter, Converged!")
        end

    end

    if n_iter > n_iter_max
        println(@bold @red "Worker value functions did not converge within $n_iter_max iterations.")
    end
end
