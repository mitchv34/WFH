using Parameters
using LinearAlgebra
using YAML
using Interpolations
using Term

@with_kw struct Primitives   
    # Parameters
    β      ::Float64 = 0.996        # discount factor
    γ      ::Float64 = 0.6          # job finding rate parameter
    κ      ::Float64 = 2.37         # vacancy posting cost
    δ_bar  ::Float64 = 0.012        # job destruction rate
    σ      ::Float64 = 0.0          # risk aversion (workers)

    A::Float64 = 1.0                # Total factor productivity 
    ψ₀::Float64 = 0.3               # Minimum efficiency threshold
    χ::Float64 = 0.5                # Remote work preference 
    b_prob ::Float64 = 0.1          # probability of losing unemployment benefits
    
    # Grids
    ## Remote work productivity grid
    n_ψ::Int64 = 50                 # Number of ψ grid points [NEW]
    ψ_min::Float64 = 0.0            # minimum ψ
    ψ_max::Float64 = 1.0            # maximum
    ψ_grid::Vector{Float64} = range(ψ_min, ψ_max, length=n_ψ)
    
    ## Utility grid
    n_x::Int64 = 100                # number of x grid points
    x_min::Float64 = 0.1            # minimum x
    x_max::Float64 = 5.0            # maximum x
    x_grid::Vector{Float64} = range(x_min, x_max, length=n_x)

    b_grid ::Vector{Float64}        # unemployment benefits grid

    # Functions 
    p::Function = θ -> θ*(1 + θ^γ)^(-1/γ)
    Y::Function = (α, ψ) -> A * ((1 - α) + α * (ψ - ψ₀))  # Production [NEW]
    u::Function
end

function getPrimitives(parameters_path::String)
    parameters = YAML.load_file(parameters_path)["Primitives"]
    
    # Parameters
    β         = parameters["beta"]
    γ         = parameters["gamma"]
    κ         = parameters["kappa"]
    δ_bar     = parameters["delta_bar"]
    σ         = parameters["sigma"]
    A         = parameters["A"]
    ψ₀        = parameters["psi_0"]
    χ         = parameters["chi"]
    b_prob    = parameters["b_prob"]
    
    # Grid parameters
    n_ψ       = parameters["n_psi"]
    ψ_min     = parameters["psi_min"]
    ψ_max     = parameters["psi_max"]
    ψ_grid    = range(ψ_min, ψ_max, length=n_ψ)
    
    n_x       = parameters["n_x"]
    x_min     = parameters["x_min"]
    x_max     = parameters["x_max"]
    x_grid    = range(x_min, x_max, length=n_x)
    
    # b_grid is a range of possible unemployment benefits 
    # (we assume that workers are thrown into unemployment with a benefit equal to their last wage)
    w_max = x_max
    w_min = x_min
    
    b_grid    = range(w_min, w_max, length=n_x)

    # Functions
    p  = (θ) -> θ .* (1 .+ θ.^γ).^(-1 ./ γ) # Job finding rate
    Y  = (α, ψ) -> A .* ((1 .- α) .+ α .* (ψ .- ψ₀))  # Production function
    
    u  = (w, α) -> w .+ α.^χ  # Utility function

    # Return the updated Primitives struct
    return Primitives(β, γ, κ, δ_bar, σ, A, ψ₀, χ, b_prob, n_ψ, ψ_min, ψ_max, ψ_grid, 
                        n_x, x_min, x_max, x_grid, b_grid, p, Y, u)
end

mutable struct Results
    J           ::Array{Float64,2}      # Value function J(ψ, x)
    θ           ::Array{Float64,2}      # Market tightness θ(ψ, x)
    α_policy    ::Array{Float64}        # Optimal remote work fraction α(ψ)
    w_policy    ::Array{Float64,2}      # Optimal wage w(ψ, x) = x - α(ψ)^χ
    δ_grid      ::Array{Float64, 2}     # job destruction grid
    W           ::Array{Float64}        # Worker value employed W(x)
    U           ::Array{Float64}        # Unemployed value U(x(b,1))
    x_policy    ::Array{Float64}        # Optimal utility search
end

function initializeModel(parameters_path::String)
    prim = getPrimitives(parameters_path)
    
    J = zeros(prim.n_ψ, prim.n_x)
    θ = zeros(prim.n_ψ, prim.n_x)
    α_policy = zeros(prim.n_ψ)
    w_policy = zeros(prim.n_ψ, prim.n_x)
    δ_grid = prim.δ_bar .* ones(prim.n_ψ, prim.n_x)
    W = zeros(prim.n_x)
    U = zeros(prim.n_x)
    x_policy = zeros(prim.n_x)
    
    return prim, Results(J, θ, α_policy, w_policy, δ_grid, W, U, x_policy)
end

function compute_optimal_α!(prim::Primitives, res::Results)
    @unpack χ, ψ_grid, ψ₀, A, n_x, n_ψ, x_grid, δ_bar, Y = prim
    
    α_pol = zeros(n_ψ)
    w_pol = zeros(n_ψ, n_x)
    δ_grid = δ_bar .* ones(n_ψ, n_x)
    

    for (i, ψ_net) in enumerate(ψ_grid .- ψ₀)
        if ψ_net <= 0
            α_pol[i] =  0.0
        else
            numerator = χ
            denominator = A * (1 - ψ_net)
            
            if χ >= 1
                α_pol[i] =  (numerator >= denominator) ? 1.0 : 0.0
            else
                α_candidate = (numerator/denominator)^(1/(1 - χ))
                α_pol[i] = clamp(α_candidate, 0.0, 1.0)
            end
        end
        for j in 1:n_x
            w_pol[i, j] = x_grid[j] - α_pol[i]^χ
            if i == 1
            @show w_pol[i, j]
            end
            # If firms's profit is negative, set δ_grid to 0 (job is not feasible)
            δ_grid[i, j] = ( Y(α_pol[i], ψ_grid[i]) - w_pol[i, j] > 0 ) ? δ_bar : 0.0
        end
    end

    # Update the α_policy in the Results struct
    res.α_policy = copy(α_pol)
    # Update the w_policy in the Results struct
    res.w_policy = copy(w_pol)
    # Update the job destruction grid
    res.δ_grid = copy(δ_grid)    
end

function iterateValueMatch(prim::Primitives, res::Results; n_iter_max::Int64=5000, tol::Float64=1e-6)
	@unpack β, δ_bar, x_grid, ψ_grid, n_ψ, n_x, Y = prim

	println(@bold @blue "Solving the firm value function J(ψ,x)")
	err   = Inf
	n_iter= 0
	
    # Update the optimal α and w policies and δ_grid
    compute_optimal_α!(prim, res)


	while err > tol && n_iter < n_iter_max
		next_J = zeros(n_ψ, n_x)
        for (i_ψ, ψ) in enumerate(ψ_grid)
            for (i_x, x) in enumerate(x_grid)
                # Get the optimal wage for the current (ψ, x)
                w = res.w_policy[i_ψ, i_x]
                # Update firm value
                α = res.α_policy[i_ψ]
                next_J[i_ψ, i_x] = Y(α, ψ) - w + β*(1 - δ_bar)*res.J[i_ψ, i_x]
            end # for i_x, x in enumerate(x_grid)
        end # for i_ψ, ψ in enumerate(ψ_grid)
		
		err  = maximum(abs.(next_J .- res.J))
		n_iter += 1
		
		if n_iter % 100 == 0
			println(@bold @yellow "Iteration: $n_iter, Error: $(round(err, digits=6))")
		elseif err < tol 
			println(@bold @green "Iteration: $n_iter, Converged!")
		end
		
		res.J = copy(next_J)
	end # while
end # function iterateValueMatch

function solveSubMarketTightness!(prim::Primitives, res::Results)
    @unpack κ, γ, n_ψ, n_x = prim
    @unpack J, δ_grid = res
    
    println(@bold @blue "Calculating market tightness θ(ψ,x)")

    for i_ψ in 1:n_ψ
        for i_x in 1:n_x
            j = J[i_ψ, i_x]
            δ = δ_grid[i_ψ, i_x]
            
            # Only calculate θ if job is viable (δ > 0) and J > κ
            if δ > 0 && j > κ
                res.θ[i_ψ, i_x] = ((j/κ)^γ - 1)^(1/γ)
            else
                res.θ[i_ψ, i_x] = 0.0
            end
        end
    end
end

# function iterateW_U(prim::Primitives, res::Results; n_iter_max::Int64=5000, tol::Float64=1e-6)
#     @unpack β, γ, δ_bar, b_prob, W_grid, b_grid, n_W, n_b = prim
#     @unpack θ = res

#     println(@bold @blue "Solving the W and U value functions")
    
#     err = Inf
#     n_iter = 0
    
#     p = prim.p.(θ)  # Job finding probability

#     while err > tol && n_iter < n_iter_max
#         next_U = zeros(n_b)
#         next_W = zeros(n_W)
        
#         # Interpolate current U for use in calculations
#         U_interp = LinearInterpolation(b_grid, res.U, extrapolation_bc=Line())
        
#         # Calculate U
#         for i in 1:n_b
#             max_val = -Inf
#             for j in 1:n_W
#                 val = p[j] * (res.W[j] - U_interp(b_grid[i]))
#                 if val > max_val
#                     max_val = val
#                     res.w[i] = W_grid[j]  # Store the wage that maximizes the value function
#                 end
#             end
#             next_U[i] = b_grid[i] + β * (U_interp((1-b_prob)*b_grid[i]) + max_val)
#         end
        
#         # Calculate W
#         for i in 1:n_W
#             next_W[i] = W_grid[i] + β * ((1-δ_bar) * res.W[i] + δ_bar * U_interp(W_grid[i]/2))
#         end
        
#         err = max(norm(next_U - res.U)/norm(next_U), norm(next_W - res.W)/norm(next_W))
        
#         res.U = copy(next_U)
#         res.W = copy(next_W)
        
#         n_iter += 1
        
#         if n_iter % 100 == 0
#             println(@bold @yellow "Iteration: $n_iter, Error: $(round(err, digits=6))")
#         elseif err < tol
#             println(@bold @green "Iteration: $n_iter, Converged!")
#         end
#     end
# end