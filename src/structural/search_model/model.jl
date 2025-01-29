using Parameters
using LinearAlgebra
using YAML
using Interpolations
using Term

@with_kw struct Primitives
    # Parameters
    β      ::Float64 = 0.996    # discount factor
    γ      ::Float64 = 0.6      # job finding rate parameter
    κ      ::Float64 = 2.37     # vacancy posting cost
    δ_bar  ::Float64 = 0.012    # job destruction rate
    σ      ::Float64 = 0.0 
    α      ::Float64 = 1.0
    χ      ::Float64 = 2.0
    b_prob ::Float64 = 0.1      # probability of losing unemployment benefits
    # Grids
    n_W    ::Int64              # number of wage grid points
    W_max  ::Float64            # maximum wage
    W_min  ::Float64            # minimum wage
    W_grid ::Vector{Float64}    # wage grid
    n_b    ::Int64              # number of unemployment benefits grid points
    b_grid ::Vector{Float64}    # unemployment benefits grid
    δ_grid ::Vector{Float64}    # job destruction grid

    # Functions 
    p      ::Function
    u      ::Function
end

function getPrimitives(parameters_path::String)
    parameters = YAML.load_file(parameters_path)["Primitives"]
    
    β         = parameters["beta"]
    γ         = parameters["gamma"]
    κ         = parameters["kappa"]
    δ_bar     = parameters["delta_bar"]
    σ         = parameters["sigma"]
    α         = parameters["alpha"]
    χ         = parameters["chi"]
    b_prob    = parameters["b_prob"]
    
    n_W       = parameters["n_W"]
    W_max     = parameters["W_max"]
    W_min     = parameters["W_min"]
    
    W_grid    = range(W_min, W_max, length=n_W)
    
    n_b       = parameters["n_b"]
    
    b_grid    = range(W_min, W_max, length=n_b)
    
    δ_grid = δ_bar .* ones(n_W)  # Set δ_grid to be δ_bar for all wages
    
    p  = (θ) -> θ*(1 + θ^γ)^(-1/γ)
    u  = (w) -> w

    # Ensure all arguments match the expected types
    return Primitives(β,γ,κ,δ_bar,σ,α,χ,b_prob,n_W,W_max,W_min,W_grid,n_b,b_grid,δ_grid,p,u)
end


mutable struct Results
    J::Vector{Float64}  # value of an ongoing match J(w)
    θ::Vector{Float64}  # market tightness θ(w)
    W::Vector{Float64}  # value of an employed household W(w)
    U::Vector{Float64}  # value of an unemployed household U(b)
    w::Vector{Float64}  # wage w(b) search policy
end

function initializeModel(parameters_path::String)
    prim = getPrimitives(parameters_path)
    
    J = zeros(prim.n_W)
    
    θ = zeros(prim.n_W)
    
    W = zeros(prim.n_W)
    
    U = zeros(prim.n_b)

    w = zeros(prim.n_b)
    
    return prim, Results(J, θ, W, U, w)
end

function iterateValueMatch(prim::Primitives, res::Results; n_iter_max::Int64=5000, tol::Float64=1e-6)
	@unpack β, δ_grid, W_grid, n_W = prim
	println(@bold @blue "Solving the firm value function J")
	err   = Inf
	n_iter= 0
	
	while err > tol && n_iter < n_iter_max
		next_J = zeros(n_W)
		for i in 1:n_W
			next_J[i] = (1 - W_grid[i]) + β * (1 - δ_grid[i]) * res.J[i]
		end
		
		err     = maximum(abs.(next_J .- res.J))
		n_iter += 1
		
		if n_iter % 100 == 0
			println(@bold @yellow "Iteration: $n_iter, Error: $(round(err, digits=6))")
		elseif err < tol 
			println(@bold @green "Iteration: $n_iter, Converged!")
		end
		
		res.J = copy(next_J)
	end 
end

function solveSubMarketTightness(prim::Primitives, res::Results)
	@unpack κ, γ = prim
	@unpack J   = res
	
	res.θ .= [(j > κ && j > 0) ? ((j / κ)^γ - 1)^(1/γ) : 0.0 for j in J]
end

function iterateW_U(prim::Primitives, res::Results; n_iter_max::Int64=5000, tol::Float64=1e-6)
    @unpack β, γ, δ_bar, b_prob, W_grid, b_grid, n_W, n_b = prim
    @unpack θ = res

    println(@bold @blue "Solving the W and U value functions")
    
    err = Inf
    n_iter = 0
    
    p = prim.p.(θ)  # Job finding probability

    while err > tol && n_iter < n_iter_max
        next_U = zeros(n_b)
        next_W = zeros(n_W)
        
        # Interpolate current U for use in calculations
        U_interp = LinearInterpolation(b_grid, res.U, extrapolation_bc=Line())
        
        # Calculate U
        for i in 1:n_b
            max_val = -Inf
            for j in 1:n_W
                val = p[j] * (res.W[j] - U_interp(b_grid[i]))
                if val > max_val
                    max_val = val
                    res.w[i] = W_grid[j]  # Store the wage that maximizes the value function
                end
            end
            next_U[i] = b_grid[i] + β * (U_interp((1-b_prob)*b_grid[i]) + max_val)
        end
        
        # Calculate W
        for i in 1:n_W
            next_W[i] = W_grid[i] + β * ((1-δ_bar) * res.W[i] + δ_bar * U_interp(W_grid[i]/2))
        end
        
        err = max(norm(next_U - res.U)/norm(next_U), norm(next_W - res.W)/norm(next_W))
        
        res.U = copy(next_U)
        res.W = copy(next_W)
        
        n_iter += 1
        
        if n_iter % 100 == 0
            println(@bold @yellow "Iteration: $n_iter, Error: $(round(err, digits=6))")
        elseif err < tol
            println(@bold @green "Iteration: $n_iter, Converged!")
        end
    end
end