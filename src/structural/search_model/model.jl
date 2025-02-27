using Parameters
using LinearAlgebra
using YAML
using Interpolations
using Term
using CSV
using DataFrames

include("generate_distribution_psi.jl")

#====================================================================
# Primitives: Model parameters, grids, and functions.
====================================================================#
@with_kw struct Primitives   
    # Parameters
    β      ::Float64                # discount factor
    γ      ::Float64                # job finding rate parameter
    κ      ::Float64                # vacancy posting cost
    δ_bar  ::Float64                # job destruction rate
    σ      ::Float64                # risk aversion (workers)
    A      ::Float64                # Total factor productivity 
    ψ₀     ::Float64                # Minimum efficiency threshold
    c0      ::Float64                # Disutility scaling factor
    c1      ::Float64                # Disutility scaling factor
    χ      ::Float64                # Remote work preference 
    η      ::Float64                # Curvature of the skill part of the dis-utility function
    # φ      ::Float64                # Fraction of current wage transferred upon separation
    # b_prob ::Float64                # probability of losing unemployment benefits
    b      ::Float64                # unemployment benefits
    ϕ      ::Float64                # Remote efficiency-worker skill factor
    # Grids (for firm remote efficiency and submarket x)
    n_ψ    ::Int64                  # Number of ψ grid points
    ψ_min  ::Float64                # minimum ψ
    ψ_max  ::Float64                # maximum ψ
    ψ_grid ::Array{Float64}         # grid of ψ values
    ψ_pdf  ::Array{Float64}         # pdf of ψ values
    ψ_cdf  ::Array{Float64}         # cdf of ψ values
    
    n_h     ::Int64                  # number of worker skill levels
    h_min   ::Float64                # minimum worker skill
    h_max   ::Float64                # maximum worker skill
    worker_skill ::Vector{Float64}  # grid of worker skill levels

    n_x    ::Int64                  # number of utility levels
    x_min  ::Float64                # minimum utility
    x_max  ::Float64                # maximum utility
    x_grid ::Vector{Float64}        # grid of utility levels

    # b_grid ::Vector{Float64}        # unemployment benefits grid


    # Functions 
    p ::Function
    # The production function here is kept as before.
    # In the optimal remote fraction, we will incorporate worker skill.
    Y ::Function
    u ::Function
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
    ϕ         = parameters["phi"]
    χ         = parameters["chi"]
    c0        = parameters["c0"]
    c1        = parameters["c1"]
    # η         = parameters["eta"]
    η         = 3.0
    # φ         = parameters["phi"]
    # b_prob    = parameters["b_prob"]
    b         = parameters["b"]
    
    # Grid parameters for ψ
    n_ψ       = parameters["n_psi"]
    ψ₀        = parameters["psi_0"] 
    
    

    # Read and process the ψ distribution from CSV.
    psi_data = CSV.File(psi_distribution_data_path) |> DataFrame
    ψ_vals = psi_data[!, :PSI] 
    weights = psi_data[!, :JOB_CREATION_BIRTHS]
    ψ_grid, ψ_pdf, ψ_cdf = fit_kde_psi(
                                        ψ_vals,
                                        weights; 
                                        num_grid_points=n_ψ, 
                                        engine = "python",
                                        boundary = false
                                    ) 
    ψ_grid = collect(ψ_grid)
    ψ_min     = minimum(ψ_grid)
    ψ_max     = maximum(ψ_grid)
    # Estimate and rescale ψ₀
    α_data = parameters["alpha_data"]
    # ψ₀ = estimate_ψ₀(ψ_grid, ψ_pdf, A, c, χ, α_data)
    
    # ψ₀  = ψ_min .+ ((ψ_max - ψ_min) .* (ψ₀ .- minimum(ψ_grid)) ./ (maximum(ψ_grid) - minimum(ψ_grid)))
    ψ_grid = ψ_min .+ ((ψ_max - ψ_min) .* (ψ_grid .- minimum(ψ_grid)) ./ (maximum(ψ_grid) - minimum(ψ_grid)))

    # Skill levels grid
    n_h       = parameters["n_h"]
    @show n_h
    h_min     = parameters["h_min"]
    h_max     = parameters["h_max"]
    worker_skill = range(h_min, h_max, length=n_h) |> collect

    # Grid for submarket x
    n_x       = parameters["n_x"]
    x_min     = parameters["x_min"]
    x_max     = parameters["x_max"]
    x_grid    = range(x_min, x_max, length=n_x)
    
    # Grid for unemployment benefits
    # b_grid    = range(0.0, φ, length=n_x)
    # worker_skill = [0.6, 1.2]
    # Functions: job finding and production as before.
    p  = (θ) -> θ .* (1 .+ θ.^γ).^(-1 ./ γ)
    Y = (h, α, ψ) -> A * g1(h) * ((1 - α) + α * (ψ * g2(h) - ψ₀))
    #? Internal complementarity
    g1 = h -> 1.0; g2 = h -> h;
    #? External complementarity
    # g1 = h -> h; g2 = h -> 1.0;
    #? Remote efficiency depends on both worker skill and firm productivity
    Y = (h, α, ψ) -> A * g1(h) * ((1 - α) + α * (ψ + g2(h) - ψ₀)) + 17
    #* Linear
    # g1 = h -> h; g2 = h -> ϕ * h;
    #* Logarithmic
    g1 = h -> h; g2 = h -> ϕ * log(h);
    correct_range = ϕ .> c0 * χ ./ (A .* worker_skill)
    correct_range_percentage =  sum(correct_range) / length(correct_range)
    if correct_range_percentage == 1
        println(@bold @green "Parameters are in the correct range.")
    else
        # Determine color based on percentage
        if correct_range_percentage >= .8
            println( @bold @yellow  "Parameters are partially out of range: $(correct_range_percentage)")

        else
            println( @bold @red "Parameters are partially out of range: $(correct_range_percentage)")
        end
    end

    #* Exponential
    # g1 = h -> h; g2 = h -> ϕ * exp(h);
    # TODO: Make the utility function flexible
    u  = (w, α) -> w .+ α.^χ  

    prim = Primitives(
    β,
    γ,
    κ,
    δ_bar,
    σ,
    A,
    ψ₀,
    c0,
    c1,
    χ,
    η,
    b,
    ϕ,
    n_ψ,
    ψ_min,
    ψ_max,
    ψ_grid,
    ψ_pdf,
    ψ_cdf,
    n_h,
    h_min,
    h_max,
    worker_skill,
    n_x,
    x_min,
    x_max,
    x_grid,
    p,
    Y,
    u
    )
    return prim
end # function getPrimitives

#====================================================================
# Results: Holds equilibrium objects. The new dimension for worker skill 
# is added (n_h = length(worker_skill)) so that arrays depending on skill 
# become three-dimensional.
====================================================================#
mutable struct Results
    J           ::Array{Float64,3}      # Value function J(ψ, h, x)
    θ           ::Array{Float64,2}      # Market tightness θ(x)
    α_policy    ::Array{Float64,2}      # Optimal remote work fraction α(ψ, h)
    w_policy    ::Array{Float64,3}      # Optimal wage w(ψ, h, x) = x + c(1 - α(ψ,h))^χ
    δ_grid      ::Array{Float64,3}      # Job destruction grid δ(ψ, h, x)
    W           ::Array{Float64,3}      # Worker value when employed W(ψ, h, x)
    U           ::Array{Float64}        # Unemployed worker value U(h)
    x_policy    ::Array{Int}            # Optimal utility search policy (over x) (storing index)
    ψ_bottom    ::Array{Float64}        # Threshold for hybrid work (depending on worker skill h)
    ψ_top       ::Array{Float64}        # Threshold for full-time remote work (depending on worker skill h)
end # struct Results

#====================================================================
# initializeModel: Sets up the model using the primitives.
# We now include the worker skill dimension.
====================================================================#
function initializeModel(parameters_path::String, psi_distribution_data_path::String)
    prim = getPrimitives(parameters_path, psi_distribution_data_path)
    
    J = zeros(prim.n_ψ, prim.n_h, prim.n_x)
    θ = zeros(prim.n_h, prim.n_x)
    α_policy = zeros(prim.n_ψ, prim.n_h)
    w_policy = zeros(prim.n_ψ, prim.n_h, prim.n_x)
    δ_grid = prim.δ_bar .* ones(prim.n_ψ, prim.n_h, prim.n_x)
    W = zeros(prim.n_ψ, prim.n_h, prim.n_x)
    U = zeros(prim.n_h)
    x_policy = zeros(prim.n_h)
    # For now, thresholds are based on the firm's parameters (could later vary with h)
    #? Internal complementarity 
    # ψ_bottom = [ (prim.ψ₀ +  (1 - (prim.c0 * prim.χ)/prim.A)) / h for h in prim.worker_skill]
    # ψ_top = [(prim.ψ₀ + 1 )/h for h in prim.worker_skill]
    #? External complementarity
    # ψ_bottom = [ prim.ψ₀ +  (1 - (prim.c0 * prim.χ)/(h * prim.A))  for h in prim.worker_skill]
    # ψ_top = [prim.ψ₀ + 1  for h in prim.worker_skill]
    # ? Remote efficiency depends on both worker skill and firm productivity
    ψ_top = [(prim.ψ₀ + 1) - prim.ϕ * log(h)  for h in prim.worker_skill]
    ψ_bottom = [ ψ_top[i_h] - prim.c0 * prim.χ / (prim.A * h) for (i_h, h) in enumerate(prim.worker_skill)]
    # ? Remote efficiency depends on both worker skill and firm productivity (linear)
    # ψ_top = [(prim.ψ₀ + 1) - prim.ϕ * h  for h in prim.worker_skill]
    # ψ_bottom = [ ψ_top[i_h] - prim.c0 * prim.χ / (prim.A * h) for (i_h, h) in enumerate(prim.worker_skill)]
    #? On‐Site Dis-utility Factor Depend on Worker Skill
    # ψ_top = [(prim.ψ₀ + 1) for h in prim.worker_skill]
    # ψ_bottom = [ ψ_top[i_h] - (prim.c0 + prim.c1 * h ^ prim.χ)* prim.χ / (prim.A * h) for (i_h, h) in enumerate(prim.worker_skill)]
    return prim, Results(J, θ, α_policy, w_policy, δ_grid, W, U, x_policy, ψ_bottom, ψ_top)
end # function initializeModel


#====================================================================

====================================================================#
function compute_optimal_α!(prim::Primitives, res::Results)
    @unpack χ, c0, c1, ψ_grid, ψ₀, A, n_x, n_ψ, x_grid, δ_bar, worker_skill, ϕ = prim
    @unpack ψ_bottom, ψ_top = res
    n_h = length(worker_skill)
    
    α_pol = zeros(n_ψ, n_h)
    w_pol = zeros(n_ψ, n_h, n_x)
    δ_grid_new = δ_bar .* ones(n_ψ, n_h, n_x)
    
    n = length(prim.c0 .* prim.χ./(prim.A .* prim.worker_skill))
    prim.c0 .* prim.χ./(prim.A .* prim.worker_skill) .< prim.ϕ
    
    # Loop over firm productivity (ψ) and worker remote skill (h)
    for (h_ind, h) in enumerate(worker_skill)
        # h_ind = 1
        # h = worker_skill[h_ind]
        for (ψ_ind, ψ) in enumerate(ψ_grid)
        # ψ_ind = 34
        # ψ = ψ_grid[ψ_ind]
            # @show ψ_ind, h_ind
        # h_ind = 1
        # h = worker_skill[h_ind]
            # Check corner conditions based on the modified production term.
            # prim.ψ_grid .> ψ_bottom[h_ind]
            # prim.ψ_grid .< ψ_top[h_ind]
            
            # 1 .- (c0 .* χ ) ./ (A * h .* (1 .+ ψ₀ .- prim.ψ_grid .- ϕ *log(h) )).^(1/(1-χ))


            #  (1 .- ( (c0 .* χ ) / (A .* h .* (1 .+ ψ₀ .- prim.ψ_grid .+ ϕ *log(h) ) )).^(1/(1-χ)))'
            
            # using Plots
            # plot( (1 .- ( (c0 .* χ ) / (A .* h .* (1 .+ ψ₀ .- prim.ψ_grid .- ϕ *log(h) ) )).^(1/(1-χ)))')
            if ψ < ψ_bottom[h_ind]
                # If the effective remote component is too low, choose full on-site work.
                α_pol[ψ_ind, h_ind] = 0.0
                # print("zero")
                # @show ψ_ind, h_ind
            elseif ψ > ψ_top[h_ind]
                # If the remote option is very attractive (or the term becomes negative), choose full remote.
                α_pol[ψ_ind, h_ind] = 1.0
            else
                # print(">zero")
                # @show ψ_ind, h_ind
                #? Internal complementarity 
                # α_pol[ψ_ind, h_ind] = 1 - ( (c0 * χ ) / ( A * (1 - (h * ψ - ψ₀) )) )^(1/(1-χ))
                #? External complementarity 
                # α_pol[ψ_ind, h_ind] = 1 - ( (c0 * χ )/(h * A * (1 - ( ψ - ψ₀) )) )^(1/(1-χ))
                #? Remote efficiency depends on both worker skill and firm productivity
                α_pol[ψ_ind, h_ind] = 1 - ( (c0 * χ ) / (A * h * (1 + ψ₀ - ψ - ϕ *log(h) ) ))^(1/(1-χ))
                #? Remote efficiency depends on both worker skill and firm productivity (linear)
                # α_pol[ψ_ind, h_ind] = 1 - ( (c0 * χ ) / (A * h * (1 + ψ₀ - ψ - ϕ * h ) ))^(1/(1-χ))
                #? On‐Site Dis-utility Factor Depend on Worker Skill
                # α_pol[ψ_ind, h_ind] = 1 - ( (prim.c0 + prim.c1 * h ^ prim.χ) * prim.χ / (A * h * (1 + ψ₀ - ψ)) )^(1/(1-χ))
            end
            # For each submarket x, compute the associated wage policy:
            for (x_ind, x) in enumerate(x_grid)
                w = x + c0*(1 - α_pol[ψ_ind, h_ind])^χ
                #? On‐Site Dis-utility Factor Depend on Worker Skill
                # w = x + (c0 + c1 * h ^ χ) * (1 - α_pol[ψ_ind, h_ind])^χ
                w_pol[ψ_ind, h_ind, x_ind] = w
            end
        end
    end
    res.α_policy = copy(α_pol)
    res.w_policy = copy(w_pol)
    res.δ_grid = copy(δ_grid_new)
end # function compute_optimal_α!

#====================================================================
# iterateValueMatch: Solves for the firm value function J(ψ,h,x)
====================================================================#
function iterateValueMatch(prim::Primitives, res::Results; n_iter_max::Int64=5000, tol::Float64=1e-6)
    @unpack β, x_grid, ψ_grid, n_ψ, n_x, worker_skill = prim
    δ_grid = res.δ_grid

    println(@bold @blue "Solving the firm value function J(ψ, h, x)")
    err   = Inf
    n_iter= 0
    
    # Update the optimal policies
    compute_optimal_α!(prim, res)

    # while err > tol && n_iter < n_iter_max
        # next_J = zeros(n_ψ, n_h, n_x)
        for i in 1:n_ψ
        # i = 25
        # ψ = ψ_grid[i]
        # h1 = 3
        # h2 = 44
            for (h_ind, h) in enumerate(worker_skill)
                # for (j, x) in enumerate(x_grid)
                    δ_val = δ_grid[i, h_ind, :]
                    w_val = res.w_policy[i, h_ind, :]
                    α_val = res.α_policy[i, h_ind]
                    res.J[i, h_ind, :] = (prim.Y(h, α_val, ψ_grid[i]) .- w_val) ./ (1 .- β.*(1 .- δ_val) )
                # end
            end
        end
        # err  = maximum(abs.(next_J .- res.J))
        # n_iter += 1
        
        # if n_iter % 100 == 0
            # println(@bold @yellow "Iteration: $n_iter, Error: $(round(err, digits=6))")
        # elseif err < tol 
            # println(@bold @green "Iteration: $n_iter, Converged!")
        # end
        
        # res.J = copy(next_J)
    # end
end # function iterateValueMatch

#====================================================================
# solveSubMarketTightness!: Computes market tightness θ(x) using the expected firm value.
# This part remains unchanged.
====================================================================#
function solveSubMarketTightness!(prim::Primitives, res::Results)
    @unpack κ, γ, n_ψ, n_x, ψ_pdf, worker_skill = prim
    @unpack J = res

    println(@bold @blue "Calculating market tightness θ(x)")
    for (i_h, h) in enumerate(worker_skill)
    # i_h = 15
    # h = worker_skill[i_h]
    # j = J[:, i_h, :]
    # Ej = sum(j .* ψ_pdf, dims = 1)  # weighted average over ψ 
    
        for i_x in 1:n_x
            j = J[:, i_h, i_x]
            Ej = dot(j, ψ_pdf)  # weighted average over ψ 
            res.θ[i_h, i_x] = (Ej > κ ) ? ( (Ej / κ )^γ - 1)^(-1/γ) : 0.0
        end
    end
end # function solveSubMarketTightness!


function iterateW_U(prim::Primitives, res::Results; n_iter_max::Int64=5000, tol::Float64=1e-6)
    # @unpack β, x_grid, ψ_grid, ψ_pdf, p, u, b_grid, b_prob, φ, worker_skill = prim
    @unpack β, x_grid, ψ_grid, ψ_pdf, p, u, b, n_h, worker_skill, n_x = prim
    # b_decay = 1 - b_prob       # Unemployment benefit decay factor
    println(@bold @blue "Solving the worker value functions (with skill dimension)")
    # Employed value function: dimensions (n_ψ, n_h, n_x)
    W_new = zeros(length(ψ_grid), n_h, length(x_grid))
    # Unemployed value function now: dimensions (n_h, length(b_grid))
    U_old = copy(res.U)  # res.U should now be a matrix of size (n_h, length(b_grid))
    U_new = similar(U_old)
    # EW: expected employed value, computed separately for each skill and submarket.
    EW = zeros(n_h, length(x_grid))
    
    err = Inf
    n_iter = 0
    # tol=1e-6
    while err > tol && n_iter < n_iter_max
        # --- 1. Update Employed Worker Value ---
        # Interpolate U for each skill level separately.
        for (i, ψ) in enumerate(ψ_grid)
            for (k, h) in enumerate(worker_skill)
                # for (j, x) in enumerate(x_grid)
                    δ_val = res.δ_grid[i, k, :]
                    w_val = res.w_policy[i, k, :]
                    # Use the unemployment value corresponding to skill k.
                    # U_interp = LinearInterpolation(collect(b_grid), U_old[k, :], extrapolation_bc=Line())
                    # U_unemp = U_interp(φ * w_val)
                    W_new[i, k, :] = ( x_grid .+ β .* δ_val .* U_old[k] ) ./ ( 1 .- β .* (1 .- δ_val) )
                # end
            end
        end

        # --- 2. Compute Expected Employed Value by Skill ---
        for h_i in 1:n_h
            # for x_i in 1:n_x
            #TODO: Maybe y need to use the old W values
            #! CHECK THIS !!!!
                EW[h_i, :] = sum( W_new[:, h_i, :] .* ψ_pdf, dims=1 )
            # end
        end
        
        # --- 3. Update Unemployed Worker Value by Skill ---
        # U_new will be computed for each skill level k and each benefit b in b_grid.
        # for h_i in 1:n_h
        # h_i = 1
        continuation_values = p(res.θ) .* EW + ( 1 .- p(res.θ) ) .* U_old

        # maximum( p(res.θ) .* EW + ( 1 .- p(res.θ) ) .* U_old, dims=2)
        # argmax( p(res.θ) .* EW + ( 1 .- p(res.θ) ) .* U_old, dims=2)

        U_new = b .+ β .* maximum(continuation_values, dims=2)
        xpol  = argmax(continuation_values, dims=2)
        res.x_policy = map(idx -> idx[2], xpol)[:]
            # Create an interpolant for U_old for the current skill level.
            # U_interp = LinearInterpolation(collect(b_grid), U_old[k, :], extrapolation_bc=Line())
            # for (i, 
            # i = 1
            # b = b_grid[i]
                # continuation_values = similar(x_grid)
                # continuation_values = p(res.θ[k, :]) .* EW[k, :] .+ (1 .- p(res.θ[k, :])).* U_interp(b_decay * b)
                # for (j, x) in enumerate(x_grid)
                #     continuation_values[j] = p(res.θ[j]) * EW[k, j] +
                #                                (1 - p(res.θ[j])) * U_interp(b_decay * b)
                # end
                # U_new[k, i] = u(b, 1) + β * maximum(continuation_values)
                # Optionally, if you want to record the optimal x for each (k, b), do so here.
                # For example:
                # res.x_policy[k, i] = x_grid[argmax(continuation_values)]
            # end
        # end
        # plot(continuation_values)

        # u_1 = p(res.θ[k, :]) .* EW[k, :] .+ (1 .- p(res.θ[k, :])).* U_interp(b_decay * b_grid[1])
        # u_100 = p(res.θ[k, :]) .* EW[k, :] .+ (1 .- p(res.θ[k, :])).* U_interp(b_decay * b_grid[100])
        # plot(x_grid, u_1, label="b = $(b_grid[1])")
        # plot!(x_grid, u_100, label="b = $(b_grid[100])")
        # argmax(u_1), argmax(u_100)
        
        # res.x_policy[k, :]
        
        
        # --- 4. Check Convergence and Update ---
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

    if n_iter >= n_iter_max
        println(@bold @red "Worker value functions did not converge within $n_iter_max iterations.")
    end
end
