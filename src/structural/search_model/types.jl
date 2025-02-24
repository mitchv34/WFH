using Parameters

"""
Model primitives and parameters
"""
@with_kw struct Primitives   
    # Fixed parameters
    β::Float64 = 0.996        # discount factor
    γ::Float64 = 0.6          # job finding rate parameter
    κ::Float64 = 2.37         # vacancy posting cost
    δ_bar::Float64 = 0.012    # job destruction rate
    σ::Float64 = 0.0          # risk aversion (workers)
    
    # Production parameters
    A::Float64 = 1.0          # Total factor productivity 
    ψ₀::Float64 = 0.3         # Minimum efficiency threshold
    c::Float64 = 1.0          # Disutility scaling factor
    χ::Float64 = 0.5          # Remote work preference 
    b_prob::Float64 = 0.1     # probability of losing unemployment benefits
    
    # Grid parameters
    n_ψ::Int64 = 50           # Number of ψ grid points
    ψ_min::Float64 = 0.0      # minimum ψ
    ψ_max::Float64 = 1.0      # maximum ψ
    ψ_grid::Vector{Float64}   # grid of ψ values
    ψ_pdf::Vector{Float64}    # pdf of ψ values
    ψ_cdf::Vector{Float64}    # cdf of ψ values
    
    n_x::Int64 = 100          # number of x grid points
    x_min::Float64 = 0.1      # minimum x
    x_max::Float64 = 5.0      # maximum x
    x_grid::Vector{Float64}   # utility grid
    b_grid::Vector{Float64}   # unemployment benefits grid
    
    # Model functions
    p::Function              # job finding probability
    Y::Function              # production function
    u::Function              # utility function
end

"""
Model results and policy functions
"""
@with_kw mutable struct Results
    # Value functions
    J::Matrix{Float64}        # Value function J(ψ, x)
    W::Matrix{Float64}        # Worker value employed W(ψ, x)
    U::Vector{Float64}        # Unemployed value U(x(b,1))
    
    # Policy functions
    θ::Vector{Float64}        # Market tightness θ(x)
    α_policy::Vector{Float64} # Optimal remote work fraction α(ψ)
    w_policy::Matrix{Float64} # Optimal wage w(ψ, x)
    x_policy::Vector{Float64} # Optimal utility search
    
    # Additional results
    δ_grid::Matrix{Float64}   # job destruction grid
    ψ_bottom::Float64        # Threshold for hybrid work
    ψ_top::Float64          # Threshold for full-time remote work
end