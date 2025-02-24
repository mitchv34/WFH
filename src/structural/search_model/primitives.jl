using YAML
using CSV
using Tables
using Term
include("types.jl")
include("utils.jl")

"""
Initialize model primitives from parameters file
"""
function getPrimitives(parameters_path::String, psi_distribution_data_path::String)
    params = YAML.load_file(parameters_path)["Primitives"]
    
    # Load and process parameters
    β, γ, κ, δ_bar, σ, A = get_base_parameters(params)
    χ, c, b_prob = get_work_parameters(params)
    n_ψ, ψ_min, ψ_max = get_productivity_parameters(params)
    
    # Process distribution data
    ψ_grid, ψ_pdf, ψ_cdf = process_distribution_data(psi_distribution_data_path, n_ψ)
    
    # Calculate ψ₀ and normalize grids
    ψ₀ = estimate_ψ₀(ψ_grid, ψ_pdf, A, c, χ, params["alpha_data"])
    ψ_grid, ψ₀ = normalize_grids(ψ_grid, ψ₀, ψ_min, ψ_max)
    
    # Setup utility and benefit grids
    x_grid, b_grid = setup_grids(params)
    
    # Define model functions
    p, Y, u = setup_model_functions(A, ψ₀)
    
    return Primitives(β=β, γ=γ, κ=κ, δ_bar=δ_bar, σ=σ, A=A, 
                    ψ₀=ψ₀, χ=χ, c=c, b_prob=b_prob,
                    n_ψ=n_ψ, ψ_min=ψ_min, ψ_max=ψ_max,
                    ψ_grid=ψ_grid, ψ_pdf=ψ_pdf, ψ_cdf=ψ_cdf,
                    x_grid=x_grid, b_grid=b_grid,
                    p=p, Y=Y, u=u)
end

"""
Initialize model results structure
"""
function initializeResults(prim::Primitives)
    n_ψ, n_x = prim.n_ψ, length(prim.x_grid)
    
    # Calculate critical thresholds
    ψ_bottom = calculate_bottom_threshold(prim)
    ψ_top = calculate_top_threshold(prim)
    
    return Results(
        J = zeros(n_ψ, n_x),
        W = zeros(n_ψ, n_x),
        U = zeros(n_x),
        θ = zeros(n_x),
        α_policy = zeros(n_ψ),
        w_policy = zeros(n_ψ, n_x),
        x_policy = zeros(n_x),
        δ_grid = prim.δ_bar .* ones(n_ψ, n_x),
        ψ_bottom = ψ_bottom,
        ψ_top = ψ_top
    )
end

"""
Initialize the complete model
"""
function initializeModel(parameters_path::String, psi_distribution_data_path::String)
    prim = getPrimitives(parameters_path, psi_distribution_data_path)
    res = initializeResults(prim)
    return prim, res
end