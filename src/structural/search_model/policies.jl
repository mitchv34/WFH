using Term
include("types.jl")
include("utils.jl")

"""
Compute optimal work-from-home policy
"""
function compute_optimal_α!(prim::Primitives, res::Results)
    @unpack χ, c, ψ_grid, ψ₀, A, n_x, Y = prim
    @unpack ψ_bottom, ψ_top = res
    
    for (i, ψ) in enumerate(ψ_grid)
        # Compute optimal WFH fraction
        res.α_policy[i] = calculate_optimal_wfh(ψ, ψ_bottom, ψ_top, A, c, χ, ψ₀)
        
        # Update wage policy and job destruction grid
        update_wage_and_destruction!(prim, res, i, ψ)
    end
end

"""
Compute submarket tightness
"""
function compute_market_tightness!(prim::Primitives, res::Results)
    @unpack κ, γ, n_x, ψ_pdf = prim
    @unpack J = res
    
    println(@bold @blue "Calculating market tightness θ(x)")
    
    for i_x in 1:n_x
        expected_value = compute_expected_value(J[:, i_x], ψ_pdf)
        res.θ[i_x] = calculate_theta(expected_value, κ, γ)
    end
end