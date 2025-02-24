using Term

"""
Helper functions for parameter processing
"""
function get_base_parameters(params)
    β = params["beta"]
    γ = params["gamma"]
    κ = params["kappa"]
    δ_bar = params["delta_bar"]
    σ = params["sigma"]
    A = params["A"]
    return β, γ, κ, δ_bar, σ, A
end

function get_work_parameters(params)
    χ = params["chi"]
    c = params["c"]
    b_prob = params["b_prob"]
    return χ, c, b_prob
end

"""
Helper functions for grid calculations
"""
function calculate_bottom_threshold(prim)
    @unpack ψ₀, c, χ, A = prim
    return ψ₀ + (1 - (c * χ) / A)
end

function calculate_top_threshold(prim)
    @unpack ψ₀ = prim
    return ψ₀ + 1
end

"""
Progress reporting
"""
function report_progress(n_iter::Int64, err::Float64, converged::Bool)
    if n_iter % 100 == 0
        println(@bold @yellow "Iteration: $n_iter, Error: $(round(err, digits=6))")
    elseif converged
        println(@bold @green "Iteration: $n_iter, Converged!")
    end
end

# Add other utility functions as needed