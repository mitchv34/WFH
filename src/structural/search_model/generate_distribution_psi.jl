#==========================================================================================
Title: Generate Distribution Psi
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-19
Description:

fit_kde_psi(ψ, num_grid_points; n_points=100, bandwidth=1, boundary=false)

Fits a kernel density estimation (KDE) for the given sample `ψ` and constructs a corresponding distribution DataFrame.
    
    # Arguments
    - `ψ::AbstractVector{<:Real}`: The input data sample for which the density is to be estimated.
    - `num_grid_points`: A parameter intended to specify the number of grid points (currently not used in the implementation).
    - `n_points::Integer=100`: The number of points to use in constructing the KDE grid.
    - `bandwidth`: The bandwidth parameter for the KDE (default is `1`).
    - `boundary`: Either `false` (default) or a tuple `(min_x, max_x)` that explicitly defines the boundaries over which the 
    KDE should be computed. If not provided, the minimum and maximum of `ψ` are used.
    
    # Returns
    A `DataFrame` with the following columns:
    - `:ψ`: The grid points at which the density is estimated.
    - `:pdf`: The normalized probability density function (PDF) values such that the sum of the PDF equals 1.
    - `:cdf`: The cumulative distribution function (CDF), computed as the cumulative sum of the PDF values.
    
    # Details
    This function carries out a KDE on `ψ` using the specified parameters and computes a normalized PDF 
        over a defined grid. The cumulative density (CDF) is then determined by cumulatively summing the PDF values. 
        Note that while `num_grid_points` is included in the function signature, only `n_points` is used to define the 
            grid resolution in the current implementation.
            
==========================================================================================#
using PyCall, Roots

function fit_kde_psi(ψ, weights; num_grid_points=100, bandwidth=1, boundary=false, engine = "julia") 
    # Fit KDE with specified grid points
    if boundary
        min_x, max_x = boundary
    else
        min_x = minimum(ψ)
        max_x = maximum(ψ)
    end

    if engine == "julia"
        kde_result = kde(
                    ψ,
                    boundary = (min_x, max_x),
                    bandwidth=bandwidth,
                    npoints=num_grid_points
                )
        # Extract grid and density values
        ψ_grid = kde_result.x
        estimated_density = kde_result.density
        # Normalize probabilities to sum to 1
        probabilities = estimated_density ./ sum(estimated_density)
    elseif engine == "python"
        # --- Step 1: Ensure required Python packages are available
        pyimport_conda("scipy.stats", "scipy")
        pyimport_conda("numpy", "numpy")        
        # Import Python libraries
        scipy_stats = pyimport("scipy.stats")
        np = pyimport("numpy")
        # --- Step 2: Convert Julia Vector to NumPy ---
        psi_values_np = np.array(ψ)
        weights_np = np.array(weights)
        # --- Step 3: Run the Python KDE ---
        kde_result = scipy_stats.gaussian_kde(psi_values_np, weights=weights_np)
        kde_grid = np.linspace(np.min(psi_values_np), np.max(psi_values_np), num_grid_points)
        estimated_density = kde_result.evaluate(kde_grid)
        prob = estimated_density / np.sum(estimated_density)  # Ensure sum to 1
        # --- Step 4: Convert Python Output to Julia ---
        ψ_grid = Vector{Float64}(kde_grid)
        probabilities = Vector{Float64}(prob)
    end
    ψ_pdf = probabilities
    # Add cdf
    ψ_cdf = cumsum(ψ_pdf)
    return ψ_grid, ψ_pdf, ψ_cdf
end

# Define the optimal policy function 
function alpha_star(ψ, ψ0, A, c, χ)
    threshold_lower = ψ0 + (1 - c*χ/A)  # for gamma = 1
    threshold_upper = ψ0 + 1
    if ψ <= threshold_lower
        return 0.0
    elseif ψ > threshold_upper
        return 1.0
    else
        return 1 - ((A*(1 - (ψ - ψ0))) / (c*χ))^(1/(χ - 1))
    end
end

# Compute the theoretical average remote work fraction
function model_average_α(ψ0, ψ_grid, ψ_pdf, A, c, χ)
    integrand = alpha_star.(ψ_grid, ψ0, A, c, χ) .* ψ_pdf
    average = sum(integrand)
    return average
end

function estimate_ψ₀(ψ_grid, ψ_pdf, A, c, χ, α_data)
    # Define the moment condition function
    moment_condition(ψ0) = model_average_α(ψ0, ψ_grid, ψ_pdf, A, c, χ) - α_data
    # Solve for ψ0 using a root-finding method
    ψ0_est = find_zero(moment_condition, (minimum(ψ_grid), maximum(ψ_grid)), Bisection())
    return ψ0_est
end
