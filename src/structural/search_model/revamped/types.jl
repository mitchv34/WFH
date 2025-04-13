#==========================================================================================
Title: types.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-26
Description: Data structures for the labor market model and simulation. Also includes 
            initialization functions for the model and simulation objects.
==========================================================================================#
module Types
    # Load required packages
    using Parameters, Random, YAML, Distributions, Term, OrderedCollections
    # Load functions to calibrate empirical distributions
    include("./calibrate_empirical_distributions.jl")
    # Load My Modules
    using ..ModelFunctions
    # Import specific types from ModelFunctions
    import ..ModelFunctions: MatchingFunction, ProductionFunction, UtilityFunction
    # Export the data structures and initialization functions
    export Primitives, Results, Worker, Economy, initializeModel, initializeEconomy
    export create_primitives_from_yaml, initializeResults
    #?=========================================================================================
    #? Model Data Structures
    #?=========================================================================================
    #==========================================================================================
    #* Primitives: Contains all parameters, grids, and functions for the labor market model. 
        #> Fields
            - `matching_function::MatchingFunction`: Function for job matching.
            - `production_function::ProductionFunction`: Function for production.
            - `utility_function::UtilityFunction`: Function for utility.
            - `κ::Float64`: Vacancy posting cost.
            - `β::Float64`: Time discount factor.
            - `δ_bar::Float64`: Baseline job destruction rate.
            - `n_ψ::Int64`: Number of remote productivity grid points.
            - `n_h::Int64`: Number of skill grid points.
            - `h_min::Float64`: Minimum worker skill.
            - `h_max::Float64`: Maximum worker skill.
            - `n_x::Int64`: Number of utility grid points.
            - `x_min::Float64`: Minimum utility.
            - `x_max::Float64`: Maximum utility.
        #> Constructor with validation
            - `Primitives(args...)`: Constructor that validates grid sizes and parameter ranges.
        ##* Validations
        - Grid sizes:
            - `ψ_grid`, `ψ_pdf`, and `ψ_cdf` must have length `n_ψ`.
            - `h_grid` and `h_pdf` must have length `n_h`.
            - `x_grid` must have length `n_x`.
        - Parameter ranges:
            - `β` must be in the range `(0,1)`.
            - `δ_bar` must be in the range `[0,1]`.
            - `κ` must be positive.
        - Grids:
            - `ψ_pdf` must be a probability distribution with non-negative values and sum to 1.
            - `h_pdf` must be a probability distribution with non-negative values and sum to 1.
    ==========================================================================================#
    @with_kw mutable struct Grid
        n::Int64                                  # Number of grid points
        min::Float64                              # Minimum value
        max::Float64                              # Maximum value
        values::Vector{Float64}  # Grid values
        pdf::Union{Function, Vector{Float64}}     # PDF of the grid values
        cdf::Union{Function, Vector{Float64}}     # CDF of the grid values
        function Grid(
                    values::Vector{Float64}; 
                    pdf::Union{Function, Vector{Float64}}=zeros(length(values)), 
                    cdf::Union{Function, Vector{Float64}}=zeros(length(values))
            )
            n = length(values)
            min = minimum(values)
            max = maximum(values)
            if typeof(pdf) == Vector{Float64}
                if (length(values) != n) || (length(pdf) != n) || (length(cdf) != n)
                    throw(ArgumentError("values must have length n"))
                end
                if any(pdf .< 0) || any(cdf .< 0)
                    throw(ArgumentError("pdf and cdf must be non-negative"))
                end
                if !isapprox(sum(pdf), 1.0, atol=1e-10)
                    throw(ArgumentError("pdf does not sum to 1."))
                end
                if !isapprox( maximum( abs.( cumsum(pdf) -  cdf)), 0.0, atol=1e-10)
                    throw(ArgumentError("pdf and cdf are not consistent"))
                end
                # Check if sorted and if not sort (also sort pdf and cdf)
                if !issorted(values)
                    sorted_indices = sortperm(values)
                    values = values[sorted_indices]
                    pdf = pdf[sorted_indices]
                    cdf = cdf[sorted_indices]
                end
            end
            # Create the grid object
            new(n, min, max, values, pdf, cdf)
        end
    end
    @with_kw mutable struct Primitives   
        #> Model functions
        matching_function::MatchingFunction
        production_function::ProductionFunction
        utility_function::UtilityFunction
        #> Market parameters
        κ::Float64      # Vacancy posting cost
        β::Float64      # Time discount factor
        δ_bar::Float64  # Baseline job destruction rate
        b::Float64      # Unemployment benefits
        #> Grids
        ψ_grid::Grid    # Remote productivity grid
        h_grid::Grid    # Skill grid
        x_grid::Grid    # Utility grid
        #> Constructor with validation
        function Primitives(args...)
            prim = new(args...)
            #?Validate parameter ranges
            ## > Discount factor
            if prim.β < 0 || prim.β > 1
                throw(ArgumentError("β must be in [0,1]"))
            end
            ## > Destruction rate
            if prim.δ_bar < 0 || prim.δ_bar > 1
                throw(ArgumentError("δ_bar must be in [0,1]"))
            end
            ## > Posting cost
            if prim.κ <= 0
                throw(ArgumentError("κ must be positive"))
            end
            return prim
        end
    end
    #==========================================================================================
    #* Results: Holds equilibrium objects from the labor market model solution.
    # Fields
        - J::Array{Float64,3}: Value of a filled vacancy (h,x) to a ψ firm: J(ψ, h, x)
        - θ::Array{Float64,2}: Market tightness θ(h, x)
        - α_policy::Array{Float64,2}: Optimal remote work a ψ firm offers an h worker: α(ψ, h)
        - w_policy::Array{Float64,3}: Optimal wage a ψ firm offers an h worker: w(ψ, h, x)
        - δ_grid::Array{Float64,3}: Job destruction grid δ(ψ, h, x) #! For now this is fixed at δ_bar (it might change in the future if dynamics are added)
        - W::Array{Float64,3}: Value of a h worker when employed in a x job at a ψ firm: W(ψ, h, x)
        - U::Array{Float64}: Value of a h worker when unemployed: U(h)
        - x_policy::Array{Int}: Optimal utility search policy (over x) (storing index)
        - ψ_bottom::Array{Float64}: Threshold for hybrid work (depending on worker skill h)
        - ψ_top::Array{Float64}: Threshold for full-time remote work (depending on worker skill h)
    ==========================================================================================#
    mutable struct Results
        #* Value functions
        J::Array{Float64,3}        # Value function J(ψ, h, x)
        θ::Array{Float64,2}        # Market tightness θ(h, x)
        W::Array{Float64,3}        # Worker value when employed W(ψ, h, x)
        U::Array{Float64}          # Unemployed worker value U(h)
        #* Policy functions
        α_policy::Array{Float64,2} # Optimal remote work fraction α(ψ, h)
        w_policy::Array{Float64,3} # Optimal wage w(ψ, h, x)
        δ_grid::Array{Float64,3}   # Job destruction grid δ(ψ, h, x)
        x_policy::Array{Int}       # Optimal utility search policy (over x) (storing index)
        ψ_bottom::Array{Float64}   # Threshold for hybrid work (depending on worker skill h)
        ψ_top::Array{Float64}      # Threshold for full-time remote work (depending on worker skill h)
    end
    #?=========================================================================================
    #? Simulation Data Structures 
    #?=========================================================================================c
    #==========================================================================================
    #* Worker: Represents an individual worker in the simulation.
    # Fields
        - id::String: Worker unique identifier
        - skill::Int: Worker skill value (index corresponding to model skill grid)
        - age::Int: Worker age (number of periods in the simulation)
        - employed::Vector{Bool}: Employment status history (true if employed)
        - wage::Vector{Float64}: Wage history
        - x_choice::Int: Worker search policy if unemployed (index in the x_policy grid)
        - firm_type::Vector{Float64}: Employer indicator/history (e.g., index from firm remote productivity grid)
        - remote_work::Vector{Float64}: Remote work fraction α history (if employed)
    ==========================================================================================#
    mutable struct Worker
        #? Worker attributes
        id::String                # Worker unique identifier
        skill::Int                # Worker skill value (index corresponding to model skill grid)
        age::Int                  # Worker age (number of periods in the simulation)
        employed::Vector{Bool}    # Employment status history (true if employed)
        wage::Vector{Float64}     # Wage history
        x_choice::Int             # Worker search policy if unemployed (index in the x_policy grid)
        firm_type::Vector{Float64} # Employer indicator/history (e.g., index from firm remote productivity grid)
        remote_work::Vector{Float64} # Remote work fraction α history (if employed)
        #? Constructor
        function Worker(
                        id::String, skill::Int, x_choice::Int; 
                        employed::Vector{Bool}=[false], wage::Vector{Float64}=[0.0],
                        firm_type::Vector{Float64}=[NaN], remote_work::Vector{Float64}=[NaN]
                    )
            #==========================================================================================
            Create a new `Worker` object.
            #* Arguments
            ##? Required
            - `id::String`: The unique identifier for the worker.
            - `skill::Int`: The skill level of the worker.
            - `x_choice::Int`: Search policy if unemployed. #! Immutable for know (might change in the future)
            ##? Optional
            - `employed::Vector{Bool}`: Employment status history. Default is `[false]` (unemployed).
            - `init_wage::Vector{Float64}`: Wage history. Default is `[0.0]` (no wage if unemployed).
            - `firm_type::Vector{Union{Float64, Missing}}`: Employer indicator/history. Default is `[NaN]` (no employer if unemployed).
            - `remote_work::Vector{Union{Float64, Missing}}`: Remote work fraction α history. Default is `[NaN]` (no remote work if unemployed).
            #* Returns
            - A new `Worker` object.
            ==========================================================================================#
            new(id, skill, 0, employed, wage, x_choice, firm_type, remote_work)
        end
    end
    #==========================================================================================
    #* Economy: Represents the economy in which workers and firms interact.
    # Fields
        - N::Int: Number of workers
        - T::Int: Number of periods to simulate
        - T_burn::Int: Number of periods to burn before collecting data
        - seed::Int: Random seed for simulation reproducibility
        - workers::Vector{Worker}: Vector of workers
        - ψ_grid::Vector{Float64}: Firm remote work efficiency grid (1D) 
        - ψ_pdf::Vector{Float64}: PDF for sampling firm types 
        - x_grid::Vector{Float64}: Utility grid from the model 
        - p::Function: Match probability function p(θ)
        - wage_fun::Function: Function to compute wage: w(x, α)
        - δ_bar::Float64: Exogenous job destruction rate #! For now this is fixed at δ_bar (it might change in the future if dynamics are added thenm it will be a model outcome)
        - θ::Array{Float64,2}: Sub-market tightness, indexed by (skill, searched_utility) #> [Model outcome]
        - α_policy::Array{Float64,2}: Optimal remote work fraction as a function of (firm type, worker skill) #>[Model outcome]
        - job_finding::Vector{Float64}: Aggregate job finding rate over time #> [Simulation outcome]
        - unemployment_rate::Vector{Float64}: Aggregate unemployment rate over time #> [Simulation outcome]
        - remote_work::Vector{Float64}: Aggregate remote work fraction over time #> [Simulation outcome]
    ==========================================================================================#
    mutable struct Economy
        #* Simulation parameters 
        N::Int                               # Number of workers
        T::Int                               # Number of periods to simulate
        T_burn::Int                          # Number of periods to burn before collecting data
        seed::Int                            # Random seed
        #* Simulation objects
        workers::Vector{Worker}              # Vector of workers
        #* Model objects
        #? Model primitives
        ψ_grid::Vector{Float64}              # Firm type grid (1D)
        ψ_pdf::Vector{Float64}               # PDF for sampling firm types
        x_grid::Vector{Float64}              # Utility grid from the model
        p::Function                          # Match probability function p(θ)
        δ_bar::Float64                       # Exogenous job destruction rate #! For now this is fixed at δ_bar (it might change in the future if dynamics are added then it will be a model solution)
        #? Model solution
        θ::Array{Float64,2}                  # Sub-market tightness, indexed by (skill, x)
        α_policy::Array{Float64,2}           # Optimal remote work fraction as a function of (firm type, worker skill)
        w_policy::Array{Float64,3}           # Optimal wage as a function of (firm type, worker skill, utility)
        #* Simulation outcomes
        job_finding::Vector{Float64}         # Aggregate job finding rate over time
        unemployment_rate::Vector{Float64}   # Aggregate unemployment rate over time
        remote_work::Vector{Float64}         # Aggregate remote work fraction over time
    end
    #?=========================================================================================
    #? Helper Functions 
    #?=========================================================================================
    #==========================================================================================
    ##* create_primitives_from_yaml(yaml_file::String)::Primitives
        Create a Primitives object by reading parameters from a YAML file.

        This function reads a YAML configuration file and initializes model functions and 
        parameters according to the configuration. It handles the initialization of matching 
        functions, production functions, and utility functions, as well as grid 
        and market parameters.

        The function reads parameters in the same order they appear in the YAML file and passes 
        them to the respective constructors in that same order. This ensures that the parameter 
        order in the YAML file matches the expected order of arguments in the constructors.

    #* Arguments
    - yaml_file::String: Path to the YAML file that contains the parameter values and model 
        specification
    #* Returns
        - Primitives: Model primitives
    ==========================================================================================#
    function create_primitives_from_yaml(yaml_file::String)::Primitives
        #> Load configuration from YAML file
        config = YAML.load_file(yaml_file, dicttype = OrderedDict)
        model_config = config["ModelConfig"]
        #> Extract configuration components
        model_parameters = model_config["ModelParameters"]
        model_grids = model_config["ModelGrids"]
        model_functions = model_config["ModelFunctions"]

        #> Extract model parameters
        #TODO: Validate model_parameters and add defaults if missing
        κ     = model_parameters["kappa"]           # Vacancy posting cost
        β     = model_parameters["beta"]            # Time discount factor
        δ_bar = model_parameters["delta_bar"]       # Baseline job destruction rate
        b = model_parameters["b"]                   # Unemployment benefit
        #> Extract model grids 
        #TODO: Validate model_grids and add defaults if missing
        #* Remote productivity grid parameters
        n_ψ       =  model_grids["RemoteProductivityGrid"]["n_psi"]                # Number of remote productivity grid points
        ψ_data    =  model_grids["RemoteProductivityGrid"]["data_file"]            # File with the data to construct the grid
        ψ_column  =  model_grids["RemoteProductivityGrid"]["data_column"]          # Column with the data to construct the grid
        ψ_weight  =  model_grids["RemoteProductivityGrid"]["weight_column"]        # Column with the weights for the data
        #* Skill grid parameters
        n_h       =  model_grids["SkillGrid"]["n_h"]                  # Number of skill grid points
        h_data    =  model_grids["SkillGrid"]["data_file"]  
        h_column  =  model_grids["SkillGrid"]["data_column"] 
        h_weight  =  model_grids["SkillGrid"]["weight_column"] 
        #* Utility grid parameters
        n_x       =  model_grids["UtilityGrid"]["n_x"]                  # Number of utility grid points
        x_min     =  model_grids["UtilityGrid"]["x_min"]                # Minimum utility #! This is hardcoded for now 
        x_max     =  model_grids["UtilityGrid"]["x_max"]                # Maximum utility #! This is hardcoded for now 
        #? Create grids
        #* Remote productivity grid
        # Compute KDE for ψ
        ψ_grid, ψ_pdf, ψ_cdf = fit_kde_psi(
                                            ψ_data,
                                            ψ_column,
                                            weights_col = ψ_weight, 
                                            num_grid_points=n_ψ,
                                            engine="python"
                                            ) 
        # Construct grid object
        ψ_grid = Grid(ψ_grid, pdf=ψ_pdf, cdf=ψ_cdf)
        #* Skill grid
        #TODO: Revise that the non-paramteric estimation is workign properly
        h_grid, h_pdf, h_cdf = fit_kde_psi(
                                            h_data,
                                            h_column,
                                            weights_col = h_weight, 
                                            num_grid_points=n_h,
                                            engine="python"
                                            ) 
        # h_grid, h_pdf, h_cdf = fit_distribution_to_data(
        #                             h_data,                 # Path to skill data 
        #                             h_column,               # Which column to use
        #                             "functions",            # Will I be returning a vector of values of functions that can be evaluated
        #                             "parametric";           # Parametric or non-Parametric estimation
        #                             distribution=Normal,    # If Parametric which distribution?
        #                             num_grid_points=n_h     # Number of grid points to fit the distribution
        #                         )
        # Construct grid object
        #if the minimum skill value h_min ≤ 0 set it to a small positive value to avoid issues with the log function
        if h_grid[1] <= 0
            h_grid[1] = 1e-6
        end
        h_grid = Grid(h_grid, pdf=h_pdf, cdf=h_cdf)
        #* Utility grid
        x_grid = range(x_min, x_max, length=n_x) |> collect
        # Create a unifor distribution for the utility grid (since it not important)
        x_pdf = fill(1/n_x, n_x)
        x_cdf = cumsum(x_pdf)
        # Construct grid object
        x_grid = Grid(x_grid, pdf=x_pdf, cdf=x_cdf)
        
        #> Extract model functions configurations
        #TODO: Validate model_functions and add defaults if missing
        #? Matching function
        matching_function_config = model_functions["MatchingFunction"]
        matching_function_type = matching_function_config["type"]
        matching_function_params = matching_function_config["params"] |> values |> collect
        # Initialize matching function
        matching_function = create_matching_function(matching_function_type, Float64.(matching_function_params) )
        #? Utility function
        utility_function_config = model_functions["UtilityFunction"]
        utility_function_type = utility_function_config["type"]
        utility_function_params = utility_function_config["params"] |> values |> collect
        # Initialize utility function
        utility_function = create_utility_function(utility_function_type, Float64.(utility_function_params) )
        #? Production function
        production_function_config = model_functions["ProductionFunction"]
        #* Productivity component
        productivity_component_config = production_function_config["ProductivityComponent"]
        productivity_component_type = productivity_component_config["type"]
        productivity_component_params = productivity_component_config["params"] |> values |> collect
        #* Remote efficiency component
        remote_work_component_config = production_function_config["RemoteEfficiencyComponent"]
        remote_work_component_type = remote_work_component_config["type"]
        remote_work_component_params = remote_work_component_config["params"] |> values |> collect
        # Initialize production function
        production_function = create_production_function(
                                                        productivity_component_type,
                                                        remote_work_component_type,
                                                        Float64.(productivity_component_params),
                                                        Float64.(remote_work_component_params)
                                                    )
        
        #> Create Primitives object
        return Primitives(
                            #> Model functions
                            matching_function,
                            production_function,
                            utility_function,
                            #> Market parameters
                            κ,
                            β,
                            δ_bar,
                            b,
                            #> Grids
                            ψ_grid,
                            h_grid,
                            x_grid
                        )
    end
    #==========================================================================================
    #* initializeResults(prim::Primitives):
        Initialize the Results object with zeros/defaults based on model primitives.
    #* Arguments
    - prim::Primitives: The model primitives
    #* Returns
        - Results: Initialized results object
    ==========================================================================================#
    function initializeResults(prim::Primitives)::Results
        #> Validate input
        if !(prim isa Primitives)
            throw(ArgumentError("Expected Primitives type"))
        end
        #* Get grid sizes
        n_ψ, n_h, n_x = prim.ψ_grid.n, prim.h_grid.n, prim.x_grid.n
        #* Initialize arrays
        J = zeros(n_ψ, n_h, n_x)
        θ = zeros(n_h, n_x)
        δ_grid = prim.δ_bar .* ones(n_ψ, n_h, n_x)
        W = zeros(n_ψ, n_h, n_x)
        U = zeros(n_h)
        x_policy = ones(Int, n_h)
        #* Calculate remote work thresholds and optimal policy based on model functions
        ψ_bottom, ψ_top, α_policy = find_thresholds_and_optimal_remote_policy(prim)
        #* Compute optimal wage policy based on model functions and optimal remote policy
        w = (i_ψ, i_h, i_x) -> find_implied_wage(prim.utility_function, prim.x_grid.values[i_x], α_policy[i_ψ, i_h], prim.h_grid.values[i_h])
        w_policy = zeros(Float64, n_ψ, n_h, n_x) 
        for i_ψ in 1:n_ψ
            for i_h in 1:n_h
                for i_x in 1:n_x
                    w_policy[i_ψ, i_h, i_x] = w(i_ψ, i_h, i_x)
                end
            end
        end
        #* Return results object
        return Results(J, θ, W, U, α_policy, w_policy, δ_grid, x_policy, ψ_bottom, ψ_top)
    end
    #==========================================================================================
    #description: 
    ==========================================================================================#
    function initializeModel(yaml_file::String)::Tuple{Primitives, Results}
        #> Create primitives from YAML file
        prim = create_primitives_from_yaml(yaml_file)
        #> Initialize results
        res = initializeResults(prim)
        #> Return primitives and results
        return prim, res
    end
    #==========================================================================================
    #* initializeEconomy(prim::Primitives, res::Results; N::Int=1000, T::Int=100, seed::Int=42)
        Initialize an Economy object for simulation based on model solution.
    #* Arguments
        #? Required
        - prim::Primitives: The model primitives
        - res::Results: The model solution results
        #? Optional
        - N::Int: Number of workers to simulate (default: 1000)
        - T::Int: Number of periods to simulate (default: 100)
        - T_burn::Int: Number of periods to burn before collecting data (default: 0)
        - seed::Int: Random seed (default: 42)
    #* Returns
        - Economy: Initialized economy for simulation
    ==========================================================================================#
    function initializeEconomy(prim::Primitives, res::Results;  N::Int=1000, T::Int=100, T_burn::Int=0, seed::Int=42)::Economy
        #* Unpack model primitives and results
        @unpack n_h, h_grid, ψ_grid, ψ_pdf, x_grid, p, δ_bar, n_h, h_pdf = prim
        @unpack θ, α_policy, x_policy, w_policy = res
        #* Initialize workers with random skill levels
        Random.seed!(seed)
        worker_skills = rand(1:n_h, Weights(h_pdf), N) # Sample worker skills from the distribution
        worker_ids = ["W$(lpad(i, 4, '0'))" for i in 1:N] # Generate worker IDs (e.g., W0001, W0002, ...)
        worker_policy = [x_policy[skill] for skill in worker_skills] # Assign search policy based on skill
        #? Create array of workers
        workers = [Worker(id, skill, policy) for (id, skill, policy) in zip(worker_ids, worker_skills, worker_policy)]
        #* Initialize time series
        job_finding = zeros(Float64, T)
        unemployment_rate = zeros(Float64, T)
        remote_work = zeros(Float64, T)
        #* Return economy object
        return Economy(
                        N, T, T_burn, seed, workers, ψ_grid, ψ_pdf,
                        x_grid, p, δ_bar, θ, α_policy, w_policy,
                        job_finding, unemployment_rate, remote_work
                    )
    end
    #?=========================================================================================
    #? Helper Functions
    #?=========================================================================================
    #*=========================================================================================
    #* find_thresholds_and_optimal_remote_policy(prim::Primitives, res::Results)::Tuple{Vector{Float64}, Vector{Float64},Matrix{Float64}}
    #*    Find the thresholds for remote work based on the model functions and parameters.
    #*    Also, compute the optimal remote work policy for each worker type.
    #* Arguments
    #*    - prim::Primitives: Model primitives
    #*=========================================================================================
    function find_thresholds_and_optimal_remote_policy(prim::Primitives)::Tuple{Vector{Float64}, Vector{Float64},Matrix{Float64}}
        # Unpack model functions
        @unpack production_function, utility_function, ψ_grid, h_grid = prim
        # Define functions for evaluation
        A = h -> evaluate(production_function.productivity, h)
        g = (ψ, h) -> evaluate(production_function.remote_efficiency, ψ, h)
        # We know that ∂Y/∂α = A(h) - g(ψ, h) 
        dY_dα = (h, ψ) -> A(h) * (g(ψ, h) - 1.0)
        # And ∂Π/∂α = ∂Y/∂α - ∂w/∂α = ∂Y/∂α - ∂w/∂α 
        # Interior solution means that ∂Π/∂α = 0 ⟹ Φ(h, ψ) = ∂w/∂α
        # By the implicit function theorem, if ∂u/∂w ≠ 0 then ∂w/∂α = - ∂u/∂α / ∂u/∂w
        #! For this function I have the derivative of the wage function with respect to α and w in general i need to do it numerically
        du_dw = (α, h) -> evaluate_derivative(utility_function, "w", 0.0, α, h) # derivative of u with respect to w (here the value of w is not important)
        du_dα = (α, h) -> evaluate_derivative(utility_function, "α", 0.0, α, h) # derivative of u with respect to α (here the value of w is not important)# 
        # Define the derivative of the wage function wrt as a function of h and ψ
        dw_dα = (α, h) -> - du_dα(α, h) / du_dw(α, h)
        # Conditions for interior solution are:
        # [-] ∂Y/∂α(h,ψ) > min{ ∂w/∂α(0, h), ∂w/∂α(1, h) }
        # [-] ∂Y/∂α(h,ψ) < max{ ∂w/∂α(0, h), ∂w/∂α(1, h) }
        # Pre-allocate arrays for thresholds ψ_bottom(h) and ψ_top(h)
        ψ_bottom = zeros(h_grid.n)
        ψ_top = zeros(h_grid.n)
        # Find the admisible range for ψ
        # Find thresholds for each skill level
        for (i, h_val) in enumerate(h_grid.values)
            # Evaluate ∂Y/∂α(h,ψ) - min{ ∂w/∂α(0, h), ∂w/∂α(1, h) } in the minimum and maximum values of ψ
            dw_min = min(dw_dα(0.0, h_val), dw_dα(1.0, h_val))
            # if (dY_dα(h_val, ψ_grid.min) - dw_min) * (dY_dα(h_val, ψ_grid.max) - dw_min) < 0
                # If the product is negative then there is a solution in the interval
                # Define the objective function to solve for the threshold
                ψ_bottom_obj = (ψ) -> dY_dα(h_val, ψ) - dw_min
                # Find the zero of the objective function Bisecting the [ψ_min, ψ_max] interval,
                # Bisection will work since conditions guarantee that wage is monotonic in α
                ψ_bottom[i] = find_zero(ψ -> ψ_bottom_obj(ψ), (-Inf, Inf), Bisection())
            # else
                # If the product is positive then there is no solution in the interval 
                # The interpretation is that there is no value of ψ ∈ [ψ_min, ψ_max] that satisfies the condition
                # This means that there all firms offer full in person work to workers with skill level h
                # This is equivalent to setting:
                # ψ_bottom[i] = Inf # Set to a value ψ > ψ_max
            # end
            # Evaluate ∂Y/∂α(h,ψ) - max{ ∂w/∂α(0, h), ∂w/∂α(1, h) } in the minimum and maximum values of ψ
            dw_max = max(dw_dα(0.0, h_val), dw_dα(1.0, h_val))
            # if (dY_dα(h_val, ψ_grid.min) - dw_max) * (dY_dα(h_val, ψ_grid.max) - dw_max) < 0
                # If the product is negative then there is a solution in the interval
                # Define the objective function to solve for the threshold
                ψ_top_obj = (ψ) -> dY_dα(h_val, ψ) - dw_max
                # Find the zero of the objective function Bisecting the [ψ_min, ψ_max] interval,
                # Bisection will work since conditions guarantee that wage is monotonic in α
                ψ_top[i] = find_zero(ψ -> ψ_top_obj(ψ), (-Inf, Inf), Bisection())
            # else
                # If the product is positive then there is no solution in the interval 
                # The interpretation is that there is no value of ψ ∈ [ψ_min, ψ_max] that satisfies the condition
                # This means that all firms offer full remote work to workers with skill level h
                # This is equivalent to setting:
                # ψ_top[i] = Inf 
            # end
        end
        # Pre-allocate array for optimal remote work policy α_policy(ψ, h)
        α_policy = zeros(ψ_grid.n, h_grid.n)
        # Find the optimal remote work policy for each firm type and skill level
        for (i_ψ, ψ_val) in enumerate(ψ_grid.values)
            for (i_h, h_val) in enumerate(h_grid.values)
                # Define the objective function for the optimal remote work policy
                α_obj = (α) -> dY_dα(h_val, ψ_val) - dw_dα(α, h_val)
                # If ψ ≤ ψ_bottom(h) then α = 0
                if ψ_val ≤ ψ_bottom[i_h]
                    α_policy[i_ψ, i_h] = 0.0
                # If ψ ≥ ψ_top(h) then α = 1
                elseif ψ_val ≥ ψ_top[i_h]
                    α_policy[i_ψ, i_h] = 1.0
                else
                    # Otherwise, find the zero of the objective function
                    try
                        α_policy[i_ψ, i_h] = find_zero(α -> α_obj(α), (0.0, 1.0), Bisection())
                    catch e
                        if isa(e, ArgumentError)
                            # Assign NaN if no solution found
                            α_policy[i_ψ, i_h] = NaN
                        else
                            rethrow(e)
                        end
                    end
                end
                
            end
        end
        return ψ_bottom, ψ_top, α_policy
    end
end # module Types