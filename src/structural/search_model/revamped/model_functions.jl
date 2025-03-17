#==========================================================================================
Title: model_functions.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-26
Description: Abstract functions for the search model:
    - ProductionFunction
    - UtilityFunction
    - MatchingFunction
    - WageFunction
=========================================ß=================================================#
module ModelFunctions
    # Load required packages
    using Parameters, Roots
    # Export functions and types 
    export AbstractModelFunction, ProductionFunction, UtilityFunction
    export MatchingFunction, WageFunction
    export evaluate, get_parameters, find_implied_wage, evaluate_derivative
    export invert_vacancy_fill, eval_prob_job_find
    # Export factory functions for creating model functions
    export create_matching_function, create_production_function, create_utility_function
    #?=========================================================================================
    #? AbstractModelFunction: Base abstract type for all model functions
    #?=========================================================================================
    #==========================================================================================
    #* AbstractModelFunction:
        -  An abstract type that serves as a common interface for all model functions.
    ==========================================================================================#
    abstract type AbstractModelFunction end
    #==========================================================================================
    #* evaluate
        Evaluate the model function `f` with the given arguments `args...`.  
        #* Arguments
        - `f::AbstractModelFunction`: The model function to be evaluated.
        - `args...`: The arguments to be passed to the model functio.
        #* Throws
        - `ErrorException`: If the `evaluate` function is not implemented for the specific type
        of `f`.
    ==========================================================================================#    
    function evaluate(f::AbstractModelFunction, args...)
        error("evaluate not implemented for $(typeof(f))")
    end
    #==========================================================================================
    #* get_parameters
        Retrieve the parameters of the model function `f`.
        #* Arguments
        - `f::AbstractModelFunction`: The model function whose parameters are to be retrieved.
        #* Throws
        - `ErrorException`: If the `get_parameters` function is not implemented for the specific 
        type of `f`.
    ==========================================================================================#
    function get_parameters(f::AbstractModelFunction)
        error("get_parameters not implemented for $(typeof(f))")
    end
    #==========================================================================================
    #* validate_parameters
        Validate the parameters of the model function `f`.
        #* Arguments
        - `f::AbstractModelFunction`: The model function whose parameters are to be validated.
        #* Throws
        - `ErrorException`: If the `validate_parameters` function is not implemented for the 
        specific type of `f`.
    ==========================================================================================#
    function validate_parameters(f::AbstractModelFunction)
        error("validate_parameters not implemented for $(typeof(f))")
    end
    #?=========================================================================================
    #? ProductionFunction: Represents production technology
    #? - Production function that captures the relationship between worker skill, remote work,
    #?   and firm remote efficiency.
    #? - Production specific methods:
    #?   - None
    #? - Components:
    #?   - ProductivityComponent: Worker skill productivity A(h)
    #?   - RemoteEfficiencyComponent: Remote work efficiency for a firm-worker pair g(ψ, h)
    #?   - CompositeProduction: Y(h, α, ψ) = A(h) * ((1 - α) + α * g(ψ, h))
    #?=========================================================================================
    abstract type ProductionFunction <: AbstractModelFunction end
    #*=========================================================================================
    #* ProductionFunction Components:
    #*  - ProductivityComponent (Captures worker skill productivity)
    #*  - RemoteEfficiencyComponent (Captures remote work efficiency for a firm-worker pair)
    #*=========================================================================================
    abstract type ProductivityComponent <: AbstractModelFunction end
    abstract type RemoteEfficiencyComponent <: AbstractModelFunction end
    #==========================================================================================
    # ProductivityComponent.
    #  Available implementations:
    #  - LinearProductivity: A(h) = A₀ + A₁ * h
    ==========================================================================================#
    # > LinearProductivity
    @with_kw struct LinearProductivity <: ProductivityComponent
        A₀::Float64              # Base productivity level
        A₁::Float64              # Skill intensity of productivity
        # Constructor: Removed undefined 'η'
        function LinearProductivity(A₀::Float64, A₁::Float64)
            prod_comp = new(A₀, A₁)
            validate_parameters(prod_comp)
            return prod_comp
        end
    end # end definition
    function validate_parameters(f::LinearProductivity)
        if f.A₀ < 0.0
            error("A₀ must be non-negative")
        end
        if f.A₁ < 0.0
            error("A₁ must be non-negative")
        end
    end # end validate_parameters
    function evaluate(f::LinearProductivity, h::Float64)
        return f.A₀ + f.A₁ * h
    end # end evaluate
    #> List of available productivity components
    productivity_components = Dict(
        "LinearProductivity" => LinearProductivity
    )
    #==========================================================================================
    # RemoteEfficiencyComponent
    #  Available implementations:
    #  - LinearFirmLogWorker g(ψ, h) = ν * ψ - ψ₀ + ϕ * log(h)
    ==========================================================================================#
    # > LinearFirmLogWorker
    @with_kw struct LinearFirmLogWorker <: RemoteEfficiencyComponent
        ν::Float64              # Remote efficiency scaling factor of firm remote efficiency
        ϕ::Float64              # Remote efficiency scaling factor of worker skill
        ψ₀::Float64             # Threshold for remote efficiency to be positive for production
        # Constructor
        function LinearFirmLogWorker(ν::Float64, ϕ::Float64, ψ₀::Float64)
            remote_eff_comp = new(ν, ϕ, ψ₀)
            # Validate parameters
            validate_parameters(remote_eff_comp)
            # Return the instance if parameters are valid
            return remote_eff_comp
        end
    end # end definition
    function validate_parameters(f::LinearFirmLogWorker)
        if f.ν < 0.0
            error("ν must be non-negative")
        end
        if f.ϕ < 0.0
            error("ϕ must be non-negative")
        end
        if f.ψ₀ < 0.0
            error("ψ₀ must be non-negative")
        end
    end
    function evaluate(f::LinearFirmLogWorker, ψ::Float64, h::Float64)
        return ψ - f.ψ₀ + f.ϕ * log(h)
    end # end evaluate
    #> List of available remote efficiency components
    remote_efficiency_components = Dict(
        "LinearFirmLogWorker" => LinearFirmLogWorker
    )
    #==========================================================================================
    # Create a composite production function that combines productivity and remote efficiency
    ==========================================================================================#
    @with_kw struct CompositeProduction <: ProductionFunction
        productivity::ProductivityComponent
        remote_efficiency::RemoteEfficiencyComponent
    end # end definition
    function evaluate(f::CompositeProduction, h::Float64, α::Float64, ψ::Float64)
        A_h = evaluate(f.productivity, h)
        g_ψh = evaluate(f.remote_efficiency, ψ, h)
        return A_h * ((1 - α) + α * g_ψh)
    end # end evaluate
    function get_parameters(f::CompositeProduction)
        return (
            productivity_params = f.productivity,
            remote_efficiency_params = f.remote_efficiency
        )
    end # end get_parameters
    #?=========================================================================================
    #? UtilityFunction: Represents worker preferences over wage and remote work: x = U(w, α, h)
    #? - Utility specific methods:
    #?   - evaluate_derivative: Evaluate the derivative of the utility function with respect to
    #?     wage w, h or α. IF THE USER SUPPLIES THE DERIVATIVE FUNCTION, ELSE: error.
    #?   - find_implied_wage: Find the wage that implies a given utility level x and remote work
    #?     level α. (IF the user defines closed form solution, ELSE: numeric solution)
    #TODO: Lets add a property to the utility function to check if we have a formula for the wage and the derivative
    #TODO: If not formula for derivative define a numerical derivative using FiniteDifference.jl
    #?=========================================================================================
    abstract type UtilityFunction <: AbstractModelFunction end
    #TODO: Consider separating the utility function into two parts: wage and remote work preferences
    #==========================================================================================
    # evaluate_derivative
    ==========================================================================================#
    function evaluate_derivative(f::UtilityFunction, args...)
        error("evaluate_derivative not implemented for $(typeof(f))")
    end
    #==========================================================================================
    # find_implied_wage 
    ==========================================================================================#
    function find_implied_wage(f::UtilityFunction, x::Float64, α::Float64, h::Float64)
        obj_fun = (w::Float64) -> evaluate(f, w, α, h) - x
        # Since parameters guarantee that utility is increasing in wage, we can use a simple
        # bisection method to find the implied wage
        return find_zero(obj_fun, (0.001, 100.0), Bisection()) # TODO: Figure out how to pin down the bounds
    end
    #==========================================================================================
    # Available implementations:
    # - PolySeparable: U(w, α, h) = (a₀ + a₁ w^η_w) - (c₀ + c₁ * h^η_h) * (1 - α)^χ
    ==========================================================================================#
    # > QuasiLinearSkill
    @with_kw struct PolySeparable <: UtilityFunction
        a₀::Float64              # Base intensity of wage preference
        a₁::Float64              # Slope of wage preference
        η_w::Float64             # Curvature of wage preference
        c₀::Float64              # Base intensity of remote work preference
        c₁::Float64              # Skill intensity of remote work preference
        η_h::Float64             # Use this for curvature on h
        χ::Float64               # Curvature of remote work preference
        # Constructor
        function PolySeparable(a₀::Float64, a₁::Float64, η_w::Float64,
                c₀::Float64, c₁::Float64, η_h::Float64, χ::Float64)
            utility = new(a₀, a₁, η_w, c₀, c₁, η_h, χ)
            # Validate parameters
            validate_parameters(utility)
            # Return the instance if parameters are valid
            return utility
        end
    end # end definition
    function validate_parameters(f::PolySeparable)
        params = [f.a₀, f.a₁, f.η_w, f.c₀, f.c₁, f.η_h, f.χ]
        param_names = ["a₀", "a₁", "η_w", "c₀", "c₁", "η_h", "χ"]
        for (param, name) in zip(params, param_names)
            if param < 0.0
                error("$name must be non-negative")
            end
        end
        if f.η_h < 1
            error("η_h must be greater than or equal to 1")
        end
    end # end validate_parameters
    function evaluate(f::PolySeparable, w::Float64, α::Float64, h::Float64)
        # Replace f.η with f.η_h
        return  (a₀ + a₁ * w^η_w) - (f.c₀ + f.c₁ * h^f.η_h) * (1 - α)^f.χ
    end # end evaluate
    function evaluate_derivative(f::PolySeparable, argument::Union{String, Symbol}, 
                                w::Float64, α::Float64, h::Float64)
        # Evaluate the derivative of the utility function with respect to the given argument
        if argument == "w"
            return f.a₁ * f.η_w * w^(f.η_w - 1) 
        elseif argument == "α"
            return  f.χ * (f.c₀ + f.c₁ * h^f.η_h) * (1 - α)^(f.χ-1)
        elseif argument == "h"
            return -f.η_h * f.c₁ * h^(f.η_h - 1) * (1 - α)^f.χ
        else
            error("Invalid argument: $argument")
        end
    end # end evaluate_derivative
    function find_implied_wage(f::PolySeparable, x::Float64, α::Float64, h::Float64)
        return ((x + (f.c₀ + f.c₁ * h^f.η_h) * (1 - α)^f.χ - f.a₀ / f.a₁))^(1 / f.η_w)
    end # end find_implied_wage
    #> List of available utility functions
    utility_functions = Dict(
        "PolySeparable" => PolySeparable
    )
    #?=========================================================================================
    #? MatchingFunction: Represents labor market matching technology
    #? - Matching function that maps vacancies and unemployed workers to matches.
    #? - Fields:
    #?   - function parameters
    #?   - maxVacancyFillRate (maximum rate at which vacancies are filled)
    #? - Matching specific methods:
    #?   - JobFindRate: Evaluate the probability of finding a job given a sub-market tightness.
    #?   - VacancyFillRate: Evaluate the probability of filling a vacancy given a sub-market
    #?                      tightness.
    #?   - InvertVacancyFill: Solve the sub-market free entry condition for the tightness.
    #?=========================================================================================
    abstract type MatchingFunction <: AbstractModelFunction end
    # invert vacancy fill
    function invert_vacancy_fill(f::MatchingFunction, κ::Float64, Ej::Float64)
        # First, check submarket activity.
        if f.maxVacancyFillRate * Ej < κ
            return 0.0
        else
            # Define the objective function:
            # We want to solve: eval_prob_vacancy_fill(f, θ) * Ej - κ = 0.
            obj_fun = (θ::Float64) -> eval_prob_vacancy_fill(f, θ) * Ej - κ
            θ_low = 1e-8  # Lower bound
            # Pin down the bounds for bisection.
            θ_high = 1.0  # Initial guess for upper bound
            # Increase θ_high until the objective becomes negative.
            while obj_fun(θ_high) > 0
                θ_high *= 2.0
                if θ_high > 1e6
                    error("Could not bracket the solution for invert_vacancy_fill.")
                end
            end
            # Use the bisection method to find the zero.
            return find_zero(obj_fun, (θ_low, θ_high), Bisection())
        end
    end
    #==========================================================================================
    # Available implementations:
    # - CobbDouglasMatching: M(V, U) = U^γ * V^(1 - γ)
    # - CESMatching: M(V, U) = (V^γ + U^γ)^(1/γ)
    # - ExponentialMatching: M(V, U) = 1 - exp(-γ * θ)
    # - LogisticMatching: M(V, U) = (V * U )^γ / (V^γ + U^γ)
    ==========================================================================================#
    # > CobbDouglasMatching
    @with_kw struct CobbDouglasMatching <: MatchingFunction
        γ::Float64               # Matching elasticity
        maxVacancyFillRate::Float64  # Maximum rate at which vacancies are filled
        # Constructor
        function CobbDouglasMatching(γ::Float64)
            matching = new(γ, Inf)
            # Validate parameters
            validate_parameters(matching)
            # Return the instance if parameters are valid
            return matching
        end
    end # end definition
    function eval_prob_job_find(f::CobbDouglasMatching, θ::Float64)
        return θ^(1 - f.γ)
    end
    function eval_prob_vacancy_fill(f::CobbDouglasMatching, θ::Float64)
        return θ^(-f.γ)
    end
    function invert_vacancy_fill(f::CobbDouglasMatching, κ::Float64, Ej::Float64)
        # Removed call to undefined 'max_q(f)'; user must supply proper bounds.
        # If needed, add proper logic here.
        if f.maxVacancyFillRate * Ej < κ
            return 0.0
        else
            return (κ / Ej)^(-1 / f.γ)
        end
    end
    function validate_parameters(f::CobbDouglasMatching)
        if f.γ <= 0.0
            error("γ must be positive")
        end
    end # end validate_parameters
    # > CESMatching
    @with_kw struct CESMatching <: MatchingFunction
        γ::Float64   # CES parameter (γ > 0)
        maxVacancyFillRate::Float64  # Maximum rate at which vacancies are filled
        # Constructor
        function CESMatching(γ::Float64)
            # Compute the maximum vacancy fill rate
            matching = new(γ, Inf)
            validate_parameters(matching)
            return matching
        end
    end
    function eval_prob_job_find(f::CESMatching, θ::Float64)
        return θ * (θ^(f.γ) + 1)^(-1/f.γ)
    end
    function eval_prob_vacancy_fill(f::CESMatching, θ::Float64)
        return ((θ^(f.γ) + 1)^(-1/f.γ))
    end
    function validate_parameters(f::CESMatching)
        if f.γ <= 0.0
            error("γ must be positive")
        end
    end
    function invert_vacancy_fill(f::CESMatching, κ::Float64, Ej::Float64)
        if Ej > κ
            return  ( ( Ej / κ )^f.γ - 1  )^(1/f.γ)
        else
            return 0.0
        end
    end
    #> ExponentialMatching
    @with_kw struct ExponentialMatching <: MatchingFunction
        γ::Float64   # parameter controlling the arrival rate (γ > 0)
        maxVacancyFillRate::Float64  # Maximum rate at which vacancies are filled
        # Constructor
        function ExponentialMatching(γ::Float64)
            # Compute the maximum vacancy fill rate
            # q(θ) = [1 - exp(-γ θ)]/θ. As θ→0, use L'Hôpital: limit = γ.
            matching = new(γ, γ)
            validate_parameters(matching)
            return matching
        end
    end
    function eval_prob_job_find(f::ExponentialMatching, θ::Float64)
        # p(θ) = 1 - exp(-γ * θ)
        return 1 - exp(-f.γ * θ)
    end
    function eval_prob_vacancy_fill(f::ExponentialMatching, θ::Float64)
        # q(θ) = p(θ)/θ = [1 - exp(-γ * θ)] / θ
        return (1 - exp(-f.γ * θ)) / θ
    end
    function validate_parameters(f::ExponentialMatching)
        if f.γ <= 0.0
            error("γ must be positive")
        end
    end
    #> LogisticMatching
    @with_kw struct LogisticMatching <: MatchingFunction
        γ::Float64   # parameter controlling the curvature (γ > 0)
        maxVacancyFillRate::Float64  # Maximum rate at which vacancies are filled
        # Constructor: use input γ instead of f.γ for setting maxVacancyFillRate
        function LogisticMatching(γ::Float64)
            local mvr = if γ < 1
                Inf
            elseif isapprox(γ, 1.0)
                1.0
            else
                0.0
            end
            matching = new(γ, mvr)
            validate_parameters(matching)
            return matching
        end
    end
    function eval_prob_job_find(f::LogisticMatching, θ::Float64)
        # p(θ) = θ^(γ) / (1 + θ^(γ))
        return θ^(f.γ) / (1 + θ^(f.γ))
    end
    function eval_prob_vacancy_fill(f::LogisticMatching, θ::Float64)
        # q(θ) = p(θ)/θ = [θ^(γ) / (1 + θ^(γ))] / θ
        return (θ^(f.γ) / (1 + θ^(f.γ))) / θ
    end
    function validate_parameters(f::LogisticMatching)
        if f.γ <= 0.0
            error("γ must be positive")
        end
    end
    #> List of available matching functions
    matching_functions = Dict(
        "CobbDouglasMatching" => CobbDouglasMatching,
        "CESMatching" => CESMatching,
        "ExponentialMatching" => ExponentialMatching,
        "LogisticMatching" => LogisticMatching
    )
    #?=========================================================================================
    #? Initialization Functions
    #? - Create matching function
    #? - Create production function
    #? - Create utility function
    #?=========================================================================================
    #*=========================================================================================
    #* create_matching_function(type::String, params::Array{Float64})::MatchingFunction
    #* Description:
    #*  - Create a matching function of the specified type with the given parameters.   
    #* Parameters:
    #*  - type::String - The type of matching function to create (e.g., "CESMatching")
    #*  - params::Array{Float64} - Array of parameters for the matching function. ORDER MATTERS!
    #* Returns:
    #*  - MatchingFunction - An initialized matching function of the specified type
    #*=========================================================================================
    function create_matching_function(type::String, params::Array{Float64})::MatchingFunction
        if haskey(matching_functions, type)
            return matching_functions[type](params...)
        else
            # TODO: figure out how to handle user provided matching functions
            error("Unknown matching function type: $type")
        end
    end
    #*=========================================================================================
    #* create_utility_function(type::String, params::Array{Float64})::UtilityFunction
    #* Description:
    #*  - Create a utility function of the specified type with the given parameters.
    #* Parameters:
    #*  - type::String - The type of utility function to create (e.g., "QuasiLinear")
    #*  - params::Array{Float64} - Array of parameters for the matching function. ORDER MATTERS!
    #* Returns:
    #*  - UtilityFunction - An initialized utility function of the specified type
    #*=========================================================================================
    function create_utility_function(type::String, params::Array{Float64})::UtilityFunction
        if haskey(utility_functions, type)
            return utility_functions[type](params...)
        else
            # TODO: figure out how to handle user provided utility functions
            error("Unknown utility function type: $type")
        end
    end
    #*=========================================================================================
    #* create_production_function(type_productivity_component::String,
    #*                           type_remote_efficiency::String,
    #*                           params_productivity_component::Array{Float64},
    #*                           params_remote_efficiency::Array{Float64})::ProductionFunction
    #* Description:
    #*  - Create a composite production function with the specified productivity and remote
    #*    efficiency components and their parameters.
    #* Parameters:
    #*  - type_productivity_component::String - The type of productivity component to create
    #*  - type_remote_efficiency::String - The type of remote efficiency component to create
    #*  - params_productivity_component::Array{Float64} - Array of parameters for the productivity
    #*    component. ORDER MATTERS!
    #*  - params_remote_efficiency::Array{Float64} - Array of parameters for the remote efficiency
    #*    component. ORDER MATTERS!
    #* Returns:
    #*  - ProductionFunction - An initialized production function with the specified components
    #*=========================================================================================
    function create_production_function(type_productivity_component::String,
                                        type_remote_efficiency::String,
                                        params_productivity_component::Array{Float64},
                                        params_remote_efficiency::Array{Float64})::ProductionFunction
        #> Create the productivity component
        if haskey(productivity_components, type_productivity_component)
            productivity_component = productivity_components[type_productivity_component](params_productivity_component...)
        else
            # TODO: figure out how to handle user provided productivity components
            error("Unknown productivity component type: $type_productivity_component")
        end
        #> Create the remote efficiency component
        if haskey(remote_efficiency_components, type_remote_efficiency)
            remote_efficiency_component = remote_efficiency_components[type_remote_efficiency](params_remote_efficiency...)
        else
            #TODO: figure out how to handle user provided remote efficiency components
            error("Unknown remote efficiency component type: $type_remote_efficiency")
        end
        #> Create the composite production function
        return CompositeProduction(productivity_component, remote_efficiency_component)
    end
end # end module