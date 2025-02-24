using StatsBase, DataFrames, Distributions
# ----------------------------
# Worker Structure
# ----------------------------
mutable struct Worker
    id          ::String            # Worker unique identifier
    skill       ::Int               # Worker skill value (index corresponding to model skill grid)
    age         ::Int               # Worker age (number of periods in the simulation)
    employed    ::Vector{Bool}      # Employment status history (true if employed)
    wage        ::Vector{Float64}   # Wage history
    x_choice    ::Int               # Worker search policy if unemployed (index in the x_policy grid)
    firm_type   ::Vector{Float64}   # Employer indicator/history (e.g., index from firm remote productivity grid)
    remote_work ::Vector{Float64}   # Remote work fraction α history (if employed)

    function Worker(id::String, skill::Int; 
                    init_employed=[false], 
                    init_wage=[0.0], 
                    x_choice=0, 
                    firm_type=[NaN], 
                    remote_work=[NaN])
        new(id, skill, 0, init_employed, init_wage, x_choice, firm_type, remote_work)
    end
end
# ----------------------------
# Economy Structure
# ----------------------------
mutable struct Economy
    N                   ::Int                      # Number of workers
    T                   ::Int                      # Number of periods to simulate
    seed                ::Int                      # Random seed
    workers             ::Vector{Worker}           # Vector of workers

    # Model objects from the block–recursive solution
    # (We assume these come from your Primitives/Results objects)
    θ                   ::Array{Float64,2}         # Sub-market tightness, indexed by (skill, x)
    ψ_grid              ::Vector{Float64}          # Firm type grid (1D)
    ψ_pdf               ::Vector{Float64}          # PDF for sampling firm types
    α_policy            ::Array{Float64,2}         # Optimal remote work fraction as a function of (firm type, worker skill)
    x_grid              ::Vector{Float64}          # Utility grid from the model

    # Functions and parameters
    p                   ::Function                 # Match probability function p(θ)
    wage_fun            ::Function                 # Function to compute wage: w(x, α)
    
    δ_bar               ::Float64                  # Exogenous job destruction rate

    # Time-series aggregates
    job_finding         ::Vector{Float64}          # Aggregate job finding rate over time
    unemployment_rate   ::Vector{Float64}          # Aggregate unemployment rate over time
    remote_work         ::Vector{Float64}          # Aggregate remote work fraction over time
end
# ----------------------------
# initEconomy Function
# ----------------------------
function initEconomy(N::Int, T::Int, seed::Int, prim::Primitives, res::Results, worker_skill_dist)
    # Unpack primitives (ensure proper commas between names!)
    @unpack x_grid, ψ_grid, ψ_pdf, p, c0, χ, Y, δ_bar, n_h = prim
    # Unpack results
    @unpack θ, x_policy, α_policy, W, U = res

    # Create workers vector. Make sure to pass a valid skill.
    workers = Vector{Worker}(undef, N)
    worker_skill_dist_pdf = pdf.(worker_skill_dist, prim.worker_skill)
    for i in 1:N
        id_str = "W" * lpad(string(i), 4, '0')
        # Draw skill from the worker skill distribution
        skill = sample(1:n_h, Weights(worker_skill_dist_pdf))
        # Get the optimal search policy for this skill (x_policy is assumed to be indexed by skill)
        x_choice = x_policy[skill]
        # Construct the worker. (Do not use type annotations in the call.)
        workers[i] = Worker(id_str, skill; x_choice = x_choice)
    end

    # Initialize time-series aggregates as empty vectors
    job_finding       = Float64[]
    unemployment_rate = Float64[]
    remote_work       = Float64[]
    # Define wage function: wage = x + c*(1-α)^χ
    wage_fun = (x, α) -> x + c0 * (1 - α)^χ

    # Initialize the economy. (Note: θ, x_policy, α_policy come from res.)
    return Economy(
        N, T, seed, workers,
        θ,                    # Matrix of sub-market tightness (indexed by skill and x)
        ψ_grid,               # 1D firm type grid
        ψ_pdf,                # PDF for firm type sampling
        α_policy,             # Matrix of optimal remote work fraction [ψ_index, skill]
        x_grid,               # Utility grid
        p, wage_fun, δ_bar,
        job_finding, unemployment_rate, remote_work
    )
end
# ----------------------------
# Helper Function: carry_over!
# ----------------------------
# This function simply pushes the last–period values into the history if no change occurs.
function carry_over!(wkr::Worker)
    push!(wkr.employed, wkr.employed[end])
    push!(wkr.wage, wkr.wage[end])
    push!(wkr.firm_type, wkr.firm_type[end])
    push!(wkr.remote_work, wkr.remote_work[end])
end
# ----------------------------
# update_worker! Function
# ----------------------------
function update_worker!(wkr::Worker, econ::Economy)
    # Increase worker's age by 1
    wkr.age += 1
    if wkr.employed[end]
        # Employed branch: check for layoff
        if rand() < econ.δ_bar
            # Worker is laid off: transition to unemployment
            push!(wkr.employed, false)
            push!(wkr.wage, 0.0)
            push!(wkr.firm_type, NaN)
            push!(wkr.remote_work, NaN)
        else
            # Remains employed: carry over last period's values
            carry_over!(wkr)
        end
    else
        # Unemployed branch:
        # Use the worker's skill to choose a submarket
        # (x_policy is assumed to be a vector indexed by skill)
        # Obtain market tightness for worker's skill and chosen submarket
        θ_val = econ.θ[wkr.skill, wkr.x_choice]
        # Compute match probability
        match_prob = econ.p(θ_val)
        if rand() < match_prob
            # Worker finds a job:
            # Sample a firm type index from the firm grid using the PDF weights.
            ψ_index = sample(1:length(econ.ψ_grid), Weights(econ.ψ_pdf))
            # Look up optimal remote work fraction: note that α_policy is indexed by (firm type, worker skill)
            α = econ.α_policy[ψ_index, wkr.skill]
            # Compute wage using the chosen submarket x_choice and remote fraction α
            w_new = econ.wage_fun( econ.x_grid[wkr.x_choice], α)
            push!(wkr.employed, true)
            push!(wkr.remote_work, α)
            push!(wkr.wage, w_new)
            push!(wkr.firm_type, econ.ψ_grid[ψ_index])
            # Optionally, update x_choice if workers update their search target on finding a job.
        else
            # Worker remains unemployed: simply carry over last period's values
            carry_over!(wkr)
        end
    end
end
# ----------------------------
# update_economy! Function
# ----------------------------
function update_economy!(econ::Economy)
    unemployed_count = 0
    job_finders = 0
    remote_work_sum = 0.0

    for wkr in econ.workers
        # Record old employment status
        old_emp = wkr.employed[end]
        update_worker!(wkr, econ)
        new_emp = wkr.employed[end]
        if new_emp && !isnan(wkr.remote_work[end])
            remote_work_sum += wkr.remote_work[end]
        end
        if !old_emp
            unemployed_count += 1
            if new_emp
                job_finders += 1
            end
        end
    end
    # Compute aggregates over the economy
    total_workers = econ.N
    current_unemp = sum(!wkr.employed[end] for wkr in econ.workers)
    unemp_rate_t = current_unemp / total_workers

    num_employed = total_workers - current_unemp
    remote_work_t = num_employed > 0 ? remote_work_sum / num_employed : 0.0

    job_find_rate_t = unemployed_count > 0 ? job_finders / unemployed_count : 0.0

    push!(econ.unemployment_rate, unemp_rate_t)
    push!(econ.job_finding, job_find_rate_t)
    push!(econ.remote_work, remote_work_t)
end
# ----------------------------
# simulate! Function
# ----------------------------
function simulate!(econ::Economy)
    println("Simulating for T = $(econ.T) periods.")
    for t in 1:econ.T
        update_economy!(econ)
        if t % 50 == 0
            println("Period: $t | Unemp Rate: $(round(econ.unemployment_rate[end], digits=3)) | Job Find Rate: $(round(econ.job_finding[end], digits=3)) | Remote Work: $(round(econ.remote_work[end], digits=3))")
        end
    end
    println("Simulation complete!")
end


# ----------------------------
# get_results Function
# ----------------------------
function get_results(econ::Economy; burn_in=0)
    all_ids       = String[]
    all_skill     = Int[]
    all_age       = Int[]
    all_periods   = Int[]
    all_employed  = Bool[]
    all_wages     = Float64[]
    all_submarket = Int[]       # if x_choice is an index, we record it as integer
    all_firm_type = Float64[]
    all_remote    = Float64[]

    for wkr in econ.workers
        for t in (burn_in+1):length(wkr.employed)
            push!(all_ids, wkr.id)
            push!(all_skill, wkr.skill)
            push!(all_age, wkr.age)
            push!(all_periods, t)
            push!(all_employed, wkr.employed[t])
            push!(all_wages, wkr.wage[t])
            push!(all_submarket, wkr.x_choice)  # x_choice remains fixed per worker in this implementation
            push!(all_firm_type, wkr.firm_type[t])
            push!(all_remote, wkr.remote_work[t])
        end
    end

    return DataFrame(
        id = all_ids,
        skill = all_skill,
        age = all_age,
        period = all_periods,
        employed = all_employed,
        wage = all_wages,
        x_choice = all_submarket,
        firm_type = all_firm_type,
        remote_work = all_remote
    )
end
# Example usage
# Initialize the economy
N = 1000
T = 100
seed = 123
#TODO: Make this distribution part flexible
worker_skill_dist = Normal(0.40887307992437705, 0.16869535321076126)
economy = initEconomy(N, T, seed, prim, res,  worker_skill_dist)
simulate!(economy)
results = get_results(economy, burn_in=10)

using StatsPlots
using Plots

# Filter out rows where remote_work is NaN
filtered_results = filter(row -> !isnan(row.remote_work), results)

@df filtered_results Plots.density(:remote_work, :wage, title="Distribution of wage")
@df filtered_results Plots.histogram(:remote_work, :remote_work, title="Distribution of remote work")
@df filtered_results Plots.density(:remote_work, :firm_type, title="Distribution of productivity")


# Group by skill and compute the mean remote work
grouped_results = groupby(filtered_results, :skill)
mean_remote_work_by_skill = combine(grouped_results, :remote_work => mean => :mean_remote_work)
mean_wage_work_by_skill = combine(grouped_results, :wage => mean => :mean_wage)
@df mean_remote_work_by_skill Plots.bar(:skill, :mean_remote_work, title="Mean remote work by skill")
@df mean_wage_work_by_skill Plots.bar(:skill, :mean_wage, title="Mean wage by skill")


filtered_results.remote_work = round.(filtered_results.remote_work, digits=1)
grouped_by_remote_skill = groupby(filtered_results, [:remote_work, :skill])
mean_wage_by_remote_skill = combine(grouped_by_remote_skill, :wage => mean => :mean_wage)
@df mean_wage_by_remote_skill Plots.density(:skill, :mean_wage, group=:remote_work, title="Mean wage by remote work and skill", lw = 3)

prim.A / (prim.c * prim.χ)