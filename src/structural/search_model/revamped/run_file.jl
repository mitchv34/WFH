import Term: install_term_stacktrace

using Term
install_term_stacktrace()

# Include the modules
include("./model_functions.jl")
include("./types.jl")
include("./model_solution.jl")
include("./plots_search_model.jl")

using .Types
using .ModelFunctions
using .ModelSolution
using .ModelPlots

# function main(; verbose::Bool=true)
#     config_path = "src/structural/search_model/revamped/parameters.yaml"
    
#     # Initialize model
#     prim, res = initializeModel(config_path)
#     print(prim.h_grid.values')
#     # Solve the model with a single function call
#     solve_model!(prim, res, n_iter_max=1, tol=1e-6, verbose=verbose)
    
#     return prim, res
# end

update_parameters!(prim, [[:utility_function, :a₀]], [20.5])

# prim, res = main(verbose=true);
model_plots = create_all_plots!("create", prim, res; verbose=true);
model_plots.job_finding_prob
model_plots.remote_work_policy
model_plots.remote_work_policy_detailed
model_plots.productivity_density
model_plots.wage_and_remote
model_plots.worker_search_policy
model_plots.worker_type_distribution

simResults_df = simulate(prim, res, simParams, simShocks; return_dataFrame=true);
# Get moments from the simulation
simulated_moments = combine(groupby(simResults_df, :Period)) do df
    DataFrame(
        Period = first(df.Period),
        # - Unemployment rate
        UnemploymentRate = mean(df.Employment .== 0),
        # - Job finding rate
        JobFindingRate = mean(df.Transition .== 1),
        # - Job separation rate
        JobSeparationRate = mean(df.Transition .== 2),
        # - Wage distribution (only for employed workers)
        ## - Mean Wage
        MeanWage = isempty(df.Wage[df.Employment .!= 0]) ? NaN : mean(df.Wage[df.Employment .!= 0]),
        ## - Median Wage
        MedianWage = isempty(df.Wage[df.Employment .!= 0]) ? NaN : median(df.Wage[df.Employment .!= 0]),
        ## - Wage 9010 percentile
        Wage9010Percentile = isempty(df.Wage[df.Employment .!= 0]) ? NaN : quantile(df.Wage[df.Employment .!= 0], 0.9) - quantile(df.Wage[df.Employment .!= 0], 0.1),
        ## - Wage 7525 percentile
        Wage7525Percentile = isempty(df.Wage[df.Employment .!= 0]) ? NaN : quantile(df.Wage[df.Employment .!= 0], 0.75) - quantile(df.Wage[df.Employment .!= 0], 0.25),
        ## - Wage Standard Deviation
        WageStandardDeviation = isempty(df.Wage[df.Employment .!= 0]) ? NaN : std(df.Wage[df.Employment .!= 0]),
        # - Remote work distribution (only for employed workers)
        ## - Mean Remote Work
        MeanRemoteWork = isempty(df.Remote_Work[df.Employment .!= 0]) ? NaN : mean(df.Remote_Work[df.Employment .!= 0]),
    )
end


using StatsPlots, Plots
histogram(prim.h_grid.values[simShocks.worker_types], title="Worker Type Distribution", xlabel="Worker Types", ylabel="Frequency")
@df simulated_moments plot(:Period, :UnemploymentRate, lw = 2, title="Unemployment Rate Over Time", xlabel="Period", ylabel="Unemployment Rate")
@df simulated_moments plot(:Period, :JobFindingRate, lw = 2, title="Job Finding Rate Over Time", xlabel="Period", ylabel="Job Finding Rate")
@df simulated_moments plot(:Period, :JobSeparationRate, lw = 2, title="Job Separation Rate Over Time", xlabel="Period", ylabel="Job Separation Rate")
@df simulated_moments plot(:Period, :MeanWage, lw = 2, label="Mean Wage", title="Wage Trends Over Time", xlabel="Period", ylabel="Wage")
@df simulated_moments plot!(:Period, :MedianWage, lw = 2, label="Median Wage")
@df simulated_moments plot(:Period, :WageStandardDeviation, lw = 2, label="Wage Standard Deviation", title="Wage Dispersion Over Time", xlabel="Period", ylabel="Standard Deviation")
@df simulated_moments plot(:Period, :Wage9010Percentile, lw = 2, label="Wage 90-10 Percentile", title="Wage Percentile Trends Over Time", xlabel="Period", ylabel="Wage Percentile Difference")
@df simulated_moments plot!(:Period, :Wage7525Percentile, lw = 2, label="Wage 75-25 Percentile")
@df simulated_moments plot(:Period, :MeanRemoteWork, lw = 2, label="Mean Remote Work", title="Mean Remote Work Over Time", xlabel="Period", ylabel="Mean Remote Work")