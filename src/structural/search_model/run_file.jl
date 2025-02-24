import Term: install_term_stacktrace
install_term_stacktrace() 


include("model_plots.jl")  # Import the module file
using .ModelPlots  # Import the module

# @time begin 
include("model.jl")
parameters_path = "src/structural/search_model/parameters.yaml"
psi_distribution_data_path = "/project/high_tech_ind/WFH/WFH/data/results/data_density_3d.csv"
prim, res = initializeModel(parameters_path, psi_distribution_data_path);
compute_optimal_α!(prim, res);

iterateValueMatch(prim, res)
solveSubMarketTightness!(prim, res);
iterateW_U(prim, res)

using Plots
plot(prim.worker_skill, res.ψ_top, label="Top", lw = 3, xlabel="Worker Skill", ylabel="Firm Remote Efficiency")
ylims!(prim.ψ_grid[1], prim.ψ_grid[end])
plot!(prim.worker_skill, res.ψ_bottom, label="Bottom", lw = 3)

heatmap(prim.worker_skill,
        prim.ψ_grid,
        res.α_policy, 
        xlabel="Worker Skill",
        ylabel="Remote Productivity",
        title="Optimal Remote Work Policy")


heatmap(
        prim.worker_skill,
        prim.x_grid,
        prim.p.(res.θ'), 
        xlabel="Worker Skill",
        ylabel="x",
        title="p(θ)")




# Model Plots
plots = ModelPlots.plot_all(prim, res);
# Density of remote work productivity
plots[:density]
# Optimal remote work
plots[:optimal_α]
# Optimal wage policy
plots[:wage_policy]
# Firm value (x continuous ψ discrete)
plots[:firm_value_x]
# Firms value (ψ continuous x discrete)
plots[:firm_value_ψ]
# Market tightness
plots[:market_tightness]
# Job finding rate
plots[:finding_rate]
# Worker value functions
plots[:unemployed_value]
# Unemployed Worker search policy
plots[:unemployed_search_pol]
# Employed Worker value (x continuous ψ discrete)
plots[:employed_value_x]

plot(
        prim.x_grid,
        prim.p.(res.θ[10, :]),
        lw = 3,
        label = "Worker Skill = 10",
        xlabel = "Promised Utility (x)",
        ylabel = "Probability of Job Offer (p(θ))",
)
ylims!(0, 1.0)
plot!(
        prim.x_grid,
        prim.p.(res.θ[40, :]),
        lw = 3,
        label = "Worker Skill = 40"
)
plot!(
        prim.x_grid,
        prim.p.(res.θ[30, :]),
        lw = 3,
        label = "Worker Skill = 30"
)
plot!(
        prim.x_grid,
        prim.p.(res.θ[20, :]),
        lw = 3,
        label = "Worker Skill = 20"
)