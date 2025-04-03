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

# Set paths
function main(; verbose::Bool=true)
    config_path = "src/structural/search_model/revamped/parameters2.yaml"
    
    # Initialize model
    prim, res = initializeModel(config_path)
    
    # Solve the model with a single function call
    solve_model!(prim, res, n_iter_max=5000, tol=1e-6, verbose=verbose)
    
    return prim, res
end

prim, res = main(verbose=true);
model_plots = create_all_plots!("create", prim, res; verbose=true);
model_plots.job_finding_prob
model_plots.remote_work_policy
model_plots.productivity_density
model_plots.wage_and_remote
model_plots.worker_search_policy