#TODO: Document this module
module ModelPlots

using CairoMakie, LaTeXStrings, Term
using ..ModelFunctions, ..Types

export ModelPlotCollection, 
        create_all_plots!,
        save_plot,
        save_all_plots

# Constants for styling
const COLORS = [
    "#23373B",   # primary
    "#EB811B",   # secondary
    "#14B03E"    # tertiary
]
const FONT_CHOICE = "CMU Serif"
const FONT_SIZE_MANUSCRIPT = 14
const FONT_SIZE_PRESENTATION = 24
const FONT_SIZE = FONT_SIZE_PRESENTATION

# Skill levels for specific plots
const SKILL_LEVELS = [10, 40]
const SKILL_LABELS = ["Low Skill", "High Skill"]

# Plot collection struct
mutable struct ModelPlotCollection
    # Output directory
    output_dir::String
    
    # Figures
    productivity_density::Union{Figure, Nothing}
    remote_work_policy::Union{Figure, Nothing}
    remote_work_policy_detailed::Union{Figure, Nothing}
    job_finding_prob::Union{Figure, Nothing}
    worker_search_policy::Union{Figure, Nothing}
    wage_and_remote::Union{Figure, Nothing}
    
    # Constructor with default nothing values
    function ModelPlotCollection(output_dir::String="figures/model_figures/")
        # Ensure the output directory exists
        mkpath(output_dir)
        
        return new(
            output_dir,
            nothing, nothing, nothing, nothing, nothing, nothing
        )
    end
end

# Helper functions for plot creation
function create_figure(; type="normal")
    if type == "wide"
        size = (1200, 800)
        scale = 1.5
    elseif type == "tall"
        size = (800, 1200)
        scale = 1.5
    elseif type == "ultrawide"
        size = (1400, 600)
        scale = 1.5
    else
        size = (800, 600)
        scale = 1.0
    end
    
    return Figure(
        size = size, 
        fontsize = FONT_SIZE * scale,
        fonts = (; regular=FONT_CHOICE, italic=FONT_CHOICE, bold=FONT_CHOICE)
    )
end

function create_axis(where_, title, xlabel, ylabel)
    ax = Axis(
        # User parameters
        where_, 
        title = title, 
        xlabel = xlabel, 
        ylabel = ylabel,
        # Tick parameters
        # xticks
        xticklabelsize = FONT_SIZE,
        xtickalign = 1, 
        xticksize = 10, 
        # yticks
        yticklabelsize = FONT_SIZE,
        ytickalign = 1, 
        yticksize = 10, 
        # Grid parameters
        xgridvisible = false, 
        ygridvisible = false, 
        topspinevisible = false, 
        rightspinevisible = false
    )
    return ax
end

# Plot generation functions
function plot_productivity_density(prim::Primitives, res::Results)
    fig = create_figure()
    ax = create_axis(fig[1, 1], "Density of Remote Work Productivity", "Worker Skill", "Density")
    
    # Plot the density of remote work productivity
    lines!(
        ax, prim.ψ_grid.values, prim.ψ_grid.pdf, color=COLORS[1], linewidth=4, label=""
    )
    
    # Add vertical line for ψ₀
    ψ₀ = prim.production_function.remote_efficiency.ψ₀
    vlines!(ax, [ψ₀], color=:gray, linestyle=:dash, linewidth=3, label=L"\psi_0")
    
    # Set custom x-ticks with LaTeX labels
    ψ_min = prim.ψ_grid.min
    ψ_max = prim.ψ_grid.max
    ax.xticks = (
        [ψ₀, ψ_min, ψ_max],
        [
            latexstring("\\psi_0 = $(round(ψ₀, digits=2))"),
            latexstring("\\psi_{\\min} = $(round(ψ_min, digits=2))"),
            latexstring("\\psi_{\\max} = $(round(ψ_max, digits=2))")
        ]
    )
    
    xlims!(ax, ψ_min - 0.15, ψ_max + 0.3)
    ax.yticks = ([], [])  # Remove y-axis ticks
    
    return fig
end

function plot_remote_thresholds_by_worker_skill(prim::Primitives, res::Results; detailed=false)
    fig = create_figure(type="wide")
    ax = create_axis(
        fig[1, 1],
        "", 
        latexstring("Worker Skill \$(h)\$"), 
        latexstring("Remote Work Productivity \$(\\psi)\$")
    )
    
    if detailed
        hmap = heatmap!(
            ax, 
            prim.h_grid.values,
            prim.ψ_grid.values,
            res.α_policy', 
            colormap = :Egypt,
        )
        Colorbar(fig[1, 2], hmap; label = L"\alpha^*(\psi,h)", width = 20, ticksize = 15, tickalign = 1)
        colsize!(fig.layout, 1, Aspect(1, 1.2))
    else
        # Fill between the thresholds (hybrid remote work)
        band!(
            ax, 
            prim.h_grid.values, 
            res.ψ_bottom, 
            res.ψ_top,
            color=(COLORS[3], 0.2),  # Same color as top line but with transparency
            label="Hybrid Work"
        )

        # Fill below the bottom threshold (in-office work)
        fill_between!(
            ax, 
            prim.h_grid.values, 
            res.ψ_bottom, 
            prim.ψ_grid.min,  # Fill to the bottom of the plot
            color=(COLORS[2], 0.2),  # Same color as bottom line but with transparency
            label="Full In-Person"
        )

        # Fill above the top threshold (full remote work)
        fill_between!(
            ax, 
            prim.h_grid.values, 
            res.ψ_top, 
            prim.ψ_grid.max,  # Fill to the top of the plot
            color=(COLORS[1], 0.2),  # Same color as top line but with transparency
            label="Full Remote"
        )
        
        Legend(fig[1, 2], ax, title="Policy", width=180, box=:off, framevisible=false, labelsize=FONT_SIZE * 0.9)
    end
    
    # Plot the remote work efficiency thresholds
    ## Top threshold (ψ_top)
    lines!(
        ax, prim.h_grid.values, res.ψ_top, color=COLORS[1], linewidth=4,
    )
    ## Bottom threshold (ψ_bottom)
    lines!(
        ax, prim.h_grid.values, res.ψ_bottom, color=COLORS[2], linewidth=4, 
    )
    
    # Set the limits of the y-axis
    ylims!(ax, prim.ψ_grid.min, prim.ψ_grid.max)
    
    # Remove spines and ticks
    ax.xticks = ([], [])  # Remove x-axis ticks
    ax.yticks = ([], [])  # Remove y-axis ticks
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.bottomspinevisible = false
    ax.leftspinevisible = false

    return fig
end

function plot_job_finding_prob(prim::Primitives, res::Results)
    fig = create_figure(type="wide")
    ax = create_axis(
        fig[1, 1],
        "Probability of Job Offer", 
        latexstring("Promised Utility (\$x\$)"),
        L"p(\theta(x,h))")

    job_skill_levels = [10, 30, 45]
    job_skill_labels = ["Low Skill", "Medium Skill", "High Skill"]

    p = θ -> eval_prob_job_find(prim.matching_function, θ)

    for (skill_i, skill) in enumerate(job_skill_levels)
        lines!(
            ax, prim.x_grid.values, p.(res.θ[skill, :]), color=COLORS[skill_i], linewidth=4, label=job_skill_labels[skill_i]
        )
    end
    
    # Set the limits of the y-axis
    ylims!(ax, 0, 1.0)
    
    # Configure ticks
    ax.yticks = (0:0.2:1)
    ax.yticklabelsize = 18      # Make y-tick labels smaller
    ylims!(ax, 0, 1.0)
    ax.xticks = ([prim.x_grid.min, prim.x_grid.max], [L"x_{min}", L"x_{max}"])
    
    # Add legend to the plot
    Legend(fig[1, 2], ax, title="Skill Level", width=180, box=:off, framevisible=false, labelsize=FONT_SIZE * 0.9)
    
    return fig
end

function plot_worker_search_policy(prim::Primitives, res::Results)
    fig = create_figure()
    ax = create_axis(
        fig[1, 1],
        "Unemployed Worker Search Policy", 
        latexstring("Worker Skill \$(h)\$"),
        latexstring("Promised Utility \$(x)\$"),
    )
    
    lines!(
        ax, 
        prim.h_grid.values,
        [prim.x_grid.values[i] for i in res.x_policy],
        color=COLORS[1], linewidth=4, label=""
    )
    
    # Set the limits of the y-axis
    ylims!(ax, prim.x_grid.min, prim.x_grid.max + 0.3)
    ax.yticks = ([prim.x_grid.min, prim.x_grid.max], [L"x_{min}", L"x_{max}"])
    ax.xticks = ([prim.h_grid.values[1], prim.h_grid.values[end]], [L"h_{min}", L"h_{max}"])
    
    return fig
end

function plot_wage_and_remote(prim::Primitives, res::Results)
    fig = create_figure(type="ultrawide")
    
    # Create axis for expected wages
    ax1 = create_axis(
        fig[1, 1],
        "Expected Wages",
        latexstring("Worker Skill \$(h)\$"),
        L"\mathbb{E}w^*(h)")
    
    # Compute expected wages
    Ewages = sum(hcat([res.w_policy[:,i_h, res.x_policy[i_h]] for (i_h, _) in enumerate(prim.h_grid.values)]...) .* prim.ψ_grid.pdf, dims=1)
    
    # Plot expected wages
    lines!(
        ax1, 
        prim.h_grid.values,
        Ewages[:],
        color=COLORS[1], linewidth=4, label=""
    )
    
    # Make remote work policy axis
    ax2 = create_axis(
        fig[1, 2],
        "Expected Remote Work",
        latexstring("Worker Skill \$(h)\$"),
        L"\mathbb{E}\alpha^*(h)")
    
    # Compute expected remote work policy
    Eα = sum(res.α_policy .* prim.ψ_grid.pdf, dims = 1)
    
    # Plot expected remote work policy
    lines!(
        ax2,
        prim.h_grid.values,
        Eα[:],
        color=COLORS[2], linewidth=4, label="Expected Remote Work Policy"
    )
    
    ax1.xticks = ([prim.h_grid.values[1], prim.h_grid.values[end]], [L"h_{min}", L"h_{max}"])
    ax2.xticks = ([prim.h_grid.values[1], prim.h_grid.values[end]], [L"h_{min}", L"h_{max}"])
    
    # Set the limits of the axis
    ylims!(ax2, 0, 1.0)

    return fig
end

# Function to create all plots
function create_all_plots!(plots::Union{ModelPlotCollection, String}, prim::Primitives, res::Results; verbose::Bool=true)
    if verbose println(@bold @blue "Creating model plots...") end
    # If the input is a string, create a new ModelPlotCollection
    if typeof(plots) == String && lowercase(plots) ∈ ["new", "create"]
        plots = ModelPlotCollection()
        if verbose
            println(@bold @green "Created new ModelPlotCollection with output directory $(plots.output_dir)")
        end
    end
    
    # Create all plots
    plots.productivity_density = plot_productivity_density(prim, res)
    plots.remote_work_policy = plot_remote_thresholds_by_worker_skill(prim, res)
    plots.remote_work_policy_detailed = plot_remote_thresholds_by_worker_skill(prim, res, detailed=true)
    plots.job_finding_prob = plot_job_finding_prob(prim, res)
    plots.worker_search_policy = plot_worker_search_policy(prim, res)
    plots.wage_and_remote = plot_wage_and_remote(prim, res)
    
    if verbose println(@bold @blue "All plots created successfully.") end
    
    return plots
end

# Function to save a specific plot
function save_plot(plots::ModelPlotCollection, plot_name::Symbol; verbose::Bool=true)
    # Get the figure from the plots collection
    fig = getproperty(plots, plot_name)
    
    if isnothing(fig)
        if verbose println(@bold @red "Plot $plot_name not found or not created yet.") end
        return false
    end
    
    # Create filename
    filename = string(plot_name) * ".pdf"
    filepath = joinpath(plots.output_dir, filename)
    
    # Save the figure
    save(filepath, fig)
    
    if verbose println(@bold @green "Saved $plot_name to $filepath") end
    
    return true
end

# Function to save all plots
function save_all_plots(plots::ModelPlotCollection; verbose::Bool=true)
    if verbose println(@bold @blue "Saving all plots to $(plots.output_dir)...") end
    
    # Get all field names except output_dir
    plot_fields = filter(f -> f != :output_dir, fieldnames(ModelPlotCollection))
    
    # Save each plot
    for field in plot_fields
        save_plot(plots, field, verbose=verbose)
    end
    
    if verbose println(@bold @blue "All plots saved successfully.") end
end

end # module ModelPlots