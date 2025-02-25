using CairoMakie, LaTeXStrings

# const COLOR1 = "#23373B"
# const FONT_CHOICE = "CMU Serif"
COLORS = [
    "#23373B",   # primary
    "#EB811B", # secondary
    "#14B03E" # tertiary
]
FONT_CHOICE = "CMU Serif"
FONT_SIZE_MANUSCRIPT = 14
FONT_SIZE_PRESENTATION = 24
FONT_SIZE = FONT_SIZE_PRESENTATION

FOLDER_TO_SAVE = "figures/model_figures/"

# primary: rgb(129, 249, 14),
# primary-light: rgb(199, 249, 150),
# secondary: rgb(230, 255, 230),
# neutral-lightest: rgb(8, 10, 8),
# neutral-dark: rgb(230, 255, 230),
# neutral-darkest: rgb(230, 255, 230)
SKILL_LEVELS = [10, 40];
SKILL_LABELS = ["Low Skill", "High Skill"];

function create_figure(;type="normal")
    if type == "wide"
        size = (1200, 800)
        scale = 1.5
    elseif type == "tall"
        size = (800, 1200)
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
        xticklabelsize = FONT_SIZE ,
        xtickalign = 1, 
        xticksize = 10, 
        # yticks
        yticklabelsize = FONT_SIZE ,
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

function plot_productivity_density(prim, res)
    fig = create_figure()
    ax = create_axis(fig[1, 1], "Density of Remote Work Productivity", "Worker Skill", "Density")
    # Plot the density of remote work productivity
    lines!(
        ax, prim.ψ_grid, prim.ψ_pdf, color=COLORS[1], linewidth=4, label=""
    )
    # Add vertical line for ψ₀
    vlines!(ax, [prim.ψ₀], color=:gray, linestyle=:dash, linewidth=3, label=L"\psi_0")
    # Add vertical line the ψ_bottom for a high skill worker and a low skill worker
    # for (skill_i, skill) in enumerate(SKILL_LEVELS)
    #     vlines!(ax, [res.ψ_bottom[skill]], color=COLORS[skill_i + 1], linestyle=:dash, linewidth=3, label=SKILL_LABELS[skill_i])
    # end
    # Set custom x-ticks with LaTeX labels
    # ax.xticks = (
    #     [prim.ψ₀, res.ψ_bottom[SKILL_LEVELS[1]], res.ψ_bottom[SKILL_LEVELS[2]]],
    #     [L"\psi_0", L"\psi_{\mathrm{low}}", L"\psi_{\mathrm{high}}"]
    # )
    ax.xticks = (
        [prim.ψ₀, prim.ψ_grid[1], prim.ψ_grid[end]],
        [L"\psi_0", latexstring("\\psi_{\\min} = $( round( prim.ψ_grid[1], digits=2 ) )"), latexstring("\\psi_{\\max} = $( round( prim.ψ_grid[end], digits=2 ) )")]
    )
    xlims!(ax, prim.ψ_grid[1] - 0.15, prim.ψ_grid[end]+0.3)
    ax.yticks = ([], [])  # Remove y-axis ticks
    return fig
end
fig_prod_density = plot_productivity_density(prim, res)
# Save to pdf to include in the manuscript
save(joinpath(FOLDER_TO_SAVE, "productivity_density.pdf"), fig_prod_density)

function plot_remote_thresholds_by_worker_skill(prim, res; detailed=false)
    fig = create_figure(type="wide")
    ax = create_axis(
        fig[1, 1],
        "", 
        latexstring("Worker Skill \$(h)\$"), 
        latexstring("Remote Work Productivity \$(\\psi)\$")
    )
    
    # Plot the remote work efficiency thresholds
    ## Top threshold (ψ_top)
    lines!(
        ax, prim.worker_skill, res.ψ_top, color=COLORS[1], linewidth=4,
    )
    ## Bottom threshold (ψ_bottom)
    lines!(
        ax, prim.worker_skill, res.ψ_bottom, color=COLORS[2], linewidth=4, 
    )
    if detailed
        hmap = heatmap!(
            ax, 
            prim.worker_skill,
            prim.ψ_grid,
            res.α_policy', 
            colormap = :navia,
        )
        Colorbar(fig[1, 2], hmap; label = L"\alpha^*(\psi,h)", width = 20, ticksize = 15, tickalign = 1)
        colsize!(fig.layout, 1, Aspect(1, 1.2))
        # colgap!(fig.layout, 7)
    else

        # Fill between the thresholds (hybrid remote work)
        band!(
            ax, 
            prim.worker_skill, 
            res.ψ_bottom, 
            res.ψ_top,
            color=(COLORS[3], 0.2),  # Same color as top line but with transparency
            label="Hybrid Work"
        )

        # Fill below the bottom threshold (in-office work)
        fill_between!(
            ax, 
            prim.worker_skill, 
            res.ψ_bottom, 
            prim.ψ_grid[1],  # Fill to the bottom of the plot
            color=(COLORS[2], 0.2),  # Same color as bottom line but with transparency
            label="Full In-Person"
        )

        # Fill above the top threshold (full remote work)
        fill_between!(
            ax, 
            prim.worker_skill, 
            res.ψ_top, 
            prim.ψ_grid[end],  # Fill to the top of the plot
            color=(COLORS[1], 0.2),  # Same color as top line but with transparency
            label="Full Remote"
        )
        Legend(fig[1, 2], ax, title="Policy", width=180, box=:off, framevisible=false, labelsize=FONT_SIZE * 0.9)
        # colsize!(fig.layout, 1, Aspect(1, 1.0))
        # colgap!(fig.layout, 7)
    end
    # Set the limits of the y-axis
    ylims!(ax, prim.ψ_grid[1], prim.ψ_grid[end])
    # Remove spines and ticks
    ax.xticks = ([], [])  # Remove x-axis ticks
    ax.yticks = ([], [])  # Remove y-axis ticks
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.bottomspinevisible = false
    ax.leftspinevisible = false

    return fig
end
fig_remote_work_policy = plot_remote_thresholds_by_worker_skill(prim, res)
fig_remote_work_policy_detailed = plot_remote_thresholds_by_worker_skill(prim, res, detailed=true)
# Save to pdf to include in the manuscript
save(joinpath(FOLDER_TO_SAVE, "remote_work_policy.pdf"), fig_remote_work_policy)
save(joinpath(FOLDER_TO_SAVE, "remote_work_policy_detailed.pdf"), fig_remote_work_policy_detailed)

function plot_job_finding_prob(prim, res)
    fig = create_figure(type="wide")
    ax = create_axis(
        fig[1, 1],
        "Probability of Job Offer", 
        latexstring("Promised Utility (\$x\$)"),
        L"p(\theta(x,h))")

    SKILL_LEVELS = [10, 30, 45];
    SKILL_LABELS = ["Low Skill", "Medium Skill", "High Skill"];

    for (skill_i, skill) in enumerate(SKILL_LEVELS)
        lines!(
            ax, prim.x_grid, prim.p.(res.θ[skill, :]), color=COLORS[skill_i], linewidth=4, label=SKILL_LABELS[skill_i]
        )
    end
    # Set the limits of the y-axis
    ylims!(ax, 0, 1.0)
    # Remove spines and ticks
    ax.yticks = (0:0.2:1)
    ax.yticklabelsize = 18      # Make y-tick labels smaller
    ylims!(ax, 0, 1.0)
    ax.xticks = ([prim.x_grid[1], prim.x_grid[end]], [L"x_{min}", L"x_{max}"])  # Remove x-axis ticks
    # Add legend to the plot
    Legend(fig[1, 2], ax, title="Skill Level", width=180, box=:off, framevisible=false, labelsize=FONT_SIZE * 0.9)
    
    return fig
end
fig_job_finding_prob = plot_job_finding_prob(prim, res)
# Save to pdf to include in the manuscript
save(joinpath(FOLDER_TO_SAVE, "job_finding_prob.pdf"), fig_job_finding_prob)


function plot_worker_search_policy(prim, res)
    fig = create_figure()
    ax = create_axis(
        fig[1, 1],
        "Unemployed Worker Search Policy", 
        latexstring("Worker Skill \$(h)\$"),
        latexstring("Promised Utility \$(x)\$"),
    )
    lines!(
        ax, 
        prim.worker_skill,
        [prim.x_grid[i] for i in res.x_policy],
        color=COLORS[1], linewidth=4, label=""
    )
    # Set the limits of the y-axis
    # Remove spines and ticks
    ylims!(ax, prim.x_grid[1], prim.x_grid[end] + 0.3)
    ax.yticks = ([prim.x_grid[1], prim.x_grid[end]], [L"x_{min}", L"x_{max}"])  # Remove y-axis ticks
    ax.xticks = ([prim.worker_skill[1], prim.worker_skill[end]], [L"h_{min}", L"h_{max}"])  # Remove x-axis ticks
    return fig
end
fig_worker_search_policy = plot_worker_search_policy(prim, res)
# Save to pdf to include in the manuscript
save(joinpath(FOLDER_TO_SAVE, "worker_search_policy.pdf"), fig_worker_search_policy)
