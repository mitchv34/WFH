module ModelPlots

using Plots
using Plots.PlotMeasures
using LaTeXStrings
using Statistics

# Define consistent styling
const COLORS = [:blue, :green, :red, :purple, :orange]
const LINE_STYLES = [:solid, :dash, :dot, :dashdot]
const LINE_WIDTH = 2

# Global selected indices
const SELECTED_INDICES = [25, 75]
const SELECTED_INDICES_SKILL = [10, 40]


"""
Initialize plotting configuration
"""
function initialize_plots()
    theme(:vibrant)
    default(
        fontfamily = "Computer Modern",
        grid = false,
        legend = true,
        framestyle = :axes,
        titlefont = font("Computer Modern", 12),
        guidefont = font("Computer Modern", 10),
        tickfont = font("Computer Modern", 10),
        legendfont = font("Computer Modern", 10),
        left_margin = 15mm,
        right_margin = 15mm,
        bottom_margin = 15mm,
        top_margin = 5mm
    )
end
# Call initialize_plots when module loads
initialize_plots()

"""
Utility function to create consistent titles with LaTeX formatting
"""
format_title(title_str) = latexstring(title_str)

"""
Utility function to create consistent axis labels with LaTeX formatting
"""
format_label(label_str) = latexstring(label_str)

"""
Plot productivity density with reference lines
"""
function plot_productivity_density(prim, res)
    p = plot(prim.ψ_grid, prim.ψ_pdf,
        label="",
        xlabel="Productivity \$\\psi\$",
        # ylabel=format_label("Density"),
        title=format_label("Density of \$\\psi\$"),
        lw=LINE_WIDTH,
        color=COLORS[1]
    )
    
    # # Add reference lines
    vline!([prim.ψ₀], label=L"\psi_0", color=:gray, linestyle=LINE_STYLES[2], lw=LINE_WIDTH)
    for (i, h_ind) in enumerate(SELECTED_INDICES_SKILL)
        vline!( [res.ψ_bottom[h_ind]],
                label=latexstring("\\underbar{\\psi}($( round( prim.worker_skill[h_ind], digits = 2) ))"),
                color=COLORS[i + 1],
                linestyle=LINE_STYLES[i + 2],
                lw=LINE_WIDTH
                )
    end
    return p
end

"""
Plot optimal α policy
"""
function plot_remote_policy(prim, res)
    p = plot(xlabel=format_label("Productivity \$\\psi\$"),
            ylabel=format_label("\$\\alpha^*(h, \\psi)\$"),
            title="Optimal Remote Work Policy")
    for (i, h_ind) in enumerate(SELECTED_INDICES_SKILL)
        h = prim.worker_skill[h_ind]
        plot!(prim.ψ_grid, res.α_policy[:, h_ind],
            label=format_label("\$\\alpha^*(h = $(round(h, digits=2))\$)"),
            color=COLORS[i],
            lw=LINE_WIDTH)
        vline!( [res.ψ_bottom[h_ind]],
            # label=latexstring("\\underbar{\\psi}($( round( h, digits = 2) ))"),
            color=:gray,
            # lw=LINE_WIDTH
            label = "",
            legend = :right
        )
    end
    # Add reference lines
    return p
end

"""
Plot wage policy for different productivity levels
"""
function plot_wage_policy(prim, res)
    p = plot(xlabel=format_label("Productivity \$\\psi\$"),
            ylabel=format_label("Wage \$w(\\psi)\$"),
            title="Optimal Wage Policy")
    
    for (i, h_ind) in enumerate(SELECTED_INDICES_SKILL)
        h = prim.worker_skill[h_ind]
        for (j, idx) in enumerate(SELECTED_INDICES)
            plot!(prim.ψ_grid, res.w_policy[:, h_ind, idx],
                label=format_label("\$(x, h)=($(round(prim.x_grid[idx], digits=2)), $(round(h, digits=2)))\$"),
                linestyle=LINE_STYLES[i],
                color=COLORS[j],
                lw=LINE_WIDTH)
        end
        # vline!( [res.ψ_bottom[h_ind]],
        #         label="",
        #         color=:gray,
        #         )
    end
    
    # Add reference lines
    # vline!([prim.ψ₀], label="", color=:gray)
    
    return p
end

# """
# Plot job finding rate heatmap
# """
# function plot_feasible_jobs(prim, res)
#     heatmap(prim.ψ_grid, prim.x_grid, res.δ_grid .== 1.0,
#         xlabel=format_label("Productivity \$\\psi\$"),
#         ylabel=format_label("Promised Utility x"),
#         title=format_label("Feasible Jobs"),
#         color=[:lightblue, :darkblue],
#         clims=(0, 1),
#         colorbar_title=L"\delta(x,\psi)",
#         colorbar_ticks=([0, 1], ["False", "True"]),
#         framestyle=:box
#     )
# end

"""
Plot value functions for different parameters
"""
function plot_firm_value_functions(prim, res)
    # Value function for selected ψ values
    p1 = plot(title=format_title("Value Function \$J(x, h, \\psi)\$ for Selected \$\\psi\$"),
            xlabel=format_label("Promised Utility \$x\$"),
            ylabel=format_label("Value \$J(x,\\psi)\$"))
    for (i, h_ind) in enumerate(SELECTED_INDICES_SKILL)
        h = prim.worker_skill[h_ind]
        for (i_ψ, ψ) in enumerate(SELECTED_INDICES)
            plot!(p1, prim.x_grid, res.J[i_ψ, h_ind, :],
                label=format_label("J(x, h =$(round(h, digits = 2)), \\psi=$(round(prim.ψ_grid[ψ], digits=2)))"),
                color=COLORS[i_ψ],
                linestyle=LINE_STYLES[i],
                lw=LINE_WIDTH
                )
        end 
    end
    
    # Value function for selected x values
    p2 = plot(title=format_title("Value Function J(\\psi, h, x) for Selected x"),
            xlabel=format_label("Productivity \\psi"),
            ylabel=format_label("Value J(\\psi,x)"))
    for (i, h_ind) in enumerate(SELECTED_INDICES_SKILL)
        h = prim.worker_skill[h_ind]
        for (i_x, x) in enumerate(SELECTED_INDICES)
            plot!(p2, prim.ψ_grid, res.J[:, h_ind, x],
                label=format_label("J(\\psi, h =$(round(h, digits = 2)), x=$(round(prim.x_grid[x], digits=2)))"),
                color=COLORS[i_x],
                linestyle=LINE_STYLES[i],
                lw=LINE_WIDTH
                )
        end
        # vline!( [res.ψ_bottom[h_ind]],
        #         label="",
        #         color=:gray,
        #         )
    end
    
    # Add reference lines to second plot
    # vline!(p2, [prim.ψ₀], label="", color=:gray)
    
    return p1, p2
end

"""
Plot market dynamics (tightness, finding rates, etc.)
"""
function plot_market_dynamics(prim, res)
    # Market tightness
    p1 = plot(
        xlabel="Promised Utility x",
        ylabel="Market Tightness",
        title=format_label("Market Tightness \$\\theta(h, x)\$")
    )
    for (i, h_ind) in enumerate(SELECTED_INDICES_SKILL)
        h = prim.worker_skill[h_ind]
        plot!(p1,
            prim.x_grid, res.θ[h_ind, :],
            label = latexstring("\\theta(h = $(round(h, digits = 2)), x)"),
            lw=LINE_WIDTH,
            color=COLORS[i]
        )
    end

    # Job finding probability
    p2 = plot(
        xlabel="Promised Utility x",
        ylabel="Job Finding Probability",
        title=format_label("Job Finding Rate \$p(\\theta(h, x))\$")
    )
    for (i, h_ind) in enumerate(SELECTED_INDICES_SKILL)
        h = prim.worker_skill[h_ind]
        plot!(p2,
            prim.x_grid, prim.p.(res.θ[h_ind, :]),
            label=latexstring("p(\\theta(h = $(round(h, digits = 2)), x))"),
            lw=LINE_WIDTH,
            color=COLORS[i]
        )
    end

    # # Matching probabilities
    # p3 = plot(res.θ, prim.p.(res.θ),
    #     label="Worker Finding Firm",
    #     xlabel="Market Tightness (v/u)",
    #     ylabel="Probability",
    #     title="Matching Probabilities",
    #     lw=LINE_WIDTH,
    #     color=COLORS[1])
    # plot!(p3, res.θ, prim.p.(res.θ) ./ res.θ,
    #     label="Firm Finding Worker",
    #     lw=LINE_WIDTH,
    #     color=COLORS[2],
    #     linestyle=LINE_STYLES[2])

    return p1, p2#, p3
end

"""
Plot worker-related functions (value, wage posting, search)
"""
function plot_worker_value_functions(prim, res)
    # Unemployed worker value
    p1 = plot(title="Unemployed Worker Value Function",
        xlabel="Unemployment Benefit b",
        ylabel="Worker Value")
    
    
    plot!(p1, prim.worker_skill, res.U,
            label="",
            lw=LINE_WIDTH,
            color=COLORS[1])


    # Employed worker value: value function for selected ψ values
    p2 = plot(title=format_label("Value Function \$W(x,\\psi)\$ for Selected \$\\psi\$"),
            xlabel=format_label("Promised Utility \$x\$"),
            ylabel=format_label("Value \$W(x,\\psi)\$"))
    for (i, h_ind) in enumerate(SELECTED_INDICES_SKILL)
        h = prim.worker_skill[h_ind]
        i_ψ = 1
        ψ = SELECTED_INDICES[i_ψ]

        # for (i, ψ) in enumerate(SELECTED_INDICES)
            plot!(p2, prim.x_grid, res.W[ψ, h_ind, :],
                label=format_label("W(x, h=$(round(h, digits=2)), \\psi=$(round(prim.ψ_grid[ψ], digits=2))"),
                lw=LINE_WIDTH,
                color=COLORS[i])
        # end
    end

    # Value function for selected x values
    p3 = plot(title=format_label("Value Function \$W(\\psi,x)\$ for Selected \$x\$"),
            xlabel=format_label("Productivity \$\\psi\$"),
            ylabel=format_label("Value \$W(\\psi,x)\$"))
    for (i, h_ind) in enumerate(SELECTED_INDICES_SKILL)
        h = prim.worker_skill[h_ind]
        for (i_x, x) in enumerate(SELECTED_INDICES)
            plot!(p3, prim.ψ_grid, res.W[:, h_ind, x],
                label=format_label("W(\\psi, h=$(round(h, digits=2)), x=$(round(prim.x_grid[x], digits=2))"),
                lw=LINE_WIDTH,
                linestyle=LINE_STYLES[i],
                color=COLORS[i_x])
        end
        vline!(p3, 
                [res.ψ_bottom[i]], 
                # label=latexstring("\\underbar{\\psi}_$(h_ind)"), 
                color=:gray, 
                linestyle=LINE_STYLES[2])
    end
    # Put legend in the left:
    p3 = plot(p3, legend=:left)
    # Add reference lines to p3
    vline!( p3, [prim.ψ₀],
            label=L"\psi_0",
            color=:gray,
            linestyle=LINE_STYLES[1])
    

    # Utility search
    p4 = plot(title="Unemployed Worker Search Policy",
        xlabel="Worker Skill h",
        ylabel="Promised Utility x")
    
    plot!(p4, prim.worker_skill, res.x_policy,
            label="",
            lw=LINE_WIDTH,
            color=COLORS[1])
    return p1, p2, p4
end

"""
Generate all plots for the model
"""
function plot_all(prim, res)
    plots = Dict{Symbol, Any}()
    
    plots[:density] = plot_productivity_density(prim, res)
    plots[:wage_policy] = plot_wage_policy(prim, res)
    plots[:optimal_α] = plot_remote_policy(prim, res)
    plots[:firm_value_x], plots[:firm_value_ψ] = plot_firm_value_functions(prim, res)
    plots[:market_tightness], plots[:finding_rate] = plot_market_dynamics(prim, res)
    plots[:unemployed_value], plots[:employed_value_x], plots[:unemployed_search_pol] = plot_worker_value_functions(prim, res)
    
    return plots
end

end  # module