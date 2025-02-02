using CairoMakie, LaTeXStrings

# Parameters
A = 1.0        # Baseline productivity
c = 1.0        # Disutility scaling factor
χ = 1.1        # Curvature parameter for disutility
ψ₀ = 0.5       # Threshold remote efficiency
γ = 1.0        # Non-linearity parameter for remote work productivity
ψ_values = 0:0.01:2  # Range of remote efficiency values

# Optimal WFH function (α*(ψ))
function optimal_wfh(ψ, A, c, χ, ψ₀, γ, ψ_c) 
    # Compute critical thresholds
    ψ_bottom = ψ₀ + (1 - (c * χ ) / A)^(1/γ)
    ψ_top = ψ₀ + 1
    # Compute optimal WFH
    if ψ <= ψ_bottom
        return 0
    elseif ψ > ψ_top
        return 1
    else
        return 1 - ( (c * χ )/(A * (1 - (ψ - ψ₀)^γ )) )^(1/(1-χ))
    end
end

# Create a new figure with more space for the legend
fig = Figure(size = (800, 700), fonts = (; regular="Computer Modern"))



ax = Axis(fig[1, 1], 
xlabel = L"\psi \text{ (Remote Work Efficiency)}", 
ylabel = L"\alpha^*(\psi) \text{ (Optimal WFH Fraction)}", 
xlabelsize = 16, ylabelsize = 16, 
xgridstyle = :dash, ygridstyle = :dash,
xtickalign = 1, xticksize = 7, ytickalign = 1, yticksize = 7)

# Add vertical lines for critical thresholds
vlines!(ax, [ψ₀, ψ₀ + 1], color = :red, linestyle = :dash, linewidth = 2)
# Plot optimal WFH as a function of ψ
lines!(ax, ψ_values, optimal_wfh.(ψ_values, A, c, χ, ψ₀, γ, ψ₀),
    label = latexstring("\$A = $A, c = $c, \\chi = $χ, \\psi_0 = $ψ₀, \\gamma = $γ\$"),
    linewidth = 3, color = :blue)
vlines!(ax, [ ψ₀ + (1 - (c * χ ) / A)^(1/γ)], color = :blue, linestyle = :dash, linewidth = 2, 
    label = latexstring("Critical Threshold for \$\\alpha^* > 0\$"))

# χ = 1.4
# # Plot optimal WFH as a function of ψ
# lines!(ax, ψ_values, optimal_wfh.(ψ_values, A, c, χ, ψ₀, γ, ψ₀),
#     label = latexstring("\$A = $A, c = $c, \\chi = $χ, \\psi_0 = $ψ₀, \\gamma = $γ\$"),
#     linewidth = 3, color = :green)
# vlines!(ax, [ ψ₀ + (1 - (c * χ ) / A)^(1/γ)], color = :green, linestyle = :dash, linewidth = 2, 
# label = latexstring("Critical Threshold for \$\\alpha^* > 0\$"))


# Add title
ax.title = L"\text{Optimal WFH Fraction as a Function of Remote Efficiency}"
ax.titlesize = 20

# Adjust the legend to be placed outside and below the plot
axislegend(ax; position = :cb, nbanks = 1, framecolor = (:grey, 0.5), labelsize = 14, padding = (5, 5, 5, 5), alignment = :center)

# Display the plot
fig
