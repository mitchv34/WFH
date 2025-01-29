using CairoMakie, LaTeXStrings

# Parameters
A = 1.0        # Baseline productivity
α = 0:0.01:1   # Fraction of work done remotely
ψ_values = [0.2, 0.8]  # Different remote work efficiencies
ψ₀_values = [0.3, 0.7] # Different ψ₀ values

# Create a new figure
fig = Figure(resolution = (800, 600), fonts = (; regular="Computer Modern"))

# Production function
production(α, ψ, ψ₀) = A * ((1 .- α) .+ α .* (ψ .- ψ₀))

# Different linestyles for different ψ₀ values
linestyles = [:solid, :dash]
# Different colors for different ψ values
colors = [:blue, :red]

# Create an axis
ax = Axis(fig[1, 1], 
    xlabel = L"\alpha \text{ (Fraction of Remote Work)}", 
    ylabel = L"Y \text{ (Firm Output)}", 
    xlabelsize = 22, ylabelsize = 22, 
    xgridstyle = :dash, ygridstyle = :dash,
    xtickalign = 1, xticksize = 10, ytickalign = 1, yticksize = 10)

# Plot the production function for different ψ₀ and ψ values
for (i, ψ) in enumerate(ψ_values)
    for (j, ψ₀) in enumerate(ψ₀_values)
        lines!(ax, α, production(α, ψ, ψ₀), 
            label=latexstring("\\phi = $(ψ), \\phi_0 = $(ψ₀)"), 
            linewidth=2, linestyle=linestyles[j], color=colors[i])
    end
end

# Add legend and customize
axislegend(ax; position=:rt, nbanks=2, framecolor=(:grey, 0.5), labelsize=14)

# Add title
ax.title = L"\text{Firm Productivity as a Function of Remote Work}"
ax.titlesize = 24


# Display the plot
fig
