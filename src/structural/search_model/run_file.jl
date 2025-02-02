using Plots
using LaTeXStrings
using Statistics

# @time begin 
include("model.jl")
path_params = "/Users/mitchv34/Work/WFH/src/structural/search_model/parameters.yaml"
prim, res = initializeModel(path_params);
iterateValueMatch(prim, res)
solveSubMarketTightness!(prim, res);
iterateW_U(prim, res)
# end


# Plotting
# Plot optimal WFH policy

plot(   prim.ψ_grid, res.α_policy, label=L"\alpha^*(\psi)",
        xlabel="Productivity ψ", ylabel="WFH policy α(ψ)",
        title="Optimal WFH policy α(ψ)",
        lw=2, color=:blue)
vline!([prim.ψ₀], label="ψ_0", color=:red, linestyle=:dash)

# Plot optimal wage policy
plot(   prim.ψ_grid, res.w_policy[:, 1], label=L"\alpha^*(\psi)",
        xlabel="Productivity ψ", ylabel="WFH policy α(ψ)",
        title="Optimal WFH policy α(ψ)",
        lw=2, color=:blue)
plot!(   prim.ψ_grid, res.w_policy[:, 100], label=L"\alpha^*(\psi)",
        xlabel="Productivity ψ", ylabel="WFH policy α(ψ)",
        title="Optimal WFH policy α(ψ)",
        lw=2, color=:red)
vline!([prim.ψ₀], label="ψ_0", color=:red, linestyle=:dash)



# Plot value function
ψ_values_ind = [25, 50, 75, 100]
plot()
for ψ in ψ_values_ind
        plot!(  prim.x_grid, res.J[ψ, :], label="J(x,ψ=$ψ)",
                xlabel="Promised utility (x)", ylabel="Value J(x,ψ)",
                title="Value function J(x,ψ)",
                lw=2)
end
plot!()


p1 = plot()
p2 = plot()
for ψ in ψ_values_ind
        # Plot market tightness
        plot!(p1,  prim.x_grid, res.θ[ψ, :], label=latexstring("\\theta(x,\\psi=$ψ)"),
                xlabel="Promised utility (x)", ylabel=L"\theta(x,\psi)",
                title="Market Tightness θ(x,ψ)",
                lw=2)
        # Plot probability of finding a job
        plot!(p2,  prim.x_grid, prim.p.(res.θ[ψ, :]), label=latexstring("p(\\theta(x,\\psi=$ψ))"),
                xlabel="Promised utility (x)", ylabel=L"p(\theta(x,\psi))",
                title="Job Finding Rate p(θ(x,ψ))",
                lw=2)
        
end
p2

# Plot probability of worker finding a job
plot(   prim.W_grid, prim.p.(res.θ), label=L"p(\theta)",
        xlabel="Searched Wage", ylabel="Probability",
        title="Job Finding Rate",
        lw=2, color=:green)
        
# Plot probability of firm filling a vacancy
plot(   prim.W_grid, prim.p.(res.θ) ./ res.θ, label=L"q(\theta)",
        xlabel="Posted Wage", ylabel="Probability",
        title="Vacancy Filling Rate",
        lw=2, color=:green)

# Plot Probability of worker finding a job vs Firm filling a vacancy
plot(   res.θ, prim.p.(res.θ), label="Probability of Worker Matching a Firm", 
        xlabel="Market tightness (v/u)", ylabel="Probability",
        title="",
        lw=2, color=:green)
plot!(  res.θ, prim.p.(res.θ) ./ res.θ , label="Probability of Firm Matching a Worker",
        xlabel="Market tightness (v/u)", ylabel="Probability",
        title="",
        lw=2, color=:red, linestyle=:dash)

# Plot worker value function
plot(   prim.b_grid, res.U, label="U(b)",
        xlabel="Unemployment benefit b", ylabel="U(b)",
        title="Unemployed Worker Value Function",
        lw=2, color=:blue)
plot(  prim.W_grid, res.W, label="W(w)",
        xlabel="Wage w", ylabel="W(w)",
        title="Employed Worker Value Function",
        lw=2, color=:red)   

# Plot wage search behavior
plot(   prim.b_grid, res.w, label="Wage w(b)",
        xlabel="Unemployment benefit b", ylabel="Wage w(b)",
        title="Wage Posting Behavior",
        lw=2, color=:blue)