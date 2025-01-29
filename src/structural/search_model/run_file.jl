using LaTeXStrings

@time begin 
    include("model.jl")
    path_params = "./para"
    prim, res = initializeModel(path_params)
    iterateValueMatch(prim, res)
    solveSubMarketTightness(prim, res);
    iterateW_U(prim, res)
end



# Plot value function
plot(   prim.W_grid, res.J, label="Value function J(w)", 
        xlabel="Wage w", ylabel="Value J(w)",
        title="Value function J(w)",
        lw=2, color=:blue)

# Plot Market tightness
plot(   prim.W_grid, res.θ, label="Market tightness θ(w)", 
        xlabel="Wage w", ylabel="Market tightness θ(w)",
        title="Market tightness θ(w)",
        lw=2, color=:red)

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