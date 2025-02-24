using OhMyREPL, Plots
## Configure OhMyREPL
colorscheme!("OneDark");

## Configure Plots
# theme(:juno)
theme(:vibrant)
# Set some global defaults (optional):
default(
    fontfamily = "Computer Modern",
    grid       = false,          # Remove grid
    legend     = false,          # Remove legend
    framestyle = :axes,          # "Despine" style (shows only left & bottom axes)
    titlefont  = font("Computer Modern", 16), # Larger title font
    guidefont  = font("Computer Modern", 14), # Larger guide font
    tickfont   = font("Computer Modern", 12),  # Larger tick font
    left_margin   = 15,
    right_margin  = 15,
    bottom_margin = 15,
    top_margin    = 5
)


using CSV, DataFrames, Printf, Statistics, StatsBase, Downloads, XLSX, Chain
using FixedEffectModels, RegressionTables, StatsPlots, KernelDensity, LaTeXStrings

# Teleworkability
wfh_share_estimation = CSV.read("/project/high_tech_ind/WFH/WFH/data/results/wfh_estimates.csv", DataFrame);
bls = CSV.read("/project/high_tech_ind/WFH/WFH/data/processed/bls/oews/oews_all_2023.csv", DataFrame);

# Convert TOT_EMP to Float64; nonparseable entries become missing
bls.TOT_EMP = [tryparse(Float64, string(x)) === nothing ? missing : tryparse(Float64, string(x)) for x in bls.TOT_EMP];

# Filter to create 3-digit and 4-digit subsets
bls_ind_3d = filter(row -> row.I_GROUP == "3-digit" && row.O_GROUP == "detailed", bls);
bls_ind_3d[!, "NAICS"] = first.(string.(bls_ind_3d.NAICS), 3);

bls_ind_4d = filter(row -> row.I_GROUP == "4-digit" && row.O_GROUP == "detailed", bls);
bls_ind_4d[!, "NAICS"] = first.(string.(bls_ind_4d.NAICS), 4);

# Read productivity data
productivity_4d = CSV.read("/project/high_tech_ind/WFH/WFH/data/processed/bls/productivity/productivity_4_Digit.csv", DataFrame; types=Dict("NAICS"=>String));
productivity_3d = CSV.read("/project/high_tech_ind/WFH/WFH/data/processed/bls/productivity/productivity_3_Digit.csv", DataFrame; types=Dict("NAICS"=>String));

# Calculate productivity coverage
subset_4d = filter(row -> row.NAICS in productivity_4d.NAICS, bls_ind_4d);
prod_coverage_4d = sum(skipmissing(subset_4d.TOT_EMP)) / sum(skipmissing(bls_ind_4d.TOT_EMP));

subset_3d = filter(row -> row.NAICS in productivity_3d.NAICS, bls_ind_3d);
prod_coverage_3d = sum(skipmissing(subset_3d.TOT_EMP)) / sum(skipmissing(bls_ind_3d.TOT_EMP));

@printf("Productivity coverage for 4-digit NAICS: %.2f%% of total employment.\n", prod_coverage_4d*100)
@printf("Productivity coverage for 3-digit NAICS: %.2f%% of total employment.\n", prod_coverage_3d*100)

# WFH rates by industry (from ACS)
wfh_rates = CSV.read("/project/high_tech_ind/WFH/WFH/data/processed/acs/acs_136_YEAR_INDNAICS.csv", DataFrame);

# Exclude rows where INDNAICS starts with "11", "92" or "99"
wfh_rates = filter(row -> !(startswith(string(row.INDNAICS), "11") ||
                            startswith(string(row.INDNAICS), "92") ||
                            startswith(string(row.INDNAICS), "99")), wfh_rates);

# Create NAICS column by taking first 3 characters of INDNAICS
wfh_rates[!, "NAICS"] = first.(string.(wfh_rates.INDNAICS), 3);

# Group by YEAR and NAICS to compute weighted average of WFH_INDEX and sum of TOTAL_WEIGHT
grp = groupby(wfh_rates, ["YEAR", "NAICS"]);
wfh_rates_3d = combine(grp, [:WFH_INDEX, :TOTAL_WEIGHT] => ((wfh_index, tot_weight) -> (
    WFH_INDEX = mean(wfh_index, Weights(tot_weight)),
    TOTAL_WEIGHT = sum(tot_weight)
)) => [:WFH_INDEX, :TOTAL_WEIGHT]);

# Read BDS data and rename column
bds_3d = CSV.read(Downloads.download("https://www2.census.gov/programs-surveys/bds/tables/time-series/2022/bds2022_vcn3.csv"), DataFrame; types=Dict("vcnaics3"=>String));
rename!(bds_3d, "vcnaics3" => "NAICS");

# Convert 3-digit NAICS to wide format with occupation composition
bls_ind_3d_wide = select(bls_ind_3d, ["NAICS", "NAICS_TITLE", "OCC_CODE", "TOT_EMP"]) |>
    df -> unstack(df, :OCC_CODE, :TOT_EMP, fill=0);

# Ensure any residual missing values (if any) are filled with zeros for occupation columns
occ_cols = names(bls_ind_3d_wide, Not(["NAICS", "NAICS_TITLE"]));
for col in occ_cols
    bls_ind_3d_wide[!, col] = coalesce.(bls_ind_3d_wide[!, col], 0)
end

occ_cols = names(bls_ind_3d_wide, Not(["NAICS", "NAICS_TITLE"]));
matrix = Matrix(bls_ind_3d_wide[:, occ_cols]);

row_sums = sum(matrix, dims=2);
bls_ind_3d_wide[:, occ_cols] = matrix ./ row_sums;

onet_soc_xwalk = CSV.read("/project/high_tech_ind/WFH/WFH/data/aux_and_croswalks/onet_soc_xwalk.csv", DataFrame);
wfh_share_estimation_occ = leftjoin(wfh_share_estimation, onet_soc_xwalk[:, ["ONET_SOC_CODE", "OCC_CODE"]], on="ONET_SOC_CODE");
wfh_share_estimation_occ = combine(groupby(wfh_share_estimation_occ, "OCC_CODE"), "ESTIMATE_WFH_ABLE" => mean => "ESTIMATE_WFH_ABLE");

# Create dictionary mapping from OCC_CODE to WFH estimates
wfh_map = Dict(row.OCC_CODE => row.ESTIMATE_WFH_ABLE for row in eachrow(wfh_share_estimation_occ));

# Get occupation columns from wide DataFrame (excluding NAICS identifiers)
occ_cols = setdiff(names(bls_ind_3d_wide), ["NAICS", "NAICS_TITLE"]);

# Create weighted copy of the DataFrame
bls_ind_3d_wide_weighted = copy(bls_ind_3d_wide);

# Apply weights or zero out columns
for occ in occ_cols
    if haskey(wfh_map, occ)
        bls_ind_3d_wide_weighted[!, occ] .= bls_ind_3d_wide_weighted[!, occ] .* wfh_map[occ]
    else
        bls_ind_3d_wide_weighted[!, occ] .= 0
    end
end

# Calculate telework scores and create final DataFrame
ind_3_tele = select(bls_ind_3d_wide_weighted, ["NAICS", "NAICS_TITLE"]);
ind_3_tele[!, :TELE] = sum(eachcol(bls_ind_3d_wide_weighted[:, occ_cols]));


# Select and rename columns from productivity_3d
data = select!(productivity_3d, 
    [
        :Sector, :Year, :NAICS, :sectoral_output_millions, 
        :hours_worked_millions, :sectoral_output_price_deflator_index, 
        :employment_thousands]) |>
    df -> rename!(df, Dict(
        :Sector => :SECTOR,
        :Year => :YEAR,
        :sectoral_output_millions => :SECTORAL_OUTPUT_MILLIONS,
        :hours_worked_millions => :HOURS_WORKED_MILLIONS,
        :sectoral_output_price_deflator_index => :SECTORAL_OUTPUT_PRICE_DEFLATOR_INDEX,
        :employment_thousands => :EMPLOYMENT_THOUSANDS
    ));

# Perform joins
data = leftjoin(data, ind_3_tele, on=:NAICS)
data = innerjoin(data, select(wfh_rates_3d, [:YEAR, :NAICS, :WFH_INDEX]), 
        on=[:YEAR, :NAICS]);

# Final renaming and calculations
rename!(data, :WFH_INDEX => :ALPHA);
data[!, :OUTPUT] = (data.SECTORAL_OUTPUT_MILLIONS ./ data.HOURS_WORKED_MILLIONS) ./
                    data.SECTORAL_OUTPUT_PRICE_DEFLATOR_INDEX;


# Drop missing values and ensure correct types
data = dropmissing(data, [:TELE, :ALPHA, :OUTPUT]);
data[!, :YEAR] = Int.(data[!, :YEAR]);

# Define formulas with proper fixed effects syntax
formula1 = @formula(OUTPUT ~ ALPHA + ALPHA&TELE)
formula2 = @formula(OUTPUT ~ ALPHA + ALPHA&TELE + fe(YEAR))  # Year fixed effects
formula3 = @formula(OUTPUT ~ ALPHA + ALPHA&TELE + fe(SECTOR)) # Sector fixed effects

# Fit models with robust SEs
model1 = reg(data, formula1, Vcov.robust())
model2 = reg(data, formula2, Vcov.robust())
model3 = reg(data, formula3, Vcov.robust())

# Weighted regression for base model
model_weighted1 = reg( data, formula1, 
                    weights = :EMPLOYMENT_THOUSANDS,
                    Vcov.robust());

# Weighted regression for model with Year fixed effects
model_weighted2 = reg(data, formula2, 
                    weights = :EMPLOYMENT_THOUSANDS,
                    Vcov.robust())

# Weighted regression for model with Sector fixed effects
model_weighted3 = reg(data, formula3, 
                    weights = :EMPLOYMENT_THOUSANDS,
                    Vcov.robust())

# Create comparison table for both unweighted and weighted models
regtable(model1, model_weighted1, model2, model_weighted2, model3, model_weighted3)

model = model_weighted1;
β₀, β₁, β₂ = coef(model);
A = β₀
δ₀ = (β₁ + 1) / A
δ₁ = β₂ / A        

# Copy ind_3_tele and calculate Ψ
data_density_3d = copy(ind_3_tele);
data_density_3d[!, "ψ"] = δ₀ .+ δ₁ .* data_density_3d[!, "TELE"];

# Filter bds_3d to the year 2022 and keep only the relevant columns
subset_bds = filter(row -> row.year == 2022, bds_3d)[!, ["job_creation_births", "NAICS"]];
# Merge ind_3_tele with the subset on the "NAICS" column (left join)
data_density_3d = leftjoin(data_density_3d, subset_bds, on = "NAICS");
# Convert "job_creation_births" column to numeric (Float64), coercing non-parsable values to missing
data_density_3d[!, "JOB_CREATION_BIRTHS"] = [tryparse(Float64, string(x)) === nothing ? missing : tryparse(Float64, string(x)) for x in data_density_3d[!, "job_creation_births"]];
# Drop the original "job_creation_births" column
select!(data_density_3d, Not("job_creation_births"));

# Drop missing values
dropmissing!(data_density_3d);

sort!(data_density_3d, :ψ)


using CairoMakie

color1 = "#23373B"

# Ensure CMU Serif is installed, or use another font
font_choice = "CMU Serif"  # Change if needed

# Compute KDE
num_grid_points = 100  
kde_result = kde(data_density_3d[!, :ψ], npoints=num_grid_points)
grid = kde_result.x
estimated_density = kde_result.density
probabilities = estimated_density ./ sum(estimated_density)

# Construct a DataFrame for PDF and CDF
psi_distribution = DataFrame(:ψ => grid, :pdf => probabilities)
psi_distribution[!, :cdf] = cumsum(psi_distribution[!, :pdf])

# Set up figure
fig = Figure(size = (1500, 500), fonts = (; regular=font_choice, italic=font_choice, bold=font_choice))

# 1. **KDE Plot**
ax1 = Axis(fig[1, 1], 
    xlabel=L"\tilde{\psi}", ylabel="Density",
    title=latexstring("Density of \$\\tilde{\\psi}\$"),
    xtickalign=1, xticksize=10, ytickalign=1, yticksize=10,
    xgridvisible=false, ygridvisible=false, topspinevisible=false, rightspinevisible=false
    )
lines!(ax1, grid, estimated_density, linewidth=4, color=color1)

# 2. **Histogram**
ax2 = Axis(fig[1, 2], xlabel="Teleworkability", ylabel="Frequency",
    title="Histogram of Teleworkability",
    xtickalign=1, xticksize=10, ytickalign=1, yticksize=10,
    xgridvisible=false, ygridvisible=false, topspinevisible=false, rightspinevisible=false
)
hist!(ax2, data_density_3d[!, :ψ], weights=data_density_3d[!, :JOB_CREATION_BIRTHS], bins=20,
    normalization=:probability, color=color1, gap=0.3)

# 3. **ECDF Plot**
ax3 = Axis(fig[1, 3], xlabel="Teleworkability", ylabel="Cumulative Probability",
    title="Empirical CDF of Teleworkability",
    xtickalign=1, xticksize=10, ytickalign=1, yticksize=10,
    xgridvisible=false, ygridvisible=false, topspinevisible=false, rightspinevisible=false
)
lines!(ax3, data_density_3d[!, :ψ], data_density_3d[!, :ECDF], linewidth=4, color=color1)

# Display figure
fig


# Fit KDE with specified grid points
num_grid_points = 100  # Adjust as needed
kde_result = kde(data_density_3d[!, :ψ],
                bandwidth=5, 
                # kernel=Normal,
                npoints=num_grid_points)

# Extract grid and density values
grid = kde_result.x
estimated_density = kde_result.density

# Normalize probabilities to sum to 1
probabilities = estimated_density ./ sum(estimated_density)

# Create distribution object
psi_distribution = DataFrame(
    [
        :ψ => grid,
        :pdf => probabilities
    ]
)

# Add cdf
psi_distribution[!, :cdf] = cumsum(psi_distribution[!, :pdf])

# Set up figure
fig = Figure(size = (1500, 500), fonts = (; regular=font_choice, italic=font_choice, bold=font_choice))

# 1. **KDE Plot**
ax1 = Axis(fig[1, 1], 
    xlabel=L"\tilde{\psi}", ylabel="Density",
    title=latexstring("Density of \$\\tilde{\\psi}\$"),
    xtickalign=1, xticksize=10, ytickalign=1, yticksize=10,
    xgridvisible=false, ygridvisible=false, topspinevisible=false, rightspinevisible=false
    )
lines!(ax1, psi_distribution[!, :ψ], psi_distribution[!, :pdf], linewidth=4, color=color1)

# 3. **ECDF Plot**
ax2 = Axis(fig[1, 2], xlabel="Teleworkability", ylabel="Cumulative Probability",
    title="Empirical CDF of Teleworkability",
    xtickalign=1, xticksize=10, ytickalign=1, yticksize=10,
    xgridvisible=false, ygridvisible=false, topspinevisible=false, rightspinevisible=false
)
lines!(ax2, psi_distribution[!, :ψ], psi_distribution[!, :cdf], linewidth=4, color=color1)

# Display figure
fig