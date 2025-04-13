# Config
import Term: install_term_stacktrace

using Term
install_term_stacktrace()

## DataPaths
struct DataPaths
    wfh_estimates::String
    bls::String
    productivity_3d::String
    productivity_4d::String
    wfh_rates::String
    onet_soc_xwalk::String
    onet_skills::String
    onet_abilities::String
    # TODO: Clean
    # skill_vectors::String
end

# Populate DataPaths    
function define_paths()
    base_path = "/project/high_tech_ind/WFH/WFH/data"
    
    return DataPaths(
        # WFH estimates
        joinpath(base_path, "results/wfh_estimates.csv"),
        
        # BLS data
        joinpath(base_path, "processed/bls/oews/oews_all_2023.csv"),
        
        # Productivity data
        joinpath(base_path, "processed/bls/productivity/productivity_3_Digit.csv"),
        joinpath(base_path, "processed/bls/productivity/productivity_4_Digit.csv"),
        
        # WFH rates data
        joinpath(base_path, "processed/acs/acs_136_YEAR_INDNAICS.csv"),
        
        # Crosswalk data
        joinpath(base_path, "aux_and_croswalks/onet_soc_xwalk.csv"),

        # Skill data
        joinpath(base_path, "onet_data/processed/measure/ABILITIES.csv"),
        joinpath(base_path, "onet_data/processed/measure/SKILLS.csv")
        # TODO: Clean
        # "/project/high_tech_ind/WFH/WFH/data/results/skill_vectors.csv"
    )
end

# data_loader.jl
module DataLoader
    using CSV, DataFrames, Downloads, StatsBase

    export load_datasets, process_bls_data, load_productivity_data, load_wfh_data, load_bds_data, load_and_process_onet_data

    function load_datasets(paths)
        wfh_share_estimation = CSV.read(paths.wfh_estimates, DataFrame)
        bls = CSV.read(paths.bls, DataFrame)
        # TODO : Clean
        # onet_skill = CSV.read(paths.skill_vectors, DataFrame)
        onet_skill = load_and_process_onet_data(paths.onet_skills, paths.onet_abilities)
        return wfh_share_estimation, bls, onet_skill
    end

    function process_bls_data(bls)
        # Convert TOT_EMP to Float64
        bls.TOT_EMP = [tryparse(Float64, string(x)) === nothing ? missing : tryparse(Float64, string(x)) for x in bls.TOT_EMP]
        
        # Create 3-digit subset
        bls_ind_3d = filter(row -> row.I_GROUP == "3-digit" && row.O_GROUP == "detailed", bls)
        bls_ind_3d[!, "NAICS"] = first.(string.(bls_ind_3d.NAICS), 3)
        
        # Create 4-digit subset
        bls_ind_4d = filter(row -> row.I_GROUP == "4-digit" && row.O_GROUP == "detailed", bls)
        bls_ind_4d[!, "NAICS"] = first.(string.(bls_ind_4d.NAICS), 4)
        
        return bls_ind_3d, bls_ind_4d
    end

    function load_productivity_data(paths)
        productivity_4d = CSV.read(paths.productivity_4d, DataFrame; types=Dict("NAICS"=>String))
        productivity_3d = CSV.read(paths.productivity_3d, DataFrame; types=Dict("NAICS"=>String))
        return productivity_3d, productivity_4d
    end

    function load_wfh_data(path)
        wfh_rates = CSV.read(path, DataFrame)
        wfh_rates = filter(row -> !(startswith(string(row.INDNAICS), "11") ||
                                    startswith(string(row.INDNAICS), "92") ||
                                    startswith(string(row.INDNAICS), "99")), wfh_rates)
        wfh_rates[!, "NAICS"] = first.(string.(wfh_rates.INDNAICS), 3)
        return wfh_rates
    end

    function load_bds_data()
        # bds_url = "https://www2.census.gov/programs-surveys/bds/tables/time-series/2022/bds2022_vcn3.csv"
        # bds_3d = CSV.read(Downloads.download(bds_url), DataFrame; types=Dict("vcnaics3"=>String))
        path = "/project/high_tech_ind/WFH/WFH/data/raw/bds/BDS 2022 VCN3.csv"
        bds_3d = CSV.read(path, DataFrame; types=Dict("vcnaics3"=>String))
        rename!(bds_3d, "vcnaics3" => "NAICS")
        return bds_3d
    end

    function load_onet_soc_xwalk(path)
        return CSV.read(path, DataFrame)
    end

    # TODO: Clean
    function load_and_process_onet_data(path_skills, path_abilities)
        skills = CSV.read(path_skills, DataFrame)
        abilities = CSV.read(path_abilities, DataFrame)
        # Pivot the skills and abilities data
        skills = unstack(skills, [:ONET_SOC_CODE, :ELEMENT_ID, :ELEMENT_NAME], :SCALE_ID, :DATA_VALUE)
        abilities = unstack(abilities, [:ONET_SOC_CODE, :ELEMENT_ID, :ELEMENT_NAME], :SCALE_ID, :DATA_VALUE)

        # Fill missing values with 0
        skills = coalesce.(skills, 0)
        abilities = coalesce.(abilities, 0)
        
        # Normalize IM and LV to the 0-1 scale
        for col in [:IM, :LV]
            abilities[!, col] = (abilities[!, col] .- minimum(abilities[!, col])) ./ (maximum(abilities[!, col]) - minimum(abilities[!, col]))
            skills[!, col] = (skills[!, col] .- minimum(skills[!, col])) ./ (maximum(skills[!, col]) - minimum(skills[!, col]))
        end

        # Compute SKILL_INDEX
        abilities.SKILL_INDEX = abilities.IM .* abilities.LV
        skills.SKILL_INDEX = skills.IM .* skills.LV

        # Concatenate skills and abilities
        all_data = vcat(skills, abilities)

        # Aggregate SKILL_INDEX by ONET_SOC_CODE
        all_data = combine(groupby(all_data, :ONET_SOC_CODE), :SKILL_INDEX => mean => :SKILL_INDEX)

        # Normalize SKILL_INDEX to the 0-1 scale
        all_data.SKILL_INDEX = (all_data.SKILL_INDEX .- minimum(all_data.SKILL_INDEX)) ./ (maximum(all_data.SKILL_INDEX) - minimum(all_data.SKILL_INDEX))

        return all_data
    end
end

# data_processor.jl
module DataProcessor
    using DataFrames, Statistics, StatsBase

    export process_wfh_rates, calculate_final_scores, prepare_regression_data

    function create_wide_format(bls_ind_3d)
        # Select relevant columns
        wide_df = select(bls_ind_3d, ["NAICS", "NAICS_TITLE", "OCC_CODE", "TOT_EMP"])
        
        # Convert to wide format
        wide_df = unstack(wide_df, :OCC_CODE, :TOT_EMP, fill=0)
        
        # Get occupation columns (all except NAICS and NAICS_TITLE)
        occ_cols = setdiff(names(wide_df), ["NAICS", "NAICS_TITLE"])
        
        # Ensure no missing values in occupation columns
        for col in occ_cols
            wide_df[!, col] = coalesce.(wide_df[!, col], 0)
        end
        
        # Calculate employment shares
        matrix = Matrix(wide_df[:, occ_cols])
        row_sums = sum(matrix, dims=2)
        wide_df[:, occ_cols] = matrix ./ row_sums
        
        return wide_df
    end

    function process_wfh_rates(wfh_rates)
        grp = groupby(wfh_rates, ["YEAR", "NAICS"])
        return combine(grp, [:WFH_INDEX, :TOTAL_WEIGHT] => ((wfh_index, tot_weight) -> (
            WFH_INDEX = mean(wfh_index, Weights(tot_weight)),
            TOTAL_WEIGHT = sum(tot_weight)
        )) => [:WFH_INDEX, :TOTAL_WEIGHT])
    end

    function create_wfh_skill_map(wfh_share_estimation, skill_data, onet_soc_xwalk)
        # Join WFH estimates with occupation crosswalk
        # @show first(wfh_share_estimation, 5)
        # @show first(skill_data, 5)
        # @show first(onet_soc_xwalk, 5)
        # TODO: Clean
        wfh_occ = leftjoin(
            wfh_share_estimation[!, [:ONET_SOC_CODE, :ESTIMATE_WFH_ABLE] ], 
            onet_soc_xwalk[:, ["ONET_SOC_CODE", "OCC_CODE"]], 
            on="ONET_SOC_CODE"
        )
        # Join skill data with occupation crosswalk
        skill_occ = leftjoin(
            skill_data, 
            onet_soc_xwalk[:, ["ONET_SOC_CODE", "OCC_CODE"]], 
            on="ONET_SOC_CODE"
        )

        # Calculate mean WFH estimate by occupation
        wfh_occ = combine(
            groupby(wfh_occ, "OCC_CODE"), 
            "ESTIMATE_WFH_ABLE" => mean => "ESTIMATE_WFH_ABLE"
        )
        # Calculate mean skill index by occupation
        skill_occ = combine(
            groupby(skill_occ, "OCC_CODE"), 
            "SKILL_INDEX" => mean => "SKILL_INDEX"
        )

        # Merge WFH and skill data
        wfh_occ = leftjoin(wfh_occ, skill_occ, on=:OCC_CODE)
        
        # Convert to dictionary for easy lookup
        occ_skill_dict = Dict(row.OCC_CODE => row.SKILL_INDEX for row in eachrow(wfh_occ))
        occ_wfh_dict = Dict(row.OCC_CODE => row.ESTIMATE_WFH_ABLE for row in eachrow(wfh_occ))

        return occ_skill_dict, occ_wfh_dict
    end

    function apply_weights(wide_df, measure_dict)
        weighted_df = copy(wide_df)
        occ_cols = setdiff(names(wide_df), ["NAICS", "NAICS_TITLE"])
        # Apply WFH weights or zero out columns
        for occ in occ_cols
            if haskey(measure_dict, occ)
                weighted_df[!, occ] .*= measure_dict[occ]
            else
                weighted_df[!, occ] .= 0
            end
        end
        
        return weighted_df
    end

    function calculate_final_scores(ind_occ_data, wfh_share_estimation, skill_data, onet_soc_xwalk)

        # Convert to wide format
        wide_df = create_wide_format(ind_occ_data)

        # Create maps from occupation to skill and WFH index
        occ_skill_dict, occ_wfh_dict = create_wfh_skill_map(wfh_share_estimation, skill_data, onet_soc_xwalk)

        # Apply skill weights
        weighted_df_skill = apply_weights(wide_df, occ_skill_dict)
        # Apply WFH weights
        weighted_df_wfh = apply_weights(wide_df, occ_wfh_dict)

        occ_cols = setdiff(names(weighted_df_wfh), ["NAICS", "NAICS_TITLE"])
        
        # Calculate final scores
        ## Wfh
        final_scores_wfh = select(weighted_df_wfh, ["NAICS", "NAICS_TITLE"])
        final_scores_wfh[!, :TELE] = sum(eachcol(weighted_df_wfh[:, occ_cols]))
        ## Skill
        final_scores_skill = select(weighted_df_skill, ["NAICS", "NAICS_TITLE"])
        final_scores_skill[!, :SKILL_INDEX] = sum(eachcol(weighted_df_skill[:, occ_cols]))

        # Merge scores
        final_scores = innerjoin(final_scores_wfh, final_scores_skill, on=[:NAICS, :NAICS_TITLE])

        return final_scores
    end

    function calculate_scores(bls_ind_3d, wfh_share_estimation, onet_soc_xwalk)
        # Create wide format DataFrame
        bls_ind_3d_wide = create_wide_format(bls_ind_3d)
        
        # Calculate WFH estimates
        occ_skill_dict, occ_wfh_dict = create_wfh_skill_map(wfh_share_estimation, skill_data, onet_soc_xwalk)
        
        # Calculate final scores
        return calculate_final_scores(bls_ind_3d_wide_weighted,occ_skill_dict, occ_wfh_dict)
    end

    function calculate_output(data)
        return (data.SECTORAL_OUTPUT_MILLIONS ./ data.HOURS_WORKED_MILLIONS) ./
                data.SECTORAL_OUTPUT_PRICE_DEFLATOR_INDEX
    end

    function process_final_data(data)
        # Drop missing values and ensure correct types
        data = dropmissing(data, [:TELE, :ALPHA, :OUTPUT])
        data[!, :YEAR] = Int.(data[!, :YEAR])
        return data
    end

    function standardize_column_names()
        return Dict(
            :Sector => :SECTOR,
            :Year => :YEAR,
            :sectoral_output_millions => :SECTORAL_OUTPUT_MILLIONS,
            :hours_worked_millions => :HOURS_WORKED_MILLIONS,
            :sectoral_output_price_deflator_index => :SECTORAL_OUTPUT_PRICE_DEFLATOR_INDEX,
            :employment_thousands => :EMPLOYMENT_THOUSANDS
        )
    end

    function prepare_regression_data(productivity_3d, ind_3_tele, wfh_skill_rates_3d)
        data = select!(productivity_3d, 
            [
                :Sector, :Year, :NAICS, :sectoral_output_millions, 
                :hours_worked_millions, :sectoral_output_price_deflator_index, 
                :employment_thousands
            ]
        )
        
        # Rename columns
        rename!(data, standardize_column_names())
        
        # Join data
        data = leftjoin(data, ind_3_tele, on=:NAICS)
        data = innerjoin(data, select(wfh_skill_rates_3d, [:YEAR, :NAICS, :WFH_INDEX]), 
                on=[:YEAR, :NAICS])
        
        # Final processing
        rename!(data, :WFH_INDEX => :ALPHA)
        rename!(data, :SKILL_INDEX => :SKILL)
        data[!, :OUTPUT] = calculate_output(data)
        data = process_final_data(data)
        
        return data
    end
end

# models.jl
module Models
    using FixedEffectModels, RegressionTables, DataFrames, KernelDensity, PyCall

    export fit_regression_models, calculate_model_parameters

    function create_model_formulas()
        return (
            # Define regression formulas
            base = @formula( OUTPUT ~ SKILL  + ALPHA&SKILL + TELE&ALPHA&SKILL + SKILL & log(SKILL) ),
            base_no_intercept = @formula(OUTPUT ~ 0 + SKILL  + ALPHA&SKILL + TELE&ALPHA&SKILL + SKILL & log(SKILL) ),
            year_fe = @formula(OUTPUT ~ SKILL  + ALPHA&SKILL + TELE&ALPHA&SKILL + SKILL & log(SKILL) + fe(YEAR)),
            sector_fe = @formula(OUTPUT ~ SKILL  + ALPHA&SKILL + TELE&ALPHA&SKILL + SKILL & log(SKILL) + fe(SECTOR)),
            sector_year_fe = @formula(OUTPUT ~ SKILL  + ALPHA&SKILL + TELE&ALPHA&SKILL + SKILL & log(SKILL) + fe(SECTOR) + fe(YEAR))
        )
    end

    function fit_unweighted_models(data, formulas)
        return (
            base = reg(data, formulas.base),#, Vcov.robust),
            base_no_intercept = reg(data, formulas.base_no_intercept),#, Vcov.robust),
            year_fe = reg(data, formulas.year_fe),#, Vcov.robust),
            sector_fe = reg(data, formulas.sector_fe),#, Vcov.robust),
            sector_year_fe = reg(data, formulas.sector_year_fe),#, Vcov.robust)
        )
    end

    function fit_weighted_models(data, formulas)
        return (
            base = reg(data, formulas.base, weights=:EMPLOYMENT_THOUSANDS),#, Vcov.robust),
            base_no_intercept = reg(data, formulas.base_no_intercept, weights=:EMPLOYMENT_THOUSANDS),#, Vcov.robust),
            year_fe = reg(data, formulas.year_fe, weights=:EMPLOYMENT_THOUSANDS),#, Vcov.robust),
            sector_fe = reg(data, formulas.sector_fe, weights=:EMPLOYMENT_THOUSANDS),#, Vcov.robust),
            sector_year_fe = reg(data, formulas.sector_year_fe, weights=:EMPLOYMENT_THOUSANDS),#, Vcov.robust)
        )
    end

    function fit_regression_models(data)
        # Define formulas
        formulas = create_model_formulas()
        
        # Fit unweighted models
        models_unweighted = fit_unweighted_models(data, formulas)
        
        # Fit weighted models
        models_weighted = fit_weighted_models(data, formulas)
        
        return models_unweighted, models_weighted
    end

    function calculate_model_parameters(model)
        coef_dict = Dict(coefnames( model ) .=> coef(model))
        β₀, β₁, β₂, β₃ = [coef_dict[c] for c in ["SKILL",  "ALPHA & SKILL", "TELE & ALPHA & SKILL", "SKILL & log(SKILL)"]]
        A = β₀ # Firn productivity
        ψ₀ = (β₁ + 1) / A 
        ψ = β₂ / A
        ϕ = β₃ / A
        return A, ψ₀, ψ, ϕ
    end

    function create_psi_distribution(ind_tele, bds, model)

        A, ψ₀, ψ, ϕ = calculate_model_parameters(model)

        # Copy ind_3_tele and calculate Ψ
        data_density = copy(ind_tele)
        data_density[!, "PSI"] = ψ .* data_density[!, "TELE"] 

        # Filter bds_3d to the year 2022 and keep only the relevant columns
        subset_bds = filter(row -> row.year == 2022, bds)[!, ["job_creation_births", "NAICS"]]
        
        # Merge ind_3_tele with the subset on the "NAICS" column (left join)
        data_density = leftjoin(data_density, subset_bds, on = "NAICS")
        
        # Convert "job_creation_births" column to numeric (Float64), coercing non-parsable values to missing
        data_density[!, "JOB_CREATION_BIRTHS"] = [tryparse(Float64, string(x)) === nothing ? missing : tryparse(Float64, string(x)) for x in data_density[!, "job_creation_births"]]
        
        # Drop the original "job_creation_births" column
        select!(data_density, Not("job_creation_births"))

        # Drop missing values and sort by ψ
        dropmissing!(data_density)
        sort!(data_density, :PSI)
        
        return data_density
    end

    function fit_kde_psi(ψ, weights; num_grid_points=100, bandwidth=1, boundary=false, engine = "julia") 
        # Fit KDE with specified grid points
        if boundary
            min_x, max_x = boundary
        else
            min_x = minimum(ψ)
            max_x = maximum(ψ)
        end

        if engine == "julia"
            kde_result = kde(
                        ψ,
                        boundary = (min_x, max_x),
                        bandwidth=bandwidth,
                        npoints=num_grid_points
                    )
            # Extract grid and density values
            grid = kde_result.x
            estimated_density = kde_result.density
            # Normalize probabilities to sum to 1
            probabilities = estimated_density ./ sum(estimated_density)
        elseif engine == "python"
            # --- Step 1: Ensure required Python packages are available
            pyimport_conda("scipy.stats", "scipy")
            pyimport_conda("numpy", "numpy")        
            # Import Python libraries
            scipy_stats = pyimport("scipy.stats")
            np = pyimport("numpy")
            # --- Step 2: Convert Julia Vector to NumPy ---
            psi_values_np = np.array(ψ)
            weights_np = np.array(weights)
            # --- Step 3: Run the Python KDE ---
            kde_result = scipy_stats.gaussian_kde(psi_values_np, weights=weights_np)
            grid = np.linspace(np.min(psi_values_np), np.max(psi_values_np), num_grid_points)
            estimated_density = kde_result.evaluate(grid)
            prob = estimated_density / np.sum(estimated_density)  # Ensure sum to 1
            # --- Step 4: Convert Python Output to Julia ---
            grid = Vector{Float64}(grid)
            probabilities = Vector{Float64}(prob)
        end
        # Create distribution object
        psi_distribution = DataFrame(
            [
                :ψ => grid,
                :pdf => probabilities
            ]
        )
        # Add cdf
        psi_distribution[!, :cdf] = cumsum(psi_distribution[!, :pdf])
        return psi_distribution
    end
end

# visualization.jl
module Visualization
    using CairoMakie, LaTeXStrings

    export create_analysis_plots

    const COLOR1 = "#23373B"
    const FONT_CHOICE = "CMU Serif"

    function create_figure()
        return Figure(
            size = (1500, 500), 
            fonts = (; regular=FONT_CHOICE, italic=FONT_CHOICE, bold=FONT_CHOICE)
        )
    end

    function create_kde_plot!(pos, psi_distribution)
        ax = Axis(pos, 
            xlabel=L"\tilde{\psi}", 
            ylabel="Density",
            title=latexstring("Density of \$\\tilde{\\psi}\$"),
            xtickalign=1, 
            xticksize=10, 
            ytickalign=1, 
            yticksize=10,
            xgridvisible=false, 
            ygridvisible=false, 
            topspinevisible=false, 
            rightspinevisible=false
        )
        lines!(
                ax, psi_distribution[!, :ψ], psi_distribution[!, :pdf], linewidth=4, color=COLOR1
            )
        return ax
    end

    function create_histogram!(pos, data_density_3d)
        ax = Axis(pos, 
            xlabel="Teleworkability", 
            ylabel="Frequency",
            title="Histogram of Teleworkability",
            xtickalign=1, 
            xticksize=10, 
            ytickalign=1, 
            yticksize=10,
            xgridvisible=false, 
            ygridvisible=false, 
            topspinevisible=false, 
            rightspinevisible=false
        )
        hist!(
                ax, data_density_3d[!, :PSI], 
                weights=data_density_3d[!, :JOB_CREATION_BIRTHS], 
                bins=20,
                normalization=:probability, 
                color=COLOR1, 
                gap=0.3
            )
        return ax
    end

    function create_ecdf_plot!(pos, psi_distribution)
        ax = Axis(pos, 
            xlabel="Teleworkability", 
            ylabel="Cumulative Probability",
            title="Empirical CDF of Teleworkability",
            xtickalign=1, 
            xticksize=10, 
            ytickalign=1, 
            yticksize=10,
            xgridvisible=false, 
            ygridvisible=false, 
            topspinevisible=false, 
            rightspinevisible=false
        )
        lines!(
                ax, psi_distribution[!, :ψ], psi_distribution[!, :cdf], linewidth=4, color=COLOR1
            )
        return ax
    end

    function create_analysis_plots(data_density, psi_distribution)
        # # Create individual plots
        fig1 = create_figure()
        
        Axis(fig1[1, 1], 
            title="Density of Teleworkability",
            xlabel="Teleworkability", 
            ylabel="Density",
            xtickalign=1, 
            xticksize=10, 
            ytickalign=1, 
            yticksize=10,
            xgridvisible=false, 
            ygridvisible=false, 
            topspinevisible=false, 
            rightspinevisible=false
        )

        density!(
            data_density[!, :PSI],
            color = (COLOR1, 0.0),
            strokecolor = COLOR1, 
            strokewidth = 3,
            strokearound = true
            )
        
        create_histogram!(fig1[1, 2], data_density)

        Axis(fig1[1, 3], 
            title="Density of Teleworkability",
            xlabel="Teleworkability", 
            ylabel="Density",
            xtickalign=1, 
            xticksize=10, 
            ytickalign=1, 
            yticksize=10,
            xgridvisible=false, 
            ygridvisible=false, 
            topspinevisible=false, 
            rightspinevisible=false
        )

        ecdfplot!(data_density[!, :PSI], color=COLOR1, linewidth=3)

        fig2 = create_figure()
        create_kde_plot!(fig2[1, 1], psi_distribution)
        create_ecdf_plot!(fig2[1, 2], psi_distribution)

        return fig1, fig2
    end
end

# main.jl    
# Load data
paths = define_paths();
wfh_share_estimation, bls, onet_skill = DataLoader.load_datasets(paths);
# onet_skill[!, :SKILL_INDEX] .= onet_skill[!, :cognitive]
# onet_skill = onet_skill[!, [:ONET_SOC_CODE, :SKILL_INDEX]]
bls_ind_3d, bls_ind_4d = DataLoader.process_bls_data(bls);
productivity_3d, productivity_4d = DataLoader.load_productivity_data(paths);
wfh_rates = DataLoader.load_wfh_data(paths.wfh_rates);
bds_3d = DataLoader.load_bds_data();
onet_soc_xwalk = DataLoader.load_onet_soc_xwalk(paths.onet_soc_xwalk);

# # Process data
wfh_rates_3d = DataProcessor.process_wfh_rates(wfh_rates);
telework_skill_scores = DataProcessor.calculate_final_scores(bls_ind_3d, wfh_share_estimation, onet_skill, onet_soc_xwalk);
regression_data = DataProcessor.prepare_regression_data(productivity_3d, telework_skill_scores, wfh_rates_3d);

# # Fit models
models_unweighted, models_weighted = Models.fit_regression_models(regression_data);

# # Get coefficients
A, ψ₀, _, ϕ = Models.calculate_model_parameters( models_unweighted.base )

# Load /project/high_tech_ind/WFH/WFH/data/processed/bls/oews/oews_all_2023.csv
using CSV, DataFrames, StatsBase
owes = CSV.read(paths.bls, DataFrame)
owes_us_cross_ind_detailed = filter(row -> row.AREA_TITLE == "U.S." && row.NAICS_TITLE == "Cross-industry" && row.O_GROUP == "detailed", owes)
# Keep only US-Cross Industry-Detailed Occupation level data 
owes_us_cross_ind_detailed = owes_us_cross_ind_detailed[!, [:OCC_CODE, :TOT_EMP]]
# Convert :TOT_EMP to Float64 (parse any string as nan)
owes_us_cross_ind_detailed.TOT_EMP = [tryparse(Float64, string(x)) === nothing ? missing : tryparse(Float64, string(x)) for x in owes_us_cross_ind_detailed.TOT_EMP]
# Add OCC_CODE to onet_skill using onet_soc_xwalk
onet_skill = innerjoin(onet_skill, onet_soc_xwalk[!, [:ONET_SOC_CODE, :OCC_CODE]], on=:ONET_SOC_CODE)
# Groupby onet_skill by OCC_CODE and average the  SKILL_INDEX
skill_by_occupation = combine(groupby(onet_skill, :OCC_CODE), :SKILL_INDEX => mean => :SKILL_INDEX)
# Add the SKILL_INDEX to the owes_us_cross_ind_detailed
owes_us_cross_ind_detailed = leftjoin(owes_us_cross_ind_detailed, skill_by_occupation, on=:OCC_CODE)
# Print a list of codes with missing skill index
missing_skill_codes = filter(row -> ismissing(row.SKILL_INDEX), owes_us_cross_ind_detailed);
println("Codes with missing skill index:")
println(missing_skill_codes[!, :OCC_CODE])
# Calculate and print the percentage of employment with missing skill index
missing_emp_pct = sum(skipmissing(missing_skill_codes.TOT_EMP)) / sum(skipmissing(owes_us_cross_ind_detailed.TOT_EMP)) * 100;
println("Percentage of employment with missing skill index: $(round(missing_emp_pct, digits=2))%")
# Drop missing values
owes_us_cross_ind_detailed = dropmissing(owes_us_cross_ind_detailed, [:SKILL_INDEX])
# Compute percentage of total employment
owes_us_cross_ind_detailed[!, :PCT_EMP] = owes_us_cross_ind_detailed.TOT_EMP ./ sum(skipmissing(owes_us_cross_ind_detailed.TOT_EMP))
# Save to /project/high_tech_ind/WFH/WFH/data/results/skill_dist.csv
CSV.write("/project/high_tech_ind/WFH/WFH/data/results/skill_dist.csv", owes_us_cross_ind_detailed)


using StatsBase, CairoMakie, DataFrames

onet_skill = innerjoin(onet_skill, onet_soc_xwalk[!, [:ONET_SOC_CODE, :OCC_CODE]], on=:ONET_SOC_CODE)
# Group by OCC_CODE and calculate the mean of SKILL_INDEX
skill_by_occupation = combine(groupby(onet_skill, :OCC_CODE), :SKILL_INDEX => mean => :SKILL_INDEX)

# Select BLS data for aggregate US Level Data all industries
bls_us_cross_industry = filter(
    row -> row.AREA_TITLE == "U.S." &&
    row.NAICS_TITLE == "Cross-industry" &&
    row.O_GROUP == "detailed",
    bls
)
bls_us_cross_industry = bls_us_cross_industry[!, [:OCC_CODE, :TOT_EMP]]
bls_us_cross_industry.TOT_EMP = [tryparse(Float64, string(x)) === nothing ? missing : tryparse(Float64, string(x)) for x in bls_us_cross_industry.TOT_EMP]
bls_us_cross_industry[!, :PCT_EMP] = bls_us_cross_industry.TOT_EMP ./ sum(skipmissing(bls_us_cross_industry.TOT_EMP))

onet_skill = innerjoin(onet_skill, bls_us_cross_industry[!, [:OCC_CODE, :PCT_EMP]], on=:OCC_CODE)
rename!(onet_skill, :PCT_EMP => :WEIGHT)

# Plot the density of SKILL in the onet_skill DataFrame using weights
fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1], title="Density of SKILL", xlabel="SKILL", ylabel="Density")
density!(ax, onet_skill.SKILL_INDEX, weights=onet_skill.WEIGHT, color=:blue)
fig

# Fit a normal distribution to the data
using Distributions

normal_dist = fit_mle(Normal, onet_skill.SKILL_INDEX, onet_skill.WEIGHT)

# Create plot
fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1],
    title = "Density of SKILL",
    xlabel = "SKILL",
    ylabel = "Density",
    titlesize = 20,
    xlabelsize = 16,
    ylabelsize = 16
)

# Plot histogram with density normalization
hist!(ax, onet_skill.SKILL_INDEX, 
    weights = onet_skill.WEIGHT,
    bins = 20,
    color = (:blue, 0.5),
    strokecolor = :black,
    strokewidth = 1,
    normalization = :pdf  # Critical for matching density scale
)

# Plot fitted distribution
x_range = range(minimum(onet_skill.SKILL_INDEX), maximum(onet_skill.SKILL_INDEX), 100)
lines!(ax, x_range, pdf.(normal_dist, x_range),
    color = :red,
    linewidth = 3,
    label = "Fitted Normal"
)

axislegend(ax, position = :rt, framecolor = :transparent)

fig

# # Get psi distribution
data_density_3d = Models.create_psi_distribution(telework_skill_scores, bds_3d, models_unweighted.base);
psi_distribution = Models.fit_kde_psi(
    data_density_3d[!, :PSI], data_density_3d[!, :JOB_CREATION_BIRTHS];
    num_grid_points=50, 
    bandwidth=1, 
    boundary=false,
    engine="python"
);

# # Create visualizations
analysis_plots = Visualization.create_analysis_plots(data_density_3d, psi_distribution);

analysis_plots[1]
analysis_plots[2]

using CSV
# Save data_density_3d to CSV
CSV.write("/project/high_tech_ind/WFH/WFH/data/results/data_density_3d.csv", data_density_3d)
coef_dict = Dict(coefnames( models_unweighted.base ) .=> coef(models_unweighted.base))

β₀, β₁, β₂, β₃ = [coef_dict[c] for c in ["SKILL",  "ALPHA & SKILL", "TELE & ALPHA & SKILL", "SKILL & log(SKILL)"]]
A = β₀ # Firn productivity
ψ₀ = (β₁ + 1) / A 
ψ₁ = β₂ / A
ϕ = β₃ / A


fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1], title="Density of SKILL", xlabel="SKILL", ylabel="Density")
density!(ax, onet_skill.SKILL_INDEX, color = (:red, 0.3), strokecolor = :red, strokewidth = 3, strokearound = true)
# fit a normal distribution to the data
normal_dist = fit_mle(Normal, onet_skill.SKILL_INDEX)
# get the pdf of the normal distribution
x_range = range(minimum(onet_skill.SKILL_INDEX), maximum(onet_skill.SKILL_INDEX), length=100)
pdf_normal = pdf(normal_dist, x_range)
# plot the normal distribution
lines!(ax, x_range, pdf_normal, color = :red, linewidth = 3, label = "Fitted Normal")
fig