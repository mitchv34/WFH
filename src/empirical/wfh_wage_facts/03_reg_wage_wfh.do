// ********************************************************************************
// * author: Mitchell Valdes-Bobes (valdsbobes@wisc.edu)
// * date created: 2025-02-16
// * Special thanks to Giselle Labrador Badía for her contributions.
// * This file sets and initializes the path for the project.
// ********************************************************************************

// * This files runs OLS regressions of the type Wage = f(WFH, X, FE) for the project.

// * Variables in raw data
// year perwt age race educ wage educd classwkrd occsoc_group occsoc_detailed occsoc_broad cbsa20 wfh teleworkable



//************************************************************************
// * Data and set up
// ********************************************************************************

encode $occup_var, gen(occup_cat)

local y_var = "$y_var"
local main_x= "wfh_cat"

local filename = "reg_wage_wfh_$add_fn"

local filename = "`filename'_$occup_var"

if "$cluster_type"=="robust"{
    di "$cluster_type"
    local vce_setting = "vce(robust)"
    local filename = "`filename'_robust"
}
else{
    di "$cluster_type"
    local vce_setting = "vce(cluster ${cluster_type})"
    di "`vce_setting'"
    local filename = "`filename'_cl_${cluster_type}"
}

if "$weight_var"!=""{
    di "$weight_var"
    local weight_cond = "[aweight =${weight_var}]"
    di "`weight_cond'"
    local filename = "`filename'_w"
}
else{
    local weight_cond = ""
}

di "`filename'"

// * Var names
label var wfh "1(WFH)"
label var wfh_cat "1(WFH)"
label var wage "Wage"
label var educ "Education"
label var educd "Education"
label var age "Age"
label var age2 "Age^2"
label var race "Race"

// ************************************************************************
// * Run regressions
// ************************************************************************

// * OLS basic FE 
eststo clear

local reg_list = ""
di "Occupation: $occup_var, yvar: `y_var', xvar: `main_x'"
di "Cluster: `cluster_type', vce: `vce_setting', weight: `weight_cond'"
foreach j in $dem_list{
    foreach i in $fe_list{
        di " ********* fe `i' ********* dem `j' *********"
        di " Running FE: ${fe`i'}  DEM: ${dem`j'}"

        local regij = "reg`i'`j'"
        local reg_list = "`reg_list' `regij'"

        local fe_l = "${fe`i'}"  // Store the fixed effects in a local macro
        local dem_l = "${dem`j'}" // Store the demographic variables in a local macro

        // # if occup in fe_`i'{ then YES} OTHERWISE {NO}
        if `i'==6{
            local dem_l = ""
        }
        if `j'==2{
            local dem_l = ""
            local fe_l = "`fe_l' `dem2'"
        }
        qui reghdfe `y_var' `main_x' `dem_l' `weight_cond', absorb(`fe_l') `vce_setting'
        est store `regij'

        // Check if each fixed effect variable is in $fe`i'
        

        // Initialize the row variables as "NO"
        local row_occup "NO"
        local row_ind "NO"
        local row_year "NO"
        local row_cbs "NO"
        local row_empt "NO"
        local row_dem "NO"

        // Check for each variable
        if strpos("`fe_l'", "occup") > 0 {
            local row_occup "YES"
            di "- Occupation FE"
        }
        if strpos("`fe_l'", "ind") > 0 {
            local row_ind "YES"
            di "- Industry FE"
        }
        if strpos("`fe_l'", "year") > 0 {
            local row_year "YES"
            di "- Year FE"
        }
        if strpos("`fe_l'", "cbsa20") > 0 {
            local row_cbs "YES"
            di "- CBSA FE"
        }
        if strpos("`fe_l'", "classwkrd") > 0 {
            local row_empt "YES"
            di "- Class worker FE"
        }
        if strpos("`fe_l'", "edu") > 0 {
            local row_dem "YES"
            di "- Advanced demographics"
        }

        // Add the row variables to the stored estimates
        estadd local row_occup "`row_occup'":`regij'
        estadd local row_ind "`row_ind'":`regij'
        estadd local row_year "`row_year'":`regij'
        estadd local row_cbs "`row_cbs'":`regij'
        estadd local row_empt "`row_empt'":`regij'
        estadd local row_dem "`row_dem'":`regij'
    }
}


esttab `reg_list', ///
    nogaps label nomtitles ///
    se star(* 0.1 ** 0.05 *** 0.001) ///
    keep(wfh_cat) ///
    stats(N r2) ///
    scalars("row_occup occupation FE" "row_ind industry FE" "row_year year FE" "row_cbs CBSA FE" "row_empt Employment FE" "row_dem Advanced demographics")



esttab `reg_list' using "$path_tables/`filename'.tex", replace ///
    nogaps ///
    se star(* 0.1 ** 0.05 *** 0.001) ///
    label ///
    nomtitles ///
    keep(wfh_cat) ///
    stats(N r2) ///
    title("Wage = f(WFH, X, FE)") ///
    addnotes("Occupation: `occup_var', yvar: `y_var', xvar: `main_x', cluster: `cluster_type'") ///
    scalars("row_occup occupation FE" "row_ind industry FE" "row_year year FE" "row_cbs CBSA FE" "row_empt Employment FE" "row_dem Advanced demographics") 
    

// Drop the new occupation category
drop occup_cat  
// ************************************************************************





