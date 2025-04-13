// ********************************************************************************
// * author: Mitchell Valdes-Bobes (valdsbobes@wisc.edu)
// * date created: 2025-02-16
// * Special thanks to Giselle Labrador Badía for her contributions.
// * This file sets and initializes the path for the project.
// ********************************************************************************

// * This files runs reduced form empirical facts for the project. 

// * Variables in raw data
// year perwt age race raced educ wage educd classwkrd
// occsoc_group occsoc_detailed occsoc_broad cbsa20 wfh teleworkable
//********************************************************************************
// * global variables and set up
// ********************************************************************************


global y_var = "wage"
global x_var = "wfh"

global weight_var = "perwt" //share log_asset

global dem1 = "age age2 i.race i.educ"
global dem2 = "age_cat_4 race educd"
global dem3 = "edu_age raced"  // TO RUN AS FE

global fe1 = "year"
global fe2 = "year cbsa20"
global fe3 = "year cbsa20 ind_cat"
global fe4 = "year cbsa20 occup_cat"
global fe5 = "year cbsa20 occup_cat classwkrd ind_cat"
global fe6 = "year cbsa20 occup_cat classwkrd ind_cat $dem3"
global fe7 = "year cbsa20 classwkrd ind_cat $dem3"


// *******************************************************
// * Iteration settings 

global year_min = 2010
global year_max = 2025
global remove_years = "" // years to remove

// what tables to run below
local cluster_list = "robust" // cbs_year occupm_educ " //robust cbs_year occupm_educ edu_age
local occu_list = "occsoc_minor occsoc_broad occsoc_detailed " //occsoc_minor

// from defines list of FE and DEM above which ones to run
global fe_list = "2 3 4 5 6"
global dem_list = "2"

// *******************************************************
// * Load and Clean the Data
// *******************************************************
//import delimited "$path_data/acs_136_processed.csv", clear
//do "$path_code/02_basic_cleaning.do"

// *******************************************************
// * Regression of wage on remote work
// *******************************************************
// global add_fn = ""
//
// foreach c in `cluster_list' { //"loan_losses" zscore2 loan_losses ratio_eqasset
//     di "Cluster: `c'" 
//     global cluster_type = "`c'"
//    
//     foreach o in `occu_list' { //occu_list
//         di "Occupation: `o'"
//         global occup_var = "`o'"
//         do "$path_code/03_reg_wage_wfh.do" 
//     }
//
// }

// *******************************************************
// * Run basic regressions with Teleworkable index
// *******************************************************

global add_fn = "Teleindex_ext_skill_occfe"

global occup_var = "occsoc_minor"

global occ_level = "occsoc_detailed"
local occ_level = "$occ_level" 

// * Run table summary for the project
global main_x1 = "teleworkable_`occ_level'"
global main_x2 = "teleworkable_`occ_level' wfh_cat"
global main_x3 = "teleworkable_`occ_level' skill_cognitive_`occ_level' skill_mechanical_`occ_level' skill_social_`occ_level'"
global main_x4 = "teleworkable_`occ_level' wfh_cat skill_cognitive_`occ_level' skill_mechanical_`occ_level' skill_social_`occ_level'"

global keep = "wfh_cat teleworkable_occsoc_detailed skill_cognitive_`occ_level' skill_mechanical_`occ_level' skill_social_`occ_level'"

global fe_list = "7 6"
global dem_list = "2"
global main_x_list = "1 2 3 4"


global cluster_list = "robust" // cbs_year occupm_educ " //robust cbs_year occupm_educ edu_age

foreach c in `cluster_list' { //"loan_losses" zscore2 loan_losses ratio_eqasset
    di "Cluster: `c'" 
    global cluster_type = "`c'"
    
	foreach y in wage{
		global y_var = "`y'"
		do "$path_code/04_reg_wage_wfh_teleindex.do"
    }

}



// //*************************************************
// // * Remove years Robustness

// // * Remove covid years 
// global remove_years = "2020 2021"

// global add_fn = "_covid"

// global y_var = "wage"

// global cluster_type = "robust"

// foreach o in `occu_list' { //occu_list
//     di "Occupation: `o'"
//     global occup_var = "`o'"
//     do "$path_code/03_reg_wage_wfh.do" 
// }

// //*************************************************

// // * Log wage

// global y_var = "log_wage"

// global add_fn = "_logwage"

// global remove_years = ""

// foreach c in `cluster_list' { 
//     di "Cluster: `c'" 
//     global cluster_type = "`c'"
    
//     foreach o in `occu_list' { //occu_list
//         di "Occupation: `o'"
//         global occup_var = "`o'"
//         do "$path_code/03_reg_wage_wfh.do" 
//     }

// }