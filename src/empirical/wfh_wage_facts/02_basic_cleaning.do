// ********************************************************************************
// * author: Mitchell Valdes-Bobes (valdsbobes@wisc.edu)
// * date created: 2025-02-16
// * Special thanks to Giselle Labrador Badía for her contributions.
// * This file sets and initializes the path for the project.
// ********************************************************************************

// This files prepares the data to run regrssions for the project.

// * Variables in raw data
// year perwt age race raced educ wage educd classwkrd occsoc_group occsoc_detailed occsoc_broad cbsa20 wfh
//************************************************************************

drop if wage == .
drop if year == .

drop if year<$year_min | year>$year_max

foreach y in $remove_years{
    drop if year == `y'
}

gen wfh_cat = 1 if wfh == "True"
replace wfh_cat = 0 if wfh == "False"

drop if wfh_cat == .

// create age categories, 4 year intervals
gen age_cat_4 = floor(age/4)
// gen age_cat_5 = floor(age/5)

// to string 
tostring occsoc_broad, replace
tostring occsoc_detailed, replace
tostring occsoc_minor, replace
replace occsoc_broad = occsoc_minor if occsoc_broad == ""
replace occsoc_detailed = occsoc_broad if occsoc_detailed == ""

replace teleworkable_occsoc_broad = teleworkable_occsoc_minor if  teleworkable_occsoc_broad == .
replace teleworkable_occsoc_detailed = teleworkable_occsoc_broad if  teleworkable_occsoc_detailed == .

// create categorical variables
// encode race, gen(race_cat)
// encode educ, gen(educ_cat)
encode $occup_var, gen(occup_cat)
encode indnaics, gen(ind_cat)
// create groups
egen edu_age = group(educ age_cat_4)
egen cbs_year = group(cbsa20 year)
egen occupm_educ = group(occsoc_minor educ)


// create log vars
gen log_wage = log(wage)
// gen log_age = log(age)
gen age2 = age^2