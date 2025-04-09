// ********************************************************************************
// * author: Mitchell Valdes-Bobes (valdsbobes@wisc.edu)
// * date created: 2025-02-16
// * Special thanks to Giselle Labrador Badía for her contributions.
// * This file sets and initializes the path for the project.
// ********************************************************************************



local unix = 0
global project_name = "WFH"

if `unix' ==1{
	global dir "/Users/mitchv34/Work/$project_name"
	cd `dir'
}
else {
	global dir = "V:\high_tech_ind\WFH/$project_name"
	cd `dir'
}

global path_data = "$dir/data/processed/acs/"
global path_code = "$dir/src/empirical/wfh_wage_facts/"
global path_tables = "$dir/docs/tables"
// *******************************************************