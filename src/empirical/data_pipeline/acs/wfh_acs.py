# %% Import packages
import pandas as pd
import numpy as np

# %% Define functions
#? Read ACS data
def read_acs_data(path, min_year=None, max_year=None):
    """
    Read ACS data from a CSV file and filter it based on the specified minimum and maximum years.

    Parameters:
    - path (str): The path to the CSV file.
    - min_year (int, optional): The minimum year to filter the data. Default is None.
    - max_year (int, optional): The maximum year to filter the data. Default is None.

    Returns:
    - data (pd.DataFrame): The filtered ACS data as a pandas DataFrame.
    """

    # Define data types
    dtypes = {
        "YEAR": int,
        "STATEFIP" : str,
        "PUMA" : str,
        "PERWT" : float,
        "RACE" : str,
        "RACED" : str,
        "EDUC" : str,
        "EDUCD" : str,
        "CLASSWKR": str,
        "CLASSWKRD": str,
        "INDNAICS": str,
        "OCCSOC" : str,
        "TRANWORK": str,
        "TRANTIME": float,
        "INCWAGE": float,
        "UHRSWORK": float,
        "INCTOT": float
    }
    # Read data
    data = pd.read_csv(path, compression='gzip', low_memory=False, dtype=dtypes)

    # Filter data
    if min_year is not None:
        data = data[data['YEAR'] >= min_year]
    if max_year is not None:
        data = data[data['YEAR'] <= max_year]

    return data

#? Filter ACS data
def filter_acs_data(data, **kwargs):
    """
    Filter ACS (American Community Survey) data based on specified criteria.

    Args:
        data (DataFrame): The input ACS data.
        **kwargs: Additional keyword arguments for specifying filter criteria.

    Returns:
        DataFrame: The filtered ACS data.

    """
    
    # Filter data
    ## Minimum hours worked
    if ( "UHRSWORK" in data.columns ) or ("hours_worked_lim" in kwargs):
        if "hours_worked_lim" not in kwargs:
            kwargs["hours_worked_lim"] = 35
        if "UHRSWORK" not in data.columns:
            print("Warning: 'UHRSWORK' column not found in the data. Skipping filter based on minimum hours worked.")
        else:
            data = data[data['UHRSWORK'] > kwargs["hours_worked_lim"]]

    # If data on wage and hours worked is available, calculate wage per hour
    if 'INCWAGE' in data.columns and 'UHRSWORK' in data.columns:
        data.loc[data.index, 'WAGE'] = data.INCWAGE / (data.UHRSWORK * 52)
        # Drop the original wage and hours worked columns
        data = data.drop(columns=['INCWAGE', 'UHRSWORK'])

    ## Minimum wage
    if ( "WAGE" in data.columns ) or "wage_lim" in kwargs:
        if "wage_lim" not in kwargs:
            kwargs["wage_lim"] = 5
        if "WAGE" not in data.columns:
            print("Warning: 'WAGE' column not found in the data. Skipping filter based on minimum wage.")
        else:
            data = data[data['WAGE'] > kwargs["wage_lim"]]

    ## Class of worker
    if ( "CLASSWKR" in data.columns ) or "class_of_worker" in kwargs:
        if "class_of_worker" not in kwargs:
            kwargs["class_of_worker"] = ['2'] # Work for wages
        if "CLASSWKR" not in data.columns:
            print("Warning: 'CLASSWKR' column not found in the data. Skipping filter based on class of worker.")
        else:
            data = data[data['CLASSWKR'].isin(kwargs["class_of_worker"])]


    return data

#? Modify the industry codes
def modify_industry_codes(data):
    """
    Modifies the industry codes in the given data by performing the following steps:
    1. Strips leading and trailing whitespace from the 'INDNAICS' column.
    2. Removes rows with 'INDNAICS' value of "0".
    3. Removes rows with military industry codes: "928110P1", "928110P2", "928110P3", "928110P4", "928110P5", "928110P6", "928110P7".
    4. Removes rows with 'INDNAICS' value of "999920" (unemployed).

    Args:
        data (pandas.DataFrame): The input data containing the 'INDNAICS' column.

    Returns:
        pandas.DataFrame: The modified data with industry codes filtered.

    """
    # Filter INDNAICS
    data.INDNAICS = data.INDNAICS.str.strip()
    # Remove rows with missing values
    data = data[data.INDNAICS != "0"]
    # Remove military industry codes
    data = data[data.INDNAICS.isin(["928110P1", "928110P2", "928110P3", "928110P4", "928110P5", "928110P6", "928110P7"]) == False]
    # Remove unemployed
    data = data[data.INDNAICS != "999920"]
    
    return data

# ? Create aggregator for occupation codes
def create_aggregator(path):
    """
    Create an aggregator for SOC (Standard Occupational Classification) data.

    Parameters:
    path (str): The path to the Excel file containing the SOC data.

    Returns:
    pandas.DataFrame: The aggregated SOC data.

    """
    soc_2018_struct = pd.read_excel(path, skiprows=7)

    group_soc_data = pd.DataFrame()
    major_occ = soc_2018_struct['Major Group'].unique().tolist()

    for mo in major_occ:
        if not isinstance(mo, str):
            continue
        minor_group = soc_2018_struct.loc[soc_2018_struct["Minor Group"].str.startswith(mo[:2]) == True, "Minor Group"].unique().tolist()

        for mg in minor_group:
            broad_group = soc_2018_struct.loc[soc_2018_struct["Broad Group"].str.startswith(mg[:4]) == True, "Broad Group"].unique().tolist()

            for bg in broad_group:
                detailed_occupation = soc_2018_struct.loc[soc_2018_struct["Detailed Occupation"].str.startswith(bg[:6]) == True, "Detailed Occupation"].unique().tolist()

                bg_list = [bg] * len(detailed_occupation)
                mg_list = [mg] * len(detailed_occupation)
                mo_list = [mo] * len(detailed_occupation)

                group_soc_data = pd.concat(
                    [   group_soc_data,
                        pd.DataFrame({   
                                "Detailed Occupation": detailed_occupation,
                                "Broad Group": bg_list, 
                                "Minor Group": mg_list, 
                                "Major Group": mo_list
                            })])

    return group_soc_data

# ? Convert using crosswalk
def convert_using_cw(code, cw, keep_original=True):
    """
    Convert a code using a code-to-value dictionary.

    Parameters:
    code (str): The code to be converted.
    cw (dict): The code-to-value dictionary.

    Returns:
    str: The converted code if it exists in the dictionary, otherwise the original code.
    """
    # If the NAICS code is not in the dictionary return the same code
    if code not in cw.keys():
        if keep_original:
            return code
        else:
            return np.nan
    # If the NAICS code is in the dictionary return the value of the dictionary
    return cw[code]

# ? Modify the occupation codes
def modify_occupation_codes(data, aggregator, occ_col = "OCCSOC", threshold=2):
    """
    Modify the occupation codes in the given data DataFrame.

    Args:
        data (pandas.DataFrame): The DataFrame containing the occupation codes.
        aggregator (pandas.DataFrame): The DataFrame containing the mapping of occupation codes.
        occ_col (str, optional): The name of the column containing the occupation codes. Defaults to "OCCSOC".
        threshold (int, optional): The maximum number of 'X' or 'Y' characters allowed in the OCCSOC column. Defaults to 2.

    Returns:
        pandas.DataFrame: The modified DataFrame with updated occupation codes.

    """
    data.dropna(subset=[occ_col], inplace=True)
    data = data[data[occ_col] != "nan"]
    # Modify the occupation codes to look like XX-XXXX
    # Check if data is in the format of XX-XXXX
    if data[occ_col].str.contains("-").all():
        pass
    else:
        data[occ_col] = data[occ_col].apply(lambda x: "-".join([x[:2], x[2:]]))
    # Count how many X or Y char in the OCCSOC column Drop rows that dont meet the threshold
    data = data[data[occ_col].str.count('X') + data[occ_col].str.count('Y') <= threshold]
    # Replace "X" with 0 in the OCCSOC column
    data[occ_col] = data[occ_col].str.replace("X", "0").str.replace("Y", "0")

    data[occ_col + "_group"] = np.nan
    data.loc[data[occ_col].isin(aggregator["Detailed Occupation"]), occ_col + "_group"] = "detailed"
    data.loc[data[occ_col].isin(aggregator["Broad Group"]), occ_col + "_group"] = "broad"
    data.loc[data[occ_col].isin(aggregator["Minor Group"]), occ_col + "_group"] = "minor"
    data.loc[data[occ_col].isin(aggregator["Major Group"]), occ_col + "_group"] = "major"
    # Drop rows where OCCSOC_group remains NaN
    data = data[data[occ_col + "_group"] == data[occ_col + "_group"]]
    # Create dictionaries to group SOC codes from detailed to Broad Group
    soc_2018_dict_broad       = dict(zip(aggregator["Detailed Occupation"], aggregator["Broad Group"]))
    soc_2018_dict_minor       = dict(zip(aggregator["Detailed Occupation"], aggregator["Minor Group"]))
    soc_2018_dict_broad_minor = dict(zip(aggregator["Broad Group"], aggregator["Minor Group"]))

    # Create Detailed Broad and Minor columns
    data[occ_col + "_detailed"] = np.nan
    data[occ_col + "_broad"]    = np.nan
    data[occ_col + "_minor"]    = np.nan

    # If OCCSOC is detailed then store it in the detailed column leave the rest as NaN
    data.loc[data[occ_col + "_group"] == "detailed", occ_col + "_detailed"] = data[occ_col]
    # If OCCSOC is broad then store it in the broad, group detailed to broad and store it in the broad column leave the rest as NaN
    data.loc[data[occ_col + "_group"] == "broad", occ_col + "_broad"] = data[occ_col]
    data.loc[data[occ_col + "_group"] == "detailed", occ_col +"_broad"] = data.loc[data[occ_col + "_group"] == "detailed", occ_col].apply(
                                                                                    lambda x: convert_using_cw(x, soc_2018_dict_broad))

    # If OCCSOC is minor then store it in the minor, group broad to minor and store it in the minor column leave the rest as NaN
    data[occ_col + "_minor"] = data[occ_col + "_broad"].apply(lambda x: convert_using_cw(x, soc_2018_dict_broad_minor))
    data.loc[data[occ_col + "_group"] == "minor", occ_col + "_minor"] = data[occ_col]

    return data

# ? Aggrefate PUMA codes to CBSA codes
def aggregate_puma_to_cbsa(data, puma_crosswalk):
    """
    Aggregates PUMA (Public Use Microdata Area) codes to CBSA (Core Based Statistical Area) codes.

    Args:
        data (pandas.DataFrame): The input data containing STATEFIP and PUMA columns.
        puma_crosswalk (str): The file path to the PUMA to CBSA crosswalk CSV file.

    Returns:
        pandas.DataFrame: The input data with an additional column 'cbsa20' containing the corresponding CBSA codes.

    """
    # Pad the state and PUMA codes with leading zeros
    data['STATEFIP'] = data['STATEFIP'].str.zfill(2)
    data['PUMA'] = data['PUMA'].str.zfill(5)

    # Create a new column with the state and PUMA code
    data['state_puma'] = data['STATEFIP'] + '-' + data['PUMA']
    # Drop the state and PUMA columns
    data = data.drop(columns=['STATEFIP', 'PUMA'])
    
    # Read the PUMA to CBSA crosswalk
    puma_to_cbsa = pd.read_csv(puma_crosswalk, dtype={'state_puma': str, 'cbsa20': str})
    # Create croswalk dictionary
    puma_to_cbsa_dict = dict(zip(puma_to_cbsa['state_puma'], puma_to_cbsa['cbsa20']))
    # Assign every PUMA to the CBSA 
    data['cbsa20'] = data['state_puma'].apply(lambda x: convert_using_cw(x, puma_to_cbsa_dict, False))
    
    return data

# ? Telework Index
def telework_index(data, soc_aggregator, occ_col = "OCCSOC"):
    """
    Calculate the telework index for different occupation codes based on the provided data and SOC aggregator.

    Parameters:
    - data (pandas.DataFrame): The input data containing occupation codes.
    - soc_aggregator (pandas.DataFrame): The SOC aggregator containing mapping between detailed, broad, and minor occupation codes.
    - occ_col (str, optional): The name of the column containing the occupation codes. Default is "OCCSOC".

    Returns:
    - data (pandas.DataFrame): The input data with additional columns representing the telework index for different occupation codes.
    """
    
    # Load Data
    dn_clasification = pd.read_csv("https://raw.githubusercontent.com/jdingel/DingelNeiman-workathome/master/occ_onet_scores/output/occupations_workathome.csv") 
    # Rename columns to upper case
    dn_clasification.rename(columns = lambda x: x.upper(), inplace = True)
    # ONETSOCCODE is the ONET code which is more detailed than the SOC code (OCC_CODE) so we split it to get the OCC_CODE and ONET_DETAIL
    # I'm going to drop the rows where ONET_DETAIL is not "00" because I want to keep only the same occupations as the BLS data
    # checked that they all have the same TELEWORKABLE value
    dn_clasification[['OCC_CODE', 'ONET_DETAIL']] = dn_clasification['ONETSOCCODE'].str.split(".", expand=True)
    # Drop where ONET_DETAIL not "00"
    dn_clasification = dn_clasification[dn_clasification['ONET_DETAIL'] == "00"]
    # Keep only relevant columns
    dn_clasification = dn_clasification[['OCC_CODE', 'TITLE', 'TELEWORKABLE']]

    # Create a dictionary to map the OCC_CODE to the TELEWORKABLE value 
    telework_dict = dict(zip(dn_clasification['OCC_CODE'], dn_clasification['TELEWORKABLE']))

    # Create a new column with the TELEWORKABLE value based on the OCCSOC
    data['TELEWORKABLE_OCCSOC'] = data['OCCSOC'].apply(lambda x: convert_using_cw(x, telework_dict, False))

    # Create dictionaries to group SOC codes from detailed to Broad Group
    soc_2018_dict_broad       = dict(zip(soc_aggregator["Detailed Occupation"], soc_aggregator["Broad Group"]))
    soc_2018_dict_minor       = dict(zip(soc_aggregator["Detailed Occupation"], soc_aggregator["Minor Group"]))
    soc_2018_dict_broad_minor = dict(zip(soc_aggregator["Broad Group"], soc_aggregator["Minor Group"]))
    
    # Using the SOC aggregator create a telework index for OCCSOC_detailed, OCCSOC_broad, OCCSOC_minor
    # Start by adding broad and minor columns to dn classification
    dn_clasification["BROAD"] = dn_clasification["OCC_CODE"].apply(lambda x: convert_using_cw(x, soc_2018_dict_broad))
    dn_clasification["MINOR"] = dn_clasification["OCC_CODE"].apply(lambda x: convert_using_cw(x, soc_2018_dict_minor))

    # Create new classifications by grouping the OCC_CODE to the BROAD and MINOR groups
    dn_clasification_BROAD = dn_clasification.groupby("BROAD")["TELEWORKABLE"].mean().reset_index()   
    dn_clasification_MINOR = dn_clasification.groupby("MINOR")["TELEWORKABLE"].mean().reset_index()  

    # Create dictionaries to map the DETAILEDm, BROAD and MINOR groups to the TELEWORKABLE value
    telework_dict_detailed = dict(zip(dn_clasification['OCC_CODE'], dn_clasification['TELEWORKABLE']))
    telework_dict_broad = dict(zip(dn_clasification_BROAD['BROAD'], dn_clasification_BROAD['TELEWORKABLE']))
    telework_dict_minor = dict(zip(dn_clasification_MINOR['MINOR'], dn_clasification_MINOR['TELEWORKABLE']))

    # Create new columns with the TELEWORKABLE value based on the OCCSOC_detailed, OCCSOC_broad, OCCSOC_minor
    data['TELEWORKABLE_' + occ_col + '_detailed'] = data[occ_col +'_detailed'].apply(lambda x: convert_using_cw(x, telework_dict_detailed, False))
    data['TELEWORKABLE_' + occ_col + '_broad']    = data[occ_col +'_broad'].apply(lambda x: convert_using_cw(x, telework_dict_broad, False))
    data['TELEWORKABLE_' + occ_col + '_minor']    = data[occ_col +'_minor'].apply(lambda x: convert_using_cw(x, telework_dict_minor, False))

    return data

# ? Add WFH index to the data (based on TRANWORK) 
def wfh_index(data):  
    """
    Calculate the work-from-home (WFH) index for each record in the given data.

    Parameters:
    data (pandas.DataFrame): The input data containing the TRANWORK column.

    Returns:
    pandas.DataFrame: The input data with an additional column 'WFH' indicating whether the record represents a work-from-home scenario.
    """

    data.loc[:, 'WFH'] = data.TRANWORK == "80"
    return data
    
# ? Save the data
def save_data(data, path):
    """
    Save the data to a CSV file.

    Args:
        data (pandas.DataFrame): The data to be saved.
        path (str): The path to save the data.

    """
    data.to_csv(path, index=False)


# %% Main
if __name__ == "__main__":
    # Specify the paths
    path_acs = "../../data/acs/usa_00135.csv.gz"
    path_cps = "../../data/cps/asec.csv.gz"
    path_soc_aggregator = "../../data/aux_and_croswalks/SOC_structure_2018.xlsx"
    path_puma_crosswalk = "../../data/aux_and_croswalks/puma_to_cbsa.csv"

    # Read the ACS data and crosswalks
    data = read_acs_data(path_acs, min_year=2013)
    soc_aggregator = create_aggregator(path_soc_aggregator)

    # Filter the ACS data
    data = filter_acs_data(data, hours_worked_lim=35, wage_lim=5, class_of_worker=['2'])
    # Modify the industry codes
    data = modify_industry_codes(data)
    # Modify the occupation codes
    data = modify_occupation_codes(data, soc_aggregator)
    # Aggregate PUMA codes to CBSA codes
    data = aggregate_puma_to_cbsa(data, path_puma_crosswalk)

    # Calculate the telework index of each individual's occupation
    data = telework_index(data, soc_aggregator)

    # Calculate the WFH index of each worker
    data = wfh_index(data)

    # Save the data
    save_data(data, "../../data/acs/proc/usa_00135_filtered.csv")

    # # Data CPS
    # data_cps = pd.read_csv(path_cps, compression='gzip', low_memory=False)
    # data_cps["OCCSOC"] = data_cps["OCCSOC"].astype(str)
    # data_cps["OCCSOCLY"] = data_cps["OCCSOCLY"].astype(str)
    # # Drop military specific occupations
    # data_cps = data_cps[~data_cps.OCCSOC.str.startswith("55")]
    # data_cps = data_cps[~data_cps.OCCSOCLY.str.startswith("55")]
    # # Modify the occupation codes
    # data_cps = modify_occupation_codes(data_cps, soc_aggregator, occ_col = "OCCSOC")
    # data_cps = modify_occupation_codes(data_cps, soc_aggregator, occ_col = "OCCSOCLY")
    # # Calculate the telework index of each individual's occupation
    # data_cps = telework_index(data_cps, soc_aggregator, occ_col = "OCCSOC")
    # data_cps = telework_index(data_cps, soc_aggregator, occ_col = "OCCSOCLY")

    # # Save the data
    # save_data(data_cps, "../../data/cps/proc/asec_final.csv.gz")

# TODO: Add age filter
# TODO: Find Tenure in job in data 
# TODO: Filter by weeks worked 

# %%
# average_wfh_by_year = data.groupby('YEAR')['WFH'].apply(lambda x: (x * data.loc[x.index, 'PERWT']).sum() / data.loc[x.index, 'PERWT'].sum())
# average_telewk_by_year = data.groupby('YEAR')['TELEWORKABLE_OCCSOC_minor'].apply(lambda x: (x * data.loc[x.index, 'PERWT']).sum() / data.loc[x.index, 'PERWT'].sum())

# %%
