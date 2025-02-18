"""
Module Overview:
----------------
This module implements a data processing pipeline designed to transform raw employment and industrial data from an Excel file
into standardized CSV files suitable for further analysis across geographic and industry/occupation dimensions. It leverages 
mapping dictionaries for U.S. states and counties to identify standardized geographical identifiers and aggregates data where 
necessary.
Key Functionalities:
--------------------
1. setup_mappings():
    - Initializes mapping dictionaries, converting state names to their abbreviations and FIPS codes using an external library.
    - Reads auxiliary county-to-CBSA crosswalk files to create mappings for county codes to CBSA codes and names.
    - Logs information and errors during the mapping setup process.
2. process_sheet(sheet_name, df, state_to_abbrev, state_fips, county_cbsa_dict, county_cbsa_name_dict):
    - Processes individual data sheets by constructing a proper datetime column (YEAR_MONTH) and standardizing the column names.
    - Ensures key columns exist and assigns default values when missing.
    - Applies filtering and transformation logic based on the sheet type (e.g., occupation data, industry data, state or geographic levels).
    - Adjusts geographical information, mapping state names to FIPS or county codes to CBSA details as appropriate.
3. process_geog_sheet(processed_df):
    - Aggregates geographic data by county and recalculates key percentages.
    - Merges aggregated statistics back into the processed DataFrame.
    - Standardizes the 'AREA_TYPE' to indicate aggregated CBSA-level data.
4. process_all_sheets(data_dir, processed_data_dir, state_to_abbrev, state_fips, county_cbsa_dict, county_cbsa_name_dict):
    - Orchestrates reading an Excel file containing multiple sheets and directs the processing of each relevant sheet.
    - Saves processed outputs for each sheet as CSV files in a predetermined column order.
    - Handles specific cases, such as geographic sheets, by invoking additional processing to compute aggregated metrics.
Usage:
------
When executed as a standalone script, the module:
    - Sets up necessary state and county mappings.
    - Processes all relevant sheets from the designated Excel file.
    - Outputs cleaned and standardized CSV files, logging each step and any encountered errors for auditability.
"""
import os
import pandas as pd
import numpy as np
import logging
import argparse
import us


# Base paths
BASE_DIR = '.'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'wfh_map')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'wfh_map')
# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.basename(__file__)
log_file = os.path.join(current_dir, f'{script_name.replace(".py", ".log")}')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def setup_mappings():
    """
    Set up mapping dictionaries for state and county information.

    This function creates four dictionaries:
    1. state_to_abbrev: Maps state names to state abbreviations using the us package.
    2. state_fips: Maps state names to their FIPS codes using the us package.
    3. county_cbsa_dict: Maps county names to their corresponding CBSA codes by reading a CSV file.
    4. county_cbsa_name_dict: Maps county names to their corresponding CBSA names by reading a CSV file.

    Returns:
        tuple: A tuple containing:
            - state_to_abbrev (dict): Dictionary mapping state names to abbreviations.
            - state_fips (dict): Dictionary mapping state names to FIPS codes.
            - county_cbsa_dict (dict): Dictionary mapping county names to CBSA codes.
            - county_cbsa_name_dict (dict): Dictionary mapping county names to CBSA names.
    """
    logging.info("Starting to set up state and county mappings.")

    # Dictionary mapping state names to state abbreviations using the us package
    state_to_abbrev = {state.name: state.abbr for state in us.states.STATES}
    state_fips = {state.name: state.fips for state in us.states.STATES}
    logging.info("State mappings created.")

    try:
        # Dictionary mapping county names to CBSA codes (for aggregating to CBSA level)
        county_cbsa = pd.read_csv(os.path.join(BASE_DIR, 'data', "aux_and_croswalks", "geocorr2022_2504706033.csv"),
                                encoding='latin1', dtype=str)
        county_cbsa_dict = county_cbsa.set_index("county")["cbsa20"].to_dict()
        county_cbsa_name_dict = county_cbsa.set_index("county")["CBSAName20"].to_dict()
        logging.info("County CBSA mappings created successfully.")
    except Exception as e:
        logging.error("Error reading county CBSA data: %s", e)
        raise

    logging.info("Completed setting up all mappings.")
    return state_to_abbrev, state_fips, county_cbsa_dict, county_cbsa_name_dict
    

def process_sheet(sheet_name, df, state_to_abbrev, state_fips, county_cbsa_dict, county_cbsa_name_dict):
    """
    Process and transform a DataFrame representing employment or industrial data according to the specified sheet type.
    This function performs several operations:
    - Constructs a new 'YEAR_MONTH' datetime column from the 'Year' and 'Month' columns, then drops redundant columns.
    - Renames various columns to standardized names.
    - Ensures that the column 'N' exists, inserting it with NaN values if missing.
    - Filters the DataFrame for USA records if a 'Country' column exists, then drops the 'Country' column.
    - Sets occupation group details (e.g., 'O_GROUP', 'OCC_CODE', and 'OCC_TITLE') based on the provided sheet name.
    - Sets industry group details (e.g., 'I_GROUP', 'NAICS', and 'NAICS_TITLE') based on the sheet name.
    - Adjusts geographical information by mapping the 'PRIM_STATE' or county codes using provided dictionaries, depending on the sheet type.
    - Converts all column names to uppercase and reorders them.
    Parameters:
        sheet_name (str): The identifier for the type of sheet being processed. It influences how occupation, industry, and geographical data are handled.
        df (pandas.DataFrame): The input DataFrame containing raw employment/industry data.
        state_to_abbrev (dict): A mapping from full state names to their standard abbreviations.
        state_fips (dict): A mapping of state abbreviations or names to their corresponding FIPS codes.
        county_cbsa_dict (dict): A dictionary mapping county codes to Corresponding Metropolitan Statistical Area (or similar) codes.
        county_cbsa_name_dict (dict): A dictionary mapping county codes to their corresponding name representations for MSAs or similar geographies.
    Returns:
        pandas.DataFrame: The transformed DataFrame with standardized column names and adjusted data based on the sheet type.
    """
    
    # Create YEAR_MONTH from Year and Month and drop redundant columns
    df['Year_Month'] = pd.to_datetime(df['Year'].astype(str) + df['Month'], format='%Y%b')
    df.drop(columns=['Year', 'Month', 'Year-Month'], inplace=True)
    
    # Rename columns to a standard naming
    df.rename(columns={
        'SOC 2018 3-Digit Minor Group': "OCC_CODE",
        'SOC 2018 3-Digit Minor Group (Name)' : "OCC_TITLE",
        "NAICS 2022 3-Digit Industry Group": "NAICS",
        "NAICS 2022 3-Digit Industry Group (Name)": "NAICS_TITLE",
        "State/Region": "PRIM_STATE",
        "Geography Name": "AREA_TITLE",
        "Geography Type": "AREA_TYPE",
        "Geography Code": "AREA",
        "City": "AREA_TITLE",
    }, inplace=True)
    
    # Ensure column "N" exists
    if "N" not in df.columns:
        df["N"] = np.nan
        
    # Filter for USA if there is a Country column
    if "Country" in df.columns:
        df.loc[:, 'O_GROUP'] = 'minor'
        df = df[df.Country == 'USA']
        df.drop(columns=['Country'], inplace=True)

    # Set occupation group based on sheet name
    if sheet_name == "us_occ_by_month":
        df.loc[:, 'O_GROUP'] = 'minor'
    else:
        df.loc[:, 'O_GROUP'] = 'total'
        df.loc[:, 'OCC_CODE'] = '00-0000'
        df.loc[:, 'OCC_TITLE'] = 'All Occupations'

    # Set industry group based on sheet name
    if sheet_name == "us_ind_by_month":
        df.loc[:, 'I_GROUP'] = "3-digit"
        df['NAICS'] = df['NAICS'].astype(str) + '000'
    else:
        df.loc[:, 'I_GROUP'] = "cross-industry"
        df.loc[:, 'NAICS'] = '000000'
        df.loc[:, 'NAICS_TITLE'] = 'Cross-industry'

    # Adjust geographical columns based on sheet type
    if sheet_name == "state_by_month":
        df.loc[:, "AREA_TYPE"] = "state"
        df.loc[:, "AREA_TITLE"] = df.loc[:, "PRIM_STATE"]
        df.loc[:, "AREA"] = df['AREA_TITLE'].map(state_fips)
        df.loc[:, "PRIM_STATE"] = df['PRIM_STATE'].map(state_to_abbrev)
    elif sheet_name == "geog_by_month":
        df.loc[:, "AREA_TYPE"] = df.loc[:, "AREA_TYPE"].apply(lambda x: x.split(" ")[1].lower())
        df["AREA"] = df['AREA'].apply(lambda x: str(x).zfill(5))
        df["COUNTY"] = df['AREA']
        df["AREA"] = df['COUNTY'].map(county_cbsa_dict)
        df["AREA_TITLE"] = df['COUNTY'].map(county_cbsa_name_dict)
    else:
        df.loc[:, "AREA"] = "99"
        df.loc[:, "AREA_TITLE"] = "U.S."
        df.loc[:, "AREA_TYPE"] = "national"
        df.loc[:, "PRIM_STATE"] = "US"

    # Rename all columns to uppercase and reorder
    df.rename(columns=lambda x: x.upper(), inplace=True)
    return df


def process_geog_sheet(processed_df):
    """
    Processes the geographic sheet data contained in the given DataFrame by computing aggregated metrics
    and adjusting column values.
    This function performs the following steps:
        - Copies the "AREA" column to a new "COUNTY" column.
        - Computes a temporary "tot_rem" value by rounding the product of "PERCENT" and "N" divided by 100.
        - Groups the data by "AREA_TITLE", "AREA", and "YEAR_MONTH" to aggregate the sum of "N" and "tot_rem".
        - Recalculates the "PERCENT" column for the grouped data as 100 times the ratio of aggregated "tot_rem" to
        aggregated "N".
        - Drops the columns "PERCENT", "PERCENT_3MA", "N", "tot_rem", and "COUNTY" from the original DataFrame to
        avoid redundancy.
        - Removes duplicate rows from the DataFrame.
        - Merges the grouped DataFrame back with the original DataFrame on the grouping keys.
        - Initializes a new "PERCENT_3MA" column with NaN values.
        - Sets the "AREA_TYPE" column to "cbsa" to ensure proper column categorization.
    Parameters:
        processed_df (pandas.DataFrame): The input DataFrame containing geographic and metric data. It is expected
                                        to include at least the following columns:
                                            - "AREA"
                                            - "AREA_TITLE"
                                            - "YEAR_MONTH"
                                            - "PERCENT"
                                            - "N"
    Returns:
        pandas.DataFrame: A modified DataFrame with updated aggregate metrics, recalculated percentage values,
                        and revised columns suitable for further analysis.
    """
    processed_df["COUNTY"] = processed_df["AREA"]
    processed_df["tot_rem"] = np.round((processed_df["PERCENT"] * processed_df["N"]) / 100)
    df_g = processed_df.groupby(["AREA_TITLE", "AREA", "YEAR_MONTH"]).agg({"N": "sum", "tot_rem": "sum"}).reset_index()
    df_g["PERCENT"] = 100 * df_g["tot_rem"] / df_g["N"]
    processed_df.drop(columns=["PERCENT", "PERCENT_3MA", "N", "tot_rem", "COUNTY"], inplace=True)
    processed_df.drop_duplicates(inplace=True)
    processed_df = processed_df.merge(df_g, on=["AREA_TITLE", "AREA", "YEAR_MONTH"])
    processed_df["PERCENT_3MA"] = np.nan
    # Ensure proper column order and set new AREA_TYPE
    processed_df["AREA_TYPE"] = "cbsa20"
    return processed_df


def process_all_sheets(data_dir, processed_data_dir, state_to_abbrev, state_fips, county_cbsa_dict, county_cbsa_name_dict):
    """
    Process all sheets from the 'remote_work_in_job_ads_signup_data.xlsx' file and output processed CSV files.
    Parameters:
        data_dir (str): The directory path containing the source Excel file.
        processed_data_dir (str): The directory path where the processed CSV files will be saved.
        state_to_abbrev (dict): A mapping from state names to their abbreviations used in processing.
        state_fips (dict): A mapping of state FIPS codes necessary for processing.
        county_cbsa_dict (dict): A dictionary mapping county identifiers to CBSA codes.
        county_cbsa_name_dict (dict): A dictionary mapping county identifiers to CBSA names.
    Returns:
        None
    Notes:
        - The function expects the Excel file to be named 'remote_work_in_job_ads_signup_data.xlsx'.
        - The processing relies on external functions 'process_sheet' and 'process_geog_sheet'.
    """

    excel_file = pd.ExcelFile(os.path.join(data_dir, 'remote_work_in_job_ads_signup_data.xlsx'))
    logging.info(f"Opened Excel file with sheets: {excel_file.sheet_names}")
    sheet_names = excel_file.sheet_names
    col_order = ["YEAR_MONTH", "AREA", "AREA_TITLE", "AREA_TYPE", "PRIM_STATE",
                "NAICS", "NAICS_TITLE", "I_GROUP", "OCC_CODE", "OCC_TITLE", 
                "O_GROUP", "PERCENT", "N", "PERCENT_3MA"]

    for i, sheet_name in enumerate(sheet_names):
        if sheet_name in ["contents", "city_by_month"]:
            logging.info(f"Skipping sheet: {sheet_name}")
            continue
        logging.info(f"Processing sheet: {sheet_name}")
        df = excel_file.parse(sheet_name)
        processed_df = process_sheet(sheet_name, df.copy(), state_to_abbrev, state_fips, county_cbsa_dict, county_cbsa_name_dict)
        processed_df = processed_df[col_order]
        output_path = os.path.join(processed_data_dir, f"{sheet_name}.csv")
        processed_df.to_csv(output_path, index=False)
        logging.info(f"Saved processed data for {sheet_name} to {output_path}")
        if sheet_name == "geog_by_month":
            processed_df = process_geog_sheet(processed_df)
            processed_df = processed_df[col_order]
            output_path = os.path.join(processed_data_dir, "cbsa_by_month.csv")
            processed_df.to_csv(output_path, index=False)
            logging.info(f"Saved processed CBSA data to {output_path}")
        


# Run the pipeline if executed as a script
if __name__ == "__main__":
    logging.info("Initializing mappings.")
    state_to_abbrev, state_fips, county_cbsa_dict, county_cbsa_name_dict = setup_mappings()
    logging.info("Mappings successfully set up.")
    logging.info("Starting processing of all sheets.")
    process_all_sheets(DATA_DIR, PROCESSED_DATA_DIR, state_to_abbrev, state_fips, county_cbsa_dict, county_cbsa_name_dict)
    logging.info("Completed processing of all sheets.")