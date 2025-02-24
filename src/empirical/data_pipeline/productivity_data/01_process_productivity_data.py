# %% Global packages and constants
import os
import pandas as pd
import logging
import re

# Current directory and logging
current_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.basename(__file__)
log_file = os.path.join(current_dir, f'{script_name.replace(".py", ".log")}')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Base paths
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'bls', 'productivity')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'bls', 'productivity')

# Ensure output directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# %% Functions

def load_productivity_data():
    """
    Reads the productivity data from Excel.
    """
    excel_path = os.path.join(BASE_DIR, 'data', 'raw', 'bls', 'productivity', 'labor-productivity-detailed-industries.xlsx')
    logger.info(f"Loading productivity data from {excel_path}")
    prod_data = pd.read_excel(excel_path, sheet_name='MachineReadable')
    return prod_data

def create_reference_df(prod_data):
    """
    Generate a reference dataframe to transform the measures and units.
    """
    reference = {
        "Col": [],
        "Measure": [],
        "Units": []
    }
    
    # Extract unique combinations and create a column label
    for _, row in prod_data[["Measure", "Units"]].drop_duplicates().iterrows():
        measure = row['Measure']
        unit = row["Units"]
        reference["Measure"].append(measure)
        reference["Units"].append(unit)
        measure_clean = measure.replace(" ", "_").lower().strip()
        unit_clean = re.sub(r'%|\(.*?\)', '', unit).lower().strip().split(" ")[0]
        reference["Col"].append(f"{measure_clean}_{unit_clean}")
        
    reference_df = pd.DataFrame(reference)
    return reference_df

def prepare_productivity_data(prod_data, reference_df):
    """
    Merge the reference dataframe, drop unnecessary columns,
    and pivot the data.
    """
    # Merge reference info
    prod_data = prod_data.merge(
        reference_df,
        how="left",
        left_on=["Measure", "Units"],
        right_on=["Measure", "Units"]
    )
    prod_data.drop(columns=["Measure", "Units", "Basis"], errors="ignore", inplace=True)
    
    # Pivot the table to get a wide dataframe
    prod_data_wide = prod_data.pivot(
        index=["Sector", "NAICS", "Industry", "Digit", "Year"],
        columns="Col",
        values="Value"
    )
    return prod_data_wide, prod_data["Digit"].unique()

def save_digit_csvs(prod_data_wide, digit_list):
    """
    For each digit in digit_list, filter the pivoted dataframe and save to CSV.
    """
    for digit in digit_list:
        logger.info(f"Processing digit: {digit}")
        sub_data = prod_data_wide.xs(digit, level="Digit")
        # Keep only columns with at least one non-NA value
        sub_data = sub_data.loc[:, sub_data.notna().sum() > 0]
        file_name = f"productivity_{digit.replace('-', '_')}.csv"
        output_path = os.path.join(PROCESSED_DATA_DIR, file_name)
        sub_data.to_csv(output_path)
        logger.info(f"Saved CSV for digit {digit} to {output_path}")

def main():
    prod_data = load_productivity_data()
    reference_df = create_reference_df(prod_data)
    # Save the reference dataframe
    ref_file = os.path.join(PROCESSED_DATA_DIR, "productivity_reference.csv")
    reference_df.to_csv(ref_file, index=False)
    logger.info(f"Saved reference dataframe to {ref_file}")
    prod_data_wide, digit_list = prepare_productivity_data(prod_data, reference_df)
    save_digit_csvs(prod_data_wide, digit_list)
    
if __name__ == "__main__":
    main()