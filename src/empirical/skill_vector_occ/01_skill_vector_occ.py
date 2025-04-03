# %% 
# Im replicating 2020 - Jeremy Lise, Fabien Postel-Vinay - Multidimensional Skills, Sorting, and Human Capital Accumulation)
import os
import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.basename(__file__)
log_file = os.path.join(current_dir, f'{script_name.replace(".py", ".log")}')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base paths
BASE_DIR = '.'
BASE_DIR = '../../../'

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'onet_data', 'processed', 'measure')
REFERENCE_DATA_DIR = os.path.join(BASE_DIR, 'data', 'onet_data', 'processed', 'reference')
ONET_SELECT_PATH = os.path.join(current_dir, 'onet_select.dta')


OUTPUT_FILE = "ydist_python.csv"
ANCHOR_COLS = ["2.C.3.e", "2.A.1.e", "2.B.1.a"] # These are unambiguous columns to anchor the PCA process

# %%
def reorder_columns(df, anchor_cols):
    """
    Reorder the DataFrame columns so that the identifier is first and
    anchor columns appear immediately after.
    """
    ordered_cols = anchor_cols + [col for col in df.columns if col not in anchor_cols]
    # Create ordered list: identifier, anchor columns, then all others.
    return df[ordered_cols].copy()

def read_and_clean_data(file_path, onet_select_path=ONET_SELECT_PATH):
    """Read and clean ONET data files"""
    logging.info("Reading and cleaning data files")
    
    # Define columns and scales to exclude
    columns_exclude = [ 'STANDARD_ERROR', 'LOWER_CI_BOUND', 'UPPER_CI_BOUND', 'DATE', 'N',
                        'RECOMMEND_SUPPRESS', 'NOT_RELEVANT', 'CATEGORY', 'DOMAIN_SOURCE',
                        'ELEMENT_NAME']
    scale_ids_to_exclude = ["IM", "CXP", "CTP"]
    
    # Read ONET_SELECT file
    onet_select = pd.read_stata(onet_select_path)

    df = pd.read_csv(file_path)
    # Drop columns that exist in the dataframe
    cols_to_drop = [col for col in columns_exclude if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    # Filter out unwanted scale IDs if SCALE_ID exists
    if 'SCALE_ID' in df.columns:
        df = df[~df['SCALE_ID'].isin(scale_ids_to_exclude)]
        # Normalize the SCALE_ID column
        df.loc[df['SCALE_ID'] == "CT", 'DATA_VALUE'] = df.loc[df['SCALE_ID'] == "CT", 'DATA_VALUE'].apply(lambda x: (x - 1) / 2)
        df.loc[df['SCALE_ID'] == "CX", 'DATA_VALUE'] = df.loc[df['SCALE_ID'] == "CX", 'DATA_VALUE'].apply(lambda x: (x - 1) / 4)
        df.loc[df['SCALE_ID'] == "LV", 'DATA_VALUE'] = df.loc[df['SCALE_ID'] == "LV", 'DATA_VALUE'].apply(lambda x: x / 7)
        
        df.drop(columns=['SCALE_ID'], inplace=True)

    return df[
            ~df.ELEMENT_ID.isin(onet_select[onet_select.CMI_label == 0].elementid.values)
        ]

def append_and_create_wide(files_to_read, anchor_cols=ANCHOR_COLS):
    """Append all dataframes and create wide format"""
    logging.info("Appending dataframes and creating wide format")
    
    # Read and clean each file and stack them 
    stacked_df = pd.DataFrame()
    for file in files_to_read:
        file_path = os.path.join(PROCESSED_DATA_DIR, f'{file.upper()}.csv')
        df = read_and_clean_data(file_path)
        stacked_df = pd.concat([stacked_df, df], ignore_index=True)

    # Convert to a wide format
    df_wide = stacked_df.pivot(
        index='ONET_SOC_CODE',
        columns='ELEMENT_ID',
        values='DATA_VALUE'
    )

    df_wide.reset_index(inplace=True)

    reorder_columns(df_wide, anchor_cols = anchor_cols)
    
    return df_wide.set_index('ONET_SOC_CODE')

def process_skill_vectors(df, anchor_cols, n_components=3,  named_factors=True):
    """
    Process skill vectors using PCA: runs PCA, reparameterizes loadings based on anchor columns,
    computes factor scores, normalizes them, and exports the result.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame containing skill measurements
    score_cols : list
        List of columns to use for PCA
    identifier_col : str
        Column name for occupational identifiers
    anchor_cols : list
        Columns to anchor the PCA process
    n_components : int, default=3
        Number of PCA components
    output_file : str, default=OUTPUT_FILE
        File path for exporting results
        
    Returns:
    --------
    pandas DataFrame
        Processed DataFrame with skill vectors
    """
    # Drop rows with missing values in the score columns
    df_pca = df.dropna()
    
    # Fit PCA with n_components components
    pca = PCA(n_components=n_components)
    pca.fit(df_pca)
    
    # Get loadings matrix
    ALPHA0 = pca.components_.T
    
    # Reparameterize loadings
    num_anchors = len(anchor_cols)
    A = ALPHA0[:num_anchors, :]
    A_inv = np.linalg.inv(A)
    ALPHA = ALPHA0.dot(A_inv)
    
    # Compute and normalize factor scores
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Calculate factor scores and normalize them
    factor_scores = scaler.fit_transform(df_pca.dot(ALPHA))
    
    # Create DataFrame with the original index
    result_df = pd.DataFrame(factor_scores, index=df_pca.index)
    
    # If named factors is true we will use CONTENT_MODEL_REFERENCE to name the factors as the anchor columns
    if named_factors:
        # Load the content model reference
        content_model_reference = pd.read_csv(os.path.join(REFERENCE_DATA_DIR, 'CONTENT_MODEL_REFERENCE.csv'))
        # Create a mapping from ELEMENT_ID to ELEMENT_NAME
        anchor_mapping = dict(zip(content_model_reference.ELEMENT_ID, content_model_reference.ELEMENT_NAME))
        # Reorder names to match the order in anchor_cols
        names = [anchor_mapping[ac] for ac in anchor_cols if ac in anchor_mapping]
        # Rename columns
        result_df.columns = names
    
    return result_df

def main():
    """Main function to process ONET data"""
    # Define file paths
    files_to_read = ['WORK_CONTEXT', 'WORK_ACTIVITIES', 'SKILLS', 'ABILITIES', 'KNOWLEDGE']

    # Append and create wide format
    onetscores_wide = append_and_create_wide(files_to_read, ANCHOR_COLS)
    # Process skill vectors
    skill_vectors = process_skill_vectors(onetscores_wide, ANCHOR_COLS)
    return onetscores_wide, skill_vectors
# %%
# if __name__ == "__main__":
    # Read and clean data
onetscores_wide, skill_vectors = main()

# Load occupation titles 
occupations = pd.read_csv(os.path.join(REFERENCE_DATA_DIR, 'OCCUPATION_DATA.csv')).set_index('ONET_SOC_CODE')
# Merge with skill vectors
skill_vectors = skill_vectors.join(occupations["OCCUPATION_TITLE"], how='left')

# For each dimension of the skill vector print the top 5 and bottom 5 occupations
for c in skill_vectors.columns:
    # Skip OCCUPATION_TITLE
    if c == "OCCUPATION_TITLE":
        continue
    print(f"Top 5 and bottom 5 occupations for {c}")
    print("TOP 5:")
    print(
        skill_vectors.sort_values(by=c, ascending=False).head(5)
    )
    print("BOTTOM 5:")
    print(
        skill_vectors.sort_values(by=c, ascending=True).head(5)
    )
    print("\n\n")


# %%
# # Plot using seaborn the correlations between three three components 
import matplotlib.pyplot as plt
import seaborn as sns


sns.pairplot(skill_vectors, diag_kind='kde', markers='o')

