# %% 
# Im replicating 2020 - Jeremy Lise, Fabien Postel-Vinay - Multidimensional Skills, Sorting, and Human Capital Accumulation)
import os
import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np


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

def stata_compatible_pca(df, score_cols, n_components=3):
    """
    Implement PCA in a way that matches Stata's approach
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the variables for PCA
    score_cols : list
        List of column names to include in PCA
    n_components : int
        Number of components to extract
        
    Returns:
    --------
    tuple
        (loadings_df, scores_df, ALPHA0, A) - all matrices needed for further processing
    """
    # Standardize the data (Stata's PCA typically uses correlation matrix)
    X = df[score_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run PCA without scaling by eigenvalues
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    
    # Get eigenvalues (variances) and eigenvectors directly
    eigenvalues = pca.explained_variance_
    
    # Stata works with eigenvectors directly (not scaled by sqrt of eigenvalues)
    # In sklearn, components_ are already scaled by sqrt(eigenvalues), so we need to undo this
    eigenvectors = pca.components_.T * np.sqrt(eigenvalues)
    
    # Create dataframe of loadings (eigenvectors)
    loadings_df = pd.DataFrame(
        eigenvectors, 
        index=score_cols,
        columns=[f'Comp{i+1}' for i in range(n_components)]
    )
    
    # Calculate scores directly using eigenvectors (not scaled components)
    # This matches Stata's approach
    scores_direct = X_scaled @ eigenvectors
    
    # Create dataframe of scores
    scores_df = pd.DataFrame(
        scores_direct,
        index=df.index,
        columns=[f'Comp{i+1}' for i in range(n_components)]
    )
    
    # Return matrices and dataframes needed for next steps
    return loadings_df, scores_df, eigenvectors, eigenvectors[:n_components, :]

def process_skill_vectors_python(df, anchor_cols, n_components=3):
    """
    Process skill vectors using Python PCA implementation: runs PCA, reparameterizes loadings based on anchor columns,
    computes factor scores, normalizes them, and exports the result.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame containing skill measurements
    anchor_cols : list
        Columns to anchor the PCA process
    n_components : int, default=3
        Number of PCA components
        
    Returns:
    --------
    pandas DataFrame
        Processed DataFrame with skill vectors (unnnamed columns)
    """
    # Reorder columns to have anchor columns first
    df_pca = reorder_columns(df, anchor_cols)
    # Drop rows with missing values in the score columns
    df_pca = df_pca.dropna()

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
    
    # Set anchor columns as names 
    result_df.columns = ANCHOR_COLS

    return result_df

def process_skill_vectors_stata(df, anchor_cols, n_components=3):
    """
    Process skill vectors using Stata's approach to PCA
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing O*NET scores with occupations as index
    anchor_cols : list
        List of column names to use as anchors for the factors
    n_components : int, default=3
        Number of components to extract
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with occupation codes as index and normalized skill indices (unnamed columns)
    """
    # Get all score columns
    score_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Reorder columns to match Stata's ordering (anchor columns first)
    cols_reordered = anchor_cols + [col for col in score_cols if col not in anchor_cols]
    
    # Run PCA Stata-style
    loadings_df, scores_df, ALPHA0, A = stata_compatible_pca(
        df, cols_reordered, n_components=n_components
    )
    
    # Continue with the transformation as in Stata
    # Compute ALPHA = ALPHA0 * inv(A)
    ALPHA = ALPHA0 @ np.linalg.inv(A)
    
    # Calculate factor scores as in Stata
    X_scaled = StandardScaler().fit_transform(df[cols_reordered].values)
    factor_scores = X_scaled @ ALPHA
    
    # Create a DataFrame with the factor scores
    result_df = pd.DataFrame(
        factor_scores, 
        index=df.index,
        columns=range(n_components)
    )
    
    # Normalize each factor to range from 0 to 1
    for col in range(n_components):
        col_min = result_df[col].min()
        col_max = result_df[col].max()
        result_df[col] = (result_df[col] - col_min) / (col_max - col_min)
    
    # Drop rows with missing values in any of the factors
    result_df = result_df.dropna()
    
    # Use the anchor columns to name the factors
    result_df.columns = anchor_cols

    return result_df

def add_names_and_titles(skill_vectors, add_occupation_titles=False, named_factors=False):
    """
    Add names to skill vector columns and/or add occupation titles
    
    Parameters:
    -----------
    skill_vectors : pandas DataFrame
        DataFrame containing the skill vectors
    anchor_cols : list, optional
        If provided, use these columns to name the factors
    named_factors : bool, default=False
        Whether to use the content model reference to name the columns
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with named columns and/or occupation titles
    """
    result_df = skill_vectors.copy()
    
    if add_occupation_titles:
        # Add occupation titles
        try:
            occupations = pd.read_csv(os.path.join(REFERENCE_DATA_DIR, 'OCCUPATION_DATA.csv')).set_index('ONET_SOC_CODE')
            result_df = result_df.join(occupations["OCCUPATION_TITLE"], how='left')
        except Exception as e:
            logging.warning(f"Could not add occupation titles: {e}")
        
    # Add names to columns if requested
    if named_factors:
        try:
            # Load the content model reference
            content_model_reference = pd.read_csv(os.path.join(REFERENCE_DATA_DIR, 'CONTENT_MODEL_REFERENCE.csv'), usecols=[ 'ELEMENT_ID' , "ELEMENT_NAME"])
            # Create a mapping from ELEMENT_ID to ELEMENT_NAME
            element_mapping = content_model_reference.set_index('ELEMENT_ID')['ELEMENT_NAME'].to_dict()
            # If we have names for all our numeric columns, rename them
            result_df.rename(columns=element_mapping, inplace=True)
        except Exception as e:
            logging.warning(f"Could not name factors: {e}")
    
    return result_df

def main(method='python', add_occupation_titles=True, named_factors=False):
    """
    Main function to process ONET data
    
    Parameters:
    -----------
    method : str, default='python'
        Method to use for processing skill vectors. Options: 'python', 'stata'
    add_names : bool, default=True
        Whether to add occupation titles
    named_factors : bool, default=False
        Whether to use the content model reference to name the columns
    
    Returns:
    --------
    tuple
        (onetscores_wide, skill_vectors) - wide format data and processed skill vectors
    """
    # Define file paths
    files_to_read = ['WORK_ACTIVITIES', 'SKILLS', 'ABILITIES', 'KNOWLEDGE']

    # Append and create wide format
    onetscores_wide = append_and_create_wide(files_to_read, ANCHOR_COLS)    
    
    # Process skill vectors according to the specified method
    if method.lower() == 'stata':
        skill_vectors = process_skill_vectors_stata(onetscores_wide, ANCHOR_COLS)
    else:  # default to python
        skill_vectors = process_skill_vectors_python(onetscores_wide, ANCHOR_COLS)
    
    # Add names and titles if requested
    skill_vectors = add_names_and_titles(skill_vectors, 
                                        add_occupation_titles = add_occupation_titles , 
                                        named_factors = named_factors)
    
    return onetscores_wide, skill_vectors

# %%
# Run the main function 
onetscores_wide, skill_vectors_python = main(method='python', add_occupation_titles=True, named_factors=True)
onetscores_wide, skill_vectors_stata = main(method='stata', add_occupation_titles=True, named_factors=True)

# %%
sns.pairplot(skill_vectors_python, diag_kind='kde')
plt.show()
# %%
sns.pairplot(skill_vectors_stata, diag_kind='kde')
plt.show()

# %%
# Compare the top 10 occupations on each skill dimension with each method
for c in skill_vectors_python.columns:
    if c == "OCCUPATION_TITLE":
        continue
    # Sort both dataframes by the current column
    sorted_python = skill_vectors_python.sort_values(by=c, ascending=False)
    sorted_stata = skill_vectors_stata.sort_values(by=c, ascending=False)
    # get the top 10 occupations and their values
    print(f"Top 10 occupations for {c} skill dimension")
    top_python = sorted_python[["OCCUPATION_TITLE", c]].head(10).reset_index(drop=True)
    top_stata = sorted_stata[["OCCUPATION_TITLE", c]].head(10).reset_index(drop=True)
    # Put them in a single dataframe use concatenate since the index is the same
    display(
        pd.concat([top_python, top_stata], axis=1)
    )
    # repeat for the bottom 10 occupations
    print(f"Bottom 10 occupations for {c} skill dimension")
    bottom_python = sorted_python[["OCCUPATION_TITLE", c]].tail(10).reset_index(drop=True)
    bottom_stata = sorted_stata[["OCCUPATION_TITLE", c]].tail(10).reset_index(drop=True)
    # Put them in a single dataframe use concatenate since the index is the same
    display(
        pd.concat([bottom_python, bottom_stata], axis=1)
    )
# %%
# Load the telewokability estimates
teleworkability = pd.read_csv(BASE_DIR + "/data/results/wfh_estimates.csv").set_index('ONET_SOC_CODE')
# Join the teleworkability estimates with the skill vectors
skill_vectors_python = skill_vectors_python.join(teleworkability["ESTIMATE_WFH_ABLE"], how='left')
skill_vectors_stata = skill_vectors_stata.join(teleworkability["ESTIMATE_WFH_ABLE"], how='left')
# %%
# Create FacetGrid plots to visualize relationships between skill indices and WFH ability
plt.figure(figsize=(15, 10))

skills_melted_python = pd.melt(
    skill_vectors_python.reset_index(), 
    id_vars=['ONET_SOC_CODE', 'OCCUPATION_TITLE', 'ESTIMATE_WFH_ABLE'],
    value_vars=['Mechanical', 'Mathematics', 'Social Perceptiveness'],
    var_name='Skill', value_name='Score'
)

# Plot for Python method
g = sns.FacetGrid(skills_melted_python[skills_melted_python.ESTIMATE_WFH_ABLE > 0],
                col='Skill', height=4, aspect=1.2)

g.map_dataframe(sns.scatterplot, y='Score', x='ESTIMATE_WFH_ABLE', alpha=0.6)
g.map_dataframe(sns.regplot, y='Score', x='ESTIMATE_WFH_ABLE', scatter=False, color='red')
g.set_axis_labels('Teleworkability', 'Skill Score')
# g.set_titles('{col} Skills vs WFH Ability (Python)')
plt.show()
# %% 
# Fit logistic regression models for each skill
python_models = {}
for col in skill_vectors_python.columns:
    if col == 'OCCUPATION_TITLE' or col == 'ESTIMATE_WFH_ABLE':
        continue
    X = skill_vectors_python[col].values.reshape(-1, 1)
    # Create a binary target for logistic regression (assuming > 0.5 means remote work)
    y = (skill_vectors_python['ESTIMATE_WFH_ABLE'] > 0.0).astype(int)
    model = LogisticRegression()
    model.fit(X, y)
    python_models[col] = model
# Plot logistic curves for each skill 
plt.figure(figsize=(15, 5))
for i, (skill_name, model) in enumerate(python_models.items()):
    plt.subplot(1, 3, i+1)
    
    # Create a range of values for the skill
    x_range = np.linspace(0, 1, 100).reshape(-1, 1)
    # Predict probabilities
    y_probs = model.predict_proba(x_range)[:, 1]
    
    # Plot the curve
    plt.plot(x_range, y_probs, 'r-', linewidth=2)
    
    plt.title(f'{skill_name} vs. Remote Work Probability')
    plt.xlabel(f'{skill_name} Skill Score')
    plt.ylabel('Probability of Remote Work')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.suptitle('Logistic Regression: Skill Impact on Remote Work (Python Method)', y=1.05)
plt.show()

# %%
