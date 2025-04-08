"""
Skill Vector Occupations Analysis

This module processes ONET data to create skill vectors for different occupations,
implementing both Python and Stata-compatible PCA approaches.
"""

import os
import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_distances

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data_exploration'))
# Add the parent directory to sys.path to import OccupationHierarchy
from soc_structure import OccupationHierarchy

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.basename(__file__)
log_file = os.path.join(current_dir, f'{script_name.replace(".py", ".log")}')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = "."

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'onet_data', 'processed', 'measure')
REFERENCE_DATA_DIR = os.path.join(BASE_DIR, 'data', 'onet_data', 'processed', 'reference')
ONET_SELECT_PATH = os.path.join(current_dir, 'onet_select.dta')

# Default anchor columns
DEFAULT_ANCHOR_COLS = ["2.C.3.e", "2.A.1.e", "2.B.1.a"]  # These are unambiguous columns to anchor the PCA process

def reorder_columns(df, anchor_cols):
    """
    Reorder the DataFrame columns so that     anchor columns appear first.
    """
    ordered_cols = anchor_cols + [col for col in df.columns if col not in anchor_cols]
    # Create ordered list: identifier, anchor columns, then all others.
    return df[ordered_cols].copy()

def read_and_clean_data(file_path, onet_select_path=ONET_SELECT_PATH, imporance_weight=False, non_relevance_exclude=False):
    """Read and clean ONET data files"""
    logging.info(f"Reading and cleaning data file: {file_path}")
    
    # Define columns and scales to exclude
    columns_exclude = ['STANDARD_ERROR', 'LOWER_CI_BOUND', 'UPPER_CI_BOUND', 'DATE', 'N',
                        'RECOMMEND_SUPPRESS', 'CATEGORY', 'DOMAIN_SOURCE', 'ELEMENT_NAME'
                    ]
    if not non_relevance_exclude:
        columns_exclude += ['NOT_RELEVANT'] # Add NOT_RELEVANT column to exclude if not already excluded
    
    scale_ids_to_exclude = ["CXP", "CTP"]
    if not imporance_weight:
        # Add scale IDs to exclude if importance weight is applied
        scale_ids_to_exclude += ["IM"]
    
    # Read ONET_SELECT file
    onet_select = pd.read_stata(onet_select_path)

    df = pd.read_csv(file_path)
    df.set_index(["ONET_SOC_CODE", "ELEMENT_ID"], inplace=True)
    # Drop columns that exist in the dataframe
    cols_to_drop = [col for col in columns_exclude if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    # Filter out unwanted scale IDs if SCALE_ID exists
    if 'SCALE_ID' in df.columns:
        # Filter out unwanted scale IDs
        df = df[~df['SCALE_ID'].isin(scale_ids_to_exclude)]
        # Normalize the SCALE_ID column
        for scale_id in df['SCALE_ID'].unique():
            if scale_id == "IM":
                # Use MinMaxScaler for the Importance scale
                scaler = MinMaxScaler()
                df.loc[df.SCALE_ID == scale_id, 'DATA_VALUE'] = scaler.fit_transform(
                    df.loc[df.SCALE_ID == scale_id, 'DATA_VALUE'].values.reshape(-1, 1)
                ).flatten()
            else:
            # Use StandardScaler for other scales
                scaler = StandardScaler()
                df.loc[df.SCALE_ID == scale_id, 'DATA_VALUE'] = scaler.fit_transform(
                    df.loc[df.SCALE_ID == scale_id, 'DATA_VALUE'].values.reshape(-1, 1)
                ).flatten()
        
        if 'NOT_RELEVANT' in df.columns:
            non_relevant_pairs = df.loc[df.NOT_RELEVANT == "Y"].index # Collect the non-relevant pairs of occupations and skills
            df.loc[non_relevant_pairs, 'DATA_VALUE'] = 0 # Set the non-relevant pairs to 0
            # Drop the NOT_RELEVANT column
            df.drop(columns=['NOT_RELEVANT'], inplace=True)

        # Compute weighted values of level 
        if imporance_weight:
            df = df.loc[df.SCALE_ID == "IM", "DATA_VALUE"]  * df.loc[df.SCALE_ID == "LV", "DATA_VALUE"]
        else:
            df = df.loc[df.SCALE_ID == "LV", "DATA_VALUE"]

    # Reset index to have ONET_SOC_CODE and ELEMENT_ID as columns
    df = df.reset_index()
    # TODO: Check without this step    
    return df[
            ~df.ELEMENT_ID.isin(onet_select[onet_select.CMI_label == 0].elementid.values)
        ]

def append_and_create_wide(files_to_read, imporance_weight=False, non_relevance_exclude=False):
    
    """Append all dataframes and create wide format"""
    logging.info("Appending dataframes and creating wide format")
    
    # Read and clean each file and stack them 
    stacked_df = pd.DataFrame()
    for file in files_to_read:
        file_path = os.path.join(PROCESSED_DATA_DIR, f'{file.upper()}.csv')
        df = read_and_clean_data(file_path, imporance_weight=imporance_weight, non_relevance_exclude=non_relevance_exclude)
        stacked_df = pd.concat([stacked_df, df], ignore_index=True)

    # Convert to a wide format
    df_wide = stacked_df.pivot(
        index=  'ONET_SOC_CODE',
        columns= 'ELEMENT_ID',
        values='DATA_VALUE'
    )
    # Return the processed dataframe
    return df_wide

def process_anchor_columns(df_wide, anchor_cols, distance_metric="euclidean"):
    """
    Process anchor columns for skill vector analysis.
    
    Parameters:
    -----------
    df_wide : pandas.DataFrame
        Wide-format dataframe where rows are occupations and columns are skills.
    anchor_cols : list or int
        Either a list of specific anchor columns to use, or an integer specifying
        the number of anchor columns to select automatically.
    distance_metric : str, optional
        Distance metric to use for clustering. Options: "euclidean", "abs_corr", 
        "raw_corr", or "cosine".
        
    Returns:
    --------
    tuple
        (df_wide, anchor_cols_list, clusters) - reordered dataframe, list of anchor columns,
        and clusters of skills around the anchors.
    """
    # If a list of anchor columns is provided, reorder the columns using that list
    # if a number of anchor columns is provided, select the anchor columns using a clustering method
    if isinstance(anchor_cols, int):
        anchor_cols_list, clusters = choose_anchors(df_wide, num_anchors=anchor_cols, distance_metric=distance_metric)
        # Reorder columns to have anchor columns first
        df_wide = reorder_columns(df_wide, anchor_cols=anchor_cols_list)
    else:
        anchor_cols_list = anchor_cols
        df_wide = reorder_columns(df_wide, anchor_cols=anchor_cols_list)
        # Get the clusters from the anchor columns
        clusters = cluster_around_anchors(df_wide, anchor_cols, distance_metric=distance_metric)
    
    return df_wide, anchor_cols_list, clusters

def pre_process_wide_data(df_wide, distance_metric="euclidean"):
    """
    Processes occupation-skill data to create a skill distance matrix.
    This function transforms a wide-format dataframe where occupations are rows and
    skills are columns. It computes the similarity between skills and converts
    these similarities to distances.
    
    Parameters
    ----------
    df_wide : pandas.DataFrame
        Wide-format dataframe where rows are occupations and columns are skills.
        Values represent the importance/level of each skill for each occupation.
    distance_metric : str, optional
        Specifies which distance metric to use. Options include:
          - "abs_corr": 1 - |correlation| (default)
          - "raw_corr": 1 - (correlation)
          - "euclidean": Euclidean distance between standardized skill vectors
          - "cosine": Cosine distance (1 minus cosine similarity)
          
    Returns
    -------
    pandas.DataFrame
        A symmetric distance matrix with skills as both rows and columns.
    """
    # Transpose to get skills as rows (each row: one skill vector across occupations)
    skill_matrix = df_wide.T  # shape: (num_skills, num_occupations)

    if distance_metric == "abs_corr":
        # Compute correlation matrix between skills
        corr_df = skill_matrix.T.corr()
        # Convert correlation to a distance matrix: 1 - |correlation|
        distance_matrix = 1 - corr_df.abs()
        
    elif distance_metric == "raw_corr":
        corr_df = skill_matrix.T.corr()
        # Convert correlation to a distance matrix: 1 - correlation
        distance_matrix = 1 - corr_df
        
    elif distance_metric == "euclidean":
        # Standardize each skill vector (row) to zero mean and unit variance
        standardized = skill_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        # Compute Euclidean distances between skills (rows)
        dists = pdist(standardized.values, metric='euclidean')
        distance_matrix = pd.DataFrame(squareform(dists), index=standardized.index, columns=standardized.index)
        
    elif distance_metric == "cosine":
        # Compute cosine distances between skills (rows)
        dists = cosine_distances(skill_matrix.values)
        distance_matrix = pd.DataFrame(dists, index=skill_matrix.index, columns=skill_matrix.index)
        
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")
        
    return distance_matrix

def cluster_around_anchors(weighted_df, anchor_list, distance_metric = "euclidean"):
    """
    Given a DataFrame of weighted skill scores and a list of predefined anchors,
    assign each skill to the closest anchor based on correlation to form non-overlapping clusters.
    
    Parameters:
    -----------
    weighted_df : pandas DataFrame
        DataFrame with weighted skill scores. Columns are ELEMENT_ID.
    anchor_list : list
        List of ELEMENT_ID values to use as anchors.
    
    Returns:
    --------
    clusters : dict
        A dictionary mapping each anchor to a list of ELEMENT_ID's in its cluster.
    """
    
    if distance_metric in ["abs_corr", "raw_corr", "euclidean", "cosine"]:
        # Pre-process the data to get the distance matrix
        distance_matrix = pre_process_wide_data(weighted_df, distance_metric=distance_metric)
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")
    
    # Make sure all anchors exist in the distance matrix
    valid_anchors = [a for a in anchor_list if a in distance_matrix.index]
    if len(valid_anchors) != len(anchor_list):
        missing = set(anchor_list) - set(valid_anchors)
        logging.warning(f"Some anchors not found in data: {missing}")
    
    # Initialize clusters dictionary with anchors
    clusters = {anchor: [anchor] for anchor in valid_anchors}
    
    # For each skill (except anchors), find the closest anchor
    for skill in distance_matrix.index:
        if skill in valid_anchors:
            continue  # Skip anchors as they're already assigned
        
        # Calculate distances to each anchor
        distances_to_anchors = {anchor: distance_matrix.loc[skill, anchor] for anchor in valid_anchors}
        
        # Find the closest anchor
        closest_anchor = min(distances_to_anchors, key=distances_to_anchors.get)
        
        # Add skill to the closest anchor's cluster
        clusters[closest_anchor].append(skill)
    
    return clusters

def choose_anchors(weighted_df, num_anchors, distance_metric = "euclidean"):
    """
    Given a DataFrame of weighted skill scores (indexed by ONET_SOC_CODE and columns = ELEMENT_ID),
    perform clustering on the columns (skills) based on their correlation (using 1 - |corr| as distance)
    and select one anchor from each cluster. For each cluster, the "central" skill is selected as the
    one with the lowest average distance to other skills in the cluster.
    
    Parameters:
    -----------
    weighted_df : pandas DataFrame
        DataFrame with weighted skill scores. Columns are ELEMENT_ID.
    num_anchors : int
        Number of anchors (clusters) to select.
    
    Returns:
    --------
    anchors : list
        List of ELEMENT_ID values selected as anchors.
    clusters : dict
        A dictionary mapping cluster labels to lists of ELEMENT_ID's in that cluster.
    """
    
    if distance_metric in ["abs_corr", "raw_corr", "euclidean", "cosine"]:
        distance_matrix = pre_process_wide_data(weighted_df, distance_metric="euclidean")
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")
    
    # Convert the full distance DataFrame to a numpy array
    distance_array = distance_matrix.values
    
    # AgglomerativeClustering with a precomputed distance matrix.
    clustering = AgglomerativeClustering(
        n_clusters=num_anchors,
        linkage='average'
    )
    # Fit clustering to the distance matrix (each skill is an observation)
    cluster_labels = clustering.fit_predict(distance_array)
    
    # Prepare a dictionary mapping cluster label -> list of ELEMENT_ID's.
    clusters = {}
    skills = distance_matrix.index.tolist()
    for skill, label in zip(skills, cluster_labels):
        clusters.setdefault(label, []).append(skill)
    

    anchors = []
    # For each cluster, choose the "central" skill.
    # Central is defined as the skill with the minimal average distance to other skills in the cluster.
    for label, skill_list in clusters.items():
        if len(skill_list) == 1:
            anchors.append(skill_list[0])
        else:
            # Extract the submatrix of distances for skills in this cluster.
            sub_dist = distance_matrix.loc[skill_list, skill_list].values
            # Compute average distance for each skill (exclude self-distance, which is zero)
            avg_dists = sub_dist.sum(axis=1) / (len(skill_list) - 1)
            # Select the skill with minimum average distance
            central_skill = skill_list[np.argmin(avg_dists)]
            anchors.append(central_skill)
    
    # Create new clusters dictionary with the central skills as keys
    new_clusters = {}
    for i, anchor in enumerate(anchors):
        # Find the original cluster label for this anchor
        original_label = [label for label, skills in clusters.items() if anchor in skills][0]
        # Map the central skill to its original cluster
        new_clusters[anchor] = clusters[original_label]

    return anchors, new_clusters

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
    # Process anchor columns
    df_wide, anchor_cols_list, clusters = process_anchor_columns(df, anchor_cols, distance_metric="euclidean")

    # Reorder columns to have anchor columns first
    df_pca = reorder_columns(df_wide, anchor_cols)
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
    result_df.columns = anchor_cols_list

    return result_df, anchor_cols_list, clusters

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

def add_names_and_titles(skill_vectors, clusters, add_occupation_titles=False, named_factors=False):
    """
    Add names to skill vector columns and/or add occupation titles
    
    Parameters:
    -----------
    skill_vectors : pandas DataFrame
        DataFrame containing the skill vectors
    clusters : dict
        Dictionary containing clusters of skills
    add_occupation_titles : bool, default=False
        Whether to add occupation titles
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

        new_clusters = {}
        for key, value in clusters.items():
            # Get the name of the cluster
            cluster_name = element_mapping.get(key, key)
            new_clusters[cluster_name] = []
            for skill in value:
                # Get the name of the skill
                skill_name = element_mapping.get(skill, skill)
                if skill != key:
                    # Print the skill name and the anchor
                    new_clusters[cluster_name].append(skill_name)

    return result_df, new_clusters

# def main(method='python', add_occupation_titles=True, named_factors=False):
#     """
#     Main function to process ONET data
    
#     Parameters:
#     -----------
#     method : str, default='python'
#         Method to use for processing skill vectors. Options: 'python', 'stata'
#     add_names : bool, default=True
#         Whether to add occupation titles
#     named_factors : bool, default=False
#         Whether to use the content model reference to name the columns
    
#     Returns:
#     --------
#     tuple
#         (onetscores_wide, skill_vectors) - wide format data and processed skill vectors
#     """
#     # Define file paths
#     files_to_read = ['WORK_ACTIVITIES', 'SKILLS', 'ABILITIES', 'KNOWLEDGE']

#     # Append and create wide format
#     # onetscores_wide = append_and_create_wide(files_to_read, ANCHOR_COLS)    
#     # onetscores_wide, ANCHOR_COLS, clusters = append_and_create_wide(files_to_read, 3)    
#     ANCHOR_COLS = ["2.C.3.e", "2.A.1.e", "2.B.1.a"] # These are unambiguous columns to anchor the PCA process
#     onetscores_wide, ANCHOR_COLS, clusters = append_and_create_wide(files_to_read, ANCHOR_COLS, imporance_weight=True, non_relevance_exclude=True)  
    
#     # Process skill vectors according to the specified method
#     if method.lower() == 'stata':
#         skill_vectors = process_skill_vectors_stata(onetscores_wide, ANCHOR_COLS)
#     else:  # default to python
#         skill_vectors = process_skill_vectors_python(onetscores_wide, ANCHOR_COLS)
    
#     # Add names and titles if requested
#     skill_vectors = add_names_and_titles(skill_vectors, 
#                                         add_occupation_titles = add_occupation_titles , 
#                                         named_factors = named_factors)
    
#     return onetscores_wide, skill_vectors, ANCHOR_COLS, clusters

# # %%
# # Run the main function 
# onetscores_wide, skill_vectors_python, ANCHOR_COLS, clusters = main(method='python', add_occupation_titles=True, named_factors=True)
# # onetscores_wide, skill_vectors_stata , ANCHOR_COLS, clusters = main(method='stata', add_occupation_titles=True, named_factors=True)

# # %%
# # load content model reference
# content_model = pd.read_csv(os.path.join(REFERENCE_DATA_DIR, 'CONTENT_MODEL_REFERENCE.csv'), usecols=['ELEMENT_ID', 'ELEMENT_NAME'])
# name_dict = content_model.set_index('ELEMENT_ID').ELEMENT_NAME.to_dict()
# for key, value in clusters.items():
#     print(f"Cluster {key}- {name_dict.get(key, 'Unknown')}")
#     # Get the skills in the cluster
#     for skill in value:
#         # Get the name of the skill
#         skill_name = name_dict.get(skill, 'Unknown')
#         if skill != key:
#             # Print the skill name and the anchor
#             print(f"\t{skill} - {skill_name}")

# # %%
# sns.pairplot(skill_vectors_python, diag_kind='kde')
# plt.show()
# # %% 
# # Import the OccupationHierarchy class
# # Create the hierarchy object
# hierarchy = OccupationHierarchy()
# # Load SOC 2018 structure from the default path
# hierarchy.load_soc_2018()
# # Map get_full_hierarchy to the index of the skill vectors dataframes

# # Initialize hierarchy columns (excluding 'onet')
# for key in ['major', 'minor', 'broad', 'detailed']:
#     skill_vectors_python[key] = None
#     skill_vectors_python[f'{key}_title'] = None

# # Process each SOC code
# for soc_code in skill_vectors_python.index:
#     try:
#         hierarchy_info, title_info = hierarchy.get_full_hierarchy(soc_code, include_titles=True)
#         # Add code information
#         for key, values in hierarchy_info.items():
#             if key != 'onet' and values:  # Skip 'onet' key
#                 skill_vectors_python.loc[soc_code, key] = values[0]  # Use the first value
#         # Add title information
#         for key, values in title_info.items():
#             if key != 'onet_title' and values:  # Skip 'onet_title' key
#                 skill_vectors_python.loc[soc_code, key] = values[0]  # Use the first value
#     except Exception as e:
#         print(f"Error processing {soc_code}: {e}")

# # Reorder columns to have code and title columns next to each other

# original_cols = [col for col in skill_vectors_python.columns if col not in ['major', 'minor', 'broad', 'detailed', 'major_title', 'minor_title', 'broad_title', 'detailed_title']]
    
# # Create ordered column list with code-title pairs
# ordered_hierarchy = []
# for level in ['major', 'minor', 'broad', 'detailed']:
#     ordered_hierarchy.extend([level, f'{level}_title'])

# # Reorder columns
# new_cols = original_cols + ordered_hierarchy
# skill_vectors_python = skill_vectors_python[new_cols]
    
# # %%
# # Repeat the pairplot using major occupation codes to group the data
# plt.figure(figsize=(15, 10))
# sns.pairplot(skill_vectors_python, diag_kind='kde', hue='major_title', palette='Set2')
# plt.show()

# # %% 
# # Load WFH data
# wfh_data = pd.read_csv("data/results/wfh_estimates.csv").set_index('ONET_SOC_CODE')
# # Merge WFH data with skill vectors
# # skill_vectors_python = skill_vectors_python.join(wfh_data["ESTIMATE_WFH_ABLE"], how='left')

# ANCHOR_COLS_NAMES = ['Mechanical', 'Mathematics', 'Social Perceptiveness']

# # Fit logistic regression models for each skill
# python_models = {}
# for col in ANCHOR_COLS_NAMES:
#     if col == 'OCCUPATION_TITLE' or col == 'ESTIMATE_WFH_ABLE':
#         continue
#     X = skill_vectors_python[col].values.reshape(-1, 1)
#     # Create a binary target for logistic regression (assuming > 0.5 means remote work)
#     y = (skill_vectors_python['ESTIMATE_WFH_ABLE'] > 0.0).astype(int)
#     model = LogisticRegression()
#     model.fit(X, y)
#     python_models[col] = model
# # Plot logistic curves for each skill 
# plt.figure(figsize=(15, 5))
# for i, (skill_name, model) in enumerate(python_models.items()):
#     plt.subplot(1, 3, i+1)
    
#     # Create a range of values for the skill
#     x_range = np.linspace(0, 1, 100).reshape(-1, 1)
#     # Predict probabilities
#     y_probs = model.predict_proba(x_range)[:, 1]
    
#     # Plot the curve
#     plt.plot(x_range, y_probs, 'r-', linewidth=2)
    
#     plt.title(f'{skill_name} vs. Remote Work Probability')
#     plt.xlabel(f'{skill_name} Skill Score')
#     plt.ylabel('Probability of Remote Work')
#     plt.grid(True, alpha=0.3)
#     plt.ylim(-0.05, 1.05)

# plt.tight_layout()
# plt.suptitle('Logistic Regression: Skill Impact on Remote Work (Python Method)', y=1.05)
# plt.show()

# # Create FacetGrid plots to visualize relationships between skill indices and WFH ability
# plt.figure(figsize=(15, 10))

# skills_melted_python = pd.melt(
#     skill_vectors_python.reset_index(), 
#     id_vars=['ONET_SOC_CODE', 'OCCUPATION_TITLE', 'ESTIMATE_WFH_ABLE'],
#     value_vars=['Mechanical', 'Mathematics', 'Social Perceptiveness'],
#     var_name='Skill', value_name='Score'
# )

# # Plot for Python method
# g = sns.FacetGrid(skills_melted_python[skills_melted_python.ESTIMATE_WFH_ABLE > 0],
#                 col='Skill', height=4, aspect=1.2)

# g.map_dataframe(sns.scatterplot, y='Score', x='ESTIMATE_WFH_ABLE', alpha=0.6)
# g.map_dataframe(sns.regplot, y='Score', x='ESTIMATE_WFH_ABLE', scatter=False, color='red')
# g.set_axis_labels('Teleworkability', 'Skill Score')
# # g.set_titles('{col} Skills vs WFH Ability (Python)')
# plt.show()


# # %%
