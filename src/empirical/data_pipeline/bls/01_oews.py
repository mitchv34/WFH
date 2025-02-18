# %% Import packages and define constants
import os
import pandas as pd
import numpy as np
import logging
import argparse
import zipfile

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.basename(__file__)
log_file = os.path.join(current_dir, f'{script_name.replace(".py", ".log")}')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base paths
BASE_DIR = '../../../../'
# BASE_DIR = '.'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'bls', 'oews')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'bls', 'oews')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# %%
for file in os.listdir(DATA_DIR):
    if file.endswith('.zip'):
        file_path = os.path.join(DATA_DIR, file)
        zipfile.ZipFile(file_path).extractall(DATA_DIR)
        folder_extracted = os.path.join(DATA_DIR, file.replace('.zip', ''))
        

# %%
