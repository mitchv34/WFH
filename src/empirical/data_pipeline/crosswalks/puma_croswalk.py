# %%
import pandas as pd
import numpy as np
import os


# Base paths
BASE_DIR = '../../../../'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks')


puma10_20 = pd.read_csv(os.path.join(DATA_DIR, 'puma2010-to-puma2020.csv'), 
                        usecols = ['state', 'puma12', 'puma22', 'afact'])
# Drop the first row
puma10_20 = puma10_20.drop(0)
# Drop Puerto Rico
puma10_20 = puma10_20[puma10_20.state != '72']
# Pad state and PUMA codes with zeros
puma10_20['state'] = puma10_20['state'].str.zfill(2)
puma10_20['puma12'] = puma10_20['puma12'].str.zfill(5)
puma10_20['puma22'] = puma10_20['puma22'].str.zfill(5)
# Create a new column with the state and PUMA code
puma10_20['state_puma_12'] = puma10_20['state'] + '-' + puma10_20['puma12']
puma10_20['state_puma_22'] = puma10_20['state'] + '-' + puma10_20['puma22']
# Drop the state and PUMA columns
puma10_20 = puma10_20.drop(columns=['state', 'puma12', 'puma22'])
# Assign every PUMA to the PUMA with the highest afact
puma10_20 = puma10_20.sort_values('afact', ascending=False).drop_duplicates(subset='state_puma_12')
# Subset the data to only keep where the PUMA codes are different
puma10_20 = puma10_20[puma10_20['state_puma_12'] != puma10_20['state_puma_22']]
# %%

puma20_CBSA = pd.read_csv(  os.path.join(DATA_DIR,  'geocorr2022_2421106844.csv'), 
                            encoding='latin1', 
                            usecols = ['state', 'puma22', 'cbsa20', 'cbsatype20', 'CBSAName20', 'afact'])
# Drop the first row
puma20_CBSA = puma20_CBSA.drop(0)
# Drop non-core based statistical areas
puma20_CBSA = puma20_CBSA[puma20_CBSA.cbsa20 != '99999']
# Drop Puerto Rico
puma20_CBSA = puma20_CBSA[puma20_CBSA.state != '72']
# Create a new column with the state and PUMA code
puma20_CBSA['state_puma'] = puma20_CBSA['state'] + '-' + puma20_CBSA['puma22']
# Drop the state and PUMA columns
puma20_CBSA = puma20_CBSA.drop(columns=['state', 'puma22'])
# Assign every PUMA to the CBSA with the highest afact
puma20_CBSA = puma20_CBSA.sort_values('afact', ascending=False).drop_duplicates(subset='state_puma')

# Add the PUMA_12 codes to the data assign to the corresponding CBSA code in 2020
codes_12, codes_22 = puma10_20['state_puma_12'].to_list(), puma10_20['state_puma_22'].to_list()

# %%
afact = np.nan
new_data = {
    'cbsa20': [],
    'cbsatype20': [],
    'CBSAName20': [],
    'afact': [],
    'state_puma': []
}
for i in range(len(codes_12)):
    # If the code is already in the 2020 PUMA CBSA crosswalk, skip
    if codes_12[i] in puma20_CBSA['state_puma'].to_list():
        continue
    # Find cbsa20 code for the 2012 PUMA using the 2020 PUMA code
    cbsa20 = puma20_CBSA[puma20_CBSA['state_puma'] == codes_22[i]]['cbsa20'].values
    if len(cbsa20) != 1:
        continue
        
    new_data["cbsa20"].append( cbsa20[0] )
    new_data["cbsatype20"].append( puma20_CBSA[puma20_CBSA['state_puma'] == codes_22[i]]['cbsatype20'].values[0] )
    new_data["CBSAName20"].append( puma20_CBSA[puma20_CBSA['state_puma'] == codes_22[i]]['CBSAName20'].values[0] )
    new_data["afact"].append( afact )
    new_data["state_puma"].append( codes_12[i] )


puma_CBSA = pd.concat([puma20_CBSA, pd.DataFrame(new_data)], axis=0)
# %%
# Save the crosswalk to a csv file drop the index and the afact column
puma_CBSA.drop(columns=['afact']).to_csv(
    os.path.join(DATA_DIR, 'puma_to_cbsa.csv'),
    index=False)

# %%
