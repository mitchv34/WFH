# %%
"""
Title: 01_wfh_share_estimation.py
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-12
Description:
"""

# Importing necessary libraries
import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# Importing machine learning libraries from scikit-learn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.inspection import permutation_importance

#*=========================================================================================
#* DEFAULTS AND CONSTANTS
#*=========================================================================================

DATA_DIR_ORS = 'data/proc/ors/'
DATA_DIR_ONET = 'data/onet_data/processed/measure/'
DATA_DIR_ONET_REFERENCE = "/project7/high_tech_ind/WFH/WFH/data/onet_data/processed/reference/"

# %%
#?=========================================================================================
#? DATA LOADER
#?=========================================================================================
class DataLoader:
    """
    Loads both the ONET and ORS datasets.
    """
    def __init__(self,
                data_dir_ors = DATA_DIR_ORS, 
                data_dir_onet = DATA_DIR_ONET, 
                data_dir_onet_reference = DATA_DIR_ONET_REFERENCE):
        """
        Initializes the DataLoader with the specified directories.
        
        Parameters:
        data_dir_ors (str): Directory path for ORS data. (Contains labels)
        data_dir_onet (str): Directory path for ONET data. (Contains features)
        data_dir_onet_reference (str): Directory path for ONET reference data. (Contains category descriptions)
        """

        
        self.data_dir_ors = data_dir_ors
        
        self.data_dir_onet = data_dir_onet
        self.data_dir_onet_reference = data_dir_onet_reference

    def load_onet_data(self, data_list):
        # TODO: Optimize this function
        # TODO: Comment all the steps
        """
        Loads and pivots ONET data from multiple sources.
        """
        # List of substrings to remove from element names
        substrings_to_remove = [
            'and','discussions','with','work','or','deal','for','frequency','of','deal',
            ',', 'in','are','very','extremely','exposed to','spend time','degree', 'being',
            'such as safety shoes, glasses, gloves, hearing protection, hard hats, or life jackets',
            'such as breathing apparatus, safety harness, full protection suits, or radiation protection', 
            'consequence of error', 'same', 'versus', 'determined by', 'duration'
        ]
        substrings_to_remove = [s + " " for s in substrings_to_remove]
        pattern = '|'.join(map(re.escape, substrings_to_remove))
        
        # Load scale reference file
        scale_ref_path = os.path.join(self.data_dir_onet_reference, 'SCALES_REFERENCE.csv')
        scale_reference = pd.read_csv(scale_ref_path).set_index('SCALE_ID')['SCALE_NAME'].to_dict()
        scales_to_exclude = ['IH', 'VH']
        
        df = pd.DataFrame()
        for data_source in data_list:
            data_path = os.path.join(self.data_dir_onet, f'{data_source}.csv')
            data = pd.read_csv(data_path)
            data['SCALE_NAME'] = data['SCALE_ID'].map(scale_reference).apply(
                lambda x: re.sub(r"\(.*\)", "", x).strip().replace(" ", "_").lower()
            )
            data = data[~data['SCALE_ID'].isin(scales_to_exclude)]
            data["ELEMENT_NAME"] = (
                data["ELEMENT_NAME"]
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("/", "_")
                .str.replace("-", "_")
            )
            data["ELEMENT_NAME"] = data["ELEMENT_NAME"].str.replace(pattern, '', regex=True)
            
            if "CATEGORY" in data.columns:
                # Prepare to pivot using a different key if CATEGORY is present.
                # columns_pivot = ['ELEMENT_NAME', 'SCALE_NAME', 'CATEGORY']
                data["CATEGORY"] = data["CATEGORY"].fillna("").apply(lambda x : str(int(x)) if x != "" else x)
                data["CATEGORY"] = data["CATEGORY"].fillna("")
                data["SCALE_NAME"] += data["CATEGORY"].apply(lambda x : "_"  if len(x) > 0 else "")  + data["CATEGORY"]
                # columns_pivot = ['ELEMENT_ID', 'SCALE_NAME', 'CATEGORY']
                # # Try to load an appropriate reference file
                # possible_reference = [fname for fname in os.listdir(self.data_dir_onet_reference)
                #                     if data_source in fname and ('CATEGOR' in fname or 'REFERENCE' in fname)]
                # if possible_reference:
                #     ref_path = os.path.join(self.data_dir_onet_reference, possible_reference[0])
                #     reference = pd.read_csv(ref_path)
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].fillna("none")
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].str.lower()
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].apply(lambda x: re.sub(r"\(.*\)", "", x))
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].apply(
                #         lambda x: re.sub(r" or more but not every \w+", "", x)
                #     )
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].str.replace("once a ", "")
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].str.replace("contact with others", "")
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].str.replace("the", "")
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].str.replace("about", "")
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].str.replace("at all", "")
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].apply(
                #         lambda x: x.replace('diploma', "").replace("than ", "").replace("a ", "").replace(
                #             "certificate ", "").replace("courses", "").replace(
                #                 "'s", "").replace("training", "").replace("degree", "")
                #     )
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].apply(
                #         lambda x: re.sub(r' \bmonth(s)?\b', 'M', x)
                #     )
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].apply(
                #         lambda x: re.sub(r' \byear(s)?\b', 'Y', x)
                #     )
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].apply(
                #         lambda x: x.replace(", up to and including ", "_").replace(
                #             "over ", "").replace(
                #                 " or short demonstration", "").replace(
                #                     'anything beyond short demonstration', "none").replace(
                #                         "up to and including", "none")
                #     )
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].apply(
                #         lambda x: re.split(r'(?<!post)-', x)[0].strip()
                #     )
                #     reference["CATEGORY_DESCRIPTION"] = reference["CATEGORY_DESCRIPTION"].str.replace(" ", "_").replace("-", "_")
                    
                #     # Create mapping: (ELEMENT_ID, SCALE_ID, CATEGORY) -> CATEGORY_DESCRIPTION
                #     category_description = reference.set_index(['ELEMENT_ID', 'SCALE_ID', 'CATEGORY'])['CATEGORY_DESCRIPTION'].to_dict()
                #     data['CATEGORY_DESCRIPTION'] = data.apply(
                #         lambda row: category_description.get((row.ELEMENT_ID, row.SCALE_ID, row.CATEGORY), 'context'),
                #         axis=1
                #     )
                #     columns_pivot = ['ELEMENT_NAME', 'CATEGORY_DESCRIPTION']
            # else:
                # columns_pivot = ['ELEMENT_NAME', 'SCALE_NAME']
            columns_pivot = ['ELEMENT_ID', 'SCALE_NAME']
            
            # Pivot the data so that each row corresponds to an occupation (ONET_SOC_CODE)
            data = data.pivot(
                index='ONET_SOC_CODE',
                columns=columns_pivot,
                values='DATA_VALUE'
            ).reset_index().set_index('ONET_SOC_CODE')
            data.columns.names = [None, None]
            df = pd.concat([df, data], axis=1)
        return df

    def load_ors_data(self):
        """
        Loads the ORS data (labels) and performs basic preprocessing.
        """
        ors_path = os.path.join(self.data_dir_ors, 'final_second_wave_2023.csv')
        ors_data = pd.read_csv(ors_path)
        ors_data.rename(columns={'SOC_2018_CODE': 'ONET_SOC_CODE', 'ESTIMATE': 'ESTIMATE_WFH_ABLE'}, inplace=True)
        ors_data['ONET_SOC_CODE'] = ors_data['ONET_SOC_CODE'] + '.00'
        ors_data['ESTIMATE_WFH_ABLE'] = ors_data['ESTIMATE_WFH_ABLE'] / 100
        return ors_data

# %%
#?=========================================================================================
#? DATA PREPROCESSOR
#?=========================================================================================  
class DataPreprocessor:
    """
    Aggregates and merges the ONET and ORS data.
    """
    def __init__(self, onet_data, data_loader: DataLoader):
        self.data = onet_data
        self.data_loader = data_loader

    def prepare_data(self, metric='importance', aggregation_level=None, aggregation_function=max):
        # Select the metric. (Currently only a string option is supported.)
        # TODO: Add support for other metrics.
        if isinstance(metric, str):
            data = self.data.xs('importance', level=1, axis=1)
        if isinstance(metric, (tuple, list)):
            data =  pd.DataFrame()
            for m in metric:
                data = pd.concat([data, self.data.xs(m, level=1, axis=1)], axis=1)
                
        else:
            raise ValueError("Metric must be a string, tuple, or list.")
        
        # Optional aggregation based on a content model reference.
        if aggregation_level:
            data = data.reset_index().melt(id_vars='ONET_SOC_CODE')
            data['LEVEL'] = data.variable.apply(lambda x : ".".join(x.split(".")[0:3]) )

            # content_model_ref_path = os.path.join(self.data_loader.data_dir_onet_reference, 'CONTENT_MODEL_REFERENCE.csv')
            # content_model_reference = pd.read_csv(content_model_ref_path)
            # content_model_reference["ELEMENT_NAME"] = content_model_reference["ELEMENT_NAME"].apply(
            #     lambda x: x.replace(" ", "_").replace("-", "_").replace("/", "_").lower()
            # )
            # sub_ref = content_model_reference.loc[
            #     content_model_reference["ELEMENT_NAME"].isin(data.columns),
            #     ['ELEMENT_NAME', 'ELEMENT_ID']
            # ].reset_index(drop=True)
            # # Determine the aggregation level based on parts of the ELEMENT_ID
            # sub_ref["LEVEL"] = sub_ref["ELEMENT_ID"].apply(lambda x: ".".join(x.split(".")[0:aggregation_level]))
            # sub_ref["LEVEL_NAME"] = sub_ref["LEVEL"].map(
            #     content_model_reference[content_model_reference["ELEMENT_ID"].isin(sub_ref["LEVEL"])]
            #     .set_index('ELEMENT_ID')["ELEMENT_NAME"].to_dict()
            # )
            # level_agg_dict = sub_ref.set_index('ELEMENT_NAME')["LEVEL_NAME"].to_dict()
            # data = data.reset_index().melt(id_vars='ONET_SOC_CODE')
            # data["level_variable"] = data["variable"].map(level_agg_dict)
            # intermediate_data = data.copy()
            data = data.groupby(['ONET_SOC_CODE', 'level_variable'])["value"].apply(
                lambda x: aggregation_function(x)
            ).reset_index()
            data = data.pivot(
                index='ONET_SOC_CODE',
                columns='level_variable', 
                values='value').rename_axis(None, axis=1)

        
        # Merge with the ORS labels
        ors_data = self.data_loader.load_ors_data()
        data = data.merge(
            ors_data[["ONET_SOC_CODE", "ESTIMATE_WFH_ABLE"]],
            left_index=True,
            right_on='ONET_SOC_CODE',
            how='left'
        ).set_index('ONET_SOC_CODE')
        self.data = data
        return self.data

# %%
#?=========================================================================================
#? ENHANCED DATAFRAME
#?=========================================================================================
class EnhancedDataFrame(pd.DataFrame):
    """
    Extended DataFrame class with additional methods and reference data.
    """
    # Class-level attributes to store reference data
    _content_model = None
    _occupation_data = None
    
    def __init__(self, *args, **kwargs):
        # Use global DATA_DIR_ONET_REFERENCE
        # self._reference_dir = DATA_DIR_ONET_REFERENCE
        super().__init__(*args, **kwargs)
        if EnhancedDataFrame._content_model is None:
            self._load_reference_data()

    @property
    def _constructor(self):
        return EnhancedDataFrame
        
    @property
    def _constructor_sliced(self):
        return pd.Series
        
    @property
    def _constructor_expanddim(self):
        return pd.DataFrame

    @classmethod
    def _load_reference_data(cls):
        """Load reference data only once and store at class level"""
        # Load content model reference
        content_model_ref = pd.read_csv(os.path.join(DATA_DIR_ONET_REFERENCE, 'CONTENT_MODEL_REFERENCE.csv'))
        cls._content_model = content_model_ref.set_index("ELEMENT_ID").to_dict()
        
        # Load occupation data
        occ_data = pd.read_csv(os.path.join(DATA_DIR_ONET_REFERENCE, 'OCCUPATION_DATA.csv'))
        cls._occupation_data = occ_data.set_index("ONET_SOC_CODE").to_dict()
    
    def get_occ_data(self, type='title'):
        """Get occupation data for the given type"""
        if EnhancedDataFrame._occupation_data is None:
            self._load_reference_data()
        if type == 'title':
            df = self.reset_index()
            df.index = df["ONET_SOC_CODE"].map(self._occupation_data["OCCUPATION_TITLE"])
            return df
        elif type == 'description':
            df = self.reset_index()
            df.index = df["ONET_SOC_CODE"].map(self._occupation_data["OCCUPATION_DESCRIPTION"])
            return df
    
    def get_content_model(self, type='title'):
        """Get content model for the given type"""
        if EnhancedDataFrame._content_model is None:
            self._load_reference_data()
        if type == 'title':
            df = self.copy()
            df.columns = [self._content_model["ELEMENT_NAME"].get(col, col) for col in df.columns]
            return df
        elif type == 'description':
            df = self.copy()
            df.columns = [self._content_model["DESCRIPTION"].get(col, col) for col in df.columns]
            return df

# %%
#?=========================================================================================
#? DATA HANDLER
#?=========================================================================================
class DataStore:
    """
    Handles storing and splitting data.

    Attributes:
        raw_data (pd.DataFrame): The complete dataset.
        labeled_data (pd.DataFrame): Rows with a non-null label.
        unlabeled_data (pd.DataFrame): Rows missing the label.
        X_train, X_test, y_train, y_test: Training and testing splits of labeled data.
    """
    def __init__(self, 
                data_list, metric = ['importance'], 
                aggregation_level=None, 
                aggregation_function=max, 
                test_size=0.2,
                random_state=42,
                data_dir_ors = DATA_DIR_ORS, 
                data_dir_onet = DATA_DIR_ONET, 
                data_dir_onet_reference = DATA_DIR_ONET_REFERENCE
                ):
        
        self.Params = {
            'data_list': data_list,
            'DataPreprocessor': {
                    'metric': metric,
                    'aggregation_level': aggregation_level,
                    'aggregation_function': aggregation_function
            },
            'test_size': test_size,
            'random_state': random_state
        }
        
        # Sub classes
        self.DataLoader = DataLoader(
            data_dir_ors = data_dir_ors,
            data_dir_onet = data_dir_onet,
            data_dir_onet_reference = data_dir_onet_reference
        )      # Data Loader instance
        self.raw_data = self.DataLoader.load_onet_data(self.Params['data_list']) # Raw data
        self.DataPreprocessor = DataPreprocessor(self.raw_data, self.DataLoader) # Data Preprocessor
        # Preprocess and merge with labels
        self.raw_data = self.DataPreprocessor.prepare_data( metric=self.Params['DataPreprocessor']['metric'],
                                                            aggregation_level=self.Params['DataPreprocessor']['aggregation_level'],
                                                            aggregation_function=self.Params['DataPreprocessor']['aggregation_function'])

        self.labeled_data = None            # Rows with a non-null label
        self.unlabeled_data = None          # Rows missing the label
        self.X_train = None                 # Training features
        self.X_test = None                  # Testing features
        self.y_train = None                 # Training labels
        self.y_test = None                  # Testing labels
        self.iz_train = None                # Training labels for Stage 1 (binary)
        self.iz_test = None                 # Testing labels for Stage 1 (binary)
        self.split_by_label()               # Split the data by label



        # Automatically split the labeled data into training and testing sets.
        self.split_train_test(test_size, random_state)

        # Convert all DataFrames to EnhancedDataFrame
        for attr in dir(self):
            if isinstance(getattr(self, attr), pd.DataFrame):
                
                setattr(self, attr, self._convert_to_enhanced_df(getattr(self, attr)))
                

    def _convert_to_enhanced_df(self, df):
        """
        Helper method to convert a regular DataFrame to EnhancedDataFrame
        """
        if isinstance(df, pd.DataFrame):
            return EnhancedDataFrame(df)
        return df

    def split_by_label(self, label='ESTIMATE_WFH_ABLE'):
        """
        Splits the raw data into labeled and unlabeled data based on the presence
        of the specified label.

        Args:
            label (str): The column name to check for labels. Defaults to 'ESTIMATE_WFH_ABLE'.
        """

        self.labeled_data = self.raw_data[self.raw_data[label].notna()].copy()
        self.unlabeled_data = self.raw_data[self.raw_data[label].isna()].copy()

    def split_train_test(self, test_size=0.2, random_state=42):
        """
        Splits the labeled data into training and testing sets.

        Args:
            test_size (float): Fraction of data to use as test set.
            random_state (int): Random seed for reproducibility.
            bootstrap (bool): If True, performs bootstrap sampling on the labeled data before splitting.

        Returns:
            X_train, X_test, y_train, y_test: The training/testing splits.
        """
        X = self.labeled_data.drop(columns=["ESTIMATE_WFH_ABLE"])
        y = self.labeled_data["ESTIMATE_WFH_ABLE"]
        
        
        is_zero = (y == 0).astype(int)  # Binary target for Stage 1
        
        # Split data for classifier
        self.X_train, self.X_test, self.y_train, self.y_test, self.iz_train, self.iz_test = train_test_split(
            X, y, is_zero, test_size=test_size, random_state=random_state
        )

        return self.X_train, self.X_test, self.y_train, self.y_test   

    def bootstrap_split(self, random_state=None):
        """
        Creates a new training and test split using bootstrap resampling.
        
        This method performs sampling with replacement on the labeled_data to
        generate a bootstrap training sample. The out-of-bag (OOB) samples
        (i.e., those not selected in the bootstrap sample) are used as the test set.
        If no OOB samples are found, it falls back to a regular train/test split.

        Args:
            random_state (int, optional): Random seed for reproducibility.

        Returns:
            X_train, X_test, y_train, y_test: The new training and testing splits.
        """
        # Generate a bootstrap sample from the labeled data.
        bootstrap_sample = self.labeled_data.sample(frac=1, replace=True, random_state=random_state)
        
        # Out-of-bag (OOB) indices are those that were not sampled.
        oob_indices = self.labeled_data.index.difference(bootstrap_sample.index)
        
        # If no OOB samples are found (which is unlikely for large datasets),
        # fallback to a simple random split using the provided test_size.
        if len(oob_indices) == 0:
            from sklearn.model_selection import train_test_split
            bootstrap_sample, oob_sample = train_test_split(
                self.labeled_data,
                test_size=self.Params['test_size'],
                random_state=random_state
            )
        else:
            oob_sample = self.labeled_data.loc[oob_indices]
        
        # Create training and test splits.
        X_train = bootstrap_sample.drop(columns=["ESTIMATE_WFH_ABLE"])
        y_train = bootstrap_sample["ESTIMATE_WFH_ABLE"]
        X_test = oob_sample.drop(columns=["ESTIMATE_WFH_ABLE"])
        y_test = oob_sample["ESTIMATE_WFH_ABLE"]
        
        # Create binary labels for Stage 1.
        iz_train = (y_train == 0).astype(int)
        iz_test = (y_test == 0).astype(int)
        
        # Update instance attributes.
        self.X_train, self.y_train, self.iz_train = X_train, y_train, iz_train
        self.X_test, self.y_test, self.iz_test = X_test, y_test, iz_test
        
        # return self.X_train, self.X_test, self.y_train, self.y_test

# %%
#?=============================================================================
#? PLOT MANAGER CLASS
#?=============================================================================
class PlotManager:
    def __init__(self):
        self.figures = {}  # Dictionary to store multiple figures
        
    def create_plot(self, name, figsize=(10, 6)):
        """Create and store a new figure"""
        fig, ax = plt.subplots(figsize=figsize)
        self.figures[name] = {'figure': fig, 'axes': ax}
        return fig, ax
    
    def get_plot(self, name):
        """Retrieve a stored figure"""
        if name in self.figures:
            return self.figures[name]['figure'], self.figures[name]['axes']
        return None, None
        
    def save_plot(self, name, path):
        """Save a stored figure to file"""
        if name in self.figures:
            self.figures[name]['figure'].savefig(path)
            
    def show_plot(self, name):
        """Display a stored figure"""
        if name in self.figures:
            return self.figures[name]['figure']
            
    def update_plot(self, name, update_func):
        """Update an existing plot using a callback function"""
        if name in self.figures:
            fig, ax = self.figures[name]['figure'], self.figures[name]['axes']
            update_func(fig, ax)
            fig.canvas.draw()

# %%
#?=============================================================================
#? MODEL PIPELINE CLASS
#?=============================================================================
class ModelPipeline:
    """
    Handles training, evaluation, prediction, and explanation of the model.
    
    Attributes:
        data (DataStore): The data container instance.
        classifier_model: The base classifier (default: RandomForestClassifier).
        regressor_model: The base regressor (default: RandomForestRegressor).
        zero_threshold (float): Threshold to decide when a prediction is zero.
    """
    def __init__(self, 
                data: DataStore, 
                classifier_model=None, 
                regressor_model=None, 
                normalize="logit",
                zero_threshold=0.8,
                random_state=42,
                suppress_messages=False
                ):
        self.data = data  
        self.zero_threshold = zero_threshold
        self.normalize = normalize
        self.suppress_messages = suppress_messages
        
        # if classifier_model is not None:
        #     if hasattr(classifier_model, "set_params"):
        #         classifier_model.set_params(random_state=random_state)
        #     self.classifier_model = classifier_model
        # else:
        #     self.classifier_model = RandomForestClassifier(random_state=random_state)
        
        if classifier_model is None:
            self.classifier_model = RandomForestClassifier(random_state=random_state)
        else:
            self.classifier_model = classifier_model

        # if regressor_model is not None:
        #     # try:
        #     #     if hasattr(regressor_model, "set_params"):
        #     #         # Note: Ridge does not have a random_state parameter, so this may be skipped
        #     #         regressor_model.set_params(random_state=random_state)
        #     # except Warning:
        #     #     print("Regressor model does not have a random_state parameter.")
        #     self.regressor_model = regressor_model
        # else:
        #     self.regressor_model = RandomForestRegressor(random_state=random_state)
        
        if regressor_model is None:
            self.regressor_model = RandomForestRegressor(random_state=random_state)
        else:
            self.regressor_model = regressor_model

        self.calibrated_classifier = None  # Calibrated version of the classifier
        self.classifier = None  # To store the classifier
        self.regressor = None  # To store the regressor
        self.train_data = None  # To store training splits for later evaluation

        self.plot_manager = PlotManager()  # Instance of PlotManager


    def train(self, include_test=False):
        """
        Trains a two-stage model:
            1. A classifier to detect zero estimates.
            2. A regressor (trained on logit-transformed non-zero data) to predict non-zero values.
        """
    
        # Train the classifier
        self.classifier = self.classifier_model
        if include_test:
            # Train on all labeled data
            X_train = self.data.labeled_data.drop(columns=["ESTIMATE_WFH_ABLE"])
            y_train = self.data.labeled_data["ESTIMATE_WFH_ABLE"]
            iz_train = (y_train == 0).astype(int)
        else:
            # Train on the training split
            X_train = self.data.X_train
            y_train = self.data.y_train
            iz_train = self.data.iz_train

        self.classifier.fit(X_train, iz_train)
        # Calibrate the classifier for better probability estimates.
        # self.calibrated_classifier = CalibratedClassifierCV(self.classifier, method="isotonic", cv=3)
        # self.calibrated_classifier.fit(self.data.X_train, self.data.iz_train)
        self.calibrated_classifier = self.classifier
        
        # Train the regressor on non-zero data only.
        X_nonzero = self.data.X_train[self.data.iz_train != 1]
        y_nonzero = self.data.y_train[self.data.iz_train != 1]

        # Logit transformation: avoid predictions outside [0,1]
        # TODO: Add transformation selection as a parameter.
        if self.normalize == "logit":
            y_nonzero_norm = np.log(y_nonzero / (1 - y_nonzero))
        else:
            y_nonzero_norm = y_nonzero
        self.regressor = self.regressor_model
        self.regressor.fit(X_nonzero, y_nonzero_norm)

    def evaluate(self, include_test=False, verbose = True):
        """
        Evaluates the two-stage model using the test split from the data object.
        """
        
        if self.suppress_messages:
            verbose = False
        if include_test:
            X_test = self.data.labeled_data.drop(columns=["ESTIMATE_WFH_ABLE"]).copy()
            y_test = self.data.labeled_data["ESTIMATE_WFH_ABLE"].copy()
            iz_test = (y_test == 0).astype(int)
        else:
            X_test = self.data.X_test
            y_test = self.data.y_test
            iz_test = self.data.iz_test

        # Stage 1: Use the classifier to predict zeros.
        zero_probs = self.calibrated_classifier.predict_proba(X_test)[:, 1]
        predicted_zero = (zero_probs > self.zero_threshold).astype(int)
        
        # Stage 2: For non-zero cases, use the regressor.
        X_non_zero = X_test.loc[predicted_zero != 1]
        if not X_non_zero.empty:
            y_nz_pred_norm = self.regressor.predict(X_non_zero)
            if self.normalize == "logit":
                y_nz_pred = 1 / (1 + np.exp(-y_nz_pred_norm))  # Inverse logit
            else:
                y_nz_pred = y_nz_pred_norm
            # TODO: Add transformation selection as a parameter. (in this case is the inverse transformation)
        else:
            y_nz_pred = np.array([])
        
        final_pred = np.zeros(len(X_test))
        final_pred[predicted_zero == 1] = 0
        final_pred[predicted_zero != 1] = y_nz_pred
        
        # Evaluation metrics.
        ## Zero-Class F1: F1 score for the binary classifier.
        f1 = f1_score(iz_test, predicted_zero)
        non_zero_mask_test = y_test != 0
        ## Non-Zero MAE: Mean absolute error for non-zero estimates.
        mae_non_zero = mean_absolute_error(y_test[non_zero_mask_test], final_pred[non_zero_mask_test])
        ## Overall MAE: Mean absolute error for all estimates.
        mae = mean_absolute_error(y_test, final_pred)
        ## Correlation: Pearson correlation between actual and predicted values. 
        corr = np.corrcoef(y_test, final_pred)[0, 1]
        ## Correlation (Non-Zero): Pearson correlation for non-zero estimates only.
        corr_non_zero = np.corrcoef(y_test[non_zero_mask_test], final_pred[non_zero_mask_test])[0, 1]
        if verbose:
            print("Zero-Class F1:", f1)
            print("Non-Zero MAE:", mae_non_zero, "Correlation (Non-Zero):", corr_non_zero)
            print("Overall MAE:", mae, "Correlation:", corr)


        # Save the scores for later use.
        self.scores = {
            'f1': f1,
            'mae_non_zero': mae_non_zero,
            'mae': mae,
            'correlation': corr,
            'correlation_non_zero': corr_non_zero
        }
        
        # Plot actual vs. predicted.
        results_df = y_test.to_frame().assign(Predicted=final_pred)
        results_df["Actual_Zero"] = results_df["ESTIMATE_WFH_ABLE"].apply(lambda x: 1 if x > 0 else 0)
        results_df["Predicted_Zero"] = results_df["Predicted"].apply(lambda x: 1 if x > 0 else 0)
        results_df['Correct_Class'] = results_df["Actual_Zero"] == results_df["Predicted_Zero"]
        
        fig, ax = self.plot_manager.create_plot('test_prediction', figsize=(8, 5))
        sns.scatterplot(x="ESTIMATE_WFH_ABLE", y="Predicted", data=results_df, ax=ax,
                        s=200, hue='Correct_Class')
        ax.set_xlabel("Actual WFH Estimate")
        ax.set_ylabel("Predicted WFH Estimate")
        ax.set_title("Actual vs. Predicted WFH Estimate")
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="gray")
        sns.despine()
        plt.close()

    def feature_importances(self, n_features=10):
        """
        Plots both standard and permutation feature importances for the classifier and regressor.
        
        Args:
            n_features (int): Number of top features to plot. Defaults to 10.
        """
        
        # Get human-readable feature names using the get_content_model method.
        features = list(self.data.X_train.get_content_model(type='title').columns)
        
        # Standard feature importances (Classifier)
        if hasattr(self.calibrated_classifier, 'base_estimator_'):
            importances = self.calibrated_classifier.base_estimator_.feature_importances_
        else:
            importances = self.classifier.feature_importances_
        clf_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        clf_imp_df = clf_imp_df.sort_values(by='Importance', ascending=False)
        fig, ax = self.plot_manager.create_plot('classifier_importances', figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=clf_imp_df.iloc[:n_features], ax=ax)
        sns.despine()
        ax.set_title("Classifier Feature Importances")
        plt.close()

        # Standard feature importances (Regressor)
        importances_reg = self.regressor.feature_importances_
        reg_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances_reg})
        reg_imp_df = reg_imp_df.sort_values(by='Importance', ascending=False)
        fig, ax = self.plot_manager.create_plot('regressor_importances', figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=reg_imp_df.iloc[:n_features], ax=ax)
        sns.despine()
        ax.set_title("Regressor Feature Importances")
        plt.close()

        # Permutation importance for classifier
        perm_clf = permutation_importance(
            self.calibrated_classifier, self.data.X_train, self.data.iz_train,
            n_repeats=10, random_state=42
        )
        perm_clf_df = pd.DataFrame({
            'Feature': features,
            'Importance': perm_clf.importances_mean
        }).sort_values(by='Importance', ascending=False)
        fig, ax = self.plot_manager.create_plot('perm_importances_classifier', figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=perm_clf_df.iloc[:n_features], ax=ax)
        ax.set_title("Permutation Importances for Classifier")
        plt.close()

        # Permutation importance for regressor
        perm_reg = permutation_importance(
            self.regressor, self.data.X_train, self.data.y_train,
            n_repeats=10, random_state=42
        )
        perm_reg_df = pd.DataFrame({
            'Feature': features,
            'Importance': perm_reg.importances_mean
        }).sort_values(by='Importance', ascending=False)
        fig, ax = self.plot_manager.create_plot('perm_importances_regressor', figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=perm_reg_df.iloc[:n_features], ax=ax)
        ax.set_title("Permutation Importances for Regressor")
        plt.close()
        

    def predict_unlabeled(self):
        """
        Makes predictions on the unlabeled portion of the data.
        """
        X = self.data.unlabeled_data.drop(columns="ESTIMATE_WFH_ABLE")
        zero_probs = self.calibrated_classifier.predict_proba(X)[:, 1]
        predicted_zero = (zero_probs > self.zero_threshold).astype(int)
        X_non_zero = X.loc[predicted_zero != 1]
        if not X_non_zero.empty:
            y_nz_pred_norm = self.regressor.predict(X_non_zero)
            if self.normalize == "logit":
                y_nz_pred = 1 / (1 + np.exp(-y_nz_pred_norm))
            else:
                y_nz_pred = y_nz_pred_norm
        else:
            y_nz_pred = np.array([])
        final_pred = np.zeros(len(X))
        final_pred[predicted_zero == 1] = 0
        final_pred[predicted_zero != 1] = y_nz_pred
        
        
        # Update the unlabeled data with the predictions.
        self.data.unlabeled_data["ESTIMATE_WFH_ABLE"] = final_pred

        # Create a prediction plot for the unlabeled data and store it in the data object.
        fig, ax = self.plot_manager.create_plot('unlabeled_prediction', figsize=(10, 6))

        sns.kdeplot(self.data.unlabeled_data["ESTIMATE_WFH_ABLE"], 
                    label='Unlabeled',
                    fill=True, 
                    alpha=0.5)
        
        sns.kdeplot(self.data.labeled_data["ESTIMATE_WFH_ABLE"],
                    label='Labeled Data',
                    fill=True,
                    alpha=0.5)
        
        sns.despine()
        plt.xlabel("Predicted WFH Estimate")
        plt.ylabel("Density")
        plt.title("Distribution of Predicted WFH Estimates")
        plt.legend()
        plt.xlim(0, 1)
        plt.close()  # Close the figure window but keep the figure object stored

    def plot(self, plot_name):
        
        return self.plot_manager.show_plot(plot_name)

