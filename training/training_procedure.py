import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def decompose_dataframe(df):
    y_all_features = config_goce.y_all_feature_keys
    z_all_features = config_goce.z_all_feature_keys
    y_all = df[y_all_features]
    z_all = df[z_all_features]
    print("z_all: ", z_all.shape)
    print("z_all: ", z_all.columns)
    x_all = df.drop(z_all_features, axis=1).drop(y_all_features, axis=1)

    return (x_all, y_all, z_all)

def extract_electric_currents_parameter_file(current_parameters_file, goce_column_description_file):
    # read in csv into pandas
    filename = goce_column_description_file
    df = pd.read_excel(filename, engine='openpyxl')
    df = df[["Parameter Name", "Unit", "Description"]]
    logger.debug(f"df.head(3): {df.head(3)}")
    logger.debug(f"df.shape: {df.shape}")

    # filter values in column "Unit" for either A, mA (Ampere, milliAmpere)
    df = df[df['Unit'].isin(["A", "mA"])]
    logger.debug(f"df.head(3): {df.head(3)}")
    logger.info(f"number of identified electric currents from description: {df.shape}")

    current_parameters = df[["Parameter Name", "Unit"]]
    current_parameters = current_parameters.set_index("Parameter Name")
    logger.debug(f"current_parameters: {current_parameters}")

    if not os.path.exists(current_parameters_file):
        os.makedirs(os.path.dirname(current_parameters_file), exist_ok=True)
    with open(current_parameters_file, 'wb') as f:
        pickle.dump(current_parameters, f)

def extract_electric_currents(df, current_parameters_file, goce_column_description_file):
    # Check for the existence of the current_parameters_file, otherwise call create method
    if not os.path.exists(current_parameters_file):
        extract_electric_currents_parameter_file(current_parameters_file, goce_column_description_file)
    with open(current_parameters_file, 'rb') as f:
        df_current_parameters = pickle.load(f)

    current_parameters_names = df_current_parameters.index
    logger.debug(f"current_parameters_names: {current_parameters_names}")
    available_currents = sorted(list(set(current_parameters_names).intersection(set(df.columns))))
    logger.info(f"number of identified electric currents from data: {len(available_currents)}")
    df_current_parameters = df_current_parameters.loc[available_currents]

    mA_currents = df_current_parameters.loc[df_current_parameters["Unit"] == "mA"].index
    logger.debug(f"electric currents with milliAmpere unit: {mA_currents}")
    df[mA_currents] = df[mA_currents] / 1000.
    electric_current_df = df[available_currents]
    df = df.drop(available_currents, axis=1)
    logger.info(f"data shape after extracting electric currents: {df.shape}")

    return df, electric_current_df

def filter_std(x_all, training_file_path, year_month_specifiers, use_cache):
    year_months = '_'.join([year_month_specifiers[0], year_month_specifiers[-1]])
    std_file = training_file_path + year_months + "/std_column_indices.pkl"

    # # TODO: What was columns used for here?
    # columns = []
    # if isinstance(x_all, pd.DataFrame):
    #     print("Converted x_all to array")
    #     columns = x_all.columns
    #     print("Got columns: ", len(columns))
    #     x_all = x_all.values

    # Check if the file exists and if use_cache is True
    if use_cache and os.path.exists(std_file):
        with open(std_file, 'rb') as f:
            std_indices = pickle.load(f)
    else:
        #std_indices = np.where(np.std(x_all, axis=0) != 0)
        # Get std_indices with pandas
        std_indices = (x_all.std(axis=0) != 0).index
        logger.debug(f"std_indices: {std_indices}")
        if not os.path.exists(std_file):
            os.makedirs(os.path.dirname(std_file), exist_ok=True)
        with open(std_file, 'wb') as f:
            pickle.dump(std_indices, f)

    logger.debug(f"x_all: {x_all}")
    logger.debug(f"x_all.shape: {x_all.shape}")

    return x_all[std_indices]


def filter_correlation(x_all, training_file_path, year_month_specifiers, use_cache):
    year_months = '_'.join([year_month_specifiers[0], year_month_specifiers[-1]])
    corr_file = training_file_path + year_months + "/corr_column_indices.pkl"
    # if isinstance(x_all, pd.DataFrame):
    #     logger.info(f"Converted x_all to array")
    #     x_all = x_all.values
    # Check if the file exists and if use_cache is True
    if use_cache and os.path.exists(corr_file):
        with open(corr_file, 'rb') as f:
            corr_indices = pickle.load(f)
    else:
        # Compute absolute of correlation matrix
        corr = pd.DataFrame(np.corrcoef(x_all.values, rowvar=False)).abs()
        # Get the upper triangle of the correlation matrix
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater/equals than 1
        # and roughly computer precision (this is needed because of how the correlation is internally computed)
        corr_indices = [column for column in upper_tri.columns if
                        any(upper_tri[column] >= 1.0 - (10 * np.finfo(np.float32).eps))]
        print("corr_indices: ", corr_indices)

        if not os.path.exists(corr_file):
            os.makedirs(os.path.dirname(corr_file), exist_ok=True)
        with open(corr_file, 'wb') as f:
            pickle.dump(corr_indices, f)

    #return np.delete(x_all, corr_indices, axis=1)
    return x_all.drop(columns=x_all.columns[corr_indices])


def scale_data(x_all, training_file_path, year_month_specifiers, use_cache):
    year_months = '_'.join([year_month_specifiers[0], year_month_specifiers[-1]])
    scaler_file = training_file_path + year_months + "/scaler.pkl"
    # if isinstance(x_all, pd.DataFrame):
    #     logger.info(f"Converted x_all to array")
    #     x_all = x_all.values
    # Check if the file exists and if use_cache is True
    if use_cache and os.path.exists(scaler_file):
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).set_output(transform="pandas")
        scaler = scaler.fit(x_all)
        if not os.path.exists(scaler_file):
            os.makedirs(os.path.dirname(scaler_file), exist_ok=True)
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)

    return scaler.transform(x_all)


def split_train_test(arrays, test_split, batch_size):
    #TODO: Test this through: Is it really necessary to limit ourselves by the batch_size because of the custom implementation
    # of the layers?
    split_arrays = train_test_split(*arrays, test_size=test_split)
    print("split_arrays:", split_arrays)
    print("type split_arrays: ", type(split_arrays))
    max_size_train = split_arrays[0].shape[0] - split_arrays[0].shape[0] % batch_size
    max_size_test = split_arrays[1].shape[0] - split_arrays[1].shape[0] % batch_size
    print("max_size_train:" , max_size_train)
    print("max_size_test:" , max_size_test)

    # Limit every second array in split_arrays by max_size
    split_arrays = [
        arr[:max_size_train] if i % 2 == 0 else arr[:max_size_test]
        for i, arr in enumerate(split_arrays)
    ]
    print("split_arrays:", split_arrays)
    print("type split_arrays: ", type(split_arrays))
    return split_arrays