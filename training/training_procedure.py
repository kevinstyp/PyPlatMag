import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def decompose_dataframe(data, y_features, meta_features):
    y_all = data[y_features]
    z_all = data[meta_features]
    x_all = data.drop(meta_features, axis=1).drop(y_features, axis=1)
    return (x_all, y_all, z_all)


def save_columns(data, config_goce, year_month_specifiers, dirname):
    training_file_path = os.path.join(dirname, config_goce.train_config.training_file_path)
    year_months = '_'.join([year_month_specifiers[0], year_month_specifiers[-1]])
    xy_columns_file = training_file_path + year_months + "/xy_columns.pkl"
    xy_columns = data.drop(config_goce.meta_features, axis=1, errors='ignore').columns.tolist()
    logger.debug("xy-columns: ", xy_columns)
    if not os.path.exists(xy_columns_file):
        os.makedirs(os.path.dirname(xy_columns_file), exist_ok=True)
    print("writing x_all_columns_file.")
    with open(xy_columns_file, 'wb') as f:
        pickle.dump(xy_columns, f)


def align_columns(data, config, config_goce, year_month_specifiers, dirname):
    training_file_path = os.path.join(dirname, config_goce.train_config.training_file_path)
    year_months = '_'.join([year_month_specifiers[0], year_month_specifiers[-1]])
    xy_columns_file = training_file_path + year_months + "/xy_columns.pkl"
    # Read in used columns without meta features for dataframe during training
    with open(xy_columns_file, 'rb') as f:
        xy_columns = pickle.load(f)

    # Read in fill-in values for nan-columns
    full_read_path = config.write_path + config.satellite_specifier + "/"
    with open(full_read_path + "features_fillna_mean.pickle", 'rb') as f:
        features_fillna_mean = pickle.load(f)

    # Check whether all available columns during training (possibly multiple months) are also available during this month,
    # otherwise fill them with the corresponding nan-fill values as was done before
    for col in xy_columns:
        if col not in data.columns:
            data[col] = features_fillna_mean[col]

    # # Same order as what is saved, to later match with processing again
    # data = data[xy_columns]
    # print("x_all-shape after extra columns: ", x_all.shape)
    return data


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


def add_solar_activity(x_all, z_all):
    x_all["F10.7"] = z_all["F10.7"]
    x_all["F10.7-81d"] = z_all["F10.7-81d"]
    return x_all


def add_day_of_year(x_all, z_all):
    x_all["DOY"] = z_all["DOY"]
    return x_all


def filter_std(x_all, training_file_path, year_month_specifiers, use_cache):
    year_months = '_'.join([year_month_specifiers[0], year_month_specifiers[-1]])
    std_file = training_file_path + year_months + "/std_column_indices.pkl"
    logger.info(f"std_file: {std_file}")

    # Check if the file exists and if use_cache is True
    if use_cache and os.path.exists(std_file):
        logger.info("Loading std_indices from file.")
        with open(std_file, 'rb') as f:
            std_indices = pickle.load(f)
    else:
        # Get std_indices
        std_indices = (x_all.std(axis=0) != 0.)
        std_indices = std_indices[std_indices].index
        if not os.path.exists(std_file):
            os.makedirs(os.path.dirname(std_file), exist_ok=True)
        with open(std_file, 'wb') as f:
            pickle.dump(std_indices, f)

    logger.debug(f"x_all: {x_all}")
    logger.debug(f"x_all.shape: {x_all.shape}")
    logger.debug(f"std_indices: {std_indices}")
    logger.debug(f"std_indices.shape: {std_indices.shape}")

    return x_all[std_indices]


def filter_correlation(x_all, training_file_path, year_month_specifiers, use_cache):
    year_months = '_'.join([year_month_specifiers[0], year_month_specifiers[-1]])
    corr_file = training_file_path + year_months + "/corr_column_indices.pkl"
    logger.debug(f"corr_file: {corr_file}")

    # Check if the file exists and if use_cache is True
    if use_cache and os.path.exists(corr_file):
        with open(corr_file, 'rb') as f:
            corr_indices = pickle.load(f)
    else:
        # Compute absolute of correlation matrix
        corr = pd.DataFrame(np.corrcoef(x_all.to_numpy(), rowvar=False)).abs()
        # Get the upper triangle of the correlation matrix without using np.bool (which is deprecated)
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        # Find index of feature columns with correlation greater/equals than 1
        # and roughly computer precision (this is needed because of how the correlation is internally computed)
        corr_indices = [column for column in upper_tri.columns if
                        any(upper_tri[column] >= 1.0 - (10 * np.finfo(np.float32).eps))]
        logger.debug(f"corr_indices: {corr_indices}")

        if not os.path.exists(corr_file):
            os.makedirs(os.path.dirname(corr_file), exist_ok=True)
        with open(corr_file, 'wb') as f:
            pickle.dump(corr_indices, f)

    return x_all.drop(columns=x_all.columns[corr_indices])


def scale_data(x_all, training_file_path, year_month_specifiers, use_cache):
    year_months = '_'.join([year_month_specifiers[0], year_month_specifiers[-1]])
    scaler_file = training_file_path + year_months + "/scaler.pkl"
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
    # TODO: Test this through: Is it really necessary to limit ourselves by the batch_size because of the custom implementation
    #  of the layers?
    split_arrays = train_test_split(*arrays, test_size=test_split)
    max_size_train = split_arrays[0].shape[0] - split_arrays[0].shape[0] % batch_size
    max_size_test = split_arrays[1].shape[0] - split_arrays[1].shape[0] % batch_size

    # Limit every second or other second array in split_arrays by max_size_train or max_size_test
    split_arrays = [
        arr[:max_size_train] if i % 2 == 0 else arr[:max_size_test]
        for i, arr in enumerate(split_arrays)
    ]
    return split_arrays


def extract_and_rebalance_weightings(z_all):
    weightings = z_all['weightings'].copy()
    # Weightings sample size
    s1 = weightings[weightings == 2].shape[0]  # low latitudes
    s2 = weightings[weightings == 1.5].shape[0]  # mid latitudes
    s3 = weightings[weightings == 1].shape[0]  # high latitudes
    logger.info(f"weightings, analysis for latitudes: low ({s1}), mid({s2}), high({s3})")
    ratio_w2 = 1 / 4
    ratio_w3 = 1 / 160
    w1 = (s1 + s2 + s3) / (s1 + s2 * ratio_w2 + s3 * ratio_w3)
    w2 = w1 * ratio_w2
    w3 = w1 * ratio_w3
    weightings.loc[weightings == 2] = w1
    weightings.loc[weightings == 1.5] = w2
    weightings.loc[weightings == 1] = w3
    logger.info(f"Calculated rebalanced weightings: low ({w1}), mid({w2}), high({w3})")
    return weightings


def prepare_data(data, config, config_goce, dirname, train_config, use_cache=True):
    training_file_path = os.path.join(dirname, train_config.training_file_path)
    # Extract power currents if use_pinn is set before scaling is applied
    electric_current_df = None
    if train_config.use_pinn:
        current_parameters_file = os.path.join(dirname, config_goce.current_parameters_file)
        goce_column_description_file = os.path.join(dirname, config_goce.goce_column_description_file)
        data, electric_current_df = extract_electric_currents(data, current_parameters_file, goce_column_description_file)

    # TODO: Why not use decompose_dataframe() function?
    # TODO: Is the modulo from the preprocess_data_array.py still needed?
    # TODO: Decompose is the much better name, split sounds like train / test which does not happen here
    # TODO: training_data, training_prcedure -> Maybe, rename them to preprare_data, prepare_procedure or smth
    x_all, y_all, z_all = decompose_dataframe(data, config_goce.y_all_feature_keys, config_goce.meta_features)

    # Add solar activity, and DOY
    x_all = add_solar_activity(x_all, z_all)
    x_all = add_day_of_year(x_all, z_all)
    logger.info(f"x_all - columns assigned after split: {x_all.columns.tolist()}")

    # Std, Corr, Scaling
    if train_config.filter_std:
        x_all = filter_std(x_all, training_file_path, config.year_month_specifiers, use_cache)
        logger.debug(f"x_all - shape after std filtering: {x_all.shape}")
    if train_config.filter_correlation:
        x_all = filter_correlation(x_all, training_file_path, config.year_month_specifiers, use_cache)
        logger.debug(f"x_all - shape after correlation filtering: {x_all.shape}")
    x_all = scale_data(x_all, training_file_path, config.year_month_specifiers, use_cache)
    logger.info(f"x_all - shape before splitting: {x_all.shape}")
    logger.info(f"Final columns: {x_all.columns.tolist()}")
    return electric_current_df, x_all, y_all, z_all