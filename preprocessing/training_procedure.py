import logging
import os
import pickle

import pandas as pd

#config_goce = Box.from_yaml(filename="./config_goce.yaml", Loader=yaml.SafeLoader)


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

