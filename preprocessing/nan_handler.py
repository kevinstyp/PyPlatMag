import logging
import os
import pickle

import numpy as np
import pandas as pd

from utils import data_io
# For "PerformanceWarning: DataFrame is highly fragmented." from pandas
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


def nan_application(year_month_specifiers_list, write_path, satellite_specifier, meta_features):
    full_read_path = write_path + satellite_specifier + "/"
    with open(full_read_path + "features_to_drop.pickle", 'rb') as f:
        features_to_drop = pickle.load(f)
    with open(full_read_path + "features_fillna_mean.pickle", 'rb') as f:
        features_fillna_mean = pickle.load(f)
    for year_month_specifier in year_month_specifiers_list:  # Go through the big list
        logger.info(f"Current year_month: {year_month_specifier}")
        df = data_io.read_df(write_path, satellite_specifier, [year_month_specifier], dataset_name="data")

        # features_to_drop
        df = df.drop(columns=features_to_drop, errors='ignore')
        df = df.reset_index(drop=True)

        # Mark all occurings of Nans which are not in the meta features
        list_of_indices = df.index.difference(df.drop(meta_features, axis=1, errors='ignore').dropna(axis='index').index).tolist()
        df['NaN_Flag'] = 0
        # z_all['NaN_Flag'][list_of_indices] = 1
        df.loc[list_of_indices, 'NaN_Flag'] = 1

        df = df.fillna(features_fillna_mean)
        # fillna() does not fill the columns which are not in the dataframe yet, but have a value in features_fillna_mean
        # add them here, for consistent data (same number of columns in every month)
        missing_feats = [feat for feat in features_fillna_mean.index if feat not in df.columns]
        for col in missing_feats:
            df.loc[:, col] = features_fillna_mean[col]

        data_io.save_df(df, write_path, satellite_specifier, year_month_specifier, dataset_name="data_nonan")


def nan_determination_merge(year_month_specifiers_list, write_path, satellite_specifier, nan_share=0.2, essential_calibration_keys=[]):
    df_column_nancount_list = []
    df_column_mean_list = []
    df_overall_list = []
    # Load monthly data for mean and count and add them up
    for year_month_specifier in year_month_specifiers_list:
        monthly_path = data_io.get_save_path(write_path, satellite_specifier) + year_month_specifier + "/"
        logger.debug(f"monthly_path: {monthly_path}")
        with open(monthly_path + "df_column_nancount.pickle", "rb") as f:
            df_column_nancount = pickle.load(f)
        with open(monthly_path + "df_column_mean.pickle", "rb") as f:
            df_column_mean = pickle.load(f)
        with open(monthly_path + "df_overall.pickle", "rb") as f:
            df_overall = pickle.load(f)

        df_column_nancount_list.append(df_column_nancount)
        df_column_mean_list.append(df_column_mean)
        df_overall_list.append(df_overall)

    # total number of samples
    total_samples = np.sum(df_overall_list, axis=0)

    # Unique list of all available columns
    df_columns = list(set.union(*(set(dic.keys()) for dic in df_column_nancount_list)))
    logger.info(f"Total number of columns: {len(df_columns)}")

    # TODO: Remove these lines, if unnecessary
    #global_df_column_mean = dict.fromkeys(df_columns, 0)
    #global_df_column_count = dict.fromkeys(df_columns, 0)

    # Nancount list contains a list for each column-key, which contains the nancounts for each month
    nancount_list = [[dictionary[column] if column in dictionary else 0 for dictionary in df_column_nancount_list] for column in
                  df_columns]
    logger.debug(f"nancount_list: {nancount_list}")
    # Mean list contains a list for each column-key, which contains the means for each month
    mean_list = [[dictionary[column] if column in dictionary else 0 for dictionary in df_column_mean_list] for column in
                 df_columns]
    logger.debug(f"mean_list: {mean_list}")
    # Sum up the nancounts for each column
    column_nancount_list = np.sum(nancount_list, axis=1)
    # Filter for columns with more than nan_share missing values
    column_indices_nanthreshold = [i for i, x in enumerate(column_nancount_list) if x >= nan_share * total_samples]
    logger.debug(f"column_indices_nanthreshold: {column_indices_nanthreshold}")
    columns_ignore_essential = essential_calibration_keys
    # get indices in df_columns of essential columns which may not be altered
    cols_ignore_essential_indices = [df_columns.index(col) for col in columns_ignore_essential if col in df_columns]
    logger.debug(f"cols_ignore_essential_indices: {cols_ignore_essential_indices}")
    # Match the indices of the columns to be removed with the indices of the essential columns
    final_removal_index_list = list(set(column_indices_nanthreshold) - set(cols_ignore_essential_indices))
    # Convert to column names
    features_to_drop = [df_columns[i] for i in final_removal_index_list]
    logger.debug(f"features_to_drop: {features_to_drop}")

    # Sum up the means for each column, taking into account their share of the total samples
    partly_mean_list = [
        np.nansum([ # Sum up the mean contributions of each month weighed by their share of the total samples
            (df_overall_list[j] - local_nancount[j]) / df_overall_list[j] * local_mean[j] * (
                    df_overall_list[j] - local_nancount[j]) / total_samples
            for j in range(len(local_nancount))
        ])
        for local_nancount, local_mean in zip(nancount_list, mean_list)
    ]
    logger.debug(f"partly_mean_list: {partly_mean_list}")

    features_fillna_mean = pd.Series(dict(zip(df_columns, partly_mean_list)))

    full_write_path = write_path + satellite_specifier + "/"
    if not os.path.exists(full_write_path):
        os.makedirs(full_write_path)
    with open(full_write_path + "features_to_drop.pickle", 'wb') as f:
        pickle.dump(features_to_drop, f)
    with open(full_write_path + "features_fillna_mean.pickle", 'wb') as f:
        pickle.dump(features_fillna_mean, f)


def nan_determination(year_month_specifiers, write_path, satellite_specifier, meta_features=[]):
    for year_month_specifier in year_month_specifiers:
        df = data_io.read_df(write_path, satellite_specifier, [year_month_specifier], dataset_name="data")

        # Remove meta features from df as they are not used for training and may contain NaNs
        df = df.drop(columns=meta_features, errors='ignore')

        # x_all, y_all, z_all, features_to_drop = training_procedure.filter_nan_share(x_all, y_all, z_all, nan_share=nan_share, return_features=True)
        df_column_nancount = dict.fromkeys(df.columns, 0)
        df_column_mean = dict.fromkeys(df.columns, 0)
        df_overall = df.shape[0]

        for i, c in enumerate(df.columns):
            logger.debug(f"{str(i)}:\t {str(c)}, NaN:  {str(df[c].isna().sum())}/ {str(df[c].shape[0])}")
            df_column_nancount[c] = df[c].isna().sum()
            df_column_mean[c] = df[c].mean()

        monthly_path = data_io.get_save_path(write_path, satellite_specifier) + year_month_specifier + "/"
        if not os.path.exists(monthly_path):
            os.makedirs(monthly_path)
        with open(monthly_path + "df_column_nancount.pickle", "wb") as f:
            pickle.dump(df_column_nancount, f)
        with open(monthly_path + "df_column_mean.pickle", "wb") as f:
            pickle.dump(df_column_mean, f)
        with open(monthly_path + "df_overall.pickle", "wb") as f:
            pickle.dump(df_overall, f)
