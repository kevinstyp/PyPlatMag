import logging
import os
import pickle
import time

import numpy as np
import pandas as pd

from utils import data_io

logger = logging.getLogger(__name__)

def get_save_path(save_path, satellite_specifier):
    return save_path + satellite_specifier + "/"

def nan_application(year_month_specifiers_list, save_path, satellite_specifier, meta_features):
    full_read_path = save_path + satellite_specifier + "/"
    with open(full_read_path + "features_to_drop.pickle", 'rb') as f:
        features_to_drop = pickle.load(f)
    with open(full_read_path + "features_fillna_mean.pickle", 'rb') as f:
        features_fillna_mean = pickle.load(f)
    for year_month_specifier in year_month_specifiers_list:  # Go through the big list

        print("year_month: ", year_month_specifier)
        df = data_io.read_df(save_path, satellite_specifier, [year_month_specifier], dataset_name="data")

        print("df.columns before features_to_drop: ", list(df.columns))
        # features_to_drop
        df = df.drop(columns=features_to_drop, errors='ignore')
        print("df.columns after features_to_drop: ", list(df.columns))

        df = df.reset_index(drop=True)

        # Mark all occurings of Nans which are not in the meta features
        list_of_indices = df.index.difference(df.drop(meta_features, axis=1).dropna(axis='index').index).tolist()
        print("len(list_of_indices): ", len(list_of_indices))
        df['NaN_Flag'] = 0
        # z_all['NaN_Flag'][list_of_indices] = 1
        df.loc[list_of_indices, 'NaN_Flag'] = 1

        df = df.fillna(features_fillna_mean)
        # fillna() does not fill the columns which are not in the dataframe yet, but have a value in features_fillna_mean
        # add them here, for consistent data (same number of columns in every month)
        missing_feats = [feat for feat in features_fillna_mean.index if feat not in df.columns]
        print("df.shape: ", df.shape)
        print("features_fillna_mean.shape: ", features_fillna_mean.shape)
        print("features_fillna_mean: ", features_fillna_mean)
        print("len(missing_feats): ", len(missing_feats))
        print("missing_feats: ", missing_feats)
        for col in missing_feats:
            df[col] = features_fillna_mean[col]

        print("df.shape: ", df.shape)
        print("df.head(2): ", df.head(2))

        data_io.save_df(df, save_path, satellite_specifier, year_month_specifier, dataset_name="data_nonan")



        print("len(df.columns): ", len(df.columns))

        print("df[RAW_Timestamp][0]: ", df["RAW_Timestamp"][0])

        print("amps values before: ", df[['amps_b_mag_x', 'amps_b_mag_y', 'amps_b_mag_z']].values)
        print("df.isna().sum(): ", df.isna().sum())
        print("df.dtypes: ", df.dtypes)
        print("df.infer_objects().dtypes: ", df.infer_objects().dtypes)
        df = df.infer_objects()
        print("amps values after: ", df[['amps_b_mag_x', 'amps_b_mag_y', 'amps_b_mag_z']].values)
        print("df.shape: ", df.shape)

def nan_determination_merge(year_month_specifiers_list, save_path, satellite_specifier, nan_share=0.2, essential_calibration_keys=[]):
    df_column_nancount_list = []
    df_column_mean_list = []
    df_overall_list = []
    # Load monthly data for mean and count and add them up
    for year_month_specifier in year_month_specifiers_list:
        monthly_path = get_save_path(save_path, satellite_specifier) + year_month_specifier + "/"
        print("monthly_path: ", monthly_path)
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
    global_df_column_mean = dict.fromkeys(df_columns, 0)
    global_df_column_count = dict.fromkeys(df_columns, 0)

    first_column = df_columns[0]
    second_column = df_columns[1]
    third_column = df_columns[2]
    print("first_column: ", first_column)
    print("second_column: ", second_column)
    print("third_column: ", third_column)
    print("df_column_count_list[0]: ", df_column_nancount_list[0])
    print("df_column_mean_list[0]: ", df_column_mean_list[0])
    print("df_column_count_list[0][first_column]: ", df_column_nancount_list[0][first_column])
    print("df_column_mean_list[0][first_column]: ", df_column_mean_list[0][first_column])
    print("df_column_count_list[0][second_column]: ", df_column_nancount_list[0][second_column])
    print("df_column_mean_list[0][second_column]: ", df_column_mean_list[0][second_column])
    print("df_column_count_list[0][third_column]: ", df_column_nancount_list[0][third_column])
    print("df_column_mean_list[0][third_column]: ", df_column_mean_list[0][third_column])

    # Nancount list contains a list for each column-key, which contains the nancounts for each month
    nancount_list = [[dictionary[column] if column in dictionary else 0 for dictionary in df_column_nancount_list] for column in
                  df_columns]
    print("nancount_list: ", nancount_list)
    print("len(nancount_list): ", len(nancount_list))
    # Mean list contains a list for each column-key, which contains the means for each month
    mean_list = [[dictionary[column] if column in dictionary else 0 for dictionary in df_column_mean_list] for column in
                 df_columns]
    print("mean_list: ", mean_list)
    print("len(mean_list): ", len(mean_list))
    # overall_list = [[dictionary[column] for dictionary in x_overall_list if column in dictionary] for column in x_columns]
    column_nancount_list = np.sum(nancount_list, axis=1)
    print("column_nancount_list: ", column_nancount_list)
    print("len(column_nancount_list): ", len(column_nancount_list))
    column_indices_nanthreshold = [i for i, x in enumerate(column_nancount_list) if x >= nan_share * total_samples]
    print("column_indices_nanthreshold: ", column_indices_nanthreshold)
    print("len(column_indices_nanthreshold): ", len(column_indices_nanthreshold))
    columns_ignore_essential = essential_calibration_keys
    # get indices in df_columns of columns_ignore_essential
    cols_ignore_essential_indices = [df_columns.index(col) for col in columns_ignore_essential if col in df_columns]
    print("cols_ignore_essential_indices: ", cols_ignore_essential_indices)
    print("len(cols_ignore_essential_indices): ", len(cols_ignore_essential_indices))
    final_removal_index_list = list(set(column_indices_nanthreshold) - set(cols_ignore_essential_indices))
    print("final_removal_index_list: ", final_removal_index_list)
    print("len(final_removal_index_list): ", len(final_removal_index_list))
    features_to_drop = [df_columns[i] for i in final_removal_index_list]
    print("features_to_drop: ", features_to_drop)

    # mean_sum_same_size_as_columns = np.sum(mean_list)
    # print("count_sum_same_size_as_columns: ", count_sum_same_size_as_columns)

    partly_mean_list = [
        np.nansum([ # Sum up the mean contributions of each month weighed by their share of the total samples
            (df_overall_list[j] - local_nancount[j]) / df_overall_list[j] * local_mean[j] * (
                    df_overall_list[j] - local_nancount[j]) / total_samples
            for j in range(len(local_nancount))
        ])
        for local_nancount, local_mean in zip(nancount_list, mean_list)
    ]
    print("partly_mean_list: ", partly_mean_list)

    features_fillna_mean = pd.Series(dict(zip(df_columns, partly_mean_list)))  # , orient='index', dtype=np.float64)
    print("features_fillna_mean.shape: ", features_fillna_mean.shape)
    print(features_fillna_mean.index)
    print(features_fillna_mean)
    print("type(features_fillna_mean): ", type(features_fillna_mean))

    full_write_path = save_path + satellite_specifier + "/"
    if not os.path.exists(full_write_path):
        os.makedirs(full_write_path)
    with open(full_write_path + "features_to_drop.pickle", 'wb') as f:
        pickle.dump(features_to_drop, f)
    with open(full_write_path + "features_fillna_mean.pickle", 'wb') as f:
        pickle.dump(features_fillna_mean, f)


def nan_determination(year_month_specifiers, save_path, satellite_specifier, meta_features=[]):
    print("year_month_specifiers: ", year_month_specifiers)
    for year_month_specifier in year_month_specifiers:
        print("year_month_specifier: ", year_month_specifier)
        df = data_io.read_df(save_path, satellite_specifier, [year_month_specifier], dataset_name="data")
        print("df[RAW_Timestamp][0]: ", df["RAW_Timestamp"][0])

        # Remove meta features from df as they are not used for training and may contain NaNs
        df = df.drop(columns=meta_features)

        # x_all, y_all, z_all, features_to_drop = training_procedure.filter_nan_share(x_all, y_all, z_all, nan_share=nan_share, return_features=True)
        df_column_nancount = dict.fromkeys(df.columns, 0)
        df_column_mean = dict.fromkeys(df.columns, 0)
        df_overall = df.shape[0]

        for i, c in enumerate(df.columns):
            print(str(i) + ":\t" + str(c) + ", NaN: " + str(df[c].isna().sum()) + "/" + str(df[c].shape[0]))
            df_column_nancount[c] = df[c].isna().sum()
            df_column_mean[c] = df[c].mean()

        monthly_path = get_save_path(save_path, satellite_specifier) + year_month_specifier + "/"
        if not os.path.exists(monthly_path):
            os.makedirs(monthly_path)
        with open(monthly_path + "df_column_nancount.pickle", "wb") as f:
            pickle.dump(df_column_nancount, f)
        with open(monthly_path + "df_column_mean.pickle", "wb") as f:
            pickle.dump(df_column_mean, f)
        with open(monthly_path + "df_overall.pickle", "wb") as f:
            pickle.dump(df_overall, f)
