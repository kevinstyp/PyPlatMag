import os
from typing import List

import pandas as pd
import logging
logger = logging.getLogger(__name__)


def get_save_path(write_path, satellite_specifier, dataset_name="data"):
    path = write_path + satellite_specifier + "/" + dataset_name + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_df(data, write_path, satellite_specifier, year_month_specifier, dataset_name="data"):
    path = get_save_path(write_path, satellite_specifier,dataset_name) + year_month_specifier
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_hdf(path + "/data.h5", key='df', mode='w')


def read_df(write_path, satellite_specifier, year_month_specifiers, dataset_name="data", specific_columns=None):
    df_list: List[pd.DataFrame] = []
    logger.info(f"year_month_specifiers: {year_month_specifiers}")
    for year_month in year_month_specifiers:
        path = get_save_path(write_path, satellite_specifier, dataset_name) + year_month + "/data.h5"
        logger.info(f"Reading file from path: {path}")
        if os.path.isfile(path):
            inter_df: pd.DataFrame = pd.read_hdf(path, "df", columns=specific_columns)
            df_list.append(inter_df)

    df = pd.concat(df_list)
    logger.info(f"Shape of training data for requested months: {df.shape}")
    return df
