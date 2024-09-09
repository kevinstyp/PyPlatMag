import logging
import os
import pickle

import pandas as pd

logger = logging.getLogger(__name__)


def one_hot_encode(data, pandas_inplace):
    str_columns = []
    str_columns_binary = []

    # TODO: Small NaN Share needs to get a flag for every removed entry
    # Entries should not be removed here but rather get a flag added which's samples can later on be deleted
    # during preprocessing!!!
    for i, c in enumerate(data.columns):
        if type(data[c][0]) is str:
            unique_number = data[c].nunique()
            logger.debug(str(i) + ":\t" + str(c) + ", uniques: " + str(unique_number))
            if unique_number > 2:
                str_columns.append(c)
                logger.debug(f"more than 2: {str(set(data[c]))}")
            else:
                str_columns_binary.append(c)

    logger.info(f"Number of string features: {len(str_columns)}")
    logger.info(f"Number of binary string features: {len(str_columns_binary)}")
    logger.debug(f"string columns: {str_columns}")

    # Convert string features to one-hot-encoding using pandas function get_dummies()
    data = pd.get_dummies(data, columns=str_columns)
    logger.debug(f"columns in data after string features: {data.columns.tolist()}")
    data = pd.get_dummies(data, columns=str_columns_binary)
    logger.debug(f"columns in data after binary string features: {data.columns.tolist()}")

    # TODO: What are these 'string_features' doing? Can they be removed?
    # TODO: There is no purpose of these string_features anymore
    string_config_file = "config_goce/string_features.pkl"
    if os.path.isfile(string_config_file):
        with open(string_config_file, 'rb') as f:
            string_features = pickle.load(f)
    else:
        string_features = set()

    string_features.update(str_columns)
    string_features.update(str_columns_binary)
    present_string_features = [str_feat for str_feat in string_features if str_feat in data.columns]
    # This should be true if get_dummies() worked correctly (-all- raw string features contain only (-all-) NaNs):
    # Then proceed to remove the columns which got one-hot-encoded
    if data[present_string_features].isna().all().all():
        data = data.drop(present_string_features, axis=1)
    else:
        for str_feat in present_string_features:
            logger.warning(f"data[str_feat].isna().all(): {data[str_feat].isna().all()}")
        raise Exception("Feature where not all were nan: ", present_string_features)
    # data = data.drop(string_features, axis=1)
    logger.info(f"string_features: {string_features}")
    logger.debug(f"Double Check for left NaNs: {data.isna().any()}")

    return data
