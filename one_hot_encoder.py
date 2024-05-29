import pandas as pd

import os
import pickle

import logging
logger = logging.getLogger(__name__)

def one_hot_encode(data, pandas_inplace):
    str_columns = []
    str_columns_binary = []

    # TODO: Small NaN Share needs to get a flag for every removed entry
    ## Entries should not be removed here but rather get a flag added which's samples can later on be deleted
    ## during preprocessing!!!
    for i, c in enumerate(data.columns):

        # if data[c].isna().any():
        #     print(str(i) + ":\t" + str(c) + ", NaN: " + str(data[c].isna().sum()) + "/" + str(data[c].shape[0]))
        #     #print("Nan-Info: " + str(data[c].isna().sum()) + "/" + str(data[c].shape[0]))
        #     # TODO: This whole NaN detection should go into goce_filter preprocessing after reading the .h5 files
        #     if data[c].isna().sum() / data[c].shape[0] < 0.2:
        #         small_nan_share.append(c)
        #print(data[c][:2])
        if type(data[c][0]) is not str:
            #print(str(np.round(np.min(data[c]), 2)) + "  -  " + str(np.round(np.max(data[c]), 2)))
            continue
        else:
            unique_number = data[c].nunique()
            logger.debug(str(i) + ":\t" + str(c) + ", uniques: " + str(unique_number))
            #print("str-data here")
            #print("len-uniques: ", unique_number)
            if unique_number > 2:
                str_columns.append(c)
                logger.debug(f"more than 2: {str(set(data[c]))}")

                # Only 1:
                # Data_validity_flag_red_1i
                # {'Measure', 'Stand_by', 'ColdStAc', 'Cold_St'}
            else:
                str_columns_binary.append(c)

    print("len-all: ", len(str_columns))
    print("len-all-bin: ", len(str_columns_binary))
    # print("small_nan_share: ", len(small_nan_share))
    # print("small_nan_share: ", small_nan_share)

    delete_strings = []
    print("str_columns:", str_columns)


    ## Convert string features to one-hot-encoding using pandas function get_dummies()
    data = pd.get_dummies(data, columns=str_columns)
    print("pd-dummies: ", data.columns.tolist())
    str_columns = [col for col in str_columns if col not in delete_strings]




    print("string-features being removed before: ", len(str_columns_binary))
    delete_strings = []

    data = pd.get_dummies(data, columns=str_columns_binary)
    print("pd-dummies: ", data.columns.tolist())
    #str_columns_binary.remove(delete_strings)
    print("'saved' features: ", delete_strings)
    str_columns_binary = [col for col in str_columns_binary if col not in delete_strings]
    print("string-features being removed after: ", len(str_columns_binary))


    # TODO: What are these 'string_features' doing? Can they be removed?
    # TODO: There is no purpose of these string_features anymore
    string_config_file = "config_goce/string_features.pkl"
    if os.path.isfile(string_config_file):
        with open(string_config_file, 'rb') as f:
            string_features = pickle.load(f)
    else:
        string_features = set()

    # Possibility 1, just drop every str
    print("type: ", type(str_columns_binary))
    print("columns: ", str_columns_binary)
    print("bef string_features drop: ", data.shape)
    # data = data.drop(str_columns_binary, axis=1)
    # data = data.drop(str_columns, axis=1)
    string_features.update(str_columns)
    string_features.update(str_columns_binary)
    present_string_features = [str_feat for str_feat in string_features if str_feat in data.columns]
    print("data[present_string_features].isna(): ", data[present_string_features].isna())
    print("data[present_string_features].isna().all(): ", data[present_string_features].isna().all())
    if data[present_string_features].isna().all().all():
        data = data.drop(present_string_features, axis=1)
    else:
        for str_feat in present_string_features:
            print("data[str_feat].isna().all(): ", data[str_feat].isna().all())
        raise Exception("Feature where not all were nan: ", present_string_features)
    # data = data.drop(string_features, axis=1)
    print("string_features: ", string_features)
    print("aft string_features drop: ", data.shape)
    print("Double Check, NaNs left?: ", data.isna().any())

    return data