
import logging

import data_filters.filter_functions as ff

logger = logging.getLogger(__name__)

def goce_filter(data, magnetic_activity=True, doy=True, training=True, training_columns=[], meta_features=[], y_features=[]):
    # x_all_columns -> training_columns

    ## Magnetic Activity - Filtering
    data = ff.flag_magnetic_activity(data, dst_period=225)

    if magnetic_activity and training:
        logger.info(f"Applying Magnetic Activity-Filtering.")
        data = ff.filter_flag(data, "Magnetic_Activity_Flag")
    else:
        logger.warning("WARNING: No Magnetic-Activity-Filtering, but Flag added.")


    # if not training:
    #     ## This basically ensures during evaluation that the same columns as 'finally' wanted are kept, the others are thrwon away
    #     ## in terms of NaN-Handling
    #     ## Then remaining NaN-values are basically what was taken because of small_nan_share, so they are filled with 0s as was
    #     ## done during training
    #     to_drop = data.drop(meta_features, axis=1, errors='ignore').isna().any()
    #     print(f"to_drop: {to_drop}")
    #     import numpy as np
    #     print(f"np.sum(to_drop): {np.sum(to_drop)}")
    #     ## x_all_columns need to be kept in the data in the end
    #     to_drop[data.columns & training_columns] = False
    #     print(f"to_drop: {to_drop}")
    #     print(f"data: {data}")
    #     data = data.loc[:, ~to_drop]
    #     logger.info(f"after non-training handling data.shape: {data.shape}")

    # TODO: All of this NaN Handling seems odd to me, maybe quickly implement it, then comment it out for testing
    # TODO: This should be rewritten: If there are Nans, the Nan-Handling did not work properly -> Throw an error here
    logger.info(f"Columns containing NaNs: {data.drop(meta_features, axis=1, errors='ignore').columns[data.drop(meta_features, axis=1, errors='ignore').isna().any()]}")
    data = data.drop(data.drop(meta_features, axis=1, errors='ignore').index.difference(data.drop(meta_features, axis=1, errors='ignore').dropna(axis='index').index))
    logger.info(f"Data shape after dropping NaNs: {data.shape}")

    if training:
        logger.info("Removing unavailable Spaceweather Flag")
        data = ff.filter_flag(data, "Spaceweather_Flag")
    else:
        logger.warning("WARNING: No Spaceweather Flag removal, but Flag added.")

    data = ff.flag_outlier(data, y_features, orbit_removal=True)
    if training:
        logger.info(f"Removing outliers from training data")
        data = ff.filter_flag(data, "Outlier_Flag")
    else:
        logger.warning(f"WARNING: No outlier removal, but Flag added.")

    data = ff.flag_samples_by_interpolation_distance(data)
    if training:
        logger.info(f"Removing samples with too large interpolation distance from training data")
        data = ff.filter_flag(data, "Interpolation_Distance_Flag")
    else:
        logger.warning(f"WARNING: No interpolation distance filtering, but Flag added.")


    # TODO: This could be tried to move to 'z_all' instead of dropping
    # Filter out all columns starting with 'str', related to star trackers and thus positional encoding
    data = data.drop(list(data.filter(regex='^str.*')), axis=1, errors='ignore')

    # TODO: Can be dropped, should be in read_files
    if doy:
        data["DOY"] = data["RAW_Timestamp"].dt.day_of_year.astype("float64")

    return data
