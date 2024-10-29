import logging

import numpy as np

logger = logging.getLogger(__name__)


def filter_flag(data, flag):
    data = data.reset_index(drop=True)
    logger.debug(f"Filtering data with flag {flag}")
    logger.info(f"Found {np.sum(data[flag] >= 1)} samples with {flag} Flag.")
    logger.debug(f"Timestamps of this data: {data.index.where(data[flag] >= 1)}")

    data = data.drop(data[data[flag] >= 1].index, axis='index')
    data = data.reset_index(drop=True)
    logger.info(f"Data shape after filtering for {flag} flag: {data.shape}")
    return data


def flag_magnetic_activity(data, dst_period=3600):
    logger.info(f"Data before first filtering: {data.shape}")
    data['Magnetic_Activity_Flag'] = 0

    # le = Lower-equals = "<=", gt = greater-than gets a 1-Flag
    indices = data["Hp30"].gt(2.)
    data.loc[indices, 'Magnetic_Activity_Flag'] = 1
    # Count KP_Dst_Flag in z_all
    logger.info(f"Number of samples identified for Hp30 filtering: {data['Magnetic_Activity_Flag'].sum()}")

    # Compare Dst with an hour before
    indices = data["Dst"].diff(periods=dst_period).fillna(0).abs().gt(4)
    data.loc[indices, 'Magnetic_Activity_Flag'] = 1
    logger.info(f"Number of samples identified for Dst change filtering: {data['Magnetic_Activity_Flag'].sum()}")
    return data


def flag_outlier(data, y_features, orbit_removal=True, mag_columns=['FGM1_X', 'FGM1_Y', 'FGM1_Z']):
    residual = data[mag_columns].values - data[y_features].values
    std = np.std(residual, axis=0)
    mean = np.mean(residual, axis=0)

    set_of_indices = set()
    for i in range(residual.shape[1]):
        std_factor = 3.
        set_of_indices.update(np.where(residual[:, i] < (mean[i] - std_factor * std[i]))[0])
        set_of_indices.update(np.where(residual[:, i] > (mean[i] + std_factor * std[i]))[0])
    logger.debug(f"Number of outliers: {len(set_of_indices)}")

    data['Outlier_Flag'] = 0
    if orbit_removal:  # Remove the entire orbit of data, if it contains at least 1 outlier
        # Calculate set of indices of outlier containing orbits, then set the Outlier_Flag to 1 for all data points in these orbits
        all_orbit_indices = data[data["ORB_OrbitNo"].isin(set(data.iloc[list(set_of_indices)]["ORB_OrbitNo"]))].index
        data.loc[all_orbit_indices, 'Outlier_Flag'] = 1
    else:  # Remove only the data points that themselves are outliers
        data.loc[list(set_of_indices), 'Outlier_Flag'] = 1
    return data


def flag_samples_by_interpolation_distance(data):
    data['Interpolation_Distance_Flag'] = 0
    data.loc[data['fgm_gapsize'] > 17, 'Interpolation_Distance_Flag'] = 1
    return data
