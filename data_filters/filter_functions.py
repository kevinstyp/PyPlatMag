import logging

import numpy as np

logger = logging.getLogger(__name__)


def filter_flag(data, flag):
    data = data.reset_index(drop=True)
    logger.debug(f"Filtering data with flag {flag}")
    logger.debug(f"data with flag >= 1: {data[flag] >= 1}")
    logger.debug(f"Timestamps of this data: {data.index.where(data[flag] >= 1)}")

    #data = data.drop((data[flag] >= 1).index, axis='index')
    data = data.drop(data[data[flag] >= 1].index, axis='index')
    data = data.reset_index(drop=True)
    logger.info(f"Data shape after filtering for {flag} flag: {data.shape}")
    return data

# TODO: Deprecated, should be removed
def filter_flag_xyz(x_all, y_all, z_all, flag):
    x_all = x_all.reset_index(drop=True)
    y_all = y_all.reset_index(drop=True)
    z_all = z_all.reset_index(drop=True)
    #indices = z_all[flag].eq(0)
    #indices = z_all[flag].eq(1)
    print("z_all[flag] >= 1: ", z_all[flag] >= 1)
    print("z_all.index[z_all[flag] >= 1]: ", z_all.index.where(z_all[flag] >= 1))
    #print("z_all.index[z_all[flag] >= 1].tolist(): ", list(z_all.index.where(z_all[flag] >= 1)))

    indices = z_all.index[z_all[flag] >= 1].tolist()
    print("number of found samples with flag: ", len(indices), flag)
    x_all = x_all.drop(indices, axis='index')#x_all.loc[indices]
    y_all = y_all.drop(indices, axis='index')#y_all.loc[indices]
    z_all = z_all.drop(indices, axis='index')#z_all.loc[indices]
    print("x_all.shape: ", x_all.shape)
    print("z_all.shape: ", z_all.shape)
    #print("z-indices: ", indices)
    #print("indices[indices == False]: ", indices[indices == False])
    #x_all.compute()

    x_all = x_all.reset_index(drop=True)
    y_all = y_all.reset_index(drop=True)
    z_all = z_all.reset_index(drop=True)

    #x_all.compute()

    return (x_all, y_all, z_all)

def flag_magnetic_activity(data, dst_period=3600):
    print("data bef KP-Dst Filtering: ", data.shape)
    # le = Lower-equals = "<="
    # gt = greater-than gets a 1-Flag
    #z_all['KP_Dst_Flag'] = 0
    data['Magnetic_Activity_Flag'] = 0

    indices = data["Hp30"].gt(2.)
    data.loc[indices, 'Magnetic_Activity_Flag'] = 1
    # Count KP_Dst_Flag in z_all
    print("data[Magnetic_Activity_Flag].sum() after KP-Filtering: ", data["Magnetic_Activity_Flag"].sum())

    # Compare Dst with an hour before
    indices = data["Dst"].diff(periods=dst_period).fillna(0).abs().gt(4)
    print("indices.sum() after new Dst-Filtering: ", indices.sum())
    data.loc[indices, 'Magnetic_Activity_Flag'] = 1
    print("data[Magnetic_Activity_Flag].sum() after new Dst-Filtering: ", data["Magnetic_Activity_Flag"].sum())

    #print("x_all aft KP-Dst Filtering: ", x_all.shape)
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
    if orbit_removal: # Remove the entire orbit of data, if it contains at least 1 outlier
        # Calculate set of indices of outlier containing orbits, then set the Outlier_Flag to 1 for all data points in these orbits
        all_orbit_indices = data[data["ORB_OrbitNo"].isin(set(data.iloc[list(set_of_indices)]["ORB_OrbitNo"]))].index
        data.loc[all_orbit_indices, 'Outlier_Flag'] = 1
    else: # Remove only the data points that themselves are outliers
        data.loc[list(set_of_indices), 'Outlier_Flag'] = 1
    return data


def filter_samples_by_interpolation_distance(data):
    data['Interp_Distance_Flag'] = 0
    data.loc[data['fgm_gapsize'] > 17, 'Interp_Distance_Flag'] = 1
    return data

