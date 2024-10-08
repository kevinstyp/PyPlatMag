
import datetime
import logging
from os.path import isfile

import numpy as np
import pandas as pd
import ppigrf.ppigrf
import pyamps
from scipy import interpolate

import utils.load_omni_data as omni_loader
import utils.time_handler as th
from lib.dipole import Dipole
from utils import quaternion_util as qu

logger = logging.getLogger(__name__)


def unpack_amps_params_file_to_df(auxiliary_data_path, year_month, use_cache=True):
    amps_params_path = omni_loader.get_output_filename(year=year_month[:4], data_spec="minute")
    # If cache shall not be used or file does not exist, create it
    if not use_cache or not isfile(amps_params_path):
        # Load the data from omni
        omni_loader.fetch_omni_data(year=year_month[:4], outdir=auxiliary_data_path, data_spec="minute")
    else:
        logger.info(f"Loading data from cache: {amps_params_path}")

    amps_params_array = np.genfromtxt(amps_params_path,
                                   dtype=('U4', 'U3', 'U2', 'U2', np.float64, np.float64, np.float64),
                                   names=['year', 'dayofyear', 'hour', 'minute', "By", "Bz", "Sw"])
    logger.debug(f"amps_params_array: {amps_params_array}")
    logger.debug(f"amps_params_array.shape: {amps_params_array.shape}")
    amps_params_df = pd.DataFrame(columns=["Amps_Timestamp", "gps_sec", "By", "Bz", "Sw"])

    timestamps = [datetime.datetime.strptime(amps_param[0] + ' ' + amps_param[1] + ' ' + amps_param[2] + ' ' + amps_param[3], '%Y %j %H %M') for
                  amps_param in amps_params_array]
    gps_times = th.datetime_to_gps(timestamps)

    amps_params_df = amps_params_df.append(pd.DataFrame({"Amps_Timestamp": timestamps, "gps_sec": gps_times,
                                       "By": amps_params_array['By'], "Bz": amps_params_array['Bz'], "Sw": amps_params_array['Sw'],
                                       }))
    return amps_params_df

def enrich_df_with_amps_params_data(data, amps_params_df):
    # Set flag where values are 999.9 or 9999. depending on the type of column because that means the value is missing
    amps_params_df['Spaceweather_Flag'] = 0.
    amps_params_df.loc[amps_params_df['By'] >= 999., 'Spaceweather_Flag'] = 1.
    amps_params_df.loc[amps_params_df['Bz'] >= 999., 'Spaceweather_Flag'] = 1.
    amps_params_df.loc[amps_params_df['Sw'] >= 9999., 'Spaceweather_Flag'] = 1.

    # Interpolate the values to the data
    by_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["By"], kind='linear', fill_value="extrapolate")
    data["By"] = by_interpolater(data["gps_sec"])
    bz_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["Bz"], kind='linear', fill_value="extrapolate")
    data["Bz"] = bz_interpolater(data["gps_sec"])
    sw_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["Sw"], kind='linear', fill_value="extrapolate")
    data["Sw"] = sw_interpolater(data["gps_sec"])

    # now set to null in reference dataframe for rolling window to ignore these values
    amps_params_df['By'] = np.where(amps_params_df.By >= 999., np.nan, amps_params_df.By)
    amps_params_df['Bz'] = np.where(amps_params_df.Bz >= 999., np.nan, amps_params_df.Bz)
    amps_params_df['Sw'] = np.where(amps_params_df.Sw >= 9999., np.nan, amps_params_df.Sw)
    # count nans in parameters -> missing values
    logger.info(f"Missing By in amps_params_df: {amps_params_df['By'].isnull().sum()}")
    logger.info(f"Missing Bz in amps_params_df: {amps_params_df['Bz'].isnull().sum()}")
    logger.info(f"Missing Sw in amps_params_df: {amps_params_df['Sw'].isnull().sum()}")
    # Fill Nans / missing values with linear interpolation
    amps_params_df['By'] = amps_params_df['By'].interpolate(method='linear')
    amps_params_df['Bz'] = amps_params_df['Bz'].interpolate(method='linear')
    amps_params_df['Sw'] = amps_params_df['Sw'].interpolate(method='linear')
    logger.debug(f"Missing By after interpolation in amps_params_df: {amps_params_df['By'].isnull().sum()}")
    logger.debug(f"Missing Bz after interpolation in amps_params_df: {amps_params_df['Bz'].isnull().sum()}")
    logger.debug(f"Missing Sw after interpolation in amps_params_df: {amps_params_df['Sw'].isnull().sum()}")
    logger.debug(f"amps_params_df.head(5): {amps_params_df.head(5)}")

    amps_params_df = amps_params_df.set_index("Amps_Timestamp", drop=False)
    # Add additional 20min rolling mean columns to the dataframe for the parameters
    amps_params_df["By-20m"] = amps_params_df["By"].rolling('20min', min_periods=10).mean()
    amps_params_df["Bz-20m"] = amps_params_df["Bz"].rolling('20min', min_periods=10).mean()
    amps_params_df["Sw-20m"] = amps_params_df["Sw"].rolling('20min', min_periods=10).mean()
    amps_params_df.loc[amps_params_df['By-20m'].isnull(), 'Spaceweather_Flag'] = 1.
    amps_params_df.loc[amps_params_df['Bz-20m'].isnull(), 'Spaceweather_Flag'] = 1.
    amps_params_df.loc[amps_params_df['Sw-20m'].isnull(), 'Spaceweather_Flag'] = 1.

    # TODO: Can this happen after values have been filled? It's exactly 9
    logger.debug(f"Sum of nulls in By-20m before filling: {amps_params_df['By-20m'].isnull().sum()}")
    logger.debug(f"Sum of nulls in Sw-20m before filling: {amps_params_df['Sw-20m'].isnull().sum()}")
    by20m_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["By-20m"], kind='linear', fill_value="extrapolate")
    data["By-20m"] = by20m_interpolater(data["gps_sec"])
    bz20m_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["Bz-20m"], kind='linear', fill_value="extrapolate")
    data["Bz-20m"] = bz20m_interpolater(data["gps_sec"])
    sw20m_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["Sw-20m"], kind='linear', fill_value="extrapolate")
    data["Sw-20m"] = sw20m_interpolater(data["gps_sec"])

    # Find Nan values in By-20m, Bz-20m, Sw-20m and set Spaceweather_Flag to 1. The rest got meaningfully interpolated
    data.loc[data['By-20m'].isnull(), 'Spaceweather_Flag'] = 1.
    data.loc[data['Bz-20m'].isnull(), 'Spaceweather_Flag'] = 1.
    data.loc[data['Sw-20m'].isnull(), 'Spaceweather_Flag'] = 1.

    # Sanity check for Spaceweather flags set to 1
    logger.debug(f"data['Spaceweather_Flag'].sum(): {data['Spaceweather_Flag'].sum()}")
    logger.debug(f"data.loc[data['Spaceweather_Flag'] == 1., 'By']: {data.loc[data['Spaceweather_Flag'] == 1., 'By']}")
    logger.debug(f"data.loc[data['Spaceweather_Flag'] == 1., 'Bz']: {data.loc[data['Spaceweather_Flag'] == 1., 'Bz']}")
    logger.debug(f"data.loc[data['Spaceweather_Flag'] == 1., 'Sw']: {data.loc[data['Spaceweather_Flag'] == 1., 'Sw']}")
    # Log max values of the BY, BZ, SW columns
    logger.debug(f"data['By-20m'].max() raw: {data['By-20m'].max()}")
    logger.debug(f"data['Bz-20m'].max() raw: {data['Bz-20m'].max()}")
    logger.debug(f"data['Sw-20m'].max() raw: {data['Sw-20m'].max()}")
    # Log max values with spaceweather flag set to 0
    logger.debug(f"data.loc[data['Spaceweather_Flag'] == 0., 'By-20m'].max(): {data.loc[data['Spaceweather_Flag'] == 0., 'By-20m'].max()}")
    logger.debug(f"data.loc[data['Spaceweather_Flag'] == 0., 'Bz-20m'].max(): {data.loc[data['Spaceweather_Flag'] == 0., 'Bz-20m'].max()}")
    logger.debug(f"data.loc[data['Spaceweather_Flag'] == 0., 'Sw-20m'].max(): {data.loc[data['Spaceweather_Flag'] == 0., 'Sw-20m'].max()}")

    return data

def enrich_df_with_amps_data(data, quaternion_columns=["q1_fgm12nec", "q2_fgm12nec", "q3_fgm12nec", "q4_fgm12nec"]):
    # TODO: The performance of this method could be increased by only calculating the AMPS model for the QDLat restriction
    ## applied at the end of the method. This would reduce the number of calculations needed.
    # We have merged all the needed data for the AMPS model now!
    # Assumption: The year of the processed data is the same for the current chunk (we compute in monthly chunks, so this
    #  should be true)
    epoch_year = data.index[0].year
    epoch_month = data.index[0].month
    logger.info(f"Enriching with AMPS data for: {epoch_year}, {epoch_month}")
    dipole_year_month = Dipole(epoch_year + (epoch_month-1.)/12.)

    logger.debug(f"epoch_year_month: {epoch_year + (epoch_month-1.)/12.}")
    #dipole_year_month = Dipole(epoch_year)
    data["tilt"] = dipole_year_month.tilt(data.index)
    logger.debug(f"tilt_values: {data['tilt']}")
    # Convert the lat/lon/height from geocentric to geodetic with geoc2geod function from the ppigrf package
    # '90 - latitudes': Converts the latitudes to the expected Colatitudes, defined from 0 to 180 degrees
    gdlat, height, _, _ = ppigrf.ppigrf.geoc2geod(90 - data["RAW_Latitude"], data["r.trs"], np.zeros_like(data["RAW_Latitude"]),
                                                  np.zeros_like(data["RAW_Latitude"]))
    logger.debug(f"gdlat: {gdlat}")
    logger.debug(f"height: {height}")
    logger.debug(f"glon: {data['RAW_Longitude']}")
    logger.debug(f"data[RAW_Timestamp].values: {data['RAW_Timestamp'].values}")
    logger.debug(f"data[Sw-20m].values: {data['Sw-20m'].values}")
    logger.debug(f"data[Sw-20m].values: {list(data['Sw-20m'].values[:40])}")
    logger.debug(f"data[By-20m].values: {data['By-20m'].values}")
    logger.debug(f"data[Bz-20m].values: {data['Bz-20m'].values}")
    logger.debug(f"data[tilt].values: {data['tilt'].values}")
    logger.debug(f"data[F10.7].values: {data['F10.7'].values}")

    # get_B_space() claims to use glat and glon as 'geographic' coordinates,
    # but the function actually uses geodetic coordinates (according to KML).
    # This is also supported by the height parameter being requested in geodetic coordinates.
    B_e, B_n, B_u = pyamps.get_B_space(glat=gdlat, glon=data["RAW_Longitude"].values, height=height,
                                       time=data["RAW_Timestamp"].values,
                                       v=data["Sw-20m"].values, By=data["By-20m"].values, Bz=data["Bz-20m"].values,
                                       tilt=data["tilt"].values, f107=data["F10.7"].values)
    logger.debug(f"b_space: {B_e}, {B_n}, {B_u}")
    # Convert the geodetic return values of the get_B_space() function to geocentric coordinates
    _, _, B_th, B_r = ppigrf.ppigrf.geod2geoc(gdlat, height, B_n, B_u)
    logger.debug(f"B_th, B_r: {B_th}, {B_r}")
    # Invert the sign of the B_u component, because then we have NEC values
    B_c = -B_r
    # The theta return needs to be inverted as well
    B_n = -B_th
    # B_n, B_e, B_c are the N, E, C components from NEC
    amps_nec = np.column_stack([B_n, B_e, B_c])
    logger.debug(f"amps_nec: {amps_nec}")
    # Now we can compute the magnetic field vector in the spacecraft frame by inverting the FGM_to_NEC quaternions and applying them
    # in the opposite direction to the B_e, B_n, B_c components
    quat = np.column_stack([data[quaternion_columns[0]].values, data[quaternion_columns[1]].values,
                            data[quaternion_columns[2]].values, data[quaternion_columns[3]].values])
    logger.debug(f"quat: {quat}")
    # Quat can contain nans
    valid_mask = ~np.isnan(quat).any(axis=1)
    quat_valid = quat[valid_mask]
    amps_nec_valid = amps_nec[valid_mask]
    amps_mag_valid = qu.rotate_nec2mag(quat_valid, amps_nec_valid)
    amps_mag = np.full_like(amps_nec, np.nan)
    amps_mag[valid_mask] = amps_mag_valid

    data["amps_b_mag_x"] = amps_mag[:, 0]
    data["amps_b_mag_y"] = amps_mag[:, 1]
    data["amps_b_mag_z"] = amps_mag[:, 2]
    data["amps_b_nec_x"] = amps_nec[:, 0]
    data["amps_b_nec_y"] = amps_nec[:, 1]
    data["amps_b_nec_z"] = amps_nec[:, 2]
    # now set the respective rows to zero for latitudes below 40 / above -40 degrees (the model is not valid for these latitudes)
    data.loc[(data["APEX_QD_LAT"] > -40) & (data["APEX_QD_LAT"] < 40),
             ["amps_b_mag_x", "amps_b_mag_y", "amps_b_mag_z", "amps_b_nec_x", "amps_b_nec_y", "amps_b_nec_z"]] = 0.
    return data
