import datetime
import logging
from os.path import isfile

import numpy as np
import pandas as pd
from scipy import interpolate

import utils.load_omni_data as omni_loader
import utils.time_handler as th
import utils.load_Hp30_data as hp30_loader

logger = logging.getLogger(__name__)


def unpack_kp_dst_file_to_df(auxiliary_data_path, year_month, use_cache=True):

    kp_dst_f107_path = omni_loader.get_output_filename(year=year_month[:4], data_spec="hourly")
    # If cache shall not be used or file does not exist, create it
    if not use_cache or not isfile(kp_dst_f107_path):
        # Load the data from omni
        omni_loader.fetch_omni_data(year=year_month[:4], outdir=auxiliary_data_path, data_spec="hourly")
    else:
        logger.info(f"Loading data from cache: {kp_dst_f107_path}")

    kp_dst_array = np.genfromtxt(kp_dst_f107_path,
                                   dtype=('U4', 'U3', 'U2', np.int32, np.int32, np.float64),
                                   names=['year', 'dayofyear', 'hour', "KP", "Dst", "F107"])
    logger.debug(f"kp_dst_array: {kp_dst_array}")
    logger.debug(f"kp_dst_array.shape: {kp_dst_array.shape}")

    kp_df = pd.DataFrame(columns=["KP_Timestamp", "gps_sec", "KP", "Dst", "F10.7"])

    timestamps = [datetime.datetime.strptime(kp_dst[0] + ' ' + kp_dst[1] + ' ' + kp_dst[2], '%Y %j %H') for kp_dst in kp_dst_array]
    gps_times = th.datetime_to_gps(timestamps).astype(np.float64)
    logger.debug(f"len timestamps: {len(timestamps)}")
    logger.debug(f"len gps_times: {len(gps_times)}")

    # np.genfromtxt returns a structured ndarray, so we can access the columns by name
    kp_df = kp_df.append(pd.DataFrame({"KP_Timestamp": timestamps, "gps_sec": gps_times, "KP": kp_dst_array['KP'],
                                       "Dst": kp_dst_array['Dst'], "F10.7": kp_dst_array['F107']}))

    # Set Spaceweather_Flag where values are 999.9 or 9999. depending on the type of column because that means the value is
    # missing as per definition on the OMNIWeb website: https://omniweb.gsfc.nasa.gov/html/ow_data.html
    kp_df['Spaceweather_Flag'] = 0.
    kp_df.loc[kp_df['KP'] >= 99., 'Spaceweather_Flag'] = 1.
    logger.info(f"Spaceweather Flag after KP: {kp_df['Spaceweather_Flag'].sum()}")
    kp_df.loc[kp_df['Dst'] >= 99999., 'Spaceweather_Flag'] = 1.
    logger.info(f"Spaceweather Flag after Dst: {kp_df['Spaceweather_Flag'].sum()}")
    kp_df.loc[kp_df['F10.7'] >= 999., 'Spaceweather_Flag'] = 1.
    logger.info(f"Spaceweather Flag after F10.7: {kp_df['Spaceweather_Flag'].sum()}")

    # set types of KP to int64 and Dst to float64
    kp_df["KP"] = kp_df["KP"].astype(np.int64)
    kp_df["Dst"] = kp_df["Dst"].astype(np.float64)
    return kp_df


def unpack_hp30_file_to_df(auxiliary_data_path, year_month, use_cache=True):
    hp30_path = hp30_loader.get_output_filename(year=year_month[:4], outdir=auxiliary_data_path)
    if not use_cache or not isfile(hp30_path):
        hp30_loader.fetch_Hp30_data(year=year_month[:4], outdir=auxiliary_data_path)
    hp30_df = pd.read_hdf(hp30_path, "df")
    hp30_df["gps_sec"] = th.datetime_to_gps(hp30_df["HP_Timestamp"]).astype(np.float64)
    logger.debug(f"hp30_df.head(10): {hp30_df.head(10)}")
    return hp30_df


def enrich_df_with_kp_data(data, kp_df, with_kp=False):
    logger.debug(f"kp_df.head(5): {kp_df.head(5)}")
    logger.debug(f"data.shape: {data.shape}")
    dst_interpolater = interpolate.interp1d(kp_df["gps_sec"], kp_df["Dst"], kind='linear', fill_value="extrapolate")
    data["Dst"] = dst_interpolater(data["gps_sec"])
    f107_interpolater = interpolate.interp1d(kp_df["gps_sec"], kp_df["F10.7"], kind='linear', fill_value="extrapolate")
    data["F10.7"] = f107_interpolater(data["gps_sec"])

    logger.info(f"Spaceweather Flag after Dst: {kp_df['Spaceweather_Flag'].sum()}")
    # TODO: This does not look too well here!?
    data = data.reset_index(drop=True)
    data = pd.merge_asof(data, kp_df[["KP_Timestamp", "Spaceweather_Flag"]], on=None, left_on="RAW_Timestamp", right_on="KP_Timestamp",
                         left_index=False, right_index=False, by=None,
                         left_by=None, right_by=None, suffixes=('_x', '_y'), tolerance=pd.Timedelta("1h"),
                         allow_exact_matches=True,
                         direction='backward')
    data = data.drop(columns=["KP_Timestamp"])

    if with_kp:
        #data = data.set_index("RAW_Timestamp", drop=False)
        data = pd.merge_asof(data, kp_df[["KP_Timestamp", "KP"]], on=None, left_on="RAW_Timestamp", right_on="KP_Timestamp",
                                  left_index=False, right_index=False, by=None,
                                  left_by=None, right_by=None, suffixes=('_x', '_y'), tolerance=pd.Timedelta("1h"),
                                  allow_exact_matches=True,
                                  direction='backward')

    data = data.set_index("RAW_Timestamp", drop=False)
    data["Spaceweather_Flag"] = data["Spaceweather_Flag"].rolling('1min', min_periods=1).max()
    logger.info(f"Spaceweather Flag after KP merge: {data['Spaceweather_Flag'].sum()}")
    return data


def enrich_df_with_hp_data(data, hp30_df):
    logger.debug(f"hp30_df.head(5): {hp30_df.head(5)}")
    data = data.reset_index(drop=True)
    data = pd.merge_asof(data, hp30_df[["HP_Timestamp", "Hp30"]], on=None, left_on="RAW_Timestamp", right_on="HP_Timestamp",
                         left_index=False, right_index=False, by=None,
                         left_by=None, right_by=None, suffixes=('_x', '_y'), tolerance=pd.Timedelta("30min"),
                         allow_exact_matches=True,
                         direction='backward')
    data = data.set_index("RAW_Timestamp", drop=False)
    logger.debug(f"hp30_df.head(5): {hp30_df.head(5)}")
    return data
