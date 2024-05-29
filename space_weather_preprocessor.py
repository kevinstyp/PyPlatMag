import pandas as pd
import datetime
import numpy as np
from scipy import interpolate
#from astropy.time import Time

import utils.time_handler as th

def unpack_kp_dst_file_to_df(kp_dst_path, use_cache=True):
    if use_cache: # If file exists, load it
        # Look up in kp_dst path of current year_month




        pass
    else: # If file does not exist, create it
        # Load the data
        pass
    kp_dst_indices = np.genfromtxt(kp_dst_path,
                                   dtype=('U4', 'U3', 'U2', np.float64, np.float64, np.float64, np.int32, np.int32, np.float64),
                                   names=['year', 'dayofyear', 'hour', "By", "Bz", "Sw", "KP", "Dst", "F107"])
    print("kp_dst_indices: ", kp_dst_indices)
    print("kp_dst_indices.shape: ", kp_dst_indices.shape)
    # omniweb_df = pd.DataFrame(columns=["KP_Timestamp", "By", "Bz", "Sw", "KP", "Dst", "F10.7"])

    # kp_dst_df = pd.DataFrame(kp_dst_indices, columns=["year", "month", "day", "By", "Bz", "Sw", "KP", "Dst", "F10.7"])
    kp_df = pd.DataFrame(columns=["KP_Timestamp", "gps_sec", #"By", "Bz", "Sw",
                                  "KP", "Dst", "F10.7"])

    # We could reduce the size of the data by filtering for the relevant dates
    timestamps = [datetime.datetime.strptime(kp_dst[0] + ' ' + kp_dst[1] + ' ' + kp_dst[2], '%Y %j %H') for kp_dst in
                  kp_dst_indices
                  # if int(kp_dst[0]) > int(year_month_specifiers[0][:4]) and int(kp_dst[0]) < int(year_month_specifiers[-1][:4])
                  ]

    #gps_times = list(map(to_gps, timestamps))
    gps_times = th.datetime_to_gps(timestamps).astype(np.float64)
    #gps_times = [float(to_gps(timestamp)) for timestamp in timestamps]
    # np.genfromtxt returns a structured ndarray, this is why we can access the columns by name
    print("timestamps.shape: ", len(timestamps))
    print("gps_times.shape: ", len(gps_times))
    print("kp_dst_indices['By'].shape: ", kp_dst_indices['By'].shape)

    kp_df = kp_df.append(pd.DataFrame({"KP_Timestamp": timestamps,
                                       "gps_sec": gps_times,
                                       #"By": kp_dst_indices['By'], "Bz": kp_dst_indices['Bz'],
                                       #"Sw": kp_dst_indices['Sw'],
                                       "KP": kp_dst_indices['KP'], "Dst": kp_dst_indices['Dst'],
                                       "F10.7": kp_dst_indices['F107']}))

    # Set flag where values are 999.9 or 9999. depending on the type of column because apparently that means the value is missing
    kp_df['Spaceweather_Flag'] = 0.
    # kp_df.loc[kp_df['By'] >= 999., 'Spaceweather_Flag'] = 1. #ok
    # kp_df.loc[kp_df['Bz'] >= 999., 'Spaceweather_Flag'] = 1. #ok
    # kp_df.loc[kp_df['Sw'] >= 9999., 'Spaceweather_Flag'] = 1. #ok
    kp_df.loc[kp_df['KP'] >= 999., 'Spaceweather_Flag'] = 1. #ok
    print("kp_df[Spaceweather_Flag].sum() KP: ", kp_df["Spaceweather_Flag"].sum())
    #kp_df.loc[kp_df['Dst'] >= 999., 'Spaceweather_Flag'] = 1. #ok
    #print("kp_df[Spaceweather_Flag].sum() Dst: ", kp_df["Spaceweather_Flag"].sum())
    #kp_df.loc[kp_df['F10.7'] >= 999., 'Spaceweather_Flag'] = 1. #ok
    #print("kp_df[Spaceweather_Flag].sum() F10.7: ", kp_df["Spaceweather_Flag"].sum())

    return kp_df


def unpack_hp30_file_to_df(hp30_path):
    hp30_array = np.genfromtxt(hp30_path,
                                   dtype=('U4', 'U2', 'U2', 'U4', 'U4', np.float64, np.float64, np.float64, np.float64),
                                   names=['year', 'month', 'day', "hour", "hour_m", "days", "days_m", "Hp30", "Ap", "D_notused"])
    hp30_df = pd.DataFrame(columns=["HP_Timestamp", "gps_sec",  # "By", "Bz", "Sw",
                                  "Hp30", "Ap"])
    # just misuse the 'hour_m' column here as we do not use it anyways
    print("hp30_array['hour']: ", hp30_array['hour'])
    print("float(['hour']): ", hp30_array['hour'].astype(float))
    #print("float(['hour']): ", float(hp30_array['hour']))

    hp30_array['hour_m'] = ((hp30_array['hour'].astype(float) % 1) * 60).astype(np.int16)
    #hp30_array['hour'] = hp30_array['hour'].astype(int).astype(str)
    print("hp30_array.dtype: ", hp30_array.dtype)
    #hp30_array['hour'] = hp30_array['hour'].astype(int)
    print("hp30_array: ", hp30_array)
    print("hp30_array['hour_m']: ", hp30_array['hour_m'])
    print("hp30_array['hour']: ", hp30_array['hour'])
    print("hp30_array.dtype: ", hp30_array.dtype)
    #hp30_array['hour'] = hp30_array['hour'].astype('|S2')
    print("hp30_array['hour']: ", hp30_array['hour'])
    hp30_array = hp30_array.astype([('year', '<U4'), ('month', '<U2'), ('day', '<U2'), ('hour', '<U2'), ('hour_m', '<U2'), ('days', '<f8'), ('days_m', '<f8'), ('Hp30', '<f8'), ('Ap', '<f8')])
    print("hp30_array.dtype: ", hp30_array.dtype)
    #abctest = [print(hp30[0] + ' ' + hp30[1] + ' ' + hp30[2] + ' ' + str(hp30[3])) for hp30 in hp30_array[:10]]
    timestamps = [datetime.datetime.strptime(hp30[0] + ' ' + hp30[1] + ' ' + hp30[2] + ' ' + hp30[3] + ' ' + hp30[4] , '%Y %m %d %H %M') for hp30 in
                  hp30_array
                  ]
    print("timestamps[:10]: ", timestamps[:10])

    #gps_times = list(map(to_gps, timestamps))
    gps_times = th.datetime_to_gps(timestamps).astype(np.float64)

    hp30_df = hp30_df.append(pd.DataFrame({"HP_Timestamp": timestamps, "gps_sec": gps_times,
                                       "Hp30": hp30_array['Hp30'], "Ap": hp30_array['Ap']}))
    print("hp30_df.head(10): ", hp30_df.head(10))

    return hp30_df

def enrich_df_with_kp_data(data, kp_df, with_kp=False):
    print("kp_df.head(5): ", kp_df.head(5))
    print("data['gps_sec']: ", data['gps_sec'])
    print("kp_df['gps_sec']: ", kp_df['gps_sec'])
    print("data.shape: ", data.shape)
    # by_interpolater = interpolate.interp1d(kp_df["gps_sec"], kp_df["By"], kind='linear', fill_value="extrapolate")
    # data["By"] = by_interpolater(data["gps_sec"])
    # bz_interpolater = interpolate.interp1d(kp_df["gps_sec"], kp_df["Bz"], kind='linear', fill_value="extrapolate")
    # data["Bz"] = bz_interpolater(data["gps_sec"])
    # sw_interpolater = interpolate.interp1d(kp_df["gps_sec"], kp_df["Sw"], kind='linear', fill_value="extrapolate")
    # data["Sw"] = sw_interpolater(data["gps_sec"])
    dst_interpolater = interpolate.interp1d(kp_df["gps_sec"], kp_df["Dst"], kind='linear', fill_value="extrapolate")
    data["Dst"] = dst_interpolater(data["gps_sec"])
    f107_interpolater = interpolate.interp1d(kp_df["gps_sec"], kp_df["F10.7"], kind='linear', fill_value="extrapolate")
    data["F10.7"] = f107_interpolater(data["gps_sec"])
    print("dataDst].shape: ", data["Dst"].shape)
    print("data[Dst]: ", data["Dst"])
    # data["By-20m"] = data["By"].rolling('20min', min_periods=1).mean()
    # data["Bz-20m"] = data["Bz"].rolling('20min', min_periods=1).mean()
    # data["Sw-20m"] = data["Sw"].rolling('20min', min_periods=1).mean()
    # print("data[By]: ", data["By"])
    # print("data[By-20m]: ", data["By-20m"])
    # print sum of spaceweather flag
    print("kp_df[Spaceweather_Flag].sum(): ", kp_df["Spaceweather_Flag"].sum())
    print("data.shape: ", data.shape)
    print("kp_df.shape: ", kp_df.shape)
    #data["Spaceweather_Flag"] = kp_df["Spaceweather_Flag"]

    data = data.reset_index(drop=True)
    data = pd.merge_asof(data, kp_df[["KP_Timestamp", "Spaceweather_Flag"]], on=None, left_on="RAW_Timestamp", right_on="KP_Timestamp",  #
                         left_index=False, right_index=False, by=None,
                         left_by=None, right_by=None, suffixes=('_x', '_y'), tolerance=pd.Timedelta("1h"),
                         allow_exact_matches=True,
                         direction='backward')
    data = data.drop(columns=["KP_Timestamp"])

    if with_kp:
        #data = data.set_index("RAW_Timestamp", drop=False)
        data = pd.merge_asof(data, kp_df[["KP_Timestamp", "KP"]], on=None, left_on="RAW_Timestamp", right_on="KP_Timestamp",  #
                                  left_index=False, right_index=False, by=None,
                                  left_by=None, right_by=None, suffixes=('_x', '_y'), tolerance=pd.Timedelta("1h"),
                                  allow_exact_matches=True,
                                  direction='backward')

    data = data.set_index("RAW_Timestamp", drop=False)
    data["Spaceweather_Flag"] = data["Spaceweather_Flag"].rolling('1min', min_periods=1).max()
    print("data[Spaceweather_Flag].sum(): ", data["Spaceweather_Flag"].sum())
    return data

def enrich_df_with_hp_data(data, hp30_df):
    print("hp30_df.head(5): ", hp30_df.head(5))
    print("data.head(5): ", data.head(5))

    data = data.reset_index(drop=True)
    data = pd.merge_asof(data, hp30_df[["HP_Timestamp", "Hp30", "Ap"]], on=None, left_on="RAW_Timestamp", right_on="HP_Timestamp",  #
                         left_index=False, right_index=False, by=None,
                         left_by=None, right_by=None, suffixes=('_x', '_y'), tolerance=pd.Timedelta("30min"),
                         allow_exact_matches=True,
                         direction='backward')
    data = data.set_index("RAW_Timestamp", drop=False)

    print("data.head(5): ", data.head(5))
    print("data['Hp30'].head(5): ", data['Hp30'].head(5))
    print("data['HP_Timestamp'].head(5): ", data['HP_Timestamp'].head(5))

    return data
