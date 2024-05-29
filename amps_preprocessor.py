
import logging

import numpy as np
import pandas as pd
import ppigrf.ppigrf
import pyamps
from pyquaternion import Quaternion
from scipy import interpolate

import utils.time_handler as th
# import mToolsQuaternion_py3 as mtq
from dipole import Dipole

logger = logging.getLogger(__name__)

def unpack_amps_params_file_to_df(amps_params_path, year_month_specifier):
    amps_params_numpy = np.genfromtxt(amps_params_path,
                                   dtype=('U4', 'U3', 'U2', 'U2', np.float32, np.float32, np.float32),
                                   names=['year', 'dayofyear', 'hour', 'minute', "By", "Bz", "Sw"])
    print("amps_params_numpy: ", amps_params_numpy)
    print("amps_params_numpy.shape: ", amps_params_numpy.shape)

    mask = (amps_params_numpy['year'] == year_month_specifier[:4])
    amps_params_numpy = amps_params_numpy[mask]

    print("amps_params_numpy: ", amps_params_numpy)
    print("amps_params_numpy.shape: ", amps_params_numpy.shape)

    amps_params_df = pd.DataFrame(columns=["Amps_Timestamp", "gps_sec", "By", "Bz", "Sw"])
    # timestamps = [datetime.datetime.strptime(kp_dst[0] + ' ' + kp_dst[1] + ' ' + kp_dst[2] + ' ' + kp_dst[3], '%Y %j %H %M') for kp_dst in
    #               amps_params_numpy
    #               # if int(kp_dst[0]) > int(year_month_specifiers[0][:4]) and int(kp_dst[0]) < int(year_month_specifiers[-1][:4])
    #               ]
    # timestamps = [datetime.datetime.strptime("{year} {dayofyear} {hour} {minute}".format(**kp_dst), '%Y %j %H %M') for kp_dst in
    #               amps_params_numpy]
    datetimes_df = pd.DataFrame(
        {"year": amps_params_numpy['year'], "dayofyear":  amps_params_numpy['dayofyear'], "hour": amps_params_numpy['hour'],
         "minute": amps_params_numpy['minute']}
    )
    print("datetimes_df: ", datetimes_df)
    print("datetimes_df.head(5): ", datetimes_df.head(5))
    print("datetimes_df.tail(5): ", datetimes_df.tail(5))
    print("datetimes_df['year']: ", datetimes_df['year'])
    datetimes_df['year'] = pd.to_numeric(datetimes_df['year'])
    datetimes_df['dayofyear'] = pd.to_numeric(datetimes_df['dayofyear'])
    datetimes_df['hour'] = pd.to_numeric(datetimes_df['hour'])
    datetimes_df['minute'] = pd.to_numeric(datetimes_df['minute'])
    print("datetimes_df['year']: ", datetimes_df['year'])

    #print("pd.Timestamp(datetimes_df['year']): ", pd.to_datetime(datetimes_df['year'], format="%Y"))
    # # datetimes_df_2 = pd.to_datetime(datetimes_df)
    # datetimes_df_2 = pd.to_datetime(datetimes_df['dayofyear'], unit='D', origin=pd.to_datetime(datetimes_df['year'], format="%Y"))
    ## Add them together by shifting them through multitplication, so that format in line below can be applied
    datetimes_df['combined'] = datetimes_df['year'] * 10000000 + datetimes_df['dayofyear'] * 10000 + \
                               datetimes_df['hour'] * 100 + datetimes_df['minute']
    print("datetimes_df['combined']: ", datetimes_df['combined'])
    timestamps = pd.to_datetime(datetimes_df['combined'], format='%Y%j%H%M')
    print("timestamps: ", timestamps)




    # timestamps = np.datetime64(amps_params_numpy['year'] + '-' + amps_params_numpy['dayofyear'] + 'T' + amps_params_numpy['hour'] + ':' +
    #     amps_params_numpy['minute'], 'm')
    #gps_times = [float(to_gps(timestamp)) for timestamp in timestamps]
    #print("first gps_times: ", gps_times[:10])

    #gps_times = list(map(to_gps, timestamps))
    gps_times = th.datetime_to_gps(timestamps).astype(np.float64)
    print("second gps_times: ", gps_times[:10])
    datetimes_df['gps_sec'] = gps_times
    pd.set_option("display.precision", 15)
    print("datetimes_df.head(10): ", datetimes_df.head(10))

    print("timestamps.shape: ", len(timestamps))
    print("gps_times.shape: ", len(gps_times))
    print("amps_params_numpy['By'].shape: ", amps_params_numpy['By'].shape)

    amps_params_df = amps_params_df.append(pd.DataFrame({"Amps_Timestamp": timestamps,
                                       "gps_sec": gps_times,
                                       "By": amps_params_numpy['By'], "Bz": amps_params_numpy['Bz'],
                                       "Sw": amps_params_numpy['Sw'],
                                       }))


    return amps_params_df

def enrich_df_with_amps_params_data(data, amps_params_df):
    # Set flag where values are 999.9 or 9999. depending on the type of column because apparently that means the value is missing
    amps_params_df['Spaceweather_Flag'] = 0.
    amps_params_df.loc[amps_params_df['By'] >= 999., 'Spaceweather_Flag'] = 1.  # ok
    amps_params_df.loc[amps_params_df['Bz'] >= 999., 'Spaceweather_Flag'] = 1.  # ok
    amps_params_df.loc[amps_params_df['Sw'] >= 9999., 'Spaceweather_Flag'] = 1.  # ok
    #amps_params_df.loc[amps_params_df['KP'] >= 999., 'Spaceweather_Flag'] = 1.  # ok
    #amps_params_df.loc[amps_params_df['Dst'] >= 999., 'Spaceweather_Flag'] = 1.  # ok
    #amps_params_df.loc[amps_params_df['F10.7'] >= 999., 'Spaceweather_Flag'] = 1.  # ok






    by_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["By"], kind='linear', fill_value="extrapolate")
    data["By"] = by_interpolater(data["gps_sec"])
    bz_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["Bz"], kind='linear', fill_value="extrapolate")
    data["Bz"] = bz_interpolater(data["gps_sec"])
    sw_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["Sw"], kind='linear', fill_value="extrapolate")
    data["Sw"] = sw_interpolater(data["gps_sec"])
    # dst_interpolater = interpolate.interp1d(kp_df["gps_sec"], kp_df["Dst"], kind='linear', fill_value="extrapolate")
    # data["Dst"] = dst_interpolater(data["gps_sec"])
    # f107_interpolater = interpolate.interp1d(kp_df["gps_sec"], kp_df["F10.7"], kind='linear', fill_value="extrapolate")
    # data["F10.7"] = f107_interpolater(data["gps_sec"])

    # now set to null in reference dataframe for rolling window to ignore these values
    amps_params_df['By'] = np.where(amps_params_df.By >= 999., np.nan, amps_params_df.By)
    amps_params_df['Bz'] = np.where(amps_params_df.Bz >= 999., np.nan, amps_params_df.Bz)
    amps_params_df['Sw'] = np.where(amps_params_df.Sw >= 9999., np.nan, amps_params_df.Sw)
    # count nans in By:
    print("amps_params_df['By'].isnull().sum(): ", amps_params_df['By'].isnull().sum())
    print("amps_params_df['Bz'].isnull().sum(): ", amps_params_df['Bz'].isnull().sum())
    print("amps_params_df['Sw'].isnull().sum(): ", amps_params_df['Sw'].isnull().sum())
    # Fill Nans in amps_params_df['By'] with linear interpolation
    amps_params_df['By'] = amps_params_df['By'].interpolate(method='linear') #,limit_direction='both')
    amps_params_df['Bz'] = amps_params_df['Bz'].interpolate(method='linear') #,limit_direction='both')
    amps_params_df['Sw'] = amps_params_df['Sw'].interpolate(method='linear') #,limit_direction='both')
    print("amps_params_df['By'].isnull().sum(): ", amps_params_df['By'].isnull().sum())
    print("amps_params_df['Bz'].isnull().sum(): ", amps_params_df['Bz'].isnull().sum())
    print("amps_params_df['Sw'].isnull().sum(): ", amps_params_df['Sw'].isnull().sum())


    print("amps_params_df.head(5): ", amps_params_df.head(5))

    amps_params_df = amps_params_df.set_index("Amps_Timestamp", drop=False)

    amps_params_df["By-20m"] = amps_params_df["By"].rolling('20min', min_periods=10).mean()
    amps_params_df["Bz-20m"] = amps_params_df["Bz"].rolling('20min', min_periods=10).mean()
    amps_params_df["Sw-20m"] = amps_params_df["Sw"].rolling('20min', min_periods=10).mean()
    amps_params_df.loc[amps_params_df['By-20m'].isnull(), 'Spaceweather_Flag'] = 1.
    amps_params_df.loc[amps_params_df['Bz-20m'].isnull(), 'Spaceweather_Flag'] = 1.
    amps_params_df.loc[amps_params_df['Sw-20m'].isnull(), 'Spaceweather_Flag'] = 1.

    print("before amps_params_df['By-20m'].isnull().sum(): ", amps_params_df['By-20m'].isnull().sum())
    print("before amps_params_df['Sw-20m'].isnull().sum(): ", amps_params_df['Sw-20m'].isnull().sum())

    by20m_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["By-20m"], kind='linear', fill_value="extrapolate")
    data["By-20m"] = by20m_interpolater(data["gps_sec"])
    bz20m_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["Bz-20m"], kind='linear', fill_value="extrapolate")
    data["Bz-20m"] = bz20m_interpolater(data["gps_sec"])
    sw20m_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["Sw-20m"], kind='linear', fill_value="extrapolate")
    data["Sw-20m"] = sw20m_interpolater(data["gps_sec"])




    print("after amps_params_df['By-20m'].isnull().sum(): ", amps_params_df['By-20m'].isnull().sum())
    print("after data['By-20m'].isnull().sum(): ", data['By-20m'].isnull().sum())
    print("after amps_params_df['Sw-20m'].isnull().sum(): ", amps_params_df['Sw-20m'].isnull().sum())
    print("after data['Sw-20m'].isnull().sum(): ", data['Sw-20m'].isnull().sum())

    # data["By-20m"] = amps_params_df["By"].rolling('20min', min_periods=10).mean()
    # data["Bz-20m"] = amps_params_df["Bz"].rolling('20min', min_periods=10).mean()
    # data["Sw-20m"] = amps_params_df["Sw"].rolling('20min', min_periods=10).mean()

    # flag_interpolater = interpolate.interp1d(amps_params_df["gps_sec"], amps_params_df["Spaceweather_Flag"], kind='linear',
    #                                          fill_value="extrapolate")
    # data["Spaceweather_Flag"] = flag_interpolater(data["gps_sec"])
    # # Set Flag everywhere back to 1. where it was interpolated, even only slightly away from 0.
    # data.loc[data['Spaceweather_Flag'] > 0., 'Spaceweather_Flag'] = 1.

    # Find Nan values in By-20m, Bz-20m, Sw-20m and set Spaceweather_Flag to 1. only there, the rest got meaningfully interpolated
    data.loc[data['By-20m'].isnull(), 'Spaceweather_Flag'] = 1.
    data.loc[data['Bz-20m'].isnull(), 'Spaceweather_Flag'] = 1.
    data.loc[data['Sw-20m'].isnull(), 'Spaceweather_Flag'] = 1.


    # Sanity check for Spaceweather flags set to 1
    print("data['Spaceweather_Flag'].sum(): ", data['Spaceweather_Flag'].sum())
    print("data.loc[data['Spaceweather_Flag'] == 1., 'By']: ", data.loc[data['Spaceweather_Flag'] == 1., 'By'])
    print("data.loc[data['Spaceweather_Flag'] == 1., 'Bz']: ", data.loc[data['Spaceweather_Flag'] == 1., 'Bz'])
    print("data.loc[data['Spaceweather_Flag'] == 1., 'Sw']: ", data.loc[data['Spaceweather_Flag'] == 1., 'Sw'])

    # Print out max values of the BY, BZ, SW columns
    print("data['By-20m'].max() raw: ", data['By-20m'].max())
    print("data['Bz-20m'].max() raw: ", data['Bz-20m'].max())
    print("data['Sw-20m'].max() raw: ", data['Sw-20m'].max())
    # Print out max values with spaceweather flag set to 0
    print("data.loc[data['Spaceweather_Flag'] == 0., 'By-20m'].max(): ", data.loc[data['Spaceweather_Flag'] == 0., 'By-20m'].max())
    print("data.loc[data['Spaceweather_Flag'] == 0., 'Bz-20m'].max(): ", data.loc[data['Spaceweather_Flag'] == 0., 'Bz-20m'].max())
    print("data.loc[data['Spaceweather_Flag'] == 0., 'Sw-20m'].max(): ", data.loc[data['Spaceweather_Flag'] == 0., 'Sw-20m'].max())

    return data

def enrich_df_with_amps_data(data, quaternion_columns=["q1_fgm12nec", "q2_fgm12nec", "q3_fgm12nec", "q4_fgm12nec"]):
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
    print("tilt_values: ", data["tilt"])
    # Convert the lat/lon/height from geocentric to geodetic with geoc2geod function from the ppigrf package
    # TODO: This ("geod2geoc") seems like a function naming mismatch,
    #  the description actually says it converts from geocentric to geodetic
    #  which is exactly what we actually need here.
    # Our lat and height are in geocentric coordinates
    #print("X, Y, Z:", data[["X", "Y", "Z"]].head(10))
    print("lat.trs, lon and r.trs:", data[["RAW_Latitude", "RAW_Longitude", "r.trs"]].head(10))
    # theta, r, B_th, B_r = ppigrf.ppigrf.geod2geoc(data["RAW_Latitude"], data["r.trs"]*1000, np.zeros_like(data["RAW_Latitude"]), np.zeros_like(data["RAW_Latitude"]))
    # 90 - latitudes: Converts the latitudes to the expected Colatitudes, defined from 0 to 180 degrees
    print("90 - data[RAW_Latitude] :", 90 - data["RAW_Latitude"])
    gdlat, height, _, _ = ppigrf.ppigrf.geoc2geod(90 - data["RAW_Latitude"], data["r.trs"], np.zeros_like(data["RAW_Latitude"]),
                                                  np.zeros_like(data["RAW_Latitude"]))
    # print("lat.trs and r.trs:", data[["RAW_Latitude", "r.trs"]].head(10))
    logger.debug(f"gdlat: {gdlat}")
    logger.debug(f"height: {height}")
    # TODO: get_B_space() claims to use glat and glon as 'geographic' coordinates,
    #  but the function actually uses geodetic coordinates (according to KML).
    #  This is also supported by the height parameter being requested in geodetic coordinates.
    B_e, B_n, B_u = pyamps.get_B_space(glat=gdlat, glon=data["RAW_Longitude"].values, height=height,
                                       time=data["RAW_Timestamp"].values,
                                       v=data["Sw-20m"].values, By=data["By-20m"].values, Bz=data["Bz-20m"].values,
                                       tilt=data["tilt"].values, f107=data["F10.7"].values)
    print("b_space: ", B_e, B_n, B_u)
    # Convert the geodetic return values of the get_B_space() function to geocentric coordinates
    # TODO: Sanity check with returned "new" geocentric coordinates
    _, _, B_th, B_r = ppigrf.ppigrf.geod2geoc(gdlat, height, B_n, B_u)
    logger.debug(f"B_th, B_r: {B_th}, {B_r}")
    # Invert the sign of the B_u component, because then we have NEC values
    B_c = -B_r
    # The theta return needs to be inverted as well
    B_n = -B_th
    # B_n is the N component from NEC
    # B_e is the E component from NEC
    # B_c is the C component from NEC
    amps_nec = np.column_stack([B_n, B_e, B_c])
    print("amps_nec: ", amps_nec)
    # Now we can compute the magnetic field vector in the spacecraft frame by inverting the FGM_to_NEC quaternions and applying them
    # in the opposite direction to the B_e, B_n, B_c components
    quat = np.column_stack([data[quaternion_columns[0]].values, data[quaternion_columns[1]].values,
                            data[quaternion_columns[2]].values, data[quaternion_columns[3]].values])
    # rotation_matrices = mtq.m_quaternion2dcMatrix(mtq.m_inverseQuaternion(quat))
    # amps_mag_old = np.empty_like(amps_nec)
    # for i in range(amps_mag_old.shape[0]):
    #     amps_mag_old[i] = np.dot(rotation_matrices[i], amps_nec[i])
    # del rotation_matrices

    #my_quaternions = Quaternion(quat)
    print("quat.shape: ", quat.shape)
    print("quat: ", quat.dtype)
    print("quat[:4]: ", quat[:4])
    #my_quaternions = Quaternion(w=quat[:, 3], x=quat[:, 0], y=quat[:, 1], z=quat[:, 2])
    #my_quaternions = [Quaternion(w, x, y, z) for x, y, z, w in quat]
    my_quaternions_inversed = [Quaternion(w, x, y, z).inverse for x, y, z, w in quat]
    print("my_quaternions_inversed[:4]: ", my_quaternions_inversed[:4])
    amps_mag = np.array([q.rotate(vec) for q, vec in zip(my_quaternions_inversed, amps_nec)])
    print("amps_nec: ", amps_nec.dtype)
    print("amps_mag: ", amps_mag.dtype)
    #amps_mag = my_quaternions_inversed.rotate(amps_nec)

    # amps_mag = mtq.m_multiplyQuaternion(quat,
    #                                     mtq.m_multiplyQuaternion(np.c_[amps_nec,
    #                                                                    np.zeros(amps_nec.shape[0])
    #                                                              ],
    #                                                              mtq.m_inverseQuaternion(quat)
    #                                                              )
    #                                     )[:, 0:3]

    print("amps_mag: ", amps_mag)

    # import rowan
    # quats_inversed = [rowan.inverse(np.array([x, y, z, w])) for x, y, z, w in quat]
    # print("quats_inversed[:4]: ", quats_inversed[:4])
    # print("first : ", rowan.rotate(quats_inversed[0], amps_nec[0]))
    # # print("q1.rotate(amps_nec): ", q1.rotate(amps_nec))
    # alternative_amps_mag = np.array([rowan.rotate(q, vec) for q, vec in zip(quats_inversed, amps_nec)])
    # print("alternative_amps_mag: ", alternative_amps_mag)


    # print("amps_mag_old: ", amps_mag_old)
    data["amps_b_mag_x"] = amps_mag[:, 0]
    data["amps_b_mag_y"] = amps_mag[:, 1]
    data["amps_b_mag_z"] = amps_mag[:, 2]
    data["amps_b_nec_x"] = amps_nec[:, 0]
    data["amps_b_nec_y"] = amps_nec[:, 1]
    data["amps_b_nec_z"] = amps_nec[:, 2]
    # now set the respective rows to zero for latitudes below 40 / above -40 degrees (the model is not valid for these latitudes)
    data.loc[(data["APEX_QD_LAT"] > -40) & (data["APEX_QD_LAT"] < 40),
             ["amps_b_mag_x", "amps_b_mag_y", "amps_b_mag_z", "amps_b_nec_x", "amps_b_nec_y", "amps_b_nec_z"]] = 0.
    # test this with prints
    print("amps_b_mag_x should be zero here: ",
          data.loc[(data["APEX_QD_LAT"] > -40) & (data["APEX_QD_LAT"] < 40)]["amps_b_mag_x"][:5])
    print("amps_b_mag_x should NOT be zero here, < -40 lat: ", data.loc[data["APEX_QD_LAT"] < -40]["amps_b_mag_x"][:5])
    print("amps_b_mag_x should NOT be zero here, > 40 lat: ", data.loc[data["APEX_QD_LAT"] > 40]["amps_b_mag_x"][:5])

    return data