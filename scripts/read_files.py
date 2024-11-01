import logging
import os
from pathlib import Path

import pandas as pd
import sys
import time
import yaml
from box import Box

import utils.time_handler as th
from data_connectors.goce_data_connector import GOCEConnector
from preprocessing import data_enricher
from preprocessing.amps_preprocessor import enrich_df_with_amps_data, enrich_df_with_amps_params_data, \
    unpack_amps_params_file_to_df
from preprocessing.one_hot_encoder import one_hot_encode
from preprocessing.space_weather_preprocessor import enrich_df_with_hp_data, enrich_df_with_kp_data, unpack_hp30_file_to_df, \
    unpack_kp_dst_file_to_df
from utils import data_io

# Read config file
dirname = os.path.dirname(Path(__file__).parent)
config = Box.from_yaml(filename=os.path.join(dirname, "./config.yaml"), Loader=yaml.SafeLoader)
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(config.log_level),
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def read_files():
    logger.info(f"config: {config}")

    for year_month_specifier in config.year_month_specifiers:
        first_time = time.process_time()

        # GOCE Data-Connector
        goce_connector = GOCEConnector(base_path=config.goce_data_path)
        # Get the data
        data = goce_connector.get_data(year_month_specifier)
        logger.debug(f"GOCE data shape: {data.shape}")
        logger.debug(f"GOCE data head(3): {data.head(3)}")

        if config.string_conversion == 'one-hot':
            data = one_hot_encode(data)
        # elif config.string_conversion == 'remove':
        else:
            pass

        logger.debug(f"GOCE data shape: {data.shape}")

        # TODO: Move to data connector
        # Convert GPS seconds to datetime

        data['RAW_Timestamp'] = th.gps_to_datetime(data['gps_sec'])

        logger.debug(f"data - Timestamp: {data['RAW_Timestamp'][0]}")
        logger.debug(f"data - GPS Sec: {data['gps_sec'][0]}")

        # Read additional files for data enrichment
        start_overall = time.process_time()

        auxiliary_data_path = os.path.join(dirname, config.auxiliary_data_path)
        sw_df = unpack_kp_dst_file_to_df(auxiliary_data_path, year_month_specifier, config.use_cache)
        # 1.1 sec
        logger.info("Time for unpacking Kp/Dst data: " + str(round(time.process_time() - start_overall, 2)) + " seconds")
        start_overall = time.process_time()
        # TODO: This is called amps_params, but actually they are space-weather parameters, why not import them all in one step?
        amps_params_df = unpack_amps_params_file_to_df(auxiliary_data_path,
                                                       year_month_specifier)  # Solar wind speed, Bx, By, F10.7
        # 22.4 sec
        logger.info("Time for unpacking Amps Params data: " + str(round(time.process_time() - start_overall, 2)) + " seconds")
        start_overall = time.process_time()
        hp30_df = unpack_hp30_file_to_df(auxiliary_data_path, year_month=year_month_specifier, use_cache=config.use_cache)
        # 2.1 sec
        logger.info("Time for unpacking Hp30 data: " + str(round(time.process_time() - start_overall, 2)) + " seconds")

        start_overall = time.process_time()
        # TODO: Maybe move KP to Hp30 processing
        data = enrich_df_with_kp_data(data, sw_df, with_kp=True)
        logger.info(f"Time for enriching with Kp data: {round(time.process_time() - start_overall, 2)} seconds")
        start_overall = time.process_time()
        data = enrich_df_with_amps_params_data(data, amps_params_df)
        logger.info(f"Time for enriching with AMPS-parameter data: {round(time.process_time() - start_overall, 2)} seconds")
        start_overall = time.process_time()
        data = enrich_df_with_hp_data(data, hp30_df)
        logger.info(f"Time for enriching with Hp data: {round(time.process_time() - start_overall, 2)} seconds")
        del amps_params_df, sw_df, hp30_df

        start_overall = time.process_time()
        # TODO: Move this line to a better place?
        #  Removes NaN entries where positional arguments are missing (e.g., latitude and quaternions)
        data = data[data['APEX_QD_LAT'].between(-90., 90.)]
        # TODO: It seems that quats are float32, whereas they were float64 before in the older code
        data = enrich_df_with_amps_data(data)
        logger.info("Time for enriching with AMPS data: " + str(round(time.process_time() - start_overall, 2)) + " seconds")

        data["F10.7-81d"] = data["F10.7"].rolling(window=1944, center=True, min_periods=1).mean()

        # TODO: Not consider features, Line 454
        do_not_consider_features = ['gps_sec', 'lt', 'mjd2000', 'mlat', 'chaos7_b_fgm2_x', 'chaos7_b_fgm2_y', 'chaos7_b_fgm2_z',
                                    'chaos7_b_fgm3_x', 'chaos7_b_fgm3_y', 'chaos7_b_fgm3_z',
                                    'chaos7_b_sc_x', 'chaos7_b_sc_y', 'chaos7_b_sc_z', 'FGM2_X_sc', 'FGM2_Y_sc', 'FGM2_Z_sc',
                                    'FGM3_X_sc', 'FGM3_Y_sc',
                                    'FGM3_Z_sc',
                                    'FGM2_X_nec', 'FGM2_Y_nec', 'FGM2_Z_nec', 'FGM3_X_nec', 'FGM3_Y_nec',
                                    'FGM3_Z_nec',
                                    ]
        data = data.drop(do_not_consider_features, axis=1, errors='ignore')
        logger.info(f"Data shape after removing not-consider features: {data.shape}")

        # TODO Orbit counter: Move to GOCE Connector
        orbit_counter_df = pd.read_hdf(config.orbit_counter_path, "df").add_prefix("ORB_")
        logger.debug(f"orbit_counter_df: {orbit_counter_df.head(2)}")
        data['copy_egg_iaq_index'] = data.index
        # data['copy_gps_sec'] = data.index
        data = data.reset_index(drop=True)
        data = pd.merge_asof(data, orbit_counter_df, on=None, left_on="RAW_Timestamp", right_on="ORB_Timestamp",  #
                             left_index=False, right_index=False, by=None,
                             left_by=None, right_by=None, suffixes=('', '_orb'), tolerance=None,
                             allow_exact_matches=True,
                             direction='backward').rename(columns={'copy_egg_iaq_index': 'egg_iaq_index'}).set_index(
            'egg_iaq_index')

        data = data.set_index("RAW_Timestamp", drop=False)
        logger.debug(f"Data head after orbit counter merge: {data.head(3)}")
        if {'ORB_Timestamp', 'ORB_Latitude', 'ORB_Longitude', 'ORB_Radius', 'ORB_MLT'}.issubset(data.columns):
            data = data.drop(['ORB_Timestamp', 'ORB_Latitude', 'ORB_Longitude', 'ORB_Radius', 'ORB_MLT'], axis=1)

        # TODO: To be honest, this should go to the training: Weightings are only needed for training (if wanted)
        data = data_enricher.add_weights(data)

        # Add DOY
        data['DOY'] = data["RAW_Timestamp"].dt.day_of_year

        # Write data to disk
        data_io.save_df(data, config.write_path, config.satellite_specifier, year_month_specifier, dataset_name="data")

        logger.info(f"data-shape: {data.shape}")
        logger.info(f"Time for overall: {round(time.process_time() - first_time, 2)} seconds")


if __name__ == "__main__":
    read_files()
