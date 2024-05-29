import pandas as pd
from box import Box
import yaml
import logging

import time
import sys

# Read config file
from data_connectors.goce_data_connector import GOCEConnector
from one_hot_encoder import one_hot_encode

import utils.time_handler as th

from space_weather_preprocessor import enrich_df_with_hp_data, enrich_df_with_kp_data, unpack_hp30_file_to_df, unpack_kp_dst_file_to_df
from amps_preprocessor import enrich_df_with_amps_data, enrich_df_with_amps_params_data, unpack_amps_params_file_to_df


config = Box.from_yaml(filename="./config.yaml", Loader=yaml.SafeLoader)
print(config)
#ch = logging.StreamHandler()
#logging.basicConfig(filename='myapp.log', level=logging.DEBUG, )
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(config.log_level))
logger = logging.getLogger(__name__)
# #logger = logging.getLogger('pyplatmag')
# #logger.setLevel(config.log_level)
# logger.setLevel(logging.DEBUG)
#
# print("__package__:", __package__)
#
# # create console handler and set level to debug
# ch = logging.StreamHandler()
# ch.setLevel(logger.level)
#
# # create formatter
# formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
#
# # add formatter to ch
# ch.setFormatter(formatter)
#
# # add ch to logger
# logger.addHandler(ch)

for year_month_specifier in config.year_month_specifiers:

    ### GOCE Data-Connector
    # Instantiate the GOCEConnector
    goce_connector = GOCEConnector(base_path=config.goce_data_path)
    # Get the data
    data = goce_connector.get_data(year_month_specifier)
    logger.debug(f"GOCE data shape: {data.shape}")
    logger.debug(f"GOCE data head(3): {data.head(3)}")

    if config.string_conversion == 'one-hot':
        data = one_hot_encode(data, config.pandas_inplace)
    #elif config.string_conversion == 'remove':
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

    sw_df = unpack_kp_dst_file_to_df(config.space_weather_path)
    # TODO: This is called amps_params, but actually they are space-weather parameters, why not import them all in one step?
    amps_params_df = unpack_amps_params_file_to_df(config.amps_params_path, year_month_specifier)  # Solar wind speed, Bx, By, F10.7
    hp30_df = unpack_hp30_file_to_df(config.hp30_path)
    logger.info("Time for unpacking auxilary data: " + str(round(time.process_time() - start_overall, 2)) + " seconds")

    start_overall = time.process_time()
    data = enrich_df_with_kp_data(data, sw_df)
    logger.info(f"Time for enriching with Kp data: {round(time.process_time() - start_overall, 2)} seconds")
    data = enrich_df_with_amps_params_data(data, amps_params_df)
    logger.info(f"Time for enriching with AMPS data: {round(time.process_time() - start_overall, 2)} seconds")
    data = enrich_df_with_hp_data(data, hp30_df)
    logger.info(f"Time for enriching with Hp data: {round(time.process_time() - start_overall, 2)} seconds")
    del amps_params_df, sw_df, hp30_df
    data = enrich_df_with_amps_data(data)

    logger.info("Time for enriching with auxilary data: " + str(round(time.process_time() - start_overall, 2)) + " seconds")

    data["F10.7-81d"] = data["F10.7"].rolling(window=1944, center=True, min_periods=1).mean()

    # Line 444
