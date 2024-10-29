import logging
import os

import sys
import yaml
from box import Box

from preprocessing import nan_handler
from utils import data_io

dirname = os.path.dirname(__file__)
config = Box.from_yaml(filename=os.path.join(dirname, "../config.yaml"), Loader=yaml.SafeLoader)
config_goce = Box.from_yaml(filename=os.path.join(dirname, "../config_goce.yaml"), Loader=yaml.SafeLoader)
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(config.log_level),
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def read_files_nan_handling():
    logger.info(f"config: {config}")
    logger.info(f"config_goce: {config_goce}")

    # determine nan
    # basically from logic wise start in the back with use_cache
    # treat all months from the directory and look into them for the features

    satellite = config.satellite_specifier
    save_path = data_io.get_save_path(config.write_path, satellite)
    # First, look up if the directory exists, and contains the needed data for filling with nans
    # If not, create the directory
    ### 1: Check availability of everything for nan_application
    if config.use_cache and os.path.exists(save_path + "features_to_drop.pickle"):
        # Apply them to all year_month_specifiers
        nan_handler.nan_application(config.year_month_specifiers, config.write_path, satellite, meta_features=config_goce.meta_features)


    else:
        # Check availability of everything needed for nan_determination_merge
        # List all directories in save_path
        logger.info(f"save_path: {save_path}")
        directories = [f for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f))]
        logger.info(f"Directories: {directories}")

        file_check = True
        for directory in directories:
            # Check if all files are available
            if not os.path.exists(save_path + directory + "/df_column_nancount.pickle") or not os.path.exists(
                    save_path + directory + "/df_column_mean.pickle") or not os.path.exists(save_path + directory + "/df_overall.pickle"):
                file_check = False
                break
        if file_check:
            # Merge, then apply
            nan_handler.nan_determination_merge(config.year_month_specifiers, config.write_path, satellite, config.nan_share)
            nan_handler.nan_application(config.year_month_specifiers, config.write_path, satellite, meta_features=config_goce.meta_features)

        else:
            # Determine, then merge, then apply
            nan_handler.nan_determination(config.year_month_specifiers, config.write_path, satellite, meta_features=config_goce.meta_features)
            nan_handler.nan_determination_merge(config.year_month_specifiers, config.write_path, satellite, config.nan_share,
                                                config_goce.essential_calibration_keys)
            nan_handler.nan_application(config.year_month_specifiers, config.write_path, satellite, meta_features=config_goce.meta_features)

        ### 3: Check availability / just apply of everything for nan_determination
        # TODO: Is there something missing here?


if __name__ == "__main__":
    read_files_nan_handling()
