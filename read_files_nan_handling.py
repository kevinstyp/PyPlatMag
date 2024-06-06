import logging
import os
import sys

import yaml
from box import Box

from preprocessing import nan_handler
from utils import data_io

config = Box.from_yaml(filename="./config.yaml", Loader=yaml.SafeLoader)
print(config)
config_goce = Box.from_yaml(filename="./config_goce.yaml", Loader=yaml.SafeLoader)
print(config_goce)

logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(config.log_level),
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# determine nan
# basically from logic wise start in the back with use_cache
# treat all months from the directory and look into them for the features

satellite = config.satellite_specifier
save_path = data_io.get_save_path(config.write_path, satellite)
# First, look up if the directory exists, and contains the needed data for filling with nans
# If not, create the directory
### 1: Check availability of everything for nan_application
if config.use_cache and os.path.exists(save_path + "features_to_drop.pickle"):
    print("first if")
    # Apply them to all year_month_specifiers
    nan_handler.nan_application(save_path, config.year_month_specifiers, satellite, meta_features=config_goce.meta_features)


else:
    print("first else")
    ### 2: Check availability of everything for nan_determination_merge
    # List all directories in save_path
    print("save_path: ", save_path)
    directories = [f for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f))]

    print("directories:", directories)
    file_check = True
    for directory in directories:
        # Check if all files are available
        if not os.path.exists(save_path + directory + "/df_column_nancount.pickle") or not os.path.exists(
                save_path + directory + "/df_column_mean.pickle") or not os.path.exists(save_path + directory + "/df_overall.pickle"):
            file_check = False
            break
    if file_check:
        print("second if")
        # Merge, then apply
        nan_handler.nan_determination_merge(config.year_month_specifiers, save_path, satellite, config.nan_share)
        nan_handler.nan_application(save_path, config.year_month_specifiers, satellite, z_all_features=config_goce.meta_features)

    else:
        print("last else")
        # Determine, then merge, then apply
        nan_handler.nan_determination(config.year_month_specifiers, config.write_path, satellite, meta_features=config_goce.meta_features)
        nan_handler.nan_determination_merge(config.year_month_specifiers, config.write_path, satellite, config.nan_share,
                                            config_goce.essential_calibration_keys)
        nan_handler.nan_application(config.year_month_specifiers, config.write_path, satellite, meta_features=config_goce.meta_features)

    ### 3: Check availability / just apply of everything for nan_determination


