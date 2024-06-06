import logging
import sys

import yaml
from box import Box

from data_filters.goce_filter import goce_filter
from utils import data_io

config = Box.from_yaml(filename="./config.yaml", Loader=yaml.SafeLoader)
print(config)
config_goce = Box.from_yaml(filename="./config_goce.yaml", Loader=yaml.SafeLoader)
print(config_goce)

logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(config.log_level),
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Read dataframe for training
data = data_io.read_df(config.write_path, config.satellite_specifier, config.year_month_specifiers, dataset_name="data_nonan")

goce_filter(data, magnetic_activity=True, doy=True, training=True, training_columns=[], satellite_specifier="GOCE",
                month_specifier="200912", euler_scaler=None,
            meta_features=config_goce.meta_features, y_features=config_goce.y_all_feature_keys)


