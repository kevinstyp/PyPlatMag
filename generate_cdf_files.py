import logging
import os
import sys

import yaml
from box import Box
from utils import data_io

from spacepy import pycdf
import publication.training_retrieval as tr
from data_filters.goce_filter import goce_filter
from training import training_procedure as tp
from training import training_data
import pandas as pd

config = Box.from_yaml(filename="./config.yaml", Loader=yaml.SafeLoader)
config_goce = Box.from_yaml(filename="./config_goce.yaml", Loader=yaml.SafeLoader)

logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(config.log_level),
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"config: {config}")
logger.info(f"config_goce: {config_goce}")

os.environ["CDF_LIB"] = "~/lib/"

# Get the year_months used to save auxilary files
if config.model_year_months == "None":
    year_months = '_'.join([config.year_month_specifiers[0], config.year_month_specifiers[-1]])
else:
    year_months = config.model_year_months

# Load the auxilary files created during training
train_config = config_goce.train_config
std_indices, corr_indices, hk_scaler = tr.read_in_pickles_small(train_config.training_file_path, year_months)

# Specify the model to be loaded
model_name = config.model_output_path + config.model_name + '_' + config.satellite_specifier + '_' + year_months

# Now similar to training
# Load the data
data = data_io.read_df(config.write_path, config.satellite_specifier, config.year_month_specifiers, dataset_name="data_nonan")

# x_all, y_all, z_all = goce_filter(data , doy=True,
#                                                         training=False, x_all_columns=x_all_columns,
#                                                         month_specifier=model_year_months)
logger.info(f"Data shape after reading: {data.shape}")
data = goce_filter(data, magnetic_activity=True, doy=True,
                   training=False, training_columns=[],
            meta_features=config_goce.meta_features, y_features=config_goce.y_all_feature_keys)
logger.info(f"Data shape after filtering: {data.shape}")


# TODO: Check whether something happens because of this
# Leave align_x_columns away?: Columns are sorted -> not necessary
# Weight handling: Weights arent needed anymore -> not necessary
# Check if someone randomizes the columns at some point :O -> not necessary

# Extract power currents if use_pinn is set
train_config = config_goce.train_config
if train_config.use_pinn:
    data, electric_current_df = tp.extract_electric_currents(data, config_goce.current_parameters_file,
                                                             config_goce.goce_column_description_file)


# TODO: training_data, training_prcedure -> Maybe, rename them to preprare_data, prepare_procedure or smth
x_all, y_all, z_all, weightings = training_data.split_dataframe(data, config_goce.y_all_feature_keys, config_goce.meta_features)

# Add solar activity, and DOY
x_all = tp.add_solar_activity(x_all, z_all)
x_all = tp.add_day_of_year(x_all, z_all)

# Std, Corr, Scaling
if train_config.filter_std:
    x_all = tp.filter_std(x_all, train_config.training_file_path, config.year_month_specifiers, config.use_cache)
    logger.debug(f"x_all - shape after std filtering: {x_all.shape}")

if train_config.filter_correlation:
    x_all = tp.filter_correlation(x_all, train_config.training_file_path, config.year_month_specifiers, config.use_cache)
    logger.debug(f"x_all - shape after correlation filtering: {x_all.shape}")

x_all = tp.scale_data(x_all, train_config.training_file_path, config.year_month_specifiers, config.use_cache)

logger.info(f"x_all - shape before splitting: {x_all.shape}")
logger.info(f"Final columns for generating predictions: {x_all.columns.tolist()}")


# Check how to split network building and network training etc.
if train_config.use_pinn:
    model_input_train = pd.concat([x_all, electric_current_df], axis=1)
else:
    model_input_train = x_all
