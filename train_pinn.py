import logging
import sys

import pandas as pd
import yaml
from box import Box

from training.training_procedure import extract_electric_currents
from training import training_data
from training import training_procedure
from data_filters.goce_filter import goce_filter
from utils import data_io
import training.neural_network_training as nn_train

config = Box.from_yaml(filename="./config.yaml", Loader=yaml.SafeLoader)
config_goce = Box.from_yaml(filename="./config_goce.yaml", Loader=yaml.SafeLoader)

logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(config.log_level),
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"config: {config}")
logger.info(f"config_goce: {config_goce}")

# Read dataframe for training
data = data_io.read_df(config.write_path, config.satellite_specifier, config.year_month_specifiers, dataset_name="data_nonan")

data = goce_filter(data, magnetic_activity=True, doy=True, training=True, training_columns=[], #satellite_specifier="GOCE",
                #month_specifier="200912", euler_scaler=None,
            meta_features=config_goce.meta_features, y_features=config_goce.y_all_feature_keys)

# TODO: is this still necessary, what was the reason for this?
# Save exact amount of preprocessed columns for later reusage as pickle file

# extract the currents & voltages for pinn before scaling is applied
data, electric_current_df = extract_electric_currents(data, config_goce.current_parameters_file, config_goce.goce_column_description_file)

# TODO: Why not use decompose_dataframe() function?
# TODO: Is the modulo from the preprocess_data_array.py still needed?
# TODO: Decompose is the much better name, split sounds like train / test which does not happen here
x_all, y_all, z_all, weightings = training_data.split_dataframe(data, config_goce.y_all_feature_keys, config_goce.meta_features)

# TODO: What about AMPS inclusion: Add CHAOS with AMPS model


logger.info(f"x_all - columns assigned after split: {x_all.columns.tolist()}")

train_config = config_goce.train_config

# TODO: What about "second_preprocess": STD filtering, Correlation filtering, only needed for x_all data as only columns will be filtered
if train_config.filter_std:
    x_all = training_procedure.filter_std(x_all, train_config.training_file_path, config.year_month_specifiers, config.use_cache)

print("x_all.shape: ", x_all.shape)

if train_config.filter_correlation:
    x_all = training_procedure.filter_correlation(x_all, train_config.training_file_path, config.year_month_specifiers, config.use_cache)

print("x_all.shape: ", x_all.shape)

x_all = training_procedure.scale_data(x_all, train_config.training_file_path, config.year_month_specifiers, config.use_cache)

print("x_all: ", x_all)
print("x_all: ", x_all.shape)
print("x_all - columns: ", x_all.columns.tolist())

# Now for train / test split
x_train, x_test, y_train, y_test, el_cu_train, el_cu_test, weightings_train, weightings_test\
    = training_procedure.split_train_test([x_all, y_all, electric_current_df, weightings], train_config.test_split, train_config.learn_config.batch_size)

print("x_train.shape: ", x_train.shape)
print("el_cu_train.shape: ", el_cu_train.shape)
print("type of x_train: ", type(x_train))
print("type of el_cu_train: ", type(el_cu_train))
# # Convert every column of el_cu_train to a list and store in a list
# el_cu_train = [el_cu_train[col].tolist() for col in el_cu_train.columns]
# el_cu_test = [el_cu_test[col].tolist() for col in el_cu_test.columns]
# model_input_train = [x_train] + el_cu_train
# model_input_test = [x_test] + el_cu_test
#print("overall model input shape: ", model_input_train[0].shape[1])
model_input_train = pd.concat([x_train, el_cu_train], axis=1)
model_input_test = pd.concat([x_test, el_cu_test], axis=1)
print("model_input_train.shape: ", model_input_train.shape)
print("model_input_test.shape: ", model_input_test.shape)

print("overall model input shape: ", model_input_train.shape)
print("number_of_bisa_neurons: ", el_cu_train.shape)

print("type of config_goce: ", type(config_goce))
print("type of train_config.learn_config: ", type(train_config.learn_config))

model, history = nn_train.goce_training(x_train=model_input_train, y_train=y_train, x_test=model_input_test, y_test=y_test,
                                        number_of_bisa_neurons=el_cu_train.shape[1],
                                        weightings_train=weightings_train,
                                   weightings_test=weightings_test,
                                    learn_config = train_config.learn_config,
                                    neural_net_variant=train_config.neural_network_variant,
                                        )
