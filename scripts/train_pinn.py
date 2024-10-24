import logging
import os
import sys

import pandas as pd
import yaml
from box import Box

from training import training_data
from training import training_procedure as tp
from training import evaluation_procedure as ep
from data_filters.goce_filter import goce_filter
from utils import data_io
import training.neural_network_training as nn_train

dirname = os.path.dirname(__file__)
config = Box.from_yaml(filename=os.path.join(dirname, "../config.yaml"), Loader=yaml.SafeLoader)
config_goce = Box.from_yaml(filename=os.path.join(dirname, "../config_goce.yaml"), Loader=yaml.SafeLoader)
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(config.log_level),
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def train_pinn():
    logger.info(f"config:   {config}")
    logger.info(f"config_goce: {config_goce}")

    # Read dataframe for training
    data = data_io.read_df(config.write_path, config.satellite_specifier, config.year_month_specifiers, dataset_name="data_nonan")

    data = goce_filter(data, magnetic_activity=True, doy=True, training=True, training_columns=[],
                       meta_features=config_goce.meta_features, y_features=config_goce.y_all_feature_keys)

    # TODO: is this still necessary, what was the reason for this?
    # Save exact amount of preprocessed columns for later reusage as pickle file

    # extract the currents & voltages for pinn before scaling is applied
    train_config = config_goce.train_config
    if train_config.use_pinn:
        current_parameters_file = os.path.join(dirname, config_goce.current_parameters_file)
        goce_column_description_file = os.path.join(dirname, config_goce.goce_column_description_file)
        data, electric_current_df = tp.extract_electric_currents(data, current_parameters_file, goce_column_description_file)
        # TODO Change back to above line
        # electric_current_df = data[["PHT12780", "PHT12800"]]
        # data = data.drop(["PHT12780", "PHT12800"], axis=1)

    # TODO: Why not use decompose_dataframe() function?
    # TODO: Is the modulo from the preprocess_data_array.py still needed?
    # TODO: Decompose is the much better name, split sounds like train / test which does not happen here
    x_all, y_all, z_all, weightings = training_data.split_dataframe(data, config_goce.y_all_feature_keys, config_goce.meta_features)

    # weightings: Rebalance for sample sizes of the different regions (low, mid, and high latitudes)
    weightings = tp.rebalance_weightings(weightings)

    # TODO: What about AMPS inclusion: Add CHAOS and AMPS model
    if train_config.use_amps:
        logger.info(f"Using AMPS model for training")
        y_all = y_all + z_all[['amps_b_mag_x', 'amps_b_mag_y', 'amps_b_mag_z']].values

    # Add solar activity, and DOY
    x_all = tp.add_solar_activity(x_all, z_all)
    x_all = tp.add_day_of_year(x_all, z_all)

    del z_all

    logger.info(f"x_all - columns assigned after split: {x_all.columns.tolist()}")

    training_file_path = os.path.join(dirname, train_config.training_file_path)
    if train_config.filter_std:
        x_all = tp.filter_std(x_all, training_file_path, config.year_month_specifiers, config.use_cache)
        logger.debug(f"x_all - shape after std filtering: {x_all.shape}")

    if train_config.filter_correlation:
        x_all = tp.filter_correlation(x_all, training_file_path, config.year_month_specifiers, config.use_cache)
        logger.debug(f"x_all - shape after correlation filtering: {x_all.shape}")

    x_all = tp.scale_data(x_all, training_file_path, config.year_month_specifiers, config.use_cache)

    logger.info(f"x_all - shape before splitting: {x_all.shape}")
    logger.info(f"Final columns for training: {x_all.columns.tolist()}")

    # Split the different parts in train / test
    if train_config.use_pinn:
        x_train, x_test, y_train, y_test, el_cu_train, el_cu_test, weightings_train, weightings_test = \
            tp.split_train_test([x_all, y_all, electric_current_df, weightings], train_config.test_split,
                                train_config.learn_config.batch_size)
        logger.debug(f"el_cu_train - shape after splitting: {el_cu_train.shape}")
    else:
        x_train, x_test, y_train, y_test, weightings_train, weightings_test = \
            tp.split_train_test([x_all, y_all, weightings], train_config.test_split, train_config.learn_config.batch_size)
    logger.info(f"x_train - shape after splitting: {x_train.shape}")
    logger.info(f"x_test - shape after splitting: {x_test.shape}")

    if train_config.use_pinn:
        model_input_train = pd.concat([x_train, el_cu_train], axis=1)
        model_input_test = pd.concat([x_test, el_cu_test], axis=1)
    else:
        model_input_train = x_train
        model_input_test = x_test
    logger.info(f"model_input_train - shape after merging electric currents: {model_input_train.shape}")

    if train_config.use_pinn:
        model, history = nn_train.goce_training(x_train=model_input_train, y_train=y_train, x_test=model_input_test, y_test=y_test,
                                                number_of_bisa_neurons=el_cu_train.shape[1],
                                                weightings_train=weightings_train,
                                                weightings_test=weightings_test,
                                                learn_config=train_config.learn_config,
                                                neural_net_variant=train_config.neural_network_variant,
                                                )
    else:
        model, history = nn_train.goce_training(x_train=model_input_train, y_train=y_train, x_test=model_input_test, y_test=y_test,
                                                weightings_train=weightings_train,
                                                weightings_test=weightings_test,
                                                learn_config=train_config.learn_config,
                                                neural_net_variant=train_config.neural_network_variant,
                                                )

    year_months = '_'.join([config.year_month_specifiers[0], config.year_month_specifiers[-1]])
    model_name = config.model_output_path + config.model_name + '_' + config.satellite_specifier + '_' + year_months
    model.save(model_name + '.h5')

    # Evaluate the model on train and test data
    ep.evaluate_model(model_input_train, y_train, model_input_test, y_test, model, config.year_month_specifiers,
                      train_config.learn_config, number_of_bisa_neurons=el_cu_train.shape[1])

if __name__ == '__main__':
    main()
