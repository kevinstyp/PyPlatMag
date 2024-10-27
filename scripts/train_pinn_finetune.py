import logging
import os
from pathlib import Path

import pandas as pd
import sys
import yaml
from box import Box

import training.neural_network_training as nn_train
from data_filters.goce_filter import goce_filter
from training import evaluation_procedure as ep, training_procedure as tp
from utils import data_io

dirname = os.path.dirname(Path(__file__).parent)
config = Box.from_yaml(filename=os.path.join(dirname, "./config.yaml"), Loader=yaml.SafeLoader)
config_goce = Box.from_yaml(filename=os.path.join(dirname, "./config_goce.yaml"), Loader=yaml.SafeLoader)
train_config = config_goce.train_config
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(config.log_level),
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def train_pinn_finetune():
    logger.info(f"config:   {config}")
    logger.info(f"config_goce: {config_goce}")

    # Get the year_months used to save auxilary files
    if config.model_year_months == "None":
        year_months = '_'.join([config.year_month_specifiers[0], config.year_month_specifiers[-1]])
    else:
        year_months = config.model_year_months

    # Specify the model to be loaded
    model_name = config.model_output_path + config.model_name + '_' + config.satellite_specifier + '_' + year_months + '.h5'
    model_path = os.path.join(dirname, model_name)
    logger.info(f"model_name: {model_name}")
    logger.info(f"model_path: {model_path}")

    for year_month_specifier in config.year_month_specifiers:
        # Read dataframe for training
        data = data_io.read_df(config.write_path, config.satellite_specifier, [year_month_specifier], dataset_name="data_nonan")
        logger.info(f"Data shape after reading: {data.shape}")

        data = goce_filter(data, magnetic_activity=True, doy=True, training=True, training_columns=[],
                           meta_features=config_goce.meta_features, y_features=config_goce.y_all_feature_keys)
        logger.info(f"Data shape after filtering: {data.shape}")

        electric_current_df, x_all, y_all, z_all = tp.prepare_data(data, config, config_goce, dirname, train_config,
                                                                   use_cache=config.use_cache)

        weightings = tp.extract_and_rebalance_weightings(z_all)
        if train_config.use_amps:
            logger.info(f"Using AMPS model for training")
            y_all = y_all + z_all[['amps_b_mag_x', 'amps_b_mag_y', 'amps_b_mag_z']].values
        del z_all

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

        ## ---
        ## Up until here, Fine I guess, now load the old model and actually finetune
        ## ---

        if train_config.use_pinn:
            model, history = nn_train.goce_training(x_train=model_input_train, y_train=y_train, x_test=model_input_test, y_test=y_test,
                                                    number_of_bisa_neurons=el_cu_train.shape[1],
                                                    weightings_train=weightings_train,
                                                    weightings_test=weightings_test,
                                                    learn_config=train_config.learn_config,
                                                    neural_net_variant=train_config.neural_network_variant,
                                                    model_path=model_path,
                                                    )
            # model, history = nn_train.goce_training(x_train=model_input_train, y_train=y_train, x_test=model_input_test, y_test=y_test,
            #                                         epochs=epochs,
            #                                         batch_size=batch_size,
            #                                         number_of_bisa_neurons=input_pinn_train.shape[1],
            #                                         model_path=model_path,
            #                                         learning_rate=finetune_learning_rate,
            #                                         weightings_train=weightings_train,
            #                                         weightings_test=weightings_test,
            #                                         neural_net_variant=4,)
        else:
            model, history = nn_train.goce_training(x_train=model_input_train, y_train=y_train, x_test=model_input_test, y_test=y_test,
                                                    weightings_train=weightings_train,
                                                    weightings_test=weightings_test,
                                                    learn_config=train_config.learn_config,
                                                    neural_net_variant=train_config.neural_network_variant,
                                                    model_path=model_path,
                                                    )

        year_months = '_'.join([year_month_specifier, year_month_specifier])
        model_name = config.model_output_path + config.model_name + '_finetune_' + config.satellite_specifier + '_' + year_months
        model_path = os.path.join(dirname, model_name)
        model.save(model_path + '.h5')

        # Evaluate the model on train and test data
        ep.evaluate_model(model_input_train, y_train, model_input_test, y_test, model, [year_month_specifier],
                          train_config.learn_config, number_of_bisa_neurons=el_cu_train.shape[1])


if __name__ == '__main__':
    train_pinn_finetune()
