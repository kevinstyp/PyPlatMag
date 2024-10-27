import logging

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler

import training.customs.stepdecay as sd
from training.model_builder import build_network_goce, build_network_goce_pinn

logger = logging.getLogger(__name__)


def goce_training(x_train, y_train, x_test, y_test, weightings_train, weightings_test, learn_config=None, neural_net_variant=0,
                  model_path=None, all_values_raw=None,
                  all_values_tq=None, timestamps_training=None, timestamps_val=None, number_of_bisa_neurons=None):
    logger.debug(f"Number of threads used by tensorflow: {tf.config.threading.get_inter_op_parallelism_threads()}")
    logger.debug(f"Tensorflow version: {tf.__version__}")
    logger.info(f"learn_config: {learn_config}")
    if model_path: # If model_path to an existing model is given, a finetune training is performed
        epochs = learn_config.epochs_finetune
        learning_rate = learn_config.learning_rate_finetune
    else:
        epochs = learn_config.epochs
        learning_rate = learn_config.learning_rate
    batch_size = learn_config.batch_size

    if neural_net_variant == 0:
        model = build_network_goce(input_shape=(x_train.shape[1]))

    elif neural_net_variant == 1:
        init_value = 1.0  # 1e-4
        model = build_interpolation_network_goce(batch_size,
                                                        all_values_raw, all_values_tq, #all_values_sa, all_values_bc,
                                                        hk_shape=(x_train.shape[1]),
                                                        init_raw=init_value, init_tq=init_value,
                                                        )
    elif neural_net_variant == 2:
        model = build_interpolation_network_goce(batch_size,
                                                        all_values_raw, all_values_tq, #all_values_sa, all_values_bc,
                                                        hk_shape=(x_train.shape[1]),
                                                        )
    elif neural_net_variant == 3:
        # x_train is a list and first element contains the non-current input data
        model = build_network_goce_pinn(input_shape=(x_train.shape[1] - number_of_bisa_neurons),
                                        #input_shape=(x_train[0].shape[1]),
                                        batch_size=batch_size,
                                                number_of_bisa_neurons=number_of_bisa_neurons,
                                                 )
    elif neural_net_variant == 4:
        model = build_network_goce_multipinn(batch_size,
                                                 #all_values_raw, all_values_tq,  # all_values_sa, all_values_bc,
                                                 hk_shape=(x_train[0].shape[1]),#x_train is a list and first element contains the non-current input data
                                                number_of_bisa_neurons=number_of_bisa_neurons,
                                                 )
    elif neural_net_variant == 5:
        model = build_interpolation_network_tubin_simple(batch_size,
                                                        # model = build_interpolation_network_goce_experimental(batch_size,  # all_values_raw, all_values_tq, #all_values_sa, all_values_bc,
                                                        hk_shape=(x_train.shape[1]), init_raw=0.0, init_tq=0.0)
    else:
        model = build_interpolation_network_goce_simple(batch_size,
                                                        # model = build_interpolation_network_goce_experimental(batch_size,  # all_values_raw, all_values_tq, #all_values_sa, all_values_bc,
                                                        hk_shape=(x_train.shape[1]), init_raw=0.0, init_tq=0.0)
    if model_path is not None:
        logger.info(f"Loading model from given path: {model_path}")
        model.load_weights(model_path)

    schedule = sd.StepDecay(initAlpha=learning_rate, factor=0.5, dropEvery=learn_config.drop_every,
                            first_extra=learn_config.first_extra)
    stepdecay = LearningRateScheduler(schedule)

    # TODO: optimizer is one of the parameters in learn_config
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=learn_config.loss, optimizer=optimizer, metrics=['mse', 'mae'],
                  weighted_metrics=[])

    logger.info(f"model summary: {model.summary()}")

    callbacks = [stepdecay]

    logger.info(f"Number of electric current features: {number_of_bisa_neurons}")
    #a = [x_train.iloc[:, :-number_of_bisa_neurons]]
    #b = [x_train.iloc[:, i] for i in range(x_train.shape[1] - number_of_bisa_neurons, x_train.shape[1])]

    if neural_net_variant == 0:
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            sample_weight=weightings_train,
            validation_data=(x_test, y_test, weightings_test),
            callbacks=callbacks)
    elif neural_net_variant == 1:
        history = model.fit(
            [timestamps_training, timestamps_training, x_train],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            sample_weight=weightings_train,
            validation_data=([timestamps_val, timestamps_val, x_test], y_test, weightings_test),
            callbacks=callbacks)
    elif neural_net_variant == 2:
        history = model.fit(
            [timestamps_training, timestamps_training, x_train],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            sample_weight=weightings_train,
            validation_data=([timestamps_val, timestamps_val, x_test], y_test, weightings_test),
            callbacks=callbacks)
    elif neural_net_variant == 3:
        history = model.fit(
            [x_train.iloc[:, :-number_of_bisa_neurons]] + [x_train.iloc[:, i] for i in range(x_train.shape[1] - number_of_bisa_neurons, x_train.shape[1])],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            sample_weight=weightings_train,
            validation_data=(
                [x_test.iloc[:, :-number_of_bisa_neurons]] + [x_test.iloc[:, i] for i in range(x_test.shape[1] - number_of_bisa_neurons, x_test.shape[1])],
                y_test, weightings_test),
            callbacks=callbacks)
    else:
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            sample_weight=weightings_train,
            validation_data=(x_test, y_test, weightings_test),
            callbacks=callbacks)

    return (model, history)
