
import tensorflow as tf

#from model_builder import build_network_goce
from training.model_builder import build_network_goce, build_network_goce_pinn
import training.customs.stepdecay as sd
from tensorflow.keras.callbacks import LearningRateScheduler

def goce_training(x_train, y_train, x_test, y_test, weightings_train, weightings_test, learn_config=None, neural_net_variant=0,
                  model_path=None, all_values_raw=None,
                  all_values_tq=None, timestamps_training=None, timestamps_val=None, number_of_bisa_neurons=None):
    # print(tf.config.threading.set_inter_op_parallelism_threads(4))
    print(tf.config.threading.get_inter_op_parallelism_threads())
    print(tf.config.threading.get_intra_op_parallelism_threads())
    print(tf.__version__)

    print("learn_config: ", learn_config)
    epochs = learn_config.epochs
    batch_size = learn_config.batch_size


    #tf.executing_eagerly()
    ###os.environ["TF_KERAS"] = '1'


    if neural_net_variant == 0:
        print("x_train: ", x_train.shape)
        model = build_network_goce(input_shape=(x_train.shape[1]))

    elif neural_net_variant == 1:
        init_value = 1.0#1e-4
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
        # TODO: x_train is not a list anymore: the last {number_of_bisa_neurons} elements are the electric current data
        model = build_network_goce_pinn(input_shape=(x_train[0].shape[1]), batch_size=batch_size,
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
        print("model_path: ", model_path)
        model.load_weights(model_path)

    for layer in model.layers[2:4]:
        print("layer: ", layer)
        print("layer.get_weights(): ", layer.get_weights())
        curr_weights = layer.get_weights()
    if neural_net_variant != 3 and neural_net_variant != 4:
        print("model.layers[2].get_weights()[0]: ", model.layers[2].get_weights()[0])
        print("model.layers[2].get_weights()[0][0]: ", model.layers[2].get_weights()[0][0])
        # curr_weights = model.layers[2].get_weights()
        # curr_weights[0][0] = 0.4
        # model.layers[2].set_weights(curr_weights)

    schedule = sd.StepDecay(initAlpha=learn_config.learning_rate, factor=0.5, dropEvery=learn_config.drop_every,
                            first_extra=learn_config.first_extra)
    stepdecay = LearningRateScheduler(schedule)
    if neural_net_variant == 3:
        print("Using clipnorm.")
        # TODO: optimizer is one of the parameters in learn_config
        # TODO: Is clipnorm still needed?
        optimizer = tf.keras.optimizers.Adam(learning_rate=learn_config.learning_rate, clipnorm=0.1)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learn_config.learning_rate)


    model.compile(loss=learn_config.loss, optimizer=optimizer, metrics=['mse', 'mae'])

    #print("input layers: %d" % x_train.shape[1])
    print(model.summary())

    callbacks = [stepdecay]

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
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            sample_weight=weightings_train,
            validation_data=(x_test, y_test, weightings_test),
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