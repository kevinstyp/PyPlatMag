import logging

import tensorflow as tf

from training.customs.custom_initializer import CustomInitializer
from training.model_build_util import biot_savart_input

logger = logging.getLogger(__name__)


def build_network_goce(input_shape):
    inputs_hk = tf.keras.Input(shape=input_shape)
    dense1 = tf.keras.layers.Dense(384, activation='elu', kernel_regularizer=tf.keras.regularizers.L1(l1=0.01))(inputs_hk)
    dense2 = tf.keras.layers.Dense(128, activation='elu', kernel_regularizer=tf.keras.regularizers.L1(l1=0.01))(dense1)
    outputs = tf.keras.layers.Dense(3)(dense2)
    return tf.keras.Model(inputs=inputs_hk, outputs=outputs)


def build_network_goce_pinn(input_shape, batch_size,
                            number_of_bisa_neurons=0, trainable_pinn=True):
    logger.debug(f"input_shape in build-method: {input_shape}")
    zeros_initializer = tf.keras.initializers.Zeros()
    custom_initializer = CustomInitializer(mean=1., number_of_biot_savart_neurons=number_of_bisa_neurons)

    # Dense part of the network for the non-electric-current data / housekeeping data
    inputs_x = tf.keras.Input(shape=input_shape)
    dense1 = tf.keras.layers.Dense(384, activation='elu')(inputs_x)  # 192 384 #512 #kernel_regularizer='l2'
    dense2 = tf.keras.layers.Dense(128, activation='elu')(dense1)  # 96 128 #192
    dense3 = tf.keras.layers.Dense(3)(dense2)

    inputs_list = []
    concatenation_list = []
    inputs_list.append(inputs_x)
    concatenation_list.append(dense3)

    # Currents/MTQ Handling
    for i in range(number_of_bisa_neurons):
        inputs_current, output_current = biot_savart_input(name=str(i), batch_size_input=batch_size, mean_init=4.0,
                                                           mean_init_2=0.0,
                                                  trainable_init=trainable_pinn)
        inputs_list.append(inputs_current)
        concatenation_list.append(output_current)

    # Concatenation and final output
    pre_final = tf.keras.layers.Concatenate()(concatenation_list)
    outputs = tf.keras.layers.Dense(3, trainable=False, kernel_initializer=custom_initializer,
                                    bias_initializer=zeros_initializer)(pre_final)

    model = tf.keras.Model(inputs=inputs_list, outputs=outputs)
    logger.debug(f"model.summary(): {model.summary()}")
    return model
