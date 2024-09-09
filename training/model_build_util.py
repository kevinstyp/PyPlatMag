import tensorflow as tf
from training.customs.pinn_biot_savart_layer import BiotSavartLayer

def biot_savart_input(name="1", batch_size_input=1, mean_init=1.0, mean_init_2=1.0, trainable_init=True, inputs_biotsavart=None):
    zeros_initializer = tf.keras.initializers.Zeros()
    negative_identity = tf.keras.initializers.Identity(gain=-1.)
    print("inputs_biotsavart before: ", inputs_biotsavart)
    if inputs_biotsavart is None:
        inputs_biotsavart = tf.keras.Input(shape=(1,), name="inputs_current_" + name, batch_size=batch_size_input)
    print("inputs_biotsavart after: ", inputs_biotsavart)
    # biot_savar = BiotSavartNeuronV3(name="bisa_" + name, metricname="current_" + name, mean_init=mean_init,
    #                                 trainable_init=trainable_init)(inputs_biotsavar)
    biot_savar = BiotSavartLayer(name="bisa_" + name, metricname="current_" + name, mean_init=mean_init, mean_init_2=mean_init_2,
                                    trainable_init=trainable_init)(inputs_biotsavart)
    output_biotsavar = tf.keras.layers.Dense(3, trainable=False, kernel_initializer=negative_identity,
                                       bias_initializer=zeros_initializer)(biot_savar)
    return inputs_biotsavart, output_biotsavar