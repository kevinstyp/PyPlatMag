import tensorflow as tf

# TODO: Go through all files in customs package and refactor them
class CustomInitializer(tf.keras.initializers.Initializer):
    def __init__(self, mean, number_of_biot_savart_neurons=1):
        #self.mean = mean
        self.number_of_biot_savart_neurons = number_of_biot_savart_neurons
        self.number_of_biot_savart_outputs = tf.multiply(number_of_biot_savart_neurons, 3)

    def __call__(self, shape, dtype=None):
        # Must return a tensor of given shape
        zeros = tf.zeros(shape=shape, dtype=dtype)
        tf.print("zeros: ", zeros)
        tf.print("dtype: ", dtype)
        #zeros[0] = tf.constant([1.], dtype=dtype)
        tf.print("shape: ", shape)
        tf.print("*shape: ", *shape)
        tf.print("shape[0]: ", shape[0])
        tf.print("shape[1]: ", shape[1])
        tf.print("tf.subtract(shape[0], self.number_of_biot_savart_neurons): ", tf.subtract(shape[0], self.number_of_biot_savart_outputs))
        upper_part = tf.eye(tf.subtract(shape[0], self.number_of_biot_savart_outputs), shape[1], dtype=dtype)
        tf.print("upper_part: ", upper_part)
        tf.print("upper_part.dtype: ", upper_part.dtype)
        for i in range(self.number_of_biot_savart_neurons):
            lower_part = tf.eye(3, shape[1], dtype=dtype)
            #tf.print("lower_part: ", lower_part)
            #tf.print("lower_part.dtype: ", lower_part.dtype)
            upper_part = tf.concat([upper_part, lower_part], axis=0)

        #result = tf.multiply(tf.eye(*shape, dtype=dtype), self.mean)
        tf.print("upper_part: ", upper_part)
        return upper_part

