import tensorflow as tf


# TODO: Go through all files in customs package and refactor them
# TODO: mean is not utilized?
class CustomInitializer(tf.keras.initializers.Initializer):
    def __init__(self, mean, number_of_biot_savart_neurons=1):
        self.mean = mean
        self.number_of_biot_savart_neurons = number_of_biot_savart_neurons
        self.number_of_biot_savart_outputs = tf.multiply(number_of_biot_savart_neurons, 3).numpy()

    # For the serialization of this custom Initializer
    def get_config(self):
        config = {"mean": self.mean,
                  "number_of_biot_savart_neurons": self.number_of_biot_savart_neurons,
                  }
        base_config = super(CustomInitializer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __call__(self, shape, dtype=None):
        # Must return a tensor of given shape
        upper_part = tf.eye(tf.subtract(shape[0], self.number_of_biot_savart_outputs), shape[1], dtype=dtype)
        for i in range(self.number_of_biot_savart_neurons):
            lower_part = tf.eye(3, shape[1], dtype=dtype)
            upper_part = tf.concat([upper_part, lower_part], axis=0)

        # Factor in self.mean
        result = tf.multiply(upper_part, self.mean)
        return result
