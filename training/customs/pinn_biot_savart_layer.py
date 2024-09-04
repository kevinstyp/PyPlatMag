import tensorflow as tf
import tensorflow.keras as keras


class BiotSavartLayer(keras.layers.Layer):
    def __init__(self, output_dim=(3,), metricname="interpol", units=3, mean_init=1.0, mean_init_2=1.0, trainable_init=True,
                 alt_batch=100, init='random_normal', name="bisa", input_dim=1, verbose=False, **kwargs):
        # print("Initializing.")

        self.output_dim = output_dim
        # Call to super method
        super(BiotSavartLayer, self).__init__(**kwargs)
        # Define number of units: expected 1
        self.units = units
        self.init = init
        self.verbose = verbose
        self.metricname = metricname
        self.alt_batch = alt_batch
        # print("Initialized.")
        print("self.compute_dtype: ", self.compute_dtype)
        print("self.dtype: ", self.dtype)

        w_1_init = tf.random_normal_initializer(mean=mean_init, stddev=0.5)
        self.w_1 = tf.Variable(
            name=name + "_radius",
            initial_value=w_1_init(shape=(input_dim, units), dtype=self.dtype),
            trainable=trainable_init,
        )
        w_2_init = tf.random_normal_initializer(mean=mean_init_2, stddev=0.5)
        self.w_2 = tf.Variable(
            name=name + "_area_orth",
            initial_value=w_2_init(shape=(input_dim, 3), dtype=self.dtype),
            trainable=trainable_init,
        )
        print("self.w_1: ", self.w_1)
        print("self.w_2: ", self.w_2)

    # For the serialization of this custom Layer/Neuron
    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "units": self.units,
                  "init": self.init,
                  "verbose": self.verbose,
                  "metricname": self.metricname,
                  "alt_batch": self.alt_batch,
                  }
        base_config = super(BiotSavartLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        # with tf.GradientTape() as tape:
        if self.verbose:
            tf.print("inputs: ", inputs)
            tf.print("self.w_1 / radius: ", self.w_1)
            tf.print("self.w_2 / area_vector: ", self.w_2)
        # r_vector = tf.matmul(inputs, self.w_1)
        r_vector = self.w_1
        area_orth_vector = self.w_2
        if self.verbose:
            print("r_vector: ", r_vector)
            tf.print("r_vector: ", r_vector)
        scaling_3 = tf.math.pow(tf.norm(r_vector, axis=1), 3)
        scaling_5 = tf.math.pow(tf.norm(r_vector, axis=1), 5)
        if self.verbose:
            tf.print("scaling_3: ", scaling_3)
            tf.print("scaling_5: ", scaling_5)
            # tf.print("scaling_2: ", scaling_2)
            tf.print("r_vector[0,0]: ", r_vector[0, 0])
            tf.print("r_vector[0,1]: ", r_vector[0, 1])
            tf.print("r_vector[0,2]: ", r_vector[0, 2])

        #print("inputs: ", inputs)
        # produce momentum vector m = I x A
        momentum = tf.multiply(inputs, area_orth_vector)

        m_times_r = tf.matmul(momentum, r_vector, transpose_b=True)
        r_times_m_times_r = tf.multiply(r_vector, m_times_r)

        if self.verbose:
            tf.print("momentum: ", momentum)
            tf.print("m_times_r: ", m_times_r)
            tf.print("r_times_m_times_r: ", r_times_m_times_r)


        ## calculate together:
        inner_term = tf.divide(
                        tf.scalar_mul(3, # 3 * ...
                                r_times_m_times_r # r * (m * r)

                         ),
                     scaling_5) # ... / r^5

        ## second momentum term: m / r^3
        second_term = tf.divide(momentum, scaling_3)
        # Put it together: 100 * (inner_term - second_term)
        output = tf.scalar_mul(100., tf.subtract(inner_term, second_term))

        if self.verbose:
            tf.print("inner_term: ", inner_term)
            tf.print("second_term: ", second_term)
            tf.print("output: ", output)

        if self.verbose:
            tf.print("output: ", output)
        return output
