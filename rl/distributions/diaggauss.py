import numpy as np
import tensorflow as tf


class DiagGaussian:

    def neglogp(self, mean, logstd, x):
        std = tf.exp(logstd)
        return 0.5 * tf.reduce_sum(tf.square((x - mean) / std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.dtypes.cast(tf.shape(x)[-1], dtype=tf.float32) \
               + tf.reduce_sum(logstd, axis=-1)

    def kl(self, mean, logstd, other_mean, other_logstd):
        std = tf.exp(logstd)
        other_std = tf.exp(other_logstd)
        return tf.reduce_sum(other_logstd - logstd + (tf.square(std) + tf.square(mean - other_mean)) / (2.0 * tf.square(other_std)) - 0.5, axis=-1)

    def entropy(self, logstd):
        return tf.reduce_sum(logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self, mean, logstd):
        std = tf.exp(logstd)
        return mean + std * tf.random.normal(tf.shape(mean))
