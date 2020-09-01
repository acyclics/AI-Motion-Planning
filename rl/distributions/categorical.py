import tensorflow as tf


class CategoricalPd:
    """
        Args:
            logits: a tensor of logits outputted from a neural network
            x: the sampled argmax action index
    """
    def mode(self, logits):
        return tf.argmax(logits, axis=-1)

    def mean(self, logits):
        return tf.nn.softmax(logits)

    def neglogp(self, logits, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x_shape_list = x.shape.as_list()
            logits_shape_list = logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

            x = tf.one_hot(x, logits.get_shape().as_list()[-1])
        else:
            # already encoded
            assert x.shape.as_list() == logits.shape.as_list()
        return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=x)

    def kl(self, logits, other_logits):
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        a1 = other_logits - tf.reduce_max(other_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)

    def entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    def sample(self, logits):
        u = tf.random.uniform(tf.shape(logits), dtype=logits.dtype)
        return tf.argmax(logits - tf.math.log(-tf.math.log(u)), axis=-1)
        