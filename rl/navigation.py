import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import numpy as np
import tensorflow as tf

from rl.distributions.categorical import CategoricalPd


class normc_initializer(tf.keras.initializers.Initializer):
    def __init__(self, std=1.0, axis=0):
        self.std = std
        self.axis = axis
    def __call__(self, shape, dtype=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= self.std / np.sqrt(np.square(out).sum(axis=self.axis, keepdims=True))
        return tf.constant(out)


class Navigation(tf.keras.Model):

    def __init__(self, batch_size=1, training=True):
        super(Navigation, self).__init__()
        self.batch_size = batch_size
        self.training = training
        self.categoricalPd = CategoricalPd()

        self.core = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
            tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
            tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
            tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
            tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
        ])

        with tf.name_scope("xyyaw"):
            self.act_core = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            ])
            self.logits_x1 = tf.keras.layers.Dense(21,
                                              kernel_initializer=normc_initializer(0.01),
                                              name="logits_x1")
            self.logits_x2 = tf.keras.layers.Dense(21,
                                              kernel_initializer=normc_initializer(0.01),
                                              name="logits_x2")
            self.logits_y1 = tf.keras.layers.Dense(21,
                                              kernel_initializer=normc_initializer(0.01),
                                              name="logits_y1")
            self.logits_y2 = tf.keras.layers.Dense(21,
                                              kernel_initializer=normc_initializer(0.01),
                                              name="logits_y2")
            self.logits_w1 = tf.keras.layers.Dense(21,
                                              kernel_initializer=normc_initializer(0.01),
                                              name="logits_w1")
            self.logits_w2 = tf.keras.layers.Dense(21,
                                              kernel_initializer=normc_initializer(0.01),
                                              name="logits_w2")


        with tf.name_scope("value"):
            self.val_core = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            ])
            self.value = tf.keras.layers.Dense(1, name="value", activation=None, kernel_initializer=normc_initializer(1.0))

    @tf.function
    def call(self, obs):
        core_output = self.core(obs)

        with tf.name_scope("xyyaw"):
            act_core = self.act_core(core_output)
            logit_x1 = self.logits_x1(act_core)
            logit_x2 = self.logits_x2(act_core)
            logit_y1 = self.logits_y1(act_core)
            logit_y2 = self.logits_y2(act_core)
            logit_w1 = self.logits_w1(act_core)
            logit_w2 = self.logits_w2(act_core)
            sampled_x1 = self.categoricalPd.sample(logit_x1)
            sampled_x2 = self.categoricalPd.sample(logit_x2)
            sampled_y1 = self.categoricalPd.sample(logit_y1)
            sampled_y2 = self.categoricalPd.sample(logit_y2)
            sampled_w1 = self.categoricalPd.sample(logit_w1)
            sampled_w2 = self.categoricalPd.sample(logit_w2)
        
        with tf.name_scope('value'):
            val_core = self.val_core(core_output)
            value = self.value(val_core)[:, 0]   # flatten value otherwise it might broadcast
        
        actions = {
            'x1': sampled_x1,
            'x2': sampled_x2,
            'y1': sampled_y1,
            'y2': sampled_y2,
            'w1': sampled_w1,
            'w2': sampled_w2
        }

        logits = {
            'x1': logit_x1,
            'x2': logit_x2,
            'y1': logit_y1,
            'y2': logit_y2,
            'w1': logit_w1,
            'w2': logit_w2
        }

        neglogp = (
            self.categoricalPd.neglogp(logit_x1, sampled_x1) +
            self.categoricalPd.neglogp(logit_x2, sampled_x2) +
            self.categoricalPd.neglogp(logit_y1, sampled_y1) +
            self.categoricalPd.neglogp(logit_y2, sampled_y2) +
            self.categoricalPd.neglogp(logit_w1, sampled_w1) +
            self.categoricalPd.neglogp(logit_w2, sampled_w2)
        )

        entropy = (
            self.categoricalPd.entropy(logit_x1) +
            self.categoricalPd.entropy(logit_x2) +
            self.categoricalPd.entropy(logit_y1) +
            self.categoricalPd.entropy(logit_y2) +
            self.categoricalPd.entropy(logit_w1) +
            self.categoricalPd.entropy(logit_w2)
        )

        return actions, neglogp, entropy, value, logits
    
    def call_build(self):
        """
        IMPORTANT: This function has to be editted so that the below input features
        have the same shape as the actual inputs, otherwise the weights would not
        be restored properly.
        """
        self(np.zeros([self.batch_size, 16]))
 