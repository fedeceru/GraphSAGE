"""
Small helpers for creating TensorFlow weight variables with a given random
initialization scheme. Used throughout ``layers.py``, ``aggregators.py`` and
``prediction.py`` instead of calling ``tf.Variable`` directly, so every
learnable weight in the codebase is initialized consistently.
"""
import tensorflow as tf
import numpy as np

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/tkipf/gcn
# which is under an identical MIT license as GraphSAGE


def uniform(shape, scale=0.05, name=None):
    """Weight matrix of the given ``shape``, drawn uniformly from [-scale, scale]."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) uniform initialization.

    This is the default initializer used for every aggregator's weight
    matrices: the sampling range is scaled by ``sqrt(6 / (fan_in + fan_out))``
    so that the variance of activations stays roughly constant across layers.
    """
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All-zeros variable, used for bias terms."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All-ones variable."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
