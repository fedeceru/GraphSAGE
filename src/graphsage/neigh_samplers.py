"""
Neighborhood sampling: the ``N(v)`` in the paper's Algorithm 1. Instead of
aggregating over a node's *entire* neighbor set (which would make the
per-batch cost unbounded), GraphSAGE draws a fixed-size random sample at
every layer -- see "Neighborhood definition" in Section 3.1 of the paper.
"""
from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


class UniformNeighborSampler(Layer):
    """Draws ``num_samples`` neighbors per node, uniformly and with
    replacement, from a precomputed padded adjacency table.

    ``adj_info`` is a dense ``[num_nodes, max_degree]`` int32 tensor (built by
    ``minibatch.py``'s ``construct_adj``) where row ``i`` lists (a
    fixed-length, padded/subsampled view of) node ``i``'s neighbors. Sampling
    without replacement isn't needed here because that padding/subsampling
    step already randomizes which neighbors are available; we simply shuffle
    the row and slice off the first ``num_samples`` entries, which is
    equivalent to sampling with replacement when ``num_samples`` exceeds the
    node's true degree.
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists
