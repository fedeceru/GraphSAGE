"""
Trains a single unsupervised GraphSAGE aggregator on PPI in one continuous
run, periodically dumping embeddings for every node, so notebook.ipynb can
show how the embedding space evolves over the course of training -- from
right after initialization to the fully trained state -- for the
best-performing unsupervised aggregator (GraphSAGE-pool, per
results/ppi_results.md).

This intentionally does NOT modify graphsage/unsupervised_train.py (which
stays a faithful, unmodified copy of the original paper's script). Instead
it reuses that module's already-registered command-line flags and its
construct_placeholders()/save_val_embeddings() helpers, and only adds a
snapshot mechanism around the same training loop shape. Because everything
runs in one TensorFlow session, model weights are never reset between
snapshots -- consecutive snapshots are genuine points along a single
training trajectory (same seeds, same minibatch order as a normal run of
`python -m graphsage.unsupervised_train --model graphsage_maxpool ...`).

Usage (same flags as unsupervised_train.py itself):
    python scripts/train_embedding_snapshots.py \
        --train_prefix data/ppi/ppi --model graphsage_maxpool \
        --model_size small --base_log_dir logs --gpu 0
"""
from __future__ import division, print_function

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import graphsage.unsupervised_train as ut  # noqa: E402  (registers all CLI flags)
from graphsage.models import SampleAndAggregate, SAGEInfo  # noqa: E402
from graphsage.minibatch import EdgeMinibatchIterator  # noqa: E402
from graphsage.neigh_samplers import UniformNeighborSampler  # noqa: E402
from graphsage.utils import load_data  # noqa: E402

FLAGS = ut.FLAGS

# Step indices (within the single training epoch, same units as the "Iter:"
# counter in unsupervised_train.py's own log output) at which to snapshot
# embeddings for every node. 0 = right after initialization, before any
# gradient step; the last value should match (or be close to) the total
# number of training batches in one epoch.
SNAPSHOT_STEPS = [0, 50, 200, 800, 3000, 8000, 17050]


def main(argv=None):
    if FLAGS.model != "graphsage_maxpool":
        raise Exception(
            "This script only wires up the 'graphsage_maxpool' branch "
            "(the best unsupervised aggregator in this reproduction); "
            "extend it if you need snapshots for another variant."
        )

    print("Loading training data..")
    G, features, id_map, walks, _ = load_data(FLAGS.train_prefix, load_walks=True)
    print("Done loading training data..")

    if features is not None:
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = walks if FLAGS.random_context else None
    placeholders = ut.construct_placeholders()
    minibatch = EdgeMinibatchIterator(
        G, id_map, placeholders,
        batch_size=FLAGS.batch_size,
        max_degree=FLAGS.max_degree,
        num_neg_samples=FLAGS.neg_sample_size,
        context_pairs=context_pairs,
    )
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    sampler = UniformNeighborSampler(adj_info)
    layer_infos = [
        SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
        SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
    ]
    model = SampleAndAggregate(
        placeholders, features, adj_info, minibatch.deg,
        layer_infos=layer_infos, aggregator_type="maxpool",
        model_size=FLAGS.model_size, identity_dim=FLAGS.identity_dim,
        logging=False,
    )

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

    out_root = ut.log_dir()
    snapshot_root = os.path.join(out_root, "snapshots")

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    def dump_snapshot(step):
        # switch to the full-graph adjacency so every node (train/val/test)
        # gets a real neighborhood to aggregate over, exactly as the final
        # save_val_embeddings() call in a normal run does
        sess.run(val_adj_info.op)
        out_dir = os.path.join(snapshot_root, "step_%05d" % step) + os.sep
        ut.save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, out_dir)
        sess.run(train_adj_info.op)
        print("Saved snapshot at step %d -> %s" % (step, out_dir))

    remaining = sorted(SNAPSHOT_STEPS)
    if remaining and remaining[0] == 0:
        dump_snapshot(0)
        remaining = remaining[1:]

    total_steps = 0
    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()
        while not minibatch.end():
            feed_dict = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders["dropout"]: FLAGS.dropout})
            sess.run([model.opt_op, model.loss], feed_dict=feed_dict)
            total_steps += 1

            while remaining and total_steps >= remaining[0]:
                dump_snapshot(remaining[0])
                remaining = remaining[1:]

            if not remaining:
                break
        if not remaining:
            break

    if remaining:
        # ran out of batches before reaching every requested step (e.g. the
        # requested final step was larger than one epoch) -- snapshot
        # whatever we ended at so the sequence still has a "final" point
        dump_snapshot(total_steps)

    print("Done. Snapshots written under", snapshot_root)


if __name__ == "__main__":
    tf.app.run()
