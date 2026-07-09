"""
Command-line entry point for unsupervised GraphSAGE: trains one of the
aggregator variants (mean / GCN / LSTM / pooling), or the Node2Vec/DeepWalk
baseline (``--model n2v``), on the skip-gram-style link-prediction objective
of Eq. 1 in the paper -- as opposed to ``supervised_train.py``'s direct
classification loss.

Typical usage (see also scripts/run_ppi_experiments.ps1 for a full example):

    python -m graphsage.unsupervised_train \\
        --train_prefix data/ppi/ppi --model graphsage_mean --gpu 0

If ``--save_embeddings`` (default True), writes the learned embeddings for
*every* node (train+val+test) to ``val.npy``/``val.txt`` in a run-specific
directory under ``--base_log_dir`` (see ``log_dir()`` below); these are what
``scripts/eval_unsupervised.py`` later trains a classifier on.

Every run also writes ``metrics.csv`` (step, epoch, train/val loss and MRR)
into the same directory -- a structured counterpart to the console log,
meant for programmatic plotting (e.g. notebook.ipynb's training-dynamics
charts) without having to parse printed text. This is purely an added
side-effect (like the existing TensorBoard summary writer below) and never
changes what gets computed.

Passing ``--embedding_snapshot_steps`` (default: empty, i.e. disabled)
additionally dumps embeddings for every node at the given training steps --
e.g. ``--embedding_snapshot_steps 0,100,1000,5000`` -- under
``<log_dir>/snapshots/step_<N>/``, on top of the usual final embeddings.
This is what notebook.ipynb's embedding-progression visuals (PCA/t-SNE
across training) consume; with the flag left empty, a run behaves exactly
like the original paper's script.
"""
from __future__ import division
from __future__ import print_function

# stdlib
import csv
import os
import time

# third-party
import tensorflow as tf
import numpy as np

# local
from graphsage.models import SampleAndAggregate, SAGEInfo, Node2VecModel
from graphsage.minibatch import EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'name of the object file that stores the training data. must be specified.')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 1, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('embedding_snapshot_steps', '',
        "Comma-separated step indices at which to additionally dump embeddings for every "
        "node during training (e.g. '0,100,1000,5000'), for visualizing how the embedding "
        "space evolves. Empty (default) disables this and matches the original script exactly.")
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def log_dir():
    """Run-specific output directory, e.g.
    ``<base_log_dir>/unsup-ppi/graphsage_mean_small_0.000010/`` -- this is
    also where the final embeddings (val.npy/val.txt) get written."""
    log_dir = FLAGS.base_log_dir + "/unsup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    """Quick MRR/loss evaluation on a random sample of `size` val edges --
    used for the periodic progress logging during training."""
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss, model.ranks, model.mrr],
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

def incremental_evaluate(sess, model, minibatch_iter, size):
    """Full MRR/loss evaluation over *all* val edges, processed in chunks of
    `size` (analogous to supervised_train.incremental_evaluate)."""
    t_test = time.time()
    finished = False
    val_losses = []
    val_mrrs = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                            feed_dict=feed_dict_val)
        val_losses.append(outs_val[0])
        val_mrrs.append(outs_val[2])
    return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)

def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    """Run the trained model in inference mode over *every* node in the
    graph (train+val+test, via `incremental_embed_feed_dict`, which uses the
    full-graph adjacency) and dump the resulting embeddings to
    `<out_dir>/val<mod>.npy` (float matrix) with the corresponding node ids
    in `<out_dir>/val<mod>.txt` (one id per line, same row order). This is
    the file pair that `scripts/eval_unsupervised.py` later reads to train a
    downstream classifier on top of the embeddings."""
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1], 
                            feed_dict=feed_dict_val)
        #ONLY SAVE FOR embeds1 because of planetoid
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i,:])
                nodes.append(edge[0])
                seen.add(edge[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)
    np.save(out_dir + name + mod + ".npy",  val_embeddings)
    with open(out_dir + name + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str,nodes)))

def construct_placeholders():
    """TensorFlow input placeholders: `batch1`/`batch2` hold the anchor and
    positive-context node ids for the current minibatch of edges (fed by
    `EdgeMinibatchIterator.next_minibatch_feed_dict()`); negative samples are
    drawn inside the graph itself (see `SampleAndAggregate._build` in
    models.py), not fed here."""
    # Define placeholders
    placeholders = {
        'batch1' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch2' : tf.placeholder(tf.int32, shape=(None), name='batch2'),
        # negative samples for all nodes in the batch
        'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def train(train_data, test_data=None):
    """Build the model for `FLAGS.model` and run the full training loop
    (FLAGS.epochs epochs over the training edges/random-walk pairs),
    periodically evaluating MRR on a validation sample, then -- unless
    disabled -- saving embeddings for every node."""
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders()
    minibatch = EdgeMinibatchIterator(G, 
            id_map,
            placeholders, batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            num_neg_samples=FLAGS.neg_sample_size,
            context_pairs = context_pairs)
    # See the matching comment in supervised_train.py: `adj_info` is a
    # tf.Variable so it can be swapped between train-only / full-graph
    # adjacency without rebuilding the graph.
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    # ---- model selection -----------------------------------------------
    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     concat=False,
                                     logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     identity_dim = FLAGS.identity_dim,
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'n2v':
        # DeepWalk/node2vec (p=q=1) baseline, reimplemented in TensorFlow so
        # its runtime is directly comparable to GraphSAGE (Appendix C);
        # doesn't use sample()/aggregate() at all, just a plain embedding
        # lookup table.
        model = Node2VecModel(placeholders, features.shape[0],
                                       minibatch.deg,
                                       #2x because graphsage uses concat
                                       nodevec_dim=2*FLAGS.dim_1,
                                       lr=FLAGS.learning_rate)
    else:
        raise Exception('Error: model name unrecognized.')

    # ---- TF session setup ------------------------------------------------
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

    # Structured, per-step counterpart to the console log below (step, epoch,
    # train/val loss and MRR) -- written unconditionally, same spirit as the
    # TensorBoard summary_writer above: an observational side effect that
    # doesn't change what gets computed, meant for programmatic plotting
    # (e.g. notebook.ipynb) without parsing printed text.
    metrics_fp = open(os.path.join(log_dir(), "metrics.csv"), "w")
    metrics_writer = csv.writer(metrics_fp)
    metrics_writer.writerow(["step", "epoch", "train_loss", "train_mrr", "val_loss", "val_mrr"])

    # ---- optional embedding snapshots (training-progression visuals) ------
    # See --embedding_snapshot_steps above: empty by default, so none of this
    # runs (and behavior is identical to the original script) unless opted in.
    remaining_snapshots = sorted({int(s) for s in FLAGS.embedding_snapshot_steps.split(",") if s.strip() != ""})

    def dump_embedding_snapshot(step, train_adj_info, val_adj_info):
        sess.run(val_adj_info.op)
        out_dir = os.path.join(log_dir(), "snapshots", "step_%05d" % step) + os.sep
        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, out_dir)
        sess.run(train_adj_info.op)
        print("Saved embedding snapshot at step %d -> %s" % (step, out_dir))

    # ---- training loop ----------------------------------------------------
    train_shadow_mrr = None
    shadow_mrr = None

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    if remaining_snapshots and remaining_snapshots[0] == 0:
        dump_embedding_snapshot(0, train_adj_info, val_adj_info)
        remaining_snapshots = remaining_snapshots[1:]

    for epoch in range(FLAGS.epochs):
        minibatch.shuffle() 

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all, 
                    model.mrr, model.outputs1], feed_dict=feed_dict)
            train_cost = outs[2]
            train_mrr = outs[5]
            if train_shadow_mrr is None:
                train_shadow_mrr = train_mrr#
            else:
                train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                val_cost, ranks, val_mrr, duration  = evaluate(sess, model, minibatch, size=FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost
            if shadow_mrr is None:
                shadow_mrr = val_mrr
            else:
                shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_mrr=", "{:.5f}".format(train_mrr),
                      "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr), # exponential moving average
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_mrr=", "{:.5f}".format(val_mrr),
                      "val_mrr_ema=", "{:.5f}".format(shadow_mrr), # exponential moving average
                      "time=", "{:.5f}".format(avg_time))
                metrics_writer.writerow([total_steps, epoch + 1, train_cost, train_mrr, val_cost, val_mrr])

            iter += 1
            total_steps += 1

            while remaining_snapshots and total_steps >= remaining_snapshots[0]:
                dump_embedding_snapshot(remaining_snapshots[0], train_adj_info, val_adj_info)
                remaining_snapshots = remaining_snapshots[1:]

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
                break

    metrics_fp.close()

    if remaining_snapshots:
        # requested a step beyond what one epoch actually covers -- snapshot
        # wherever training ended so the sequence still has a "final" point
        dump_embedding_snapshot(total_steps, train_adj_info, val_adj_info)

    print("Optimization Finished!")
    if FLAGS.save_embeddings:
        sess.run(val_adj_info.op)

        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir())

        if FLAGS.model == "n2v":
            # DeepWalk/node2vec has no inductive mechanism: embeddings are
            # looked up by id, so a never-seen node simply has no embedding.
            # To let it "handle" test-time nodes at all (Appendix C, "Notes
            # on the DeepWalk implementation"), we now run a *second* round
            # of training restricted to fresh random walks touching val/test
            # nodes, while freezing (stop_gradient) the embeddings of nodes
            # already optimized above -- otherwise this second round would
            # keep drifting the training embeddings the classifier was
            # already fit on. This whole block is specific to the n2v
            # baseline and doesn't run for any GraphSAGE variant.
            # stopping the gradient for the already trained nodes
            train_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if not G.node[n]['val'] and not G.node[n]['test']],
                    dtype=tf.int32)
            test_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if G.node[n]['val'] or G.node[n]['test']], 
                    dtype=tf.int32)
            update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(test_ids))
            no_update_nodes = tf.nn.embedding_lookup(model.context_embeds,tf.squeeze(train_ids))
            update_nodes = tf.scatter_nd(test_ids, update_nodes, tf.shape(model.context_embeds))
            no_update_nodes = tf.stop_gradient(tf.scatter_nd(train_ids, no_update_nodes, tf.shape(model.context_embeds)))
            model.context_embeds = update_nodes + no_update_nodes
            sess.run(model.context_embeds)

            # run random walks
            from graphsage.utils import run_random_walks
            nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
            start_time = time.time()
            pairs = run_random_walks(G, nodes, num_walks=50)
            walk_time = time.time() - start_time

            test_minibatch = EdgeMinibatchIterator(G, 
                id_map,
                placeholders, batch_size=FLAGS.batch_size,
                max_degree=FLAGS.max_degree, 
                num_neg_samples=FLAGS.neg_sample_size,
                context_pairs = pairs,
                n2v_retrain=True,
                fixed_n2v=True)
            
            start_time = time.time()
            print("Doing test training for n2v.")
            test_steps = 0
            for epoch in range(FLAGS.n2v_test_epochs):
                test_minibatch.shuffle()
                while not test_minibatch.end():
                    feed_dict = test_minibatch.next_minibatch_feed_dict()
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all, 
                        model.mrr, model.outputs1], feed_dict=feed_dict)
                    if test_steps % FLAGS.print_every == 0:
                        print("Iter:", '%04d' % test_steps, 
                              "train_loss=", "{:.5f}".format(outs[1]),
                              "train_mrr=", "{:.5f}".format(outs[-2]))
                    test_steps += 1
            train_time = time.time() - start_time
            save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir(), mod="-test")
            print("Total time: ", train_time+walk_time)
            print("Walk time: ", walk_time)
            print("Train time: ", train_time)

    

def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix, load_walks=True)
    print("Done loading training data..")
    train(train_data)

if __name__ == '__main__':
    tf.app.run()
