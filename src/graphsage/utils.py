"""
Dataset loading and random-walk generation.

A dataset is identified by a ``prefix`` (e.g. ``data/ppi/ppi``) and expected
to provide four files: ``<prefix>-G.json`` (a networkx graph in
node-link format, with a boolean ``val``/``test`` attribute on every node
marking which split it belongs to), ``<prefix>-id_map.json`` (node id ->
integer row index into the feature matrix), ``<prefix>-class_map.json`` (node
id -> label, either a single class index or, for multi-label datasets like
PPI, a list of 0/1 values), and optionally ``<prefix>-feats.npy`` (the
feature matrix itself; if absent, the model falls back to identity/featureless
mode).

Pinned to ``networkx<=1.11`` because the graph API used here (``G.node[n]``,
``G.neighbors(n)`` returning a list, etc.) changed in networkx 2.x.
"""
from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5
N_WALKS=50

def load_data(prefix, normalize=True, load_walks=False):
    """Load a dataset given its file prefix.

    Returns:
        G: the networkx graph, with every node tagged 'val'/'test' (nodes
           missing these attributes are dropped -- see the networkx-version
           note below) and every edge tagged 'train_removed' (True if the
           edge touches a val/test node; used by minibatch.py to build the
           train-only adjacency table).
        feats: [num_nodes, feat_dim] array, or None in featureless mode. If
           `normalize`, features are standardized (zero mean/unit variance)
           using statistics computed on the training nodes only.
        id_map: dict, node id -> row index into `feats`.
        walks: list of (node, context_node) pairs from `<prefix>-walks.txt`,
           only populated if `load_walks=True` (used for the unsupervised
           objective's positive pairs).
        class_map: dict, node id -> label (int class index, or a list for
           multi-label datasets).
    """
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    # An edge is "train_removed" if it touches a val/test node -- this is the
    # flag minibatch.construct_adj() checks to build the train-only
    # adjacency table, so training nodes never see val/test nodes through
    # the graph structure.
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map

def run_random_walks(G, nodes, num_walks=N_WALKS):
    """Generate the (node, context_node) co-occurrence pairs used as
    "positive" examples for the unsupervised loss (Eq. 1), by running
    `num_walks` independent random walks of length `WALK_LEN` from every node
    in `nodes` and pairing the start node with every node visited along the
    way (self-pairs are skipped). Matches Appendix C: "we ran 50 random
    walks of length 5 from each node"."""
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
