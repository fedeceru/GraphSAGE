"""
Evaluation of GraphSAGE's unsupervised embeddings on PPI, following the
procedure described in Appendix C of the paper ("Logistic regression model"):
a scikit-learn logistic SGDClassifier (era-matched settings -- see
ERA_MATCHED_SGD_PARAMS below) is trained only on the training nodes, and
evaluated on the test nodes -- with no fine-tuning on the embeddings
generated for the test nodes.

Consumes the output of `graphsage.unsupervised_train` (which saves `val.npy` /
`val.txt` in the log dir, containing the embeddings and ids of ALL nodes,
train+val+test) and does not modify src/graphsage.

``--split`` picks which held-out node set the fitted classifier is scored on:
``test`` (default) is the number that gets reported (matches the original,
single-run pipeline's behavior exactly). ``val`` is used by
``scripts/sweep_ppi_experiments.py`` to select the best (learning_rate,
model_size) config per Appendix C's "performance on a validation set"
criterion, without ever looking at test-set performance while choosing a
config -- the classifier is still fit on train embeddings only either way.
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graphsage.utils import load_data  # noqa: E402

SEED = 123

# Matches scripts/baseline_ppi.py's ERA_MATCHED_SGD_PARAMS (see its docstring):
# reproduces scikit-learn's pre-0.19 SGDClassifier default (fixed 5 passes,
# no early stopping) rather than the installed version's current default,
# so this evaluation classifier behaves like the one Appendix C describes.
ERA_MATCHED_SGD_PARAMS = dict(max_iter=5, tol=None)


def load_embeddings(embed_dir):
    emb = np.load(os.path.join(embed_dir, "val.npy"))
    with open(os.path.join(embed_dir, "val.txt")) as fp:
        ids = [line.strip() for line in fp if line.strip() != ""]
    # PPI node ids are integers (json numeric keys -> read as int in load_data)
    ids = [int(i) for i in ids]
    return emb, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_prefix", default="data/ppi/ppi")
    parser.add_argument("--embed_dir", required=True,
                         help="folder with val.npy/val.txt produced by unsupervised_train.py")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                         help="which held-out node set to score the classifier on "
                              "(default: test, the number reported in results/). "
                              "Use 'val' for hyperparameter selection.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("Loading PPI graph/labels...")
    G, _, id_map, _, class_map = load_data(args.train_prefix, load_walks=False)

    print("Loading embeddings from", args.embed_dir)
    emb, ids = load_embeddings(args.embed_dir)
    id_to_row = {node_id: row for row, node_id in enumerate(ids)}

    train_ids = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"] and n in id_to_row]
    eval_ids = [n for n in G.nodes() if G.node[n][args.split] and n in id_to_row]

    X_train = np.array([emb[id_to_row[n]] for n in train_ids])
    y_train = np.array([class_map[n] for n in train_ids])
    X_test = np.array([emb[id_to_row[n]] for n in eval_ids])
    y_test = np.array([class_map[n] for n in eval_ids])

    print("train nodes:", X_train.shape[0], "%s nodes:" % args.split, X_test.shape[0])

    clf = OneVsRestClassifier(SGDClassifier(loss="log", random_state=SEED, **ERA_MATCHED_SGD_PARAMS), n_jobs=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    result = {
        "f1_micro": f1_score(y_test, y_pred, average="micro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }
    print(result)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(result, fp, indent=2)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
