"""
"Random" and "Raw features" baselines for the PPI dataset, as described in the
GraphSAGE paper (Section 4, "Experimental set-up", and Appendix C).

- Random: for each of the 121 GO labels, an independent Bernoulli prediction
  with probability equal to the positive rate observed on the training set.
- Raw features: OneVsRestClassifier(SGDClassifier(loss="log")) -- the "logistic
  SGDClassifier... with default settings" described in Appendix C -- trained
  on the raw features of the training nodes, evaluated on the test nodes.

  "Default settings" meant something different in 2017 than it does with the
  scikit-learn version pinned in requirements.txt (1.0.2): SGDClassifier's
  default was a fixed `n_iter=5` (5 passes over the data, no early stopping)
  until that parameter was deprecated in favor of `max_iter`/`tol`, with the
  default eventually becoming `max_iter=1000, tol=1e-3` (i.e. up to 1000
  passes, but usually stopped early once the loss improvement falls below
  `tol` for 5 consecutive epochs) -- a materially different optimizer, not
  just a version bump. `ERA_MATCHED_SGD_PARAMS` below reproduces the
  paper-era fixed-5-epoch behavior explicitly (`max_iter=5, tol=None` --
  `tol=None` is scikit-learn's documented way to disable the tol-based
  early-stopping check so `max_iter` passes always run in full) rather than
  relying on whatever "default" happens to mean in the installed
  scikit-learn version.

Does not modify src/graphsage: it only reuses graphsage.utils.load_data to
load the dataset with the same preprocessing (StandardScaler fit on train)
used by the official training scripts.
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

# See module docstring for rationale.
ERA_MATCHED_SGD_PARAMS = dict(max_iter=5, tol=None)


def get_split_labels(G, id_map, class_map, feats):
    train_ids = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    test_ids = [n for n in G.nodes() if G.node[n]["test"]]

    train_idx = [id_map[n] for n in train_ids]
    test_idx = [id_map[n] for n in test_ids]

    y_train = np.array([class_map[n] for n in train_ids])
    y_test = np.array([class_map[n] for n in test_ids])

    X_train = feats[train_idx]
    X_test = feats[test_idx]
    return X_train, y_train, X_test, y_test


def random_baseline(y_train, y_test, seed=SEED):
    rng = np.random.RandomState(seed)
    pos_rate = y_train.mean(axis=0)
    y_pred = (rng.rand(*y_test.shape) < pos_rate[None, :]).astype(int)
    return {
        "f1_micro": f1_score(y_test, y_pred, average="micro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }


def raw_features_baseline(X_train, y_train, X_test, y_test, seed=SEED):
    clf = OneVsRestClassifier(
        SGDClassifier(loss="log", random_state=seed, **ERA_MATCHED_SGD_PARAMS), n_jobs=1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "f1_micro": f1_score(y_test, y_pred, average="micro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_prefix", default="data/ppi/ppi")
    parser.add_argument("--out", default="results/baseline_ppi.json")
    args = parser.parse_args()

    print("Loading PPI data...")
    G, feats, id_map, _, class_map = load_data(args.train_prefix, load_walks=False)
    # same "dummy" node handling used by the training scripts (zero-vector padding)
    feats = np.vstack([feats, np.zeros((feats.shape[1],))])

    X_train, y_train, X_test, y_test = get_split_labels(G, id_map, class_map, feats)
    print("train nodes:", X_train.shape[0], "test nodes:", X_test.shape[0])

    print("Computing Random baseline...")
    random_res = random_baseline(y_train, y_test)
    print("Random:", random_res)

    print("Computing Raw features baseline...")
    raw_res = raw_features_baseline(X_train, y_train, X_test, y_test)
    print("Raw features:", raw_res)

    results = {"random": random_res, "raw_features": raw_res}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(results, fp, indent=2)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
