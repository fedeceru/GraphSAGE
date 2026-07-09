"""
Baseline "Random" e "Raw features" per il dataset PPI, come descritti nel paper
GraphSAGE (Section 4, "Experimental set-up", e Appendix C).

- Random: per ciascuna delle 121 label GO, predizione Bernoulli indipendente con
  probabilita' pari al tasso di positivi osservato nel training set.
- Raw features: OneVsRestClassifier(SGDClassifier(loss="log")) -- la "logistic
  SGDClassifier... con impostazioni di default" descritta in Appendix C -- allenato
  sulle feature grezze dei nodi di training, valutato sui nodi di test.

Non modifica src/graphsage: riusa solo graphsage.utils.load_data per caricare
il dataset con lo stesso preprocessing (StandardScaler fit sul train) usato
dai training script ufficiali.
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
    pos_rate = y_train.mean(axis=0)  # per-label positive rate on train
    y_pred = (rng.rand(*y_test.shape) < pos_rate[None, :]).astype(int)
    return {
        "f1_micro": f1_score(y_test, y_pred, average="micro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }


def raw_features_baseline(X_train, y_train, X_test, y_test, seed=SEED):
    clf = OneVsRestClassifier(
        SGDClassifier(loss="log", random_state=seed), n_jobs=1
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
    # stessa gestione del nodo "dummy" usata dai training script (padding zero vec)
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
