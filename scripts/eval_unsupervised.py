"""
Valutazione degli embedding non supervisionati di GraphSAGE su PPI, seguendo
la procedura descritta in Appendix C del paper ("Logistic regression model"):
si allena una logistic SGDClassifier di scikit-learn (impostazioni di default)
solo sui nodi di training, e la si valuta sui nodi di test -- senza alcun
fine-tuning sugli embedding generati per i nodi di test.

Consuma l'output di `graphsage.unsupervised_train` (che salva `val.npy` /
`val.txt` nella log dir, contenenti gli embedding e gli id di TUTTI i nodi,
train+val+test) e non modifica src/graphsage.
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


def load_embeddings(embed_dir):
    emb = np.load(os.path.join(embed_dir, "val.npy"))
    with open(os.path.join(embed_dir, "val.txt")) as fp:
        ids = [line.strip() for line in fp if line.strip() != ""]
    # gli id dei nodi di PPI sono interi (json numeric keys -> letti come int in load_data)
    ids = [int(i) for i in ids]
    return emb, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_prefix", default="data/ppi/ppi")
    parser.add_argument("--embed_dir", required=True,
                         help="cartella con val.npy/val.txt prodotta da unsupervised_train.py")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("Loading PPI graph/labels...")
    G, _, id_map, _, class_map = load_data(args.train_prefix, load_walks=False)

    print("Loading embeddings from", args.embed_dir)
    emb, ids = load_embeddings(args.embed_dir)
    id_to_row = {node_id: row for row, node_id in enumerate(ids)}

    train_ids = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"] and n in id_to_row]
    test_ids = [n for n in G.nodes() if G.node[n]["test"] and n in id_to_row]

    X_train = np.array([emb[id_to_row[n]] for n in train_ids])
    y_train = np.array([class_map[n] for n in train_ids])
    X_test = np.array([emb[id_to_row[n]] for n in test_ids])
    y_test = np.array([class_map[n] for n in test_ids])

    print("train nodes:", X_train.shape[0], "test nodes:", X_test.shape[0])

    clf = OneVsRestClassifier(SGDClassifier(loss="log", random_state=SEED), n_jobs=1)
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
