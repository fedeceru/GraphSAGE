"""
Compila i risultati della riproduzione PPI (baseline + 4 aggregatori GraphSAGE,
supervisionato e non supervisionato) in una tabella Markdown confrontata con
le colonne PPI della Table 1 del paper.
"""
from __future__ import division, print_function

import json
import os
import re

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

MODELS = [
    ("graphsage_mean", "GraphSAGE-mean"),
    ("gcn", "GraphSAGE-GCN"),
    ("graphsage_seq", "GraphSAGE-LSTM"),
    ("graphsage_maxpool", "GraphSAGE-pool"),
]

# Valori riportati nel paper (Table 1, colonne PPI)
PAPER = {
    "Random": (0.396, 0.396),
    "Raw features": (0.422, 0.422),
    "GraphSAGE-GCN": (0.465, 0.500),
    "GraphSAGE-mean": (0.486, 0.598),
    "GraphSAGE-LSTM": (0.482, 0.612),
    "GraphSAGE-pool": (0.502, 0.600),
}


def parse_test_stats(path):
    with open(path) as fp:
        content = fp.read()
    m = re.search(r"f1_micro=([\d.]+)", content)
    return float(m.group(1)) if m else None


def main():
    baseline_path = os.path.join(RESULTS_DIR, "baseline_ppi.json")
    with open(baseline_path) as fp:
        baseline = json.load(fp)

    rows = {}
    rows["Random"] = (baseline["random"]["f1_micro"], baseline["random"]["f1_micro"])
    rows["Raw features"] = (baseline["raw_features"]["f1_micro"], baseline["raw_features"]["f1_micro"])

    for model_flag, display_name in MODELS:
        unsup_path = os.path.join(RESULTS_DIR, "eval_unsup_%s.json" % model_flag)
        with open(unsup_path) as fp:
            unsup_f1 = json.load(fp)["f1_micro"]

        sup_path = os.path.join(LOGS_DIR, "sup-ppi", "%s_small_0.0100" % model_flag, "test_stats.txt")
        sup_f1 = parse_test_stats(sup_path)

        rows[display_name] = (unsup_f1, sup_f1)

    lines = []
    lines.append("# Risultati riprodotti — PPI (confronto con Table 1 del paper)\n")
    lines.append("| Name | Unsup. F1 (riprodotto) | Unsup. F1 (paper) | Sup. F1 (riprodotto) | Sup. F1 (paper) |")
    lines.append("|---|---|---|---|---|")
    order = ["Random", "Raw features", "GraphSAGE-GCN", "GraphSAGE-mean", "GraphSAGE-LSTM", "GraphSAGE-pool"]
    for name in order:
        unsup, sup = rows[name]
        paper_unsup, paper_sup = PAPER[name]
        unsup_s = "%.3f" % unsup if unsup is not None else "n/a"
        sup_s = "%.3f" % sup if sup is not None else "n/a"
        lines.append("| %s | %s | %.3f | %s | %.3f |" % (name, unsup_s, paper_unsup, sup_s, paper_sup))

    lines.append("")
    lines.append("Nota: run singola per variante con gli iperparametri di default del codice "
                  "(non lo sweep completo di Appendix C); dataset PPI pubblico da "
                  "http://snap.stanford.edu/graphsage/.")

    out_path = os.path.join(RESULTS_DIR, "ppi_results.md")
    with open(out_path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWrote", out_path)


if __name__ == "__main__":
    main()
