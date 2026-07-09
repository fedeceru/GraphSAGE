"""
Compiles the PPI reproduction results (baselines + 4 GraphSAGE aggregators,
supervised and unsupervised) into a Markdown table compared against the PPI
columns of Table 1 in the paper.
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

# Values reported in the paper (Table 1, PPI columns)
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
    with open(baseline_path, encoding="utf-8") as fp:
        baseline = json.load(fp)

    rows = {}
    rows["Random"] = (baseline["random"]["f1_micro"], baseline["random"]["f1_micro"])
    rows["Raw features"] = (baseline["raw_features"]["f1_micro"], baseline["raw_features"]["f1_micro"])

    for model_flag, display_name in MODELS:
        unsup_path = os.path.join(RESULTS_DIR, "eval_unsup_%s.json" % model_flag)
        with open(unsup_path, encoding="utf-8") as fp:
            unsup_f1 = json.load(fp)["f1_micro"]

        sup_path = os.path.join(LOGS_DIR, "sup-ppi", "%s_small_0.0100" % model_flag, "test_stats.txt")
        sup_f1 = parse_test_stats(sup_path)

        rows[display_name] = (unsup_f1, sup_f1)

    lines = []
    lines.append("# Reproduced results — PPI (comparison with Table 1 of the paper)\n")
    lines.append("| Name | Unsup. F1 (reproduced) | Unsup. F1 (paper) | Sup. F1 (reproduced) | Sup. F1 (paper) |")
    lines.append("|---|---|---|---|---|")
    order = ["Random", "Raw features", "GraphSAGE-GCN", "GraphSAGE-mean", "GraphSAGE-LSTM", "GraphSAGE-pool"]
    for name in order:
        unsup, sup = rows[name]
        paper_unsup, paper_sup = PAPER[name]
        unsup_s = "%.3f" % unsup if unsup is not None else "n/a"
        sup_s = "%.3f" % sup if sup is not None else "n/a"
        lines.append("| %s | %s | %.3f | %s | %.3f |" % (name, unsup_s, paper_unsup, sup_s, paper_sup))

    lines.append("")
    lines.append("Note: a single run per variant with the code's default hyperparameters "
                  "(not the full sweep from Appendix C); public PPI dataset from "
                  "http://snap.stanford.edu/graphsage/.")

    out_path = os.path.join(RESULTS_DIR, "ppi_results.md")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWrote", out_path)


if __name__ == "__main__":
    main()
