"""
Compiles the PPI reproduction results (baselines + 4 GraphSAGE aggregators,
supervised and unsupervised) into a Markdown table compared against the PPI
columns of Table 1 in the paper.

If ``results/best_hparams_ppi.json`` exists (written by
``scripts/sweep_ppi_experiments.py``, the Appendix C learning-rate x
model-size sweep with validation-based selection), the table is built from
each variant's *validation-selected* configuration instead of the single
fixed-hyperparameter run -- i.e. from the log directory
``sup-ppi/<model>_<size>_<lr>/`` that manifest entry actually points to,
rather than always assuming ``<model>_small_0.0100/``. A variant missing
from the manifest (e.g. the sweep was only run for some models) falls back
to the single-run path, so partial sweeps degrade gracefully instead of
erroring.
"""
from __future__ import division, print_function

import json
import os
import re

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
BEST_HPARAMS_PATH = os.path.join(RESULTS_DIR, "best_hparams_ppi.json")

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

    best_hparams = {}
    if os.path.exists(BEST_HPARAMS_PATH):
        with open(BEST_HPARAMS_PATH, encoding="utf-8") as fp:
            best_hparams = json.load(fp)

    rows = {}
    rows["Random"] = (baseline["random"]["f1_micro"], baseline["random"]["f1_micro"])
    rows["Raw features"] = (baseline["raw_features"]["f1_micro"], baseline["raw_features"]["f1_micro"])

    swept_flags = {name: False for _, name in MODELS}
    for model_flag, display_name in MODELS:
        model_manifest = best_hparams.get(model_flag, {})

        # Unsupervised: sweep_ppi_experiments.py writes the validation-selected
        # winner's *test* score to this exact path (same filename the
        # single-run pipeline uses), so no path needs to change here -- it's
        # already the right file whether or not a sweep has been run.
        unsup_path = os.path.join(RESULTS_DIR, "eval_unsup_%s.json" % model_flag)
        with open(unsup_path, encoding="utf-8") as fp:
            unsup_f1 = json.load(fp)["f1_micro"]

        # Supervised: unlike the unsupervised case, each sweep candidate gets
        # its own log directory (sup-ppi/<model>_<size>_<lr>/), so the
        # validation-selected winner's directory has to be looked up via the
        # manifest instead of the fixed "<model>_small_0.0100/" path.
        sup_entry = model_manifest.get("supervised")
        if sup_entry is not None:
            sup_dir_name = "%s_%s_%0.4f" % (model_flag, sup_entry["model_size"], sup_entry["learning_rate"])
            swept_flags[display_name] = True
        else:
            sup_dir_name = "%s_small_0.0100" % model_flag
        sup_path = os.path.join(LOGS_DIR, "sup-ppi", sup_dir_name, "test_stats.txt")
        sup_f1 = parse_test_stats(sup_path)

        if "unsupervised" in model_manifest:
            swept_flags[display_name] = True

        rows[display_name] = (unsup_f1, sup_f1)

    lines = []
    lines.append("# Reproduced results — PPI (comparison with Table 1 of the paper)\n")
    lines.append("| Name | Unsup. F1 (reproduced) | Unsup. F1 (paper) | Sup. F1 (reproduced) | Sup. F1 (paper) | Selection |")
    lines.append("|---|---|---|---|---|---|")
    order = ["Random", "Raw features", "GraphSAGE-GCN", "GraphSAGE-mean", "GraphSAGE-LSTM", "GraphSAGE-pool"]
    for name in order:
        unsup, sup = rows[name]
        paper_unsup, paper_sup = PAPER[name]
        unsup_s = "%.3f" % unsup if unsup is not None else "n/a"
        sup_s = "%.3f" % sup if sup is not None else "n/a"
        if name in ("Random", "Raw features"):
            selection = "n/a"
        else:
            selection = "Appendix C sweep" if swept_flags[name] else "single default run"
        lines.append("| %s | %s | %.3f | %s | %.3f | %s |" % (name, unsup_s, paper_unsup, sup_s, paper_sup, selection))

    lines.append("")
    if os.path.exists(os.path.join(RESULTS_DIR, "scaled_run_manifest.json")):
        lines.append("Note: \"Appendix C sweep\" rows here come from the **scaled, 2-hour-budget** "
                      "sweep (scripts/run_ppi_experiments_scaled.py), not the full 48-run grid -- "
                      "2 learning rates x \"small\" size only, not 3 x {small, big}. See "
                      "results/scaled_run_manifest.json and README.md's \"Scaled local "
                      "reproduction\" section for exactly what ran.")
    elif any(swept_flags.values()):
        lines.append("Note: \"Appendix C sweep\" rows use the learning_rate/model_size selected by "
                      "scripts/sweep_ppi_experiments.py on validation performance (matching Appendix "
                      "C's hyperparameter selection procedure); \"single default run\" rows are still "
                      "the code's fixed defaults (run scripts/sweep_ppi_experiments.py for that "
                      "variant to replace them). Public PPI dataset from http://snap.stanford.edu/graphsage/.")
    else:
        lines.append("Note: a single run per variant with the code's default hyperparameters "
                      "(not the full sweep from Appendix C -- run scripts/sweep_ppi_experiments.py "
                      "and re-run this script to use it); public PPI dataset from "
                      "http://snap.stanford.edu/graphsage/.")

    out_path = os.path.join(RESULTS_DIR, "ppi_results.md")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWrote", out_path)


if __name__ == "__main__":
    main()
