"""
Compiles the PPI reproduction results (baselines + 4 GraphSAGE aggregators,
supervised and unsupervised) into a Markdown table compared against the PPI
columns of Table 1 in the paper.

If ``results/best_hparams_ppi.json`` exists (written by
``scripts/sweep_ppi_experiments.py``, the Appendix C learning-rate x
model-size sweep with validation-based selection), the **supervised** half of
the table is built from each variant's *validation-selected* configuration
instead of the single fixed-hyperparameter run -- i.e. from the log
directory ``sup-ppi/<model>_<size>_<lr>/`` that manifest entry actually
points to, rather than always assuming ``<model>_small_0.0100/``. A variant
missing from the manifest (e.g. the sweep was only run for some models)
falls back to the single-run path, so partial sweeps degrade gracefully
instead of erroring.

The **unsupervised** half deliberately does *not* follow the sweep's
selection, even though ``sweep_ppi_experiments.py`` also runs one: judging
unsupervised candidates at that script's reduced step cap was verified to
produce an unstable, non-representative ranking (see README.md's "Scaled
local reproduction" section). So ``results/eval_unsup_<model>.json`` is
always read as-is -- and is always written by evaluating each model's
already-existing full-epoch default run, never by the sweep (which instead
records its own diagnostic pick in ``results/eval_unsup_sweep_<model>.json``,
unused here). This asymmetry is why the table below reports separate
``Sup. Selection`` / ``Unsup. Selection`` columns instead of one shared one.
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

    sup_swept_flags = {name: False for _, name in MODELS}
    for model_flag, display_name in MODELS:
        model_manifest = best_hparams.get(model_flag, {})

        # Unsupervised: always the model's full-epoch default run, evaluated
        # by scripts/eval_unsupervised.py directly -- never the sweep's
        # (reduced-step-cap) selection. See module docstring for why.
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
            sup_swept_flags[display_name] = True
        else:
            sup_dir_name = "%s_small_0.0100" % model_flag
        sup_path = os.path.join(LOGS_DIR, "sup-ppi", sup_dir_name, "test_stats.txt")
        sup_f1 = parse_test_stats(sup_path)

        rows[display_name] = (unsup_f1, sup_f1)

    lines = []
    lines.append("# Reproduced results — PPI (comparison with Table 1 of the paper)\n")
    lines.append("| Name | Unsup. F1 (reproduced) | Unsup. F1 (paper) | Sup. F1 (reproduced) | Sup. F1 (paper) | Sup. Selection | Unsup. Selection |")
    lines.append("|---|---|---|---|---|---|---|")
    order = ["Random", "Raw features", "GraphSAGE-GCN", "GraphSAGE-mean", "GraphSAGE-LSTM", "GraphSAGE-pool"]
    for name in order:
        unsup, sup = rows[name]
        paper_unsup, paper_sup = PAPER[name]
        unsup_s = "%.3f" % unsup if unsup is not None else "n/a"
        sup_s = "%.3f" % sup if sup is not None else "n/a"
        if name in ("Random", "Raw features"):
            sup_selection = "n/a"
            unsup_selection = "n/a"
        else:
            sup_selection = "Appendix C sweep" if sup_swept_flags[name] else "single default run"
            unsup_selection = "default (full epoch)"
        lines.append("| %s | %s | %.3f | %s | %.3f | %s | %s |" %
                      (name, unsup_s, paper_unsup, sup_s, paper_sup, sup_selection, unsup_selection))

    lines.append("")
    lines.append("Note: \"Sup. Selection\" = \"Appendix C sweep\" means the learning_rate/model_size "
                  "was selected by scripts/sweep_ppi_experiments.py on validation performance "
                  "(the scaled, 2-hour-budget grid -- 2 learning rates x \"small\" size only, not "
                  "Appendix C's full 3 x {small, big} -- see results/scaled_run_manifest.json and "
                  "README.md's \"Scaled local reproduction\" section); \"single default run\" means "
                  "the code's fixed defaults were used as-is.")
    lines.append("")
    lines.append("Note: \"Unsup. Selection\" is always \"default (full epoch)\" -- unlike the "
                  "supervised half, the unsupervised number is deliberately **not** the sweep's "
                  "selection. Judging unsupervised candidates at the sweep's reduced step cap was "
                  "verified to produce an unstable, non-representative ranking; the headline number "
                  "instead comes from each model's full-epoch default run "
                  "(logs/unsup-ppi/<model>_small_1.00e-05/). The sweep's own diagnostic pick is kept, "
                  "unused, in results/eval_unsup_sweep_<model>.json. Public PPI dataset from "
                  "http://snap.stanford.edu/graphsage/.")

    out_path = os.path.join(RESULTS_DIR, "ppi_results.md")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWrote", out_path)


if __name__ == "__main__":
    main()
