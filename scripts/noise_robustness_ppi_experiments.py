"""
Feature-noise robustness experiment on PPI -- an analogue of Figure 3 /
Theorem 1 in the paper.

Section 5 proves (Theorem 1, Appendix E) that, given sufficiently distinct
node features, the *pooling* aggregator can approximate a node's clustering
coefficient -- i.e. it can learn to exploit graph structure, not just
feature content. Figure 3 backs this empirically by progressively replacing
the feature matrix with Gaussian noise and showing GraphSAGE-pool degrades
more gracefully than GraphSAGE-GCN and the raw-features baseline. Until this
script, that empirical check existed in this repo only as prose
(notebook.ipynb §9, "Theoretical analysis (overview)") describing the
paper's own Figure 3 -- nothing here actually ran it.

Figure 3 itself is drawn "on the citation data" specifically, which is out
of scope for this PPI-only reproduction, so this script re-runs the same
kind of experiment on PPI instead: does the same *qualitative* pattern
(pool > GCN > raw-features as noise increases) show up here too?

Noise injection: for a given --noise_prop p, each node's feature vector is,
independently with probability p, replaced wholesale by a fresh Gaussian
draw matching that feature dimension's real mean/std (computed once from the
unmodified data). This is a modeling choice (the paper doesn't specify its
exact corruption procedure) made so that a corrupted node's *neighbors* can
still be clean -- i.e. the experiment can actually distinguish "ignores its
own noisy features and leans on clean neighbors" (structural robustness)
from "denoises its own features" (a different, less interesting capability).
At p=1.0, every node's features are pure noise.

A fresh supervised GraphSAGE (GCN and pool variants) is trained from scratch
at each noise level -- not just evaluated with noisy test-time inputs on a
clean-trained model -- since the claim under test is about what the
*aggregator* can learn to do when features are uninformative throughout
training, matching how Figure 3 is framed ("GraphSAGE-pool was in fact
capable of maintaining modest performance by leveraging graph structure").
The era-matched raw-features SGDClassifier baseline (see baseline_ppi.py) is
also re-fit at each noise level, standing in for the paper's "feat." line.

Per-noise-level datasets are materialized under data/_noise_tmp/p<X.XX>/ as
hardlinks to the unmodified graph/id-map/class-map files (so no unmodified
data is duplicated) plus one newly generated, noise-injected ppi-feats.npy.
Passing a distinct --train_prefix this way also means supervised_train.py's
own log_dir() naturally puts each noise level under its own
logs/sup-p<X.XX>/ directory -- no separate disambiguation logic needed here
(unlike the seed/sample-size cases in the other sweep scripts).

This script's own defaults (5 noise levels, full 10 epochs) describe a
broader check than this repo runs -- `scripts/run_ppi_experiments_scaled.py`
invokes it with `--noise_props 0.0 0.5 1.0 --epochs 5` (3 of the 5 levels,
reduced epochs), which is what this repo's committed
`results/noise_robustness_ppi.json` reflects.

Writes results/noise_robustness_ppi.json:
    {"noise_props": [0.0, 0.25, ...],
     "GraphSAGE-GCN": {"test_f1_micro": [...]},
     "GraphSAGE-pool": {"test_f1_micro": [...]},
     "Raw features": {"test_f1_micro": [...]}}

Usage:
    .venv\\Scripts\\python.exe scripts\\noise_robustness_ppi_experiments.py --dry_run
    .venv\\Scripts\\python.exe scripts\\noise_robustness_ppi_experiments.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
SRC_DIR = REPO_ROOT / "src"

MODELS = [("gcn", "GraphSAGE-GCN"), ("graphsage_maxpool", "GraphSAGE-pool")]
NOISE_PROPS = [0.0, 0.25, 0.5, 0.75, 1.0]
LR = 0.01
SIZE = "small"
# Seed for the noise-injection RNG itself -- independent of --seed (which
# controls the GraphSAGE model's own training randomness).
DATA_SEED = 123


def run_streamed(cmd: list[str], log_path: Path, env: dict) -> None:
    print("\n$ " + " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Command failed (exit %d): %s" % (process.returncode, " ".join(cmd)))


def parse_test_stats(path: Path):
    if not path.exists():
        return None
    m = re.search(r"f1_micro=([\d.]+)", path.read_text(encoding="utf-8"))
    return float(m.group(1)) if m else None


def make_noisy_dataset(noise_prop: float) -> Path:
    """Materialize data/_noise_tmp/p<noise_prop>/ppi-* for this noise level
    (hardlinked graph/id-map/class-map files + a freshly generated noisy
    feats.npy), returning the resulting --train_prefix. A no-op if it
    already exists, so re-running the sweep doesn't regenerate noise."""
    src = REPO_ROOT / "data" / "ppi"
    out_dir = REPO_ROOT / "data" / "_noise_tmp" / ("p%.2f" % noise_prop)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["ppi-G.json", "ppi-id_map.json", "ppi-class_map.json"]:
        dst = out_dir / fname
        if not dst.exists():
            try:
                os.link(src / fname, dst)
            except OSError:
                shutil.copy2(src / fname, dst)

    feats_path = out_dir / "ppi-feats.npy"
    if not feats_path.exists():
        orig = np.load(src / "ppi-feats.npy")
        rng = np.random.RandomState(DATA_SEED)
        col_mean = orig.mean(axis=0)
        col_std = orig.std(axis=0)
        col_std[col_std == 0] = 1.0
        noisy = orig.copy()
        corrupt_mask = rng.rand(orig.shape[0]) < noise_prop
        n_corrupt = int(corrupt_mask.sum())
        if n_corrupt > 0:
            noise = rng.normal(loc=col_mean, scale=col_std,
                                size=(n_corrupt, orig.shape[1])).astype(orig.dtype)
            noisy[corrupt_mask] = noise
        np.save(feats_path, noisy)
        print(f"noise_prop={noise_prop}: corrupted {n_corrupt}/{orig.shape[0]} "
              f"node feature vectors -> {feats_path}")

    return out_dir / "ppi"  # train_prefix (forward slashes: see module docstring)


def sup_log_dir(train_prefix_posix: str, model: str) -> Path:
    # Mirrors supervised_train.py's log_dir(): base_log_dir/sup-<prefix's parent dir name>/{model}_{size}_{lr}/
    tag = train_prefix_posix.split("/")[-2]
    return REPO_ROOT / "logs" / ("sup-" + tag) / ("%s_%s_%0.4f" % (model, SIZE, LR))


def run_supervised(train_prefix_posix, model, gpu, env, dry_run, force, epochs, max_total_steps):
    log_dir = sup_log_dir(train_prefix_posix, model)
    log_file = REPO_ROOT / "logs" / ("noise_sup_%s_%s.log" % (model, train_prefix_posix.split("/")[-2]))
    if not force and (log_dir / "test_stats.txt").exists():
        print(f"[skip] {log_dir} already has test_stats.txt")
    else:
        cmd = [str(VENV_PYTHON), "-m", "graphsage.supervised_train",
               "--train_prefix", train_prefix_posix, "--model", model,
               "--sigmoid", "true", "--model_size", SIZE, "--learning_rate", str(LR),
               "--base_log_dir", "logs", "--gpu", str(gpu)]
        if epochs is not None:
            cmd += ["--epochs", str(epochs)]
        if max_total_steps is not None:
            cmd += ["--max_total_steps", str(max_total_steps)]
        if dry_run:
            print("[dry_run] " + " ".join(cmd))
            return None
        run_streamed(cmd, log_file, env)
    if dry_run:
        return None
    return parse_test_stats(log_dir / "test_stats.txt")


def run_raw_features_baseline(train_prefix_posix, dry_run):
    if dry_run:
        print(f"[dry_run] raw-features SGDClassifier baseline on {train_prefix_posix}")
        return None
    sys.path.insert(0, str(SRC_DIR))
    from graphsage.utils import load_data  # noqa: E402
    from baseline_ppi import get_split_labels, raw_features_baseline  # noqa: E402

    G, feats, id_map, _, class_map = load_data(train_prefix_posix, load_walks=False)
    feats = np.vstack([feats, np.zeros((feats.shape[1],))])
    X_train, y_train, X_test, y_test = get_split_labels(G, id_map, class_map, feats)
    return raw_features_baseline(X_train, y_train, X_test, y_test)["f1_micro"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise_props", nargs="+", type=float, default=NOISE_PROPS)
    parser.add_argument("--epochs", type=int, default=None,
                         help="override --epochs for every candidate (default: the script's "
                              "own default, 10). For a quick smoke test.")
    parser.add_argument("--max_total_steps", type=int, default=None,
                         help="override --max_total_steps for every candidate. For a quick smoke test.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="retrain every config even if already present.")
    parser.add_argument("--out", default="results/noise_robustness_ppi.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not VENV_PYTHON.exists():
        raise FileNotFoundError(f".venv not found at {REPO_ROOT / '.venv'} -- see README.md Setup.")

    env = os.environ.copy()
    if not args.dry_run:
        env["PATH"] = str(VENV_PYTHON.parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(SRC_DIR)

    results = {"noise_props": args.noise_props,
               "GraphSAGE-GCN": {"test_f1_micro": []},
               "GraphSAGE-pool": {"test_f1_micro": []},
               "Raw features": {"test_f1_micro": []}}

    for noise_prop in args.noise_props:
        print(f"\n=== noise_prop={noise_prop} ===")
        if args.dry_run:
            train_prefix_posix = "data/_noise_tmp/p%.2f/ppi" % noise_prop
        else:
            train_prefix_posix = make_noisy_dataset(noise_prop).as_posix()

        for model_flag, display_name in MODELS:
            f1 = run_supervised(train_prefix_posix, model_flag, args.gpu, env,
                                 args.dry_run, args.force, args.epochs, args.max_total_steps)
            if f1 is not None:
                results[display_name]["test_f1_micro"].append(f1)
                print(f"{display_name}: test_f1_micro={f1:.4f}")

        raw_f1 = run_raw_features_baseline(train_prefix_posix, args.dry_run)
        if raw_f1 is not None:
            results["Raw features"]["test_f1_micro"].append(raw_f1)
            print(f"Raw features: test_f1_micro={raw_f1:.4f}")

    if args.dry_run:
        print("\n[dry_run] nothing was executed, no files were written.")
        return

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
