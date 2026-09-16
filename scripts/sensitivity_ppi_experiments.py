"""
K and neighborhood-sample-size sensitivity sweep on PPI (Section 4.3 of the
paper), using GraphSAGE-mean, supervised.

The paper reports two specific quantitative claims in Section 4.3, backed by
Figure 2:
  - "setting K = 2 provided a consistent boost in accuracy of around 10-15%,
    on average, compared to K = 1; however, increasing K beyond 2 gave
    marginal returns in performance (0-5%) while increasing the runtime by a
    prohibitively large factor of 10-100x" (K sweep);
  - "we also found diminishing returns for sampling large neighborhoods"
    (Figure 2B: F1 vs. neighborhood sample size S1=S2, at K=2).

Figure 2 itself is drawn on the citation and Reddit datasets specifically
("Timing experiments on Reddit data" / "on the citation data using
GraphSAGE-mean") -- both out of scope for this PPI-only reproduction -- so
this script re-runs the same two sweeps on PPI instead, to check whether the
same *qualitative* claims hold on the one dataset this repo actually has,
rather than leaving them as an unverified claim repeated only in prose.

Model/setting choice: graphsage_mean, supervised. K itself is only
adjustable for the mean aggregator -- supervised_train.py's other three
aggregator branches (gcn, graphsage_seq, graphsage_maxpool) always build a
fixed 2-layer stack regardless of --samples_2/--samples_3 -- which is also,
not coincidentally, the paper's own choice of aggregator for Figure 2B.
Supervised (not unsupervised) is used so each candidate's score is a direct
read of test_stats.txt, without an extra classifier-fitting step.

K is varied via --samples_2/--samples_3 (K=1: samples_2=0; K=2 (default):
samples_2=10, samples_3=0; K=3: samples_2=10, samples_3=10 -- the paper
doesn't specify what S3 it used for K=3, so this mirrors S2 by symmetry).
Neighborhood sample size is varied via --samples_1=--samples_2=S at fixed
K=2 -- note this is symmetric (S1=S2=S), so it does NOT reuse the K sweep's
K=2 point, which is the code's actual (asymmetric) default S1=25, S2=10, not
S1=S2=25. Both dimensions share log_dir()'s `_S<s1>-<s2>-<s3>`
disambiguation (see supervised_train.py) so distinct configs never collide;
only the K sweep's K=2 entry exactly matches the existing single default
run's (25, 10, 0) and is reused rather than retrained (like
sweep_ppi_experiments.py / multiseed_ppi_experiments.py, a config already
trained -- test_stats.txt present -- is skipped unless --force).

Per-run training time (average seconds/iteration, parsed from the same
"time=" trailer supervised_train.py already prints -- see
notebook.ipynb's extract_last_avg_time) is recorded alongside F1, so the
runtime-cost half of the K claim above can be checked too, not just accuracy.

This script's own defaults are both sweeps at the full 10 epochs.
`scripts/run_ppi_experiments_scaled.py` invokes it at the reduced
`--epochs 5` budget (like every other scaled-reproduction tier) but no
longer passes `--k_only` -- both sweeps run, which is what this repo's
committed `results/sensitivity_ppi.json` reflects. `--k_only` remains
available for a cheaper K-only pass (e.g. a quick smoke test).

Writes results/sensitivity_ppi.json:
    {"K_sweep": [{"K": 1, "samples": [25,0,0], "test_f1_micro": ..., "sec_per_iter": ...}, ...],
     "sample_size_sweep": [{"sample_size": 5, "test_f1_micro": ..., "sec_per_iter": ...}, ...]}

Usage:
    .venv\\Scripts\\python.exe scripts\\sensitivity_ppi_experiments.py --dry_run
    .venv\\Scripts\\python.exe scripts\\sensitivity_ppi_experiments.py
    .venv\\Scripts\\python.exe scripts\\sensitivity_ppi_experiments.py --k_only --epochs 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
# This .venv's own site-packages isn't on sys.path by default (its
# pyvenv.cfg "home" points at a conda base whose site-packages lacks
# tensorflow) -- every subprocess spawned via VENV_PYTHON needs this on
# PYTHONPATH explicitly, or `import tensorflow` fails with
# ModuleNotFoundError despite tensorflow being installed right here.
VENV_SITE_PACKAGES = VENV_PYTHON.parent.parent / "Lib" / "site-packages"

MODEL = "graphsage_mean"
LR = 0.01
SIZE = "small"

# (K, samples_1, samples_2, samples_3). K=2 here (25, 10, 0) is the code's
# actual default -- reused, not retrained.
K_SWEEP = [(1, 25, 0, 0), (2, 25, 10, 0), (3, 25, 10, 10)]
# neighborhood sample size S1=S2, at fixed K=2. Symmetric by construction, so
# none of these (including S=25) coincide with the asymmetric (25, 10)
# default above -- every point here is trained fresh.
SAMPLE_SIZE_SWEEP = [5, 10, 25, 50, 75]


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


def parse_last_time(log_path: Path):
    """Same trailer supervised_train.py prints on every logged step and
    notebook.ipynb already parses (extract_last_avg_time): the last
    'time=X.XXXXX' at the end of a line, i.e. the running average
    seconds/iteration by the end of training."""
    if not log_path.exists():
        return None
    last = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"time=\s*([\d.]+)\s*$", line.strip())
        if m:
            last = float(m.group(1))
    return last


def sup_log_dir(s1: int, s2: int, s3: int) -> Path:
    # Mirrors supervised_train.py's log_dir(): the "_S<s1>-<s2>-<s3>" suffix
    # only appears when (s1, s2, s3) != (25, 10, 0).
    d = "%s_%s_%0.4f" % (MODEL, SIZE, LR)
    if (s1, s2, s3) != (25, 10, 0):
        d += "_S%d-%d-%d" % (s1, s2, s3)
    return REPO_ROOT / "logs" / "sup-ppi" / d


def run_config(s1, s2, s3, gpu, env, dry_run, force, epochs, max_total_steps):
    log_dir = sup_log_dir(s1, s2, s3)
    log_file = REPO_ROOT / "logs" / ("sensitivity_sup_S%d-%d-%d.log" % (s1, s2, s3))
    if not force and (log_dir / "test_stats.txt").exists():
        print(f"[skip] {log_dir} already has test_stats.txt")
    else:
        cmd = [str(VENV_PYTHON), "-m", "graphsage.supervised_train",
               "--train_prefix", "data/ppi/ppi", "--model", MODEL,
               "--sigmoid", "true", "--model_size", SIZE, "--learning_rate", str(LR),
               "--samples_1", str(s1), "--samples_2", str(s2), "--samples_3", str(s3),
               "--base_log_dir", "logs", "--gpu", str(gpu)]
        if epochs is not None:
            cmd += ["--epochs", str(epochs)]
        if max_total_steps is not None:
            cmd += ["--max_total_steps", str(max_total_steps)]
        if dry_run:
            print("[dry_run] " + " ".join(cmd))
            return None, None
        run_streamed(cmd, log_file, env)
    if dry_run:
        return None, None
    f1 = parse_test_stats(log_dir / "test_stats.txt")
    sec_per_iter = parse_last_time(log_file) if log_file.exists() else None
    return f1, sec_per_iter


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=None,
                         help="override --epochs for every candidate (default: the script's "
                              "own default, 10). For a quick smoke test.")
    parser.add_argument("--max_total_steps", type=int, default=None,
                         help="override --max_total_steps for every candidate. For a quick smoke test.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="retrain every config even if already present.")
    parser.add_argument("--k_only", action="store_true",
                         help="run only the K sweep, skipping the neighborhood-sample-size "
                              "sweep entirely -- for a smaller time budget, since K is the "
                              "paper's headline Section 4.3 claim and sample size the "
                              "secondary one.")
    parser.add_argument("--out", default="results/sensitivity_ppi.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not VENV_PYTHON.exists():
        raise FileNotFoundError(f".venv not found at {REPO_ROOT / '.venv'} -- see README.md Setup.")

    env = os.environ.copy()
    if not args.dry_run:
        env["PATH"] = str(VENV_PYTHON.parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(VENV_SITE_PACKAGES) + os.pathsep + str(REPO_ROOT / "src")

    results = {"K_sweep": [], "sample_size_sweep": []}

    print("=== K sweep (graphsage_mean, supervised, S1=25, S2=S3=10 where used) ===")
    for K, s1, s2, s3 in K_SWEEP:
        f1, sec_per_iter = run_config(s1, s2, s3, args.gpu, env, args.dry_run, args.force,
                                       args.epochs, args.max_total_steps)
        if f1 is not None:
            results["K_sweep"].append({"K": K, "samples": [s1, s2, s3],
                                        "test_f1_micro": f1, "sec_per_iter": sec_per_iter})
            print(f"K={K}: test_f1_micro={f1:.4f} sec_per_iter={sec_per_iter}")

    if args.k_only:
        print("\n[--k_only] skipping the neighborhood-sample-size sweep.")
        sample_size_sweep = []
    else:
        sample_size_sweep = SAMPLE_SIZE_SWEEP

    print("\n=== Neighborhood sample size sweep (graphsage_mean, supervised, K=2, S1=S2=S) ===")
    for S in sample_size_sweep:
        f1, sec_per_iter = run_config(S, S, 0, args.gpu, env, args.dry_run, args.force,
                                       args.epochs, args.max_total_steps)
        if f1 is not None:
            results["sample_size_sweep"].append({"sample_size": S, "test_f1_micro": f1,
                                                   "sec_per_iter": sec_per_iter})
            print(f"S1=S2={S}: test_f1_micro={f1:.4f} sec_per_iter={sec_per_iter}")

    if args.dry_run:
        print("\n[dry_run] nothing was executed, no files were written.")
        return

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
