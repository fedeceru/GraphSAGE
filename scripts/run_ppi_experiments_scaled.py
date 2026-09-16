"""
Time-boxed, locally-feasible reproduction of the paper's PPI methodology,
fit to a strict 2-hour wall-clock budget on a single consumer GPU.

Context: running the other 4 `scripts/*_ppi_experiments.py` entry points at
their own (much broader) defaults is not feasible on a local machine in one
sitting -- this script exists to fit the same checks into a strict 2-hour
budget instead. This script does NOT change what any of those 4 scripts
compute or how -- Algorithm 1/2, the aggregator formulas, the unsupervised
loss, and the K=2/S1=25/S2=10 architecture defaults are exactly as
implemented everywhere else in this repo. What's reduced here is purely the
*experimental budget*: how many hyperparameter candidates, seeds, K/noise
values are tried, and how many epochs/steps each gets -- a computational
concession, not an algorithmic one.

Priority-ordered work, highest first (see README.md for the full rationale
table): (1) a reduced Appendix C sweep -- 2 learning rates x "small" size
only (dropping "big": it's a no-op for mean/GCN and roughly doubles
LSTM/pool's cost) per model/setting, supervised candidates at the full 10
epochs (cheap), unsupervised candidates capped at 3000 steps (this project's
own notebook.ipynb Sec.8.2 embedding-snapshot analysis already found
cluster structure "largely in place by step 200" -- three orders of
magnitude short of a full ~17,050-step epoch -- so 3000 steps is a
principled cutoff for *ranking* candidates, not an arbitrary shortcut);
(2) one additional seed (124) alongside the existing seed=123 results, at
5 epochs (supervised) / 3000 steps (unsupervised), using whatever tier 1
selected; (3) the full sensitivity script -- both the K-sweep (K in {1, 3};
K=2 reused) and the neighborhood-sample-size sweep (S1=S2=S in
{5, 10, 25, 50, 75}, K=2 fixed, graphsage_mean supervised) -- 5 epochs each;
(4) noise robustness trimmed to 3 of 5 noise levels, 5 epochs each -- lowest
priority, first cut if time is short.

This is an orchestrator, not a reimplementation: every item below is a
subprocess call into one of the 4 existing fidelity scripts with reduced
arguments (their own `--epochs`/`--max_total_steps`/`--sup_lrs`/`--sizes`/
etc. flags, already built for exactly this kind of override). Before
starting each item, elapsed wall-clock time is checked against
`--time_budget_minutes` (default 100, leaving ~20 min of the 2-hour cap as
buffer for compile_results.py + notebook re-execution afterward); once the
budget is exhausted, remaining items are skipped (logged as such, never
silently dropped) rather than started and left to overrun -- this is the
"Feasibility Fallback" as actual, checked code, not just a hand-estimate
that might be wrong. Every item is independently resumable/idempotent (the
underlying scripts skip already-completed candidates unless --force), so a
second run after an interruption picks up roughly where the first left off.

Writes `results/scaled_run_manifest.json`: every planned item, whether it
ran or was skipped (and why), and actual elapsed time -- the single source
of truth for what this repo's scaled-run results actually reflect. Nothing
in README.md or notebook.ipynb's description of the scaled configuration
should say anything this manifest doesn't back up.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_ppi_experiments_scaled.py --dry_run
    .venv\\Scripts\\python.exe scripts\\run_ppi_experiments_scaled.py
    .venv\\Scripts\\python.exe scripts\\run_ppi_experiments_scaled.py --time_budget_minutes 60
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
# This .venv's own site-packages isn't on sys.path by default (its
# pyvenv.cfg "home" points at a conda base whose site-packages lacks
# tensorflow) -- every subprocess spawned via VENV_PYTHON needs this on
# PYTHONPATH explicitly, or `import tensorflow` fails with
# ModuleNotFoundError despite tensorflow being installed right here.
VENV_SITE_PACKAGES = VENV_PYTHON.parent.parent / "Lib" / "site-packages"

# Reduced grid: see module docstring for why these specific values.
SUP_LRS = ["0.01", "0.001"]           # 0.01 reuses the existing default run
UNSUP_LRS = ["0.000002", "0.0000002"]  # 2e-6, 2e-7 (2e-8 dropped)
UNSUP_STEP_CAP = "3000"
MULTISEED_EPOCHS = "5"
SENSITIVITY_EPOCHS = "5"
NOISE_EPOCHS = "5"
NOISE_PROPS = ["0.0", "0.5", "1.0"]


def run_and_time(cmd: list[str], env: dict, dry_run: bool) -> tuple[float, bool]:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return 0.0, True
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    if not ok:
        print(f"WARNING: command exited with code {result.returncode} after {elapsed:.0f}s")
    return elapsed, ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--time_budget_minutes", type=float, default=100.0,
                         help="soft wall-clock budget for starting new items (default: 100, "
                              "leaving ~20 min of the 2-hour hard cap as buffer for "
                              "compile_results.py + notebook re-execution afterward).")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_manifest", default="results/scaled_run_manifest.json")
    parser.add_argument("--dry_run", action="store_true",
                         help="print every planned item's command without executing anything "
                              "or touching logs/results (budget checks are skipped: every "
                              "item is shown, regardless of the time budget).")
    args = parser.parse_args()

    if not args.dry_run and not VENV_PYTHON.exists():
        raise FileNotFoundError(f".venv not found at {REPO_ROOT / '.venv'} -- see README.md Setup.")

    env = os.environ.copy()
    if not args.dry_run:
        env["PATH"] = str(VENV_PYTHON.parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(VENV_SITE_PACKAGES) + os.pathsep + str(REPO_ROOT / "src")

    budget_s = args.time_budget_minutes * 60
    start = time.time()
    manifest = {"time_budget_minutes": args.time_budget_minutes, "items": []}

    def remaining_s():
        return budget_s - (time.time() - start)

    def maybe_run(name: str, tier: int, cmd: list[str]):
        if not args.dry_run and remaining_s() <= 0:
            print(f"\n[skip] {name}: time budget exhausted "
                  f"({(time.time() - start) / 60:.1f} min elapsed of {args.time_budget_minutes:.0f} min budget)")
            manifest["items"].append({
                "name": name, "tier": tier, "cmd": " ".join(cmd),
                "ran": False, "ok": None, "elapsed_sec": 0.0,
                "reason": "time budget exhausted",
                "elapsed_total_sec_at_finish": time.time() - start,
            })
            return
        elapsed, ok = run_and_time(cmd, env, args.dry_run)
        manifest["items"].append({
            "name": name, "tier": tier, "cmd": " ".join(cmd),
            "ran": not args.dry_run, "ok": ok, "elapsed_sec": elapsed,
            "reason": "dry_run (shown only, not executed)" if args.dry_run else None,
            "elapsed_total_sec_at_finish": time.time() - start,
        })

    py = str(VENV_PYTHON)

    # --- Tier 1: reduced Appendix C sweep -----------------------------------
    maybe_run("sweep-supervised", 1, [
        py, "scripts/sweep_ppi_experiments.py",
        "--settings", "supervised", "--sup_lrs", *SUP_LRS, "--sizes", "small",
        "--gpu", str(args.gpu),
    ])
    maybe_run("sweep-unsupervised", 1, [
        py, "scripts/sweep_ppi_experiments.py",
        "--settings", "unsupervised", "--unsup_lrs", *UNSUP_LRS, "--sizes", "small",
        "--max_total_steps", UNSUP_STEP_CAP, "--gpu", str(args.gpu),
    ])

    # --- Tier 2: reduced multi-seed (uses tier 1's best_hparams_ppi.json if
    #     it exists; falls back to code defaults otherwise) -----------------
    maybe_run("multiseed-supervised", 2, [
        py, "scripts/multiseed_ppi_experiments.py",
        "--settings", "supervised", "--seeds", "123", "124",
        "--epochs", MULTISEED_EPOCHS, "--gpu", str(args.gpu),
    ])
    maybe_run("multiseed-unsupervised", 2, [
        py, "scripts/multiseed_ppi_experiments.py",
        "--settings", "unsupervised", "--seeds", "123", "124",
        "--max_total_steps", UNSUP_STEP_CAP, "--gpu", str(args.gpu),
    ])

    # --- Tier 3: full sensitivity sweep (K + neighborhood sample size) ------
    maybe_run("sensitivity-full-sweep", 3, [
        py, "scripts/sensitivity_ppi_experiments.py",
        "--epochs", SENSITIVITY_EPOCHS, "--gpu", str(args.gpu),
    ])

    # --- Tier 4: noise robustness, trimmed to 3 of 5 noise levels -----------
    maybe_run("noise-robustness", 4, [
        py, "scripts/noise_robustness_ppi_experiments.py",
        "--noise_props", *NOISE_PROPS, "--epochs", NOISE_EPOCHS, "--gpu", str(args.gpu),
    ])

    manifest["total_elapsed_sec"] = time.time() - start
    manifest["total_elapsed_min"] = manifest["total_elapsed_sec"] / 60

    n_ran = sum(1 for it in manifest["items"] if it["ran"])
    n_skipped = sum(1 for it in manifest["items"] if not it["ran"])
    print(f"\n=== Done: {n_ran} item(s) ran, {n_skipped} skipped, "
          f"{manifest['total_elapsed_min']:.1f} min total elapsed ===")
    for it in manifest["items"]:
        if it["ran"]:
            status = f"ran ({it['elapsed_sec']/60:.1f} min)"
        elif it["reason"] and "dry_run" in it["reason"]:
            status = "shown (--dry_run)"
        else:
            status = f"SKIPPED ({it['reason']})"
        print(f"  tier {it['tier']} | {it['name']}: {status}")

    if args.dry_run:
        print("\n[dry_run] nothing was executed, results/scaled_run_manifest.json not written.")
        return

    out_path = REPO_ROOT / args.out_manifest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("Next: scripts/compile_results.py, then re-execute notebook.ipynb, then update "
          "README.md/notebook.ipynb prose to describe this scaled configuration -- see "
          "this manifest for exactly what ran vs. was skipped.")


if __name__ == "__main__":
    main()
