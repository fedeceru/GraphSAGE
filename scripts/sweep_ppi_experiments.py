"""
Appendix C hyperparameter sweep + validation-based model selection for PPI.

``scripts/run_ppi_experiments.py`` trains every variant with a single, fixed
hyperparameter setting (the code's defaults). The paper's own Table 1 numbers
are *not* a single default run -- Appendix C ("Hyperparameter selection")
describes a real model-selection procedure:

    "In all settings, we performed hyperparameter selection on the learning
    rate and the model dimension ... we performed a parameter sweep on
    initial learning rates {0.01, 0.001, 0.0001} for the supervised models
    and {2e-6, 2e-7, 2e-8} for the unsupervised models ... we tested a 'big'
    and 'small' version of each model ... choosing the best setting for each
    variant according to performance on a validation set."

This script reproduces that procedure for the 4 PPI aggregators, in both the
supervised and unsupervised setting:

1. Train every (learning_rate, model_size) combination from the grid above.
2. Score each candidate on the *validation* split only:
   - supervised: `val_stats.txt`'s f1_micro, already computed by
     `supervised_train.py`'s own end-of-training `incremental_evaluate` call
     (no extra step needed).
   - unsupervised: fit the Appendix C logistic-regression probe on train
     embeddings and score it on *val* embeddings via
     `eval_unsupervised.py --split val` (never on test -- selecting by test
     performance would be exactly the data leakage this script exists to
     avoid).
3. Pick the (learning_rate, model_size) with the best validation score per
   (model, setting), and record its *test* score:
   - supervised: already sitting in that winning run's `test_stats.txt`.
   - unsupervised: one more `eval_unsupervised.py` call, this time
     `--split test` (the default), writing to the exact same
     `results/eval_unsup_<model>.json` path the single-run pipeline already
     uses -- so nothing downstream needs to change to consume it.
4. Write `results/best_hparams_ppi.json`: for every (model, setting), the
   winning config plus its validation and test scores. `compile_results.py`
   and `notebook.ipynb` read this file when present to know which log
   directory holds the paper-faithful, validation-selected run, instead of
   assuming the single default config's directory.

This script does NOT pass `--embedding_snapshot_steps` to any of its
candidate runs -- the embedding-progression PCA/t-SNE visuals in
notebook.ipynb are a qualitative illustration of training dynamics, not a
numeric fidelity result, and re-running a snapshot-instrumented candidate for
every point in a 6-point grid would only add I/O with no benefit. That
visualization stays tied to `run_ppi_experiments.py`'s single default run.

This script's own defaults describe the full 4-model x 2-setting x 2-size x
3-learning-rate grid (48 runs), but this repo does not run that grid --
`--models`/`--settings`/`--sizes`/`--sup_lrs`/`--unsup_lrs` narrow it, and
`--epochs`/`--max_total_steps` cap each candidate's training length, which
is what `scripts/run_ppi_experiments_scaled.py` uses to invoke this script
within a strict 2-hour local budget; see that script for the exact
configuration this repo's committed results reflect. `--dry_run` prints the
planned commands without running anything. A candidate that's already been
trained (its log_dir already has finished results) is reused rather than
retrained unless `--force`, so a run can be resumed after an interruption
without redoing finished work.

Usage:
    .venv\\Scripts\\python.exe scripts\\sweep_ppi_experiments.py --dry_run
    .venv\\Scripts\\python.exe scripts\\sweep_ppi_experiments.py
    .venv\\Scripts\\python.exe scripts\\sweep_ppi_experiments.py --models graphsage_mean --settings supervised
"""
from __future__ import annotations

import argparse
import json
import os
import re
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

MODELS = ["graphsage_mean", "gcn", "graphsage_seq", "graphsage_maxpool"]

# Appendix C's exact grids.
SUP_LRS = [0.01, 0.001, 0.0001]
UNSUP_LRS = [2e-6, 2e-7, 2e-8]
SIZES = ["small", "big"]


def run_streamed(cmd: list[str], log_path: Path, env: dict) -> None:
    """Run `cmd`, streaming its combined stdout+stderr to the console *and*
    to `log_path` live as it's produced, then raise if it exited non-zero.
    (Same helper as run_ppi_experiments.py, duplicated here to keep the two
    entry points independent scripts.)"""
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


def parse_stats_file(path: Path):
    """Parse the `key=value key=value ...` line written by
    supervised_train.py's val_stats.txt / test_stats.txt into a dict of floats."""
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8")
    return {k: float(v) for k, v in re.findall(r"(\w+)=([\d.eE+-]+)", content)}


def sup_log_dir(model: str, size: str, lr: float) -> Path:
    # Mirrors supervised_train.py's log_dir(): base_log_dir/sup-ppi/{model}_{size}_{lr:0.4f}/
    return REPO_ROOT / "logs" / "sup-ppi" / ("%s_%s_%0.4f" % (model, size, lr))


def unsup_log_dir(model: str, size: str, lr: float) -> Path:
    # Mirrors unsupervised_train.py's log_dir(): base_log_dir/unsup-ppi/{model}_{size}_{lr:.2e}/
    # (scientific notation, not fixed-point -- see the comment on that
    # function for why: Appendix C's grid includes 2e-7 and 2e-8, which
    # collide at 6 decimal places.)
    return REPO_ROOT / "logs" / "unsup-ppi" / ("%s_%s_%s" % (model, size, format(lr, ".2e")))


def sweep_supervised(model: str, sizes, lrs, epochs, max_total_steps, gpu, env, dry_run, force=False):
    """Train every (size, lr) candidate for `model`, then return the
    validation-best candidate's config + val/test f1_micro. Uses the
    supervised end-of-training val_stats.txt/test_stats.txt that
    supervised_train.py already writes -- no separate evaluation step.

    A candidate already trained (val_stats.txt and test_stats.txt both
    present) is reused rather than retrained unless `force` -- this matters
    whenever a grid value coincides with an already-completed run (e.g. the
    code's own lr=0.01 default is also the first value in Appendix C's
    supervised grid), and lets an interrupted/partial sweep resume cheaply
    instead of redoing already-finished candidates from scratch."""
    candidates = []
    for size in sizes:
        for lr in lrs:
            log_dir = sup_log_dir(model, size, lr)
            cmd = [str(VENV_PYTHON), "-m", "graphsage.supervised_train",
                   "--train_prefix", "data/ppi/ppi", "--model", model,
                   "--sigmoid", "true", "--model_size", size,
                   "--learning_rate", str(lr), "--base_log_dir", "logs", "--gpu", str(gpu)]
            if epochs is not None:
                cmd += ["--epochs", str(epochs)]
            if max_total_steps is not None:
                cmd += ["--max_total_steps", str(max_total_steps)]
            if dry_run:
                print("[dry_run] " + " ".join(cmd))
                continue
            already_done = (log_dir / "val_stats.txt").exists() and (log_dir / "test_stats.txt").exists()
            if already_done and not force:
                print(f"[skip] {log_dir} already has val_stats.txt/test_stats.txt")
            else:
                run_streamed(cmd, REPO_ROOT / "logs" / ("sweep_sup_%s_%s_%0.4f.log" % (model, size, lr)), env)
            val_stats = parse_stats_file(log_dir / "val_stats.txt")
            test_stats = parse_stats_file(log_dir / "test_stats.txt")
            candidates.append({
                "model_size": size, "learning_rate": lr,
                "val_f1_micro": val_stats["f1_micro"], "test_f1_micro": test_stats["f1_micro"],
                "log_dir": str(log_dir),
            })
    if dry_run or not candidates:
        return None
    best = max(candidates, key=lambda c: c["val_f1_micro"])
    return best


def sweep_unsupervised(model: str, sizes, lrs, max_total_steps, gpu, env, dry_run, force=False):
    """Train every (size, lr) candidate for `model`, score each on the *val*
    split via eval_unsupervised.py --split val (never test), pick the
    validation-best candidate, then run one final eval_unsupervised.py
    --split test (default) on that winner only, writing to the same
    results/eval_unsup_<model>.json path the single-run pipeline uses.

    A candidate whose embeddings already exist (val.npy present) is reused
    rather than retrained unless `force` -- see sweep_supervised()'s
    docstring for why this matters (resuming a partial/interrupted sweep
    without redoing finished work)."""
    candidates = []
    for size in sizes:
        for lr in lrs:
            cmd = [str(VENV_PYTHON), "-m", "graphsage.unsupervised_train",
                   "--train_prefix", "data/ppi/ppi", "--model", model,
                   "--model_size", size, "--learning_rate", str(lr),
                   "--base_log_dir", "logs", "--gpu", str(gpu)]
            if max_total_steps is not None:
                cmd += ["--max_total_steps", str(max_total_steps)]
            log_dir = unsup_log_dir(model, size, lr)
            # Sweep-internal log/tmp filenames use the same collision-safe
            # scientific-notation tag as unsup_log_dir() -- see its comment.
            tag = "%s_%s_%s" % (model, size, format(lr, ".2e"))
            if dry_run:
                print("[dry_run] " + " ".join(cmd))
                continue
            if (log_dir / "val.npy").exists() and not force:
                print(f"[skip] {log_dir} already has val.npy")
            else:
                run_streamed(cmd, REPO_ROOT / "logs" / ("sweep_unsup_%s.log" % tag), env)

            val_out = REPO_ROOT / "results" / "_sweep_tmp" / ("val_%s.json" % tag)
            if val_out.exists() and not force:
                print(f"[skip] {val_out} already computed")
            else:
                run_streamed(
                    [str(VENV_PYTHON), "scripts/eval_unsupervised.py",
                     "--train_prefix", "data/ppi/ppi", "--embed_dir", str(log_dir),
                     "--split", "val", "--out", str(val_out)],
                    REPO_ROOT / "logs" / ("sweep_eval_val_%s.log" % tag), env,
                )
            val_f1 = json.loads(val_out.read_text(encoding="utf-8"))["f1_micro"]
            candidates.append({"model_size": size, "learning_rate": lr, "val_f1_micro": val_f1, "log_dir": str(log_dir)})

    if dry_run or not candidates:
        return None
    best = max(candidates, key=lambda c: c["val_f1_micro"])

    test_out = REPO_ROOT / "results" / ("eval_unsup_%s.json" % model)
    run_streamed(
        [str(VENV_PYTHON), "scripts/eval_unsupervised.py",
         "--train_prefix", "data/ppi/ppi", "--embed_dir", best["log_dir"],
         "--split", "test", "--out", str(test_out)],
        REPO_ROOT / "logs" / ("sweep_eval_test_%s.log" % model), env,
    )
    best["test_f1_micro"] = json.loads(test_out.read_text(encoding="utf-8"))["f1_micro"]
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--settings", nargs="+", default=["supervised", "unsupervised"],
                         choices=["supervised", "unsupervised"])
    parser.add_argument("--sup_lrs", nargs="+", type=float, default=SUP_LRS)
    parser.add_argument("--unsup_lrs", nargs="+", type=float, default=UNSUP_LRS)
    parser.add_argument("--sizes", nargs="+", default=SIZES, choices=SIZES)
    parser.add_argument("--epochs", type=int, default=None,
                         help="override supervised_train.py --epochs for every candidate "
                              "(default: leave at the script's own default, 10). Useful "
                              "with a small value for a quick end-to-end smoke test.")
    parser.add_argument("--max_total_steps", type=int, default=None,
                         help="override --max_total_steps for every candidate, supervised "
                              "and unsupervised alike (default: no cap). Useful for a "
                              "quick smoke test of the sweep/selection plumbing.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_manifest", default="results/best_hparams_ppi.json")
    parser.add_argument("--force", action="store_true",
                         help="retrain every candidate even if already present (default: "
                              "reuse a candidate whose log_dir already has finished results, "
                              "so an interrupted/partial sweep can resume cheaply).")
    parser.add_argument("--dry_run", action="store_true",
                         help="print every training/eval command the full sweep would run, "
                              "without executing anything or touching logs/results.")
    args = parser.parse_args()

    if not args.dry_run and not VENV_PYTHON.exists():
        raise FileNotFoundError(
            f".venv not found at {REPO_ROOT / '.venv'} -- follow the Setup section in README.md first."
        )

    n_configs = len(args.sizes) * (
        (len(args.sup_lrs) if "supervised" in args.settings else 0)
        + (len(args.unsup_lrs) if "unsupervised" in args.settings else 0)
    ) * len(args.models)
    print(f"Sweeping {n_configs} (model, setting, size, lr) training runs "
          f"across {len(args.models)} model(s) x {len(args.settings)} setting(s).")

    env = os.environ.copy()
    if not args.dry_run:
        env["PATH"] = str(VENV_PYTHON.parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(VENV_SITE_PACKAGES) + os.pathsep + str(REPO_ROOT / "src")
        (REPO_ROOT / "results" / "_sweep_tmp").mkdir(parents=True, exist_ok=True)

    manifest_path = REPO_ROOT / args.out_manifest
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    start = time.time()
    for model in args.models:
        manifest.setdefault(model, {})
        if "supervised" in args.settings:
            print(f"\n=== Supervised sweep: {model} "
                  f"({len(args.sizes)} sizes x {len(args.sup_lrs)} learning rates) ===")
            best = sweep_supervised(model, args.sizes, args.sup_lrs, args.epochs,
                                     args.max_total_steps, args.gpu, env, args.dry_run, force=args.force)
            if best is not None:
                manifest[model]["supervised"] = best
                print(f"Best supervised config for {model}: "
                      f"size={best['model_size']} lr={best['learning_rate']} "
                      f"val_f1_micro={best['val_f1_micro']:.4f} test_f1_micro={best['test_f1_micro']:.4f}")

        if "unsupervised" in args.settings:
            print(f"\n=== Unsupervised sweep: {model} "
                  f"({len(args.sizes)} sizes x {len(args.unsup_lrs)} learning rates) ===")
            best = sweep_unsupervised(model, args.sizes, args.unsup_lrs,
                                       args.max_total_steps, args.gpu, env, args.dry_run, force=args.force)
            if best is not None:
                manifest[model]["unsupervised"] = best
                print(f"Best unsupervised config for {model}: "
                      f"size={best['model_size']} lr={best['learning_rate']} "
                      f"val_f1_micro={best['val_f1_micro']:.4f} test_f1_micro={best['test_f1_micro']:.4f}")

    if args.dry_run:
        print("\n[dry_run] nothing was executed, no files were written.")
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {manifest_path} ({time.time() - start:.0f}s total)")
    print("Re-run scripts/compile_results.py (or notebook.ipynb) to refresh "
          "results/ppi_results.md with the validation-selected configs.")


if __name__ == "__main__":
    main()
