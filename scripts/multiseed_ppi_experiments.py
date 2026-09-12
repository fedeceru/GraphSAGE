"""
Repeats each PPI variant under several random seeds and reports mean +/- std,
instead of the single, uncontrolled-seed point estimate every other script in
this repo produces.

This relies on `supervised_train.py` / `unsupervised_train.py`'s `--seed`
flag and `log_dir()` appending `_seed<N>` for any seed other than 123, so
repeats don't collide with each other or with the existing seed=123 runs.
This script trains one run per value in `--seeds` (default {123, 124, 125})
per (model, setting) and aggregates their test F1.

This script's own defaults (3 seeds, full epochs/steps) describe a broader
study than this repo runs -- `scripts/run_ppi_experiments_scaled.py` invokes
it with `--seeds 123 124` (2 total) and reduced epochs/steps, which is what
this repo's committed `results/seed_variance_ppi.json` reflects.

Hyperparameters: for each (model, setting), this script uses whatever
`results/best_hparams_ppi.json` records for it (the Appendix C
validation-selected learning_rate/model_size -- see
sweep_ppi_experiments.py) if present, otherwise the code's own defaults
(the same ones run_ppi_experiments.py uses). Either way, only the seed
varies across repeats of a given variant -- this isolates variance coming
from training-time randomness (weight init, minibatch order, negative/walk
sampling) from variance coming from a different hyperparameter choice.

Already-trained seeds are reused (skipped) rather than retrained: a run is
considered done if its log_dir already contains `test_stats.txt` (supervised)
or its embeddings (unsupervised); this includes the existing seed=123 runs
from run_ppi_experiments.py / sweep_ppi_experiments.py, so `--seeds 123 124
125` only pays the training cost of the two genuinely new seeds. Pass
`--force` to retrain everything anyway.

Writes `results/seed_variance_ppi.json`: for every (model, setting), the
list of seeds used, their individual test f1_micro, and the mean/std across
them.

Usage:
    .venv\\Scripts\\python.exe scripts\\multiseed_ppi_experiments.py --dry_run
    .venv\\Scripts\\python.exe scripts\\multiseed_ppi_experiments.py
    .venv\\Scripts\\python.exe scripts\\multiseed_ppi_experiments.py --models graphsage_mean --seeds 123 124 125
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

MODELS = ["graphsage_mean", "gcn", "graphsage_seq", "graphsage_maxpool"]
DEFAULT_SEEDS = [123, 124, 125]

# Fallback hyperparameters when results/best_hparams_ppi.json has no entry
# for a given (model, setting) -- matches run_ppi_experiments.py's defaults.
DEFAULT_SUP_LR = 0.01
DEFAULT_UNSUP_LR = 0.00001
DEFAULT_SIZE = "small"


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


def parse_stats_file(path: Path):
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8")
    return {k: float(v) for k, v in re.findall(r"(\w+)=([\d.eE+-]+)", content)}


def sup_log_dir(model: str, size: str, lr: float, seed: int) -> Path:
    d = "%s_%s_%0.4f" % (model, size, lr)
    if seed != 123:
        d += "_seed%d" % seed
    return REPO_ROOT / "logs" / "sup-ppi" / d


def unsup_log_dir(model: str, size: str, lr: float, seed: int) -> Path:
    d = "%s_%s_%s" % (model, size, format(lr, ".2e"))
    if seed != 123:
        d += "_seed%d" % seed
    return REPO_ROOT / "logs" / "unsup-ppi" / d


def variant_hparams(best_hparams: dict, model: str, setting: str, default_lr: float):
    """(learning_rate, model_size) for (model, setting): from
    best_hparams_ppi.json if it has an entry, else the code defaults."""
    entry = best_hparams.get(model, {}).get(setting)
    if entry is not None:
        return entry["learning_rate"], entry["model_size"]
    return default_lr, DEFAULT_SIZE


def run_supervised_seed(model, size, lr, seed, gpu, env, dry_run, force, epochs=None, max_total_steps=None):
    log_dir = sup_log_dir(model, size, lr, seed)
    if not force and (log_dir / "test_stats.txt").exists():
        print(f"[skip] {log_dir} already has test_stats.txt")
    else:
        cmd = [str(VENV_PYTHON), "-m", "graphsage.supervised_train",
               "--train_prefix", "data/ppi/ppi", "--model", model,
               "--sigmoid", "true", "--model_size", size,
               "--learning_rate", str(lr), "--seed", str(seed),
               "--base_log_dir", "logs", "--gpu", str(gpu)]
        if epochs is not None:
            cmd += ["--epochs", str(epochs)]
        if max_total_steps is not None:
            cmd += ["--max_total_steps", str(max_total_steps)]
        if dry_run:
            print("[dry_run] " + " ".join(cmd))
            return None
        run_streamed(cmd, REPO_ROOT / "logs" / f"multiseed_sup_{model}_seed{seed}.log", env)
    if dry_run:
        return None
    stats = parse_stats_file(log_dir / "test_stats.txt")
    return stats["f1_micro"] if stats else None


def run_unsupervised_seed(model, size, lr, seed, gpu, env, dry_run, force, max_total_steps=None):
    log_dir = unsup_log_dir(model, size, lr, seed)
    eval_out = REPO_ROOT / "results" / "_seed_tmp" / f"eval_{model}_seed{seed}.json"
    if not force and eval_out.exists():
        print(f"[skip] {eval_out} already computed")
    else:
        if force or not (log_dir / "val.npy").exists():
            cmd = [str(VENV_PYTHON), "-m", "graphsage.unsupervised_train",
                   "--train_prefix", "data/ppi/ppi", "--model", model,
                   "--model_size", size, "--learning_rate", str(lr), "--seed", str(seed),
                   "--base_log_dir", "logs", "--gpu", str(gpu)]
            if max_total_steps is not None:
                cmd += ["--max_total_steps", str(max_total_steps)]
            if dry_run:
                print("[dry_run] " + " ".join(cmd))
                return None
            run_streamed(cmd, REPO_ROOT / "logs" / f"multiseed_unsup_{model}_seed{seed}.log", env)
        elif dry_run:
            print(f"[dry_run] (embeddings already exist at {log_dir}, would reuse them)")
            return None
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        run_streamed(
            [str(VENV_PYTHON), "scripts/eval_unsupervised.py",
             "--train_prefix", "data/ppi/ppi", "--embed_dir", str(log_dir),
             "--split", "test", "--out", str(eval_out)],
            REPO_ROOT / "logs" / f"multiseed_eval_{model}_seed{seed}.log", env,
        )
    if dry_run:
        return None
    return json.loads(eval_out.read_text(encoding="utf-8"))["f1_micro"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--settings", nargs="+", default=["supervised", "unsupervised"],
                         choices=["supervised", "unsupervised"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=None,
                         help="override supervised_train.py --epochs for every seed run "
                              "(default: the script's own default, 10). For a quick smoke test.")
    parser.add_argument("--max_total_steps", type=int, default=None,
                         help="override --max_total_steps for every seed run, supervised and "
                              "unsupervised alike (default: no cap). For a quick smoke test.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="retrain every seed even if already present.")
    parser.add_argument("--out_manifest", default="results/seed_variance_ppi.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not VENV_PYTHON.exists():
        raise FileNotFoundError(f".venv not found at {REPO_ROOT / '.venv'} -- see README.md Setup.")

    env = os.environ.copy()
    if not args.dry_run:
        env["PATH"] = str(VENV_PYTHON.parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(REPO_ROOT / "src")

    best_hparams_path = REPO_ROOT / "results" / "best_hparams_ppi.json"
    best_hparams = json.loads(best_hparams_path.read_text(encoding="utf-8")) if best_hparams_path.exists() else {}

    manifest_path = REPO_ROOT / args.out_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    for model in args.models:
        manifest.setdefault(model, {})
        if "supervised" in args.settings:
            lr, size = variant_hparams(best_hparams, model, "supervised", DEFAULT_SUP_LR)
            f1s = {}
            for seed in args.seeds:
                f1 = run_supervised_seed(model, size, lr, seed, args.gpu, env, args.dry_run, args.force,
                                          epochs=args.epochs, max_total_steps=args.max_total_steps)
                if f1 is not None:
                    f1s[seed] = f1
            if f1s:
                values = list(f1s.values())
                manifest[model]["supervised"] = {
                    "model_size": size, "learning_rate": lr,
                    "seeds": list(f1s.keys()), "f1_micro": values,
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
                print(f"{model}/supervised: {values} -> mean={statistics.mean(values):.4f} "
                      f"std={(statistics.stdev(values) if len(values) > 1 else 0.0):.4f}")

        if "unsupervised" in args.settings:
            lr, size = variant_hparams(best_hparams, model, "unsupervised", DEFAULT_UNSUP_LR)
            f1s = {}
            for seed in args.seeds:
                f1 = run_unsupervised_seed(model, size, lr, seed, args.gpu, env, args.dry_run, args.force,
                                            max_total_steps=args.max_total_steps)
                if f1 is not None:
                    f1s[seed] = f1
            if f1s:
                values = list(f1s.values())
                manifest[model]["unsupervised"] = {
                    "model_size": size, "learning_rate": lr,
                    "seeds": list(f1s.keys()), "f1_micro": values,
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
                print(f"{model}/unsupervised: {values} -> mean={statistics.mean(values):.4f} "
                      f"std={(statistics.stdev(values) if len(values) > 1 else 0.0):.4f}")

    if args.dry_run:
        print("\n[dry_run] nothing was executed, no files were written.")
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
