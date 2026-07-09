# Single entry point that reproduces the whole PPI experiment end to end:
# dataset download (if missing) -> baselines -> 4 aggregators x
# {supervised, unsupervised} + eval -> results table -> notebook build +
# execute. One run per variant, using the original code's default
# hyperparameters (see README.md for the full write-up and rationale).
#
# The GraphSAGE-pool unsupervised run additionally passes
# --embedding_snapshot_steps so that single run also produces the embedding
# checkpoints notebook.ipynb uses for its training-progression visuals (see
# unsupervised_train.py) -- no separate training run needed for that.
#
# Usage:
#   powershell -File scripts\run_ppi_experiments.ps1
#
# Requires the .venv described in README.md to already exist at the repo
# root (this script does not create it).

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PATH = "$RepoRoot\.venv\Scripts;$env:PATH"
$env:PYTHONPATH = "$RepoRoot\src"
$python = "$RepoRoot\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw ".venv not found at $RepoRoot\.venv -- follow the Setup section in README.md first."
}

New-Item -ItemType Directory -Force -Path "$RepoRoot\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "$RepoRoot\results" | Out-Null
New-Item -ItemType Directory -Force -Path "$RepoRoot\data" | Out-Null

$models = @("graphsage_mean", "gcn", "graphsage_seq", "graphsage_maxpool")

if (-not (Test-Path "data\ppi\ppi-G.json")) {
    Write-Host "=== Downloading PPI dataset (public, linked from the paper) ===" -ForegroundColor Cyan
    Invoke-WebRequest -Uri "http://snap.stanford.edu/graphsage/ppi.zip" -OutFile "data\ppi.zip"
    Expand-Archive -Path "data\ppi.zip" -DestinationPath "data" -Force
    Remove-Item "data\ppi.zip"
}

Write-Host "=== Baselines (Random, Raw features) ===" -ForegroundColor Cyan
& $python scripts\baseline_ppi.py --train_prefix data/ppi/ppi --out results/baseline_ppi.json 2>&1 |
    Tee-Object -FilePath "logs\baseline_ppi.log"
if ($LASTEXITCODE -ne 0) { throw "baseline_ppi.py failed" }

foreach ($m in $models) {
    Write-Host "=== Supervised: $m ===" -ForegroundColor Cyan
    & $python -m graphsage.supervised_train `
        --train_prefix data/ppi/ppi `
        --model $m `
        --sigmoid true `
        --model_size small `
        --base_log_dir logs `
        --gpu 0 2>&1 | Tee-Object -FilePath "logs\sup_$m.log"
    if ($LASTEXITCODE -ne 0) { throw "supervised_train failed for $m" }
}

# Every unsupervised variant also gets embedding snapshots at these training
# steps (see --embedding_snapshot_steps in unsupervised_train.py -- it's a
# generic mechanism, not tied to any one aggregator), so notebook.ipynb can
# plot the training-progression view for whichever aggregator you pick via
# its EMBED_MODEL setting, not just one fixed variant.
$snapshotSteps = "0,100,200,800,3000,8000,17050"

foreach ($m in $models) {
    Write-Host "=== Unsupervised: $m ===" -ForegroundColor Cyan
    & $python -m graphsage.unsupervised_train `
        --train_prefix data/ppi/ppi `
        --model $m `
        --model_size small `
        --base_log_dir logs `
        --embedding_snapshot_steps $snapshotSteps `
        --gpu 0 2>&1 | Tee-Object -FilePath "logs\unsup_$m.log"
    if ($LASTEXITCODE -ne 0) { throw "unsupervised_train failed for $m" }

    $embedDir = "logs\unsup-ppi\${m}_small_0.000010"
    Write-Host "=== Eval unsupervised embeddings: $m ===" -ForegroundColor Cyan
    & $python scripts\eval_unsupervised.py `
        --train_prefix data/ppi/ppi `
        --embed_dir $embedDir `
        --out "results\eval_unsup_$m.json" 2>&1 | Tee-Object -FilePath "logs\eval_unsup_$m.log"
    if ($LASTEXITCODE -ne 0) { throw "eval_unsupervised.py failed for $m" }
}

Write-Host "=== Compiling results table ===" -ForegroundColor Cyan
& $python scripts\compile_results.py 2>&1 | Tee-Object -FilePath "logs\compile_results.log"

Write-Host "=== Building notebook.ipynb ===" -ForegroundColor Cyan
& $python scripts\build_notebook.py 2>&1 | Tee-Object -FilePath "logs\build_notebook.log"
if ($LASTEXITCODE -ne 0) { throw "build_notebook.py failed" }

Write-Host "=== Executing notebook.ipynb ===" -ForegroundColor Cyan
& $python -m nbconvert --to notebook --execute --inplace notebook.ipynb 2>&1 |
    Tee-Object -FilePath "logs\notebook_execute.log"
if ($LASTEXITCODE -ne 0) { throw "notebook execution failed" }

Write-Host "DONE -- see results\ppi_results.md and notebook.ipynb" -ForegroundColor Green
