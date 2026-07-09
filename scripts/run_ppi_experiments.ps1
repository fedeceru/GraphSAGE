# Orchestrates the GraphSAGE reproduction on PPI:
# baselines (Random, Raw features) + 4 aggregators x {supervised, unsupervised},
# a single run per variant with the original code's default hyperparameters.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PATH = "$RepoRoot\.venv\Scripts;$env:PATH"
$env:PYTHONPATH = "$RepoRoot\src"
$python = "$RepoRoot\.venv\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path "$RepoRoot\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "$RepoRoot\results" | Out-Null

$models = @("graphsage_mean", "gcn", "graphsage_seq", "graphsage_maxpool")

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

foreach ($m in $models) {
    Write-Host "=== Unsupervised: $m ===" -ForegroundColor Cyan
    & $python -m graphsage.unsupervised_train `
        --train_prefix data/ppi/ppi `
        --model $m `
        --model_size small `
        --base_log_dir logs `
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

Write-Host "=== Executing notebook.ipynb ===" -ForegroundColor Cyan
& $python -m nbconvert --to notebook --execute --inplace notebook.ipynb 2>&1 | Tee-Object -FilePath "logs\notebook_execute.log"

Write-Host "DONE" -ForegroundColor Green
