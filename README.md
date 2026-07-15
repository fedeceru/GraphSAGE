# GraphSAGE - PPI

A reproduction of **"Inductive Representation Learning on Large Graphs"**
(Hamilton, Ying, Leskovec, NeurIPS 2017, [`GraphSAGE.pdf`](GraphSAGE.pdf)),
specifically scoped to the **PPI** (protein-protein interaction) benchmark.

## Results

Table 1 (PPI columns) reproduced with a **single run per variant**, using
the code's default hyperparameters rather than the full hyperparameter sweep
from Appendix C:

| Name | Unsup. F1 (reproduced) | Unsup. F1 (paper) | Sup. F1 (reproduced) | Sup. F1 (paper) |
|---|---|---|---|---|
| Random | 0.398 | 0.396 | 0.398 | 0.396 |
| Raw features | 0.434 | 0.422 | 0.434 | 0.422 |
| GraphSAGE-GCN | 0.454 | 0.465 | 0.521 | 0.500 |
| GraphSAGE-mean | 0.464 | 0.486 | 0.585 | 0.598 |
| GraphSAGE-LSTM | 0.446 | 0.482 | 0.607 | 0.612 |
| GraphSAGE-pool | 0.482 | 0.502 | 0.602 | 0.600 |

Same qualitative ordering as the paper (GraphSAGE ≫ Raw features > Random,
GCN weakest, LSTM/pool strongest, supervised ≥ unsupervised).

## Repository structure

```
GraphSAGE.pdf                                      the paper
notebook.ipynb                                     paper walkthrough + real reproduced results
requirements.txt           

src/graphsage/                                     GraphSAGE implementation
  models.py                                        Algorithm 1/2 (sample + aggregate), unsupervised loss
  aggregators.py                                   AGGREGATE_k variants: mean, GCN, pool, LSTM
  supervised_models.py                             supervised classification head on top of models.py
  minibatch.py                                     graph -> padded-adjacency minibatches
  neigh_samplers.py                                fixed-size neighborhood sampling
  prediction.py                                    skip-gram link-prediction loss (Eq. 1)
  layers.py, inits.py, metrics.py, utils.py        supporting building blocks
  supervised_train.py, unsupervised_train.py       CLI training entry points

scripts/                                           Reproduction pipeline 
  baseline_ppi.py                                  Random + Raw features baselines
  eval_unsupervised.py                             logistic-regression eval of unsupervised embeddings
  compile_results.py                               collects everything into results/ppi_results.md
  run_ppi_experiments.py                           end to end run  

data/ppi/                                          PPI dataset 
logs/                                              training logs, TensorBoard events, metrics.csv, saved embeddings + snapshots
results/                                           final metrics + figures
```

## Reproduction

```powershell
.venv\Scripts\python.exe scripts\run_ppi_experiments.py
```

This single script downloads the PPI dataset (if not already present),
computes the Random/Raw-features baselines, runs all 4 aggregators ×
{supervised, unsupervised} on PPI, evaluates the unsupervised embeddings,
compiles `results/ppi_results.md`, and finally re-executes `notebook.ipynb`
in place. This is deliberately a smaller, single-dataset, single-hyperparameter-setting
slice of that.

Every unsupervised run is also passed `--embedding_snapshot_steps`, so each
one produces both its final embeddings (for the F1 table above) *and* the
intermediate checkpoints notebook.ipynb's PCA/t-SNE training-progression
visuals need. The flag is generic, so this works for all four
aggregators; `notebook.ipynb` §8.2 lets you pick which one to look at via its
`EMBED_MODEL` setting, defaulting to the best unsupervised performer. Every
run, supervised or unsupervised, also writes a structured `metrics.csv`
(step, epoch, loss, F1/MRR) into its log directory alongside the console
output, which is what the notebook's training-dynamics charts read from.

Each step can also be run individually; see the commands inside
`scripts/run_ppi_experiments.py`, or `notebook.ipynb` §7 for the reasoning
behind each one.
