# Reproduced results — PPI (comparison with Table 1 of the paper)

| Name | Unsup. F1 (reproduced) | Unsup. F1 (paper) | Sup. F1 (reproduced) | Sup. F1 (paper) | Sup. Selection | Unsup. Selection |
|---|---|---|---|---|---|---|
| Random | 0.398 | 0.396 | 0.398 | 0.396 | n/a | n/a |
| Raw features | 0.432 | 0.422 | 0.432 | 0.422 | n/a | n/a |
| GraphSAGE-GCN | 0.456 | 0.465 | 0.521 | 0.500 | Appendix C sweep | default (full epoch) |
| GraphSAGE-mean | 0.459 | 0.486 | 0.585 | 0.598 | Appendix C sweep | default (full epoch) |
| GraphSAGE-LSTM | 0.448 | 0.482 | 0.599 | 0.612 | Appendix C sweep | default (full epoch) |
| GraphSAGE-pool | 0.473 | 0.502 | 0.602 | 0.600 | Appendix C sweep | default (full epoch) |

Note: "Sup. Selection" = "Appendix C sweep" means the learning_rate/model_size was selected by scripts/sweep_ppi_experiments.py on validation performance (the scaled, 2-hour-budget grid -- 2 learning rates x "small" size only, not Appendix C's full 3 x {small, big} -- see results/scaled_run_manifest.json and README.md's "Scaled local reproduction" section); "single default run" means the code's fixed defaults were used as-is.

Note: "Unsup. Selection" is always "default (full epoch)" -- unlike the supervised half, the unsupervised number is deliberately **not** the sweep's selection. Judging unsupervised candidates at the sweep's reduced step cap was verified to produce an unstable, non-representative ranking; the headline number instead comes from each model's full-epoch default run (logs/unsup-ppi/<model>_small_1.00e-05/). The sweep's own diagnostic pick is kept, unused, in results/eval_unsup_sweep_<model>.json. Public PPI dataset from http://snap.stanford.edu/graphsage/.
