# Reproduced results — PPI (comparison with Table 1 of the paper)

| Name | Unsup. F1 (reproduced) | Unsup. F1 (paper) | Sup. F1 (reproduced) | Sup. F1 (paper) | Selection |
|---|---|---|---|---|---|
| Random | 0.398 | 0.396 | 0.398 | 0.396 | n/a |
| Raw features | 0.432 | 0.422 | 0.432 | 0.422 | n/a |
| GraphSAGE-GCN | 0.473 | 0.465 | 0.521 | 0.500 | Appendix C sweep |
| GraphSAGE-mean | 0.464 | 0.486 | 0.585 | 0.598 | Appendix C sweep |
| GraphSAGE-LSTM | 0.439 | 0.482 | 0.599 | 0.612 | Appendix C sweep |
| GraphSAGE-pool | 0.466 | 0.502 | 0.602 | 0.600 | Appendix C sweep |

Note: "Appendix C sweep" rows here come from the **scaled, 2-hour-budget** sweep (scripts/run_ppi_experiments_scaled.py), not the full 48-run grid -- 2 learning rates x "small" size only, not 3 x {small, big}. See results/scaled_run_manifest.json and README.md's "Scaled local reproduction" section for exactly what ran.
