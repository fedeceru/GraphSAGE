# Reproduced results: PPI (comparison with Table 1 of the paper)

| Name | Unsup. F1 (reproduced) | Unsup. F1 (paper) | Sup. F1 (reproduced) | Sup. F1 (paper) |
|---|---|---|---|---|
| Random | 0.398 | 0.396 | 0.398 | 0.396 |
| Raw features | 0.434 | 0.422 | 0.434 | 0.422 |
| GraphSAGE-GCN | 0.464 | 0.465 | 0.521 | 0.500 |
| GraphSAGE-mean | 0.460 | 0.486 | 0.585 | 0.598 |
| GraphSAGE-LSTM | 0.447 | 0.482 | 0.599 | 0.612 |
| GraphSAGE-pool | 0.481 | 0.502 | 0.602 | 0.600 |

Note: a single run per variant with the code's default hyperparameters (not the full sweep from Appendix C); public PPI dataset from http://snap.stanford.edu/graphsage/.
