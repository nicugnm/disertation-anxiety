# External (cross-corpus) validation

TF-IDF anxiety model trained on OUR corpus, applied ZERO-SHOT to independent corpora. RMHD = anxiety-related subreddits (anxiety, healthanxiety, socialanxiety) vs controls (fitness, parenting, meditation, conspiracy); subreddit-as-label. ANGST = 3-expert-psychologist labels (gated; run when available).

_Regenerate: `python scripts/external_validation.py`_

| dataset | n | n_pos | auroc | auprc | f1@0.5 |
|---|---|---|---|---|---|
| RMHD (Low 2020) | 20733 | 8619 | 0.9195 | 0.9086 | 0.8194 |
| ANGST (Hengle 2024, experts) | 2872 | 701 | 0.8215 | 0.5185 | 0.6039 |

## RMHD per-subreddit (mean predicted anxiety score)

| dataset | subreddit | label | n | mean_anxiety_score | pred_pos_rate@0.5 |
|---|---|---|---|---|---|
| RMHD | anxiety | 1 | 4999 | 0.7978 | 0.8298 |
| RMHD | healthanxiety | 1 | 795 | 0.7085 | 0.7157 |
| RMHD | socialanxiety | 1 | 2825 | 0.6011 | 0.6039 |
| RMHD | fitness | 0 | 3488 | 0.1449 | 0.0175 |
| RMHD | parenting | 0 | 3080 | 0.1236 | 0.0393 |
| RMHD | meditation | 0 | 2204 | 0.3112 | 0.1987 |
| RMHD | conspiracy | 0 | 3342 | 0.0857 | 0.0048 |

![external](figures/external_validation.png)