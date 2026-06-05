# Experiment 11 — hierarchical user-level model

HierUserModel (`src/models/hier.py`): frozen MentalRoBERTa post-encoder → learned attention/mean aggregation over a user's post stream → user head. Trained on author-grouped corpus posts (weak user labels), evaluated at the USER level on the self-disclosure test set (disclosure posts masked). Reference: TF-IDF mean-pool ≈ 0.74 user-AUROC.

_Regenerate: `python scripts/exp_hier_user.py`_

| aggregator | target | n_users | n_pos | user_auroc | user_f1 |
|---|---|---|---|---|---|
| attention | anxiety | 3516 | 258 | 0.7061 | 0.2488 |
| attention | health_anxiety | 3516 | 141 | 0.7244 | 0.1941 |
| attention | depression | 3516 | 513 | 0.5707 | 0.2815 |
| mean | anxiety | 3516 | 258 | 0.7447 | 0.2732 |
| mean | health_anxiety | 3516 | 141 | 0.763 | 0.2191 |
| mean | depression | 3516 | 513 | 0.6138 | 0.2933 |

![hierarchical user model](figures/hier_user.png)