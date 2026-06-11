# Supervised user-level anxiety detection (the non-circular benchmark)

Masked self-disclosure task: independent label (self-reported diagnosis, disclosure post hidden), subreddit-matched controls, author-disjoint user folds. Trained DIRECTLY on the disclosure label (not weak labels). Mean +/- std over 5 seeds. `scripts/exp_user_level.py`.

| method | auroc | auroc_std | ap |
|---|---|---|---|
| mean_score (baseline) | 0.7349 | 0.0 | 0.1754 |
| score_aggs (LR) | 0.7604 | 0.0031 | 0.1969 |
| user_feats ling+behav (XGB) | 0.8019 | 0.0045 | 0.248 |
| user_feats ling+behav (LR) | 0.7458 | 0.0072 | 0.2031 |
| embeddings (LR) | 0.686 | 0.0102 | 0.1438 |
| all: scores+feats+emb (XGB) | 0.7947 | 0.005 | 0.2284 |
| all: scores+feats+emb (LR) | 0.6948 | 0.0117 | 0.1538 |
| tfidf_userdoc (LR) | 0.728 | 0.004 | nan |
| deepset/attention (emb) | 0.6747 | 0.0127 | 0.13 |

![user level](figures/user_level.png)

## A genuine, non-circular positive result

This is the one task in the whole project where a method materially **beats** the long-standing baseline on a label our heuristics could not have produced.

- **Headline:** XGBoost over aggregated user features reaches **0.802 ± 0.005** AUROC vs the **0.735** mean-of-post-scores baseline (and vs ~0.74 for the earlier mean-pool hierarchical model) — a **+0.067** improvement, stable across 5 seeds.
- **Why it works:** the prior approaches mean-pooled, which discards the signal that matters at the user level — the **max / top-k** of post scores (a user is at-risk if *any* post is, not on average), behavioural patterns (subreddit mix, posting volume), and nonlinear feature interactions. Even a learned linear aggregation of the scores already beats the naive mean (0.760 > 0.735).
- **What does *not* help (an honest, slightly counter-intuitive finding):** frozen MentalRoBERTa user embeddings *underperform* (0.686), adding them to the feature model *hurts* (0.802→0.795), and a deepset/attention network over embeddings overfits (0.675). At this data scale (258 positive users, median 5 posts each) careful aggregated features + a gradient-boosted tree beat transformer embeddings and deep set models.
- **The lever vs the earlier null:** training **directly on the disclosure label** (author-disjoint user folds) rather than on weak labels, plus learned (not mean) aggregation. The earlier hierarchical model failed because it trained on a mismatched weak any-post-positive label and mean-pooled.

**Caveats (honest):** the label is still a self-disclosure proxy, not a clinician diagnosis; the positive class is small (258 users), though the multi-seed cross-validated std (0.005) makes the gain robust; AP stays low (0.25) because positives are rare.