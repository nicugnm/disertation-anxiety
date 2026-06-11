# Supervised user-level anxiety detection (the non-circular benchmark)

Masked self-disclosure task: independent label (self-reported diagnosis, disclosure post hidden), subreddit-matched controls, author-disjoint user folds. Trained DIRECTLY on the disclosure label (not weak labels). Mean +/- std over 5 seeds. `scripts/exp_user_level.py`.

| method | auroc | auroc_std | ap |
|---|---|---|---|
| mean_score (baseline) | 0.7349 | 0.0 | 0.1754 |
| score_aggs (LR) | 0.7692 | 0.0027 | 0.2039 |
| user_feats ling+behav (XGB) | 0.8316 | 0.0048 | 0.3089 |
| user_feats ling+behav (LR) | 0.7671 | 0.007 | 0.2297 |
| tfidf_userdoc (LR) | 0.728 | 0.004 | nan |

![user level](figures/user_level.png)

## Significance and top features

Paired bootstrap (2000 resamples, same folds) of the XGBoost feature model vs the mean-score baseline: AUROC difference = **+0.093** (95% CI [+0.060, +0.126], p = 0). The improvement is statistically significant (CI excludes 0).

Most important features (XGBoost gain): `anx_sub_frac`, `n_posts`, `weak_health_anxiety_mean`, `s_top3`, `ipi_std`, `weak_suicidality_mean`, `s_mean`, `weak_depression_max`, `s_max`, `eng_ncomm_mean`, `blen_mean`, `shai_medical_help_seeking_max`.

## How to read this

- Compare each method to the **mean_score baseline** (the ~0.74 mean-of-post-scores approach used by prior work). A method materially above it beats the one benchmark our heuristics cannot inflate.
- The lever is training **directly on the disclosure label** (author-disjoint user folds) plus **learned** aggregation. Naive mean-pooling discards the max/top-k signal (a user is at-risk if any post is) and behavioural patterns.
- Heavier representations *underperform* here: in the full battery, MentalRoBERTa user embeddings scored **0.686** and a deepset/attention net **0.675** — well below the 0.83 feature model. At this data scale (few hundred positive users, short histories) aggregated features + a gradient-boosted tree win.
- Caveats: the label is a self-disclosure proxy, not a clinician diagnosis; positives are few, so rely on the multi-seed std, and AP stays low because the class is rare.