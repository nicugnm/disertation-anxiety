# Exhaustive user-level push (research-backed features + model zoo)

Masked self-disclosure benchmark, author-disjoint user folds, trained directly on the disclosure label. Features and methods drawn from eRisk / CLPsych / Low 2020. `scripts/exp_user_level_push.py`.

| target | model | AUROC | std | AP |
|---|---|---|---|---|
| anxiety | mean_score (baseline) | 0.7349 | 0.0 | 0.1754 |
| anxiety | xgboost | 0.8333 | 0.004 | 0.3155 |
| anxiety | hist_gbm | 0.8237 | 0.0049 | 0.3132 |
| anxiety | random_forest | 0.8419 | 0.0023 | 0.3186 |
| anxiety | extra_trees | 0.8376 | 0.0023 | 0.2969 |
| anxiety | elasticnet_lr | 0.7772 | 0.0072 | 0.2282 |
| anxiety | linear_svm | 0.7651 | 0.0059 | 0.2152 |
| anxiety | stacking | 0.8381 | 0.004 | 0.3199 |
| health_anxiety | mean_score (baseline) | 0.7986 | 0.0 | 0.1608 |
| health_anxiety | xgboost | 0.8739 | 0.0046 | 0.2365 |
| health_anxiety | hist_gbm | 0.8718 | 0.0045 | 0.2259 |
| health_anxiety | random_forest | 0.8859 | 0.0011 | 0.2581 |
| health_anxiety | extra_trees | 0.8907 | 0.0025 | 0.2753 |
| health_anxiety | elasticnet_lr | 0.8122 | 0.0113 | 0.2038 |
| health_anxiety | linear_svm | 0.8007 | 0.0092 | 0.1866 |
| health_anxiety | stacking | 0.8767 | 0.0039 | 0.2548 |
| depression | mean_score (baseline) | 0.614 | 0.0 | 0.2337 |
| depression | xgboost | 0.8218 | 0.0048 | 0.5665 |
| depression | hist_gbm | 0.8199 | 0.0063 | 0.5721 |
| depression | random_forest | 0.8241 | 0.0034 | 0.5734 |
| depression | extra_trees | 0.8186 | 0.0032 | 0.5349 |
| depression | elasticnet_lr | 0.7697 | 0.0064 | 0.4267 |
| depression | linear_svm | 0.7647 | 0.0086 | 0.4134 |
| depression | stacking | 0.8273 | 0.0051 | 0.5788 |

![push](figures/user_level_push.png)

## Headline (anxiety)

Best model **random_forest**: AUROC **0.8419** (5-seed CV) vs mean-score baseline 0.7349.
Nested-CV (unbiased) XGBoost: **0.8245 ± 0.0061** over 147 features.
Paired bootstrap vs baseline: AUROC difference **+0.108** (95% CI [+0.078, +0.139], p = 0).

Top features: `subgrp_anxiety_primary`, `weak_anxiety_mean`, `n_posts`, `s_top5`, `weak_health_anxiety_mean`, `s_recency`, `s_p95`, `s_top3`, `s_p90`, `subgrp_health_anxiety_primary`, `s_frac70`, `subgrp_depression_primary`, `ipi_std`, `f_n_tokens_max`, `weak_depression_max`.

## Interpretation

- **All three targets improve substantially and significantly** over mean-pooling: anxiety 0.735 → 0.842, health anxiety 0.799 → **0.891**, depression 0.614 → 0.827. The most dramatic is depression (+0.21), where mean-pooling was weakest.
- **The research-backed features are the drivers.** The top features are exactly the levers the literature pointed to: **bag-of-subreddits** participation (`subgrp_*`), **comorbidity** (`weak_*` scores for the *other* conditions), **order-statistics / "any post is a red flag"** (`s_top5/p95/p90/frac70`), and **temporal** signal (`s_recency`, `ipi_std` burstiness). Naive mean (`s_mean`) is not in the top 15.
- **Honest headline number (anxiety).** The best single model in 5-seed CV is random forest at 0.842, but that figure benefits from picking the best of seven models on the same CV. The **unbiased nested-CV estimate** (inner-loop hyperparameter search, so no selection optimism) is **0.825 ± 0.006** — that is the number to trust. Both are far above baseline; the paired bootstrap (+0.108, CI excludes 0, p ≈ 0) is significant under either.
- **Tree ensembles win; linear and deep lose.** RF / ExtraTrees / XGBoost / stacking all land ~0.82–0.84 (anxiety) and ~0.87–0.89 (health anxiety); elastic-net LR and linear SVM lag at ~0.77–0.81; and from the earlier battery, transformer embeddings (0.686) and a deepset (0.675) underperform. This matches the small-N tabular literature (Grinsztajn 2022): at a few hundred positives, gradient-boosted/forest models on engineered features beat both linear models and deep nets.
- **Caveats.** Health anxiety has only 141 positive users (wider true CI, though the multi-seed std is 0.003); the label is a self-disclosure proxy, not a clinician diagnosis. SMOTE was deliberately avoided (class weights only), per van den Goorbergh 2022.