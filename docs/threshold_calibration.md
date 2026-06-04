# Per-subreddit threshold calibration

TF-IDF baseline (anxiety) on an **author-disjoint** train/calibration/test split of the current corpus. A single global best-F1 threshold (**0.647**) vs **per-subreddit** thresholds (tuned where a subreddit has >= 25 positives & negatives in calibration, else global fallback). The model is identical; only the operating point changes.

- **Macro-F1 (mean over scored subreddits): 0.719 -> 0.781** (Δ +0.062)
- Pooled F1: 0.830 -> 0.888
- Subreddits tuned (vs global fallback): 19 / 19

![per-subreddit thresholds](figures/threshold_calibration.png)

_Regenerate: `python scripts/threshold_calibration.py`_

| subreddit | n_test | n_pos | prevalence | global_thr | subreddit_thr | tuned | f1_global | f1_persub | delta_f1 |
|---|---|---|---|---|---|---|---|---|---|
| Agoraphobia | 178 | 136 | 0.764 | 0.647 | 0.381 | True | 0.9294 | 0.9268 | -0.0025 |
| Anxiety | 3275 | 1618 | 0.494 | 0.647 | 0.387 | True | 0.9171 | 0.9547 | 0.0376 |
| AnxietyDepression | 680 | 247 | 0.3632 | 0.647 | 0.302 | True | 0.8474 | 0.9051 | 0.0578 |
| Anxietyhelp | 818 | 426 | 0.5208 | 0.647 | 0.343 | True | 0.8928 | 0.9404 | 0.0476 |
| COVID19_support | 2198 | 77 | 0.035 | 0.647 | 0.85 | True | 0.6019 | 0.6582 | 0.0563 |
| CPTSD | 4725 | 42 | 0.0089 | 0.647 | 0.951 | True | 0.3871 | 0.3889 | 0.0018 |
| CovidLongHaulers | 2512 | 29 | 0.0115 | 0.647 | 0.856 | True | 0.5102 | 0.5574 | 0.0472 |
| Emetophobia | 1200 | 322 | 0.2683 | 0.647 | 0.454 | True | 0.8211 | 0.8878 | 0.0668 |
| HealthAnxiety | 1924 | 999 | 0.5192 | 0.647 | 0.419 | True | 0.9098 | 0.9482 | 0.0384 |
| OCD | 2172 | 134 | 0.0617 | 0.647 | 0.783 | True | 0.5891 | 0.6476 | 0.0585 |
| PTSD | 2382 | 27 | 0.0113 | 0.647 | 0.914 | True | 0.3247 | 0.5373 | 0.2126 |
| PanicAttack | 1604 | 988 | 0.616 | 0.647 | 0.45 | True | 0.9552 | 0.9679 | 0.0127 |
| agoraphobia | 1341 | 522 | 0.3893 | 0.647 | 0.375 | True | 0.8884 | 0.9366 | 0.0482 |
| dpdr | 876 | 99 | 0.113 | 0.647 | 0.839 | True | 0.7072 | 0.7389 | 0.0317 |
| emetophobia | 176 | 108 | 0.6136 | 0.647 | 0.575 | True | 0.8756 | 0.8829 | 0.0073 |
| ibs | 2119 | 40 | 0.0189 | 0.647 | 0.953 | True | 0.3378 | 0.5051 | 0.1673 |
| mentalhealth | 2822 | 38 | 0.0135 | 0.647 | 0.958 | True | 0.3426 | 0.557 | 0.2144 |
| panicdisorder | 1135 | 721 | 0.6352 | 0.647 | 0.521 | True | 0.9602 | 0.9694 | 0.0092 |
| socialanxiety | 2048 | 668 | 0.3262 | 0.647 | 0.351 | True | 0.8638 | 0.9196 | 0.0558 |