# SHAI-item symptom decomposition

Each post scored on the seven SHAI clinical dimensions (Salkovskis 2002), then averaged per subreddit and correlated with the health-anxiety weak label. `src/features/shai.py`. Rates = matched terms/phrases per token.

_Regenerate: `python scripts/shai_decomposition.py`_

## SHAI dimension ↔ health-anxiety label (point-biserial r)

| dimension | r |
|---|---:|
| serious_illness_fear | +0.3040 |
| illness_worry | +0.1140 |
| bodily_vigilance | +0.1054 |
| symptom_checking | +0.0226 |
| difficulty_reassured | +0.0130 |
| medical_help_seeking | +0.0046 |
| reassurance_seeking | +0.0039 |

## Mean dimension rate — selected subreddits (×1000)

| subreddit | illness_worry | bodily_vigilance | serious_illness_fear | symptom_checking | reassurance_seeking | difficulty_reassured | medical_help_seeking |
|---|---|---|---|---|---|---|---|
| HealthAnxiety | 3.84 | 0.76 | 3.20 | 0.18 | 0.69 | 0.07 | 0.22 |
| Anxiety | 0.34 | 0.47 | 0.80 | 0.03 | 0.39 | 0.11 | 0.09 |
| socialanxiety | 0.04 | 0.02 | 0.05 | 0.00 | 0.25 | 0.04 | 0.00 |
| PanicAttack | 0.31 | 1.36 | 1.39 | 0.02 | 0.24 | 0.10 | 0.23 |
| depression | 0.02 | 0.01 | 0.46 | 0.00 | 0.24 | 0.07 | 0.00 |
| COVID19positive | 0.08 | 1.20 | 0.51 | 0.01 | 1.90 | 0.11 | 0.49 |
| cooking | 0.00 | 0.00 | 0.07 | 0.00 | 0.00 | 0.01 | 0.00 |
| personalfinance | 0.01 | 0.01 | 0.07 | 0.04 | 0.02 | 0.02 | 0.03 |

![SHAI decomposition](figures/shai_decomposition.png)