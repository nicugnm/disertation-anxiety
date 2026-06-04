# eRisk early-detection metrics (disclosure test set)

Per-post classifier reframed as an early-detection system over user timelines (threshold=0.5). The disclosure post is **masked** — the system must detect from the rest of the user's stream. Decision = flag at the first post crossing the threshold. **ERDE₅/₅₀ lower is better**; `median_latency` = posts-to-detection on true positives.

_Regenerate: `python scripts/erisk_eval.py`_

| model | target | n_users | n_pos | erde_5 | erde_50 | precision | recall | f1 | latency_weighted_f1 | median_latency |
|---|---|---|---|---|---|---|---|---|---|---|
| mentalbert_anxiety | anxiety | 3516 | 258 | 0.0565 | 0.0503 | 0.1091 | 0.7829 | 0.1916 | 0.1916 | 1.0 |
| multitask_anxiety_health_dep_suic | anxiety | 3516 | 258 | 0.0568 | 0.0507 | 0.1084 | 0.7791 | 0.1903 | 0.1903 | 1.0 |
| multitask_anxiety_health_dep_suic | depression | 3516 | 513 | 0.136 | 0.1181 | 0.2396 | 0.3587 | 0.2873 | 0.285 | 3.0 |
| multitask_anxiety_health_dep_suic | health_anxiety | 3516 | 141 | 0.0398 | 0.0384 | 0.2759 | 0.0567 | 0.0941 | 0.0894 | 14.0 |
| tfidf_logreg | anxiety | 3516 | 258 | 0.0569 | 0.0514 | 0.1084 | 0.7636 | 0.1898 | 0.1898 | 1.0 |

![eRisk ERDE](figures/erisk.png)