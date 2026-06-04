"""Statistical significance for paired model comparisons.

Two complementary tests on the SAME test rows:
  - mcnemar_test     — compares two classifiers' error patterns (the discordant
                       pairs). Exact binomial when discordances are few, else
                       chi-square with continuity correction.
  - paired_bootstrap — resamples the test set to put a confidence interval and a
                       two-sided p-value on a metric DIFFERENCE (ΔAUROC/ΔF1/...).

Both are paired: pass aligned predictions/scores for the same examples.
"""
from __future__ import annotations

import numpy as np


def mcnemar_test(y_true, pred_a, pred_b, exact_threshold: int = 25) -> dict:
    """McNemar's test for two classifiers on the same examples.

    b = #(A correct, B wrong), c = #(A wrong, B correct). Significant p means the
    two models' error rates genuinely differ (b ≠ c beyond chance)."""
    y = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y
    correct_b = np.asarray(pred_b) == y
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if n < exact_threshold:
        from scipy.stats import binomtest

        p = binomtest(max(b, c), n, 0.5, alternative="two-sided").pvalue
        return {"b": b, "c": c, "n_discordant": n, "statistic": float(max(b, c)),
                "p_value": float(p), "method": "exact_binomial"}
    from scipy.stats import chi2

    stat = (abs(b - c) - 1) ** 2 / n          # continuity correction
    return {"b": b, "c": c, "n_discordant": n, "statistic": float(stat),
            "p_value": float(chi2.sf(stat, 1)), "method": "chi2_continuity"}


def _metric_fn(metric: str, threshold: float):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if metric == "auroc":
        return lambda y, s: roc_auc_score(y, s)
    if metric == "auprc":
        return lambda y, s: average_precision_score(y, s)
    thr = {
        "f1": lambda y, s: f1_score(y, (s >= threshold).astype(int), zero_division=0),
        "precision": lambda y, s: precision_score(y, (s >= threshold).astype(int), zero_division=0),
        "recall": lambda y, s: recall_score(y, (s >= threshold).astype(int), zero_division=0),
        "accuracy": lambda y, s: accuracy_score(y, (s >= threshold).astype(int)),
    }
    if metric not in thr:
        raise ValueError(f"unknown metric {metric!r}")
    return thr[metric]


def paired_bootstrap(
    y_true, score_a, score_b, metric: str = "auroc", n_boot: int = 2000,
    seed: int = 42, threshold: float = 0.5, ci: float = 0.95, progress: bool = False,
) -> dict:
    """Paired bootstrap CI + two-sided p-value for metric(A) - metric(B).

    p_value is the fraction of resamples whose delta crosses zero (×2, clipped) —
    a CI that excludes 0 corresponds to p < (1-ci)."""
    y = np.asarray(y_true)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    fn = _metric_fn(metric, threshold)
    m_a, m_b = float(fn(y, a)), float(fn(y, b))

    rng = np.random.default_rng(seed)
    n = len(y)
    it = range(n_boot)
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(it, desc=f"bootstrap:{metric}", leave=False)
    deltas = []
    for _ in it:
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(fn(yy, a[idx]) - fn(yy, b[idx]))
    deltas = np.asarray(deltas)
    lo = float(np.percentile(deltas, (1 - ci) / 2 * 100))
    hi = float(np.percentile(deltas, (1 + ci) / 2 * 100))
    p = min(1.0, 2 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean())))
    return {"metric": metric, "metric_a": m_a, "metric_b": m_b, "delta": m_a - m_b,
            "ci_lo": lo, "ci_hi": hi, "p_value": p, "n_boot": int(len(deltas))}
