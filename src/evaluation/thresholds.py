"""Per-subreddit decision-threshold calibration.

A single global threshold is a compromise across communities with very different
base rates and language intensity. Fitting a best-F1 threshold per subreddit (with
a global fallback for sparse ones) and applying each community's own threshold
recovers F1 lost to the operating-point mismatch — without changing the model.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import best_threshold_f1


def fit_per_subreddit_thresholds(
    scores, y, subreddits, min_pos: int = 25, global_threshold: float | None = None
) -> tuple[dict, float]:
    """Best-F1 threshold per subreddit; subreddits with < `min_pos` positives OR
    negatives fall back to the global best-F1 threshold. Returns (thresholds, global)."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    subreddits = np.asarray(subreddits)
    if global_threshold is None:
        global_threshold = best_threshold_f1(y, scores)[0] if y.sum() > 0 else 0.5
    global_threshold = float(global_threshold)

    thresholds: dict = {}
    for s in np.unique(subreddits):
        m = subreddits == s
        ys, ss = y[m], scores[m]
        if int(ys.sum()) >= min_pos and int((ys == 0).sum()) >= min_pos:
            thresholds[s] = float(best_threshold_f1(ys, ss)[0])
        else:
            thresholds[s] = global_threshold  # fallback — not enough data to tune
    return thresholds, global_threshold


def apply_per_subreddit(scores, subreddits, thresholds: dict, global_threshold: float) -> np.ndarray:
    """Binarise each score against its subreddit's threshold (global fallback)."""
    scores = np.asarray(scores, dtype=float)
    subreddits = np.asarray(subreddits)
    thr = np.array([thresholds.get(s, global_threshold) for s in subreddits], dtype=float)
    return (scores >= thr).astype(int)


def macro_f1_by_subreddit(y, preds, subreddits, min_pos: int = 1) -> dict:
    """Per-subreddit F1 (only subreddits with >= `min_pos` positives)."""
    from sklearn.metrics import f1_score

    y = np.asarray(y, dtype=int)
    preds = np.asarray(preds, dtype=int)
    subreddits = np.asarray(subreddits)
    out: dict = {}
    for s in np.unique(subreddits):
        m = subreddits == s
        if int(y[m].sum()) >= min_pos:
            out[s] = float(f1_score(y[m], preds[m], zero_division=0))
    return out
