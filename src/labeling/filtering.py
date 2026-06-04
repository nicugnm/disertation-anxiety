"""Weak-label filtering (confident-learning style; cf. Whisper's training-data cleanup).

Weak labels (subreddit prior + lexicon) are noisy: off-topic posts in an anxiety
subreddit get labelled positive, and genuine anxiety in a neutral subreddit gets
labelled negative. Given out-of-fold model scores, flag examples where the model
*confidently disagrees* with the weak label — the likely mislabels — so they can be
removed before retraining.
"""
from __future__ import annotations

import numpy as np


def confident_label_issues(y_weak, oof_scores, low: float = 0.10, high: float = 0.90):
    """Flag likely-mislabeled examples from confident model-vs-weak-label disagreement.

    likely false positive: weak=1 but model score < `low`.
    likely false negative: weak=0 but model score > `high`.
    Returns (boolean mask of issues, counts dict)."""
    y = np.asarray(y_weak, dtype=int)
    s = np.asarray(oof_scores, dtype=float)
    fp = (y == 1) & (s < low)
    fn = (y == 0) & (s > high)
    mask = fp | fn
    return mask, {"likely_false_pos": int(fp.sum()), "likely_false_neg": int(fn.sum()),
                  "total_flagged": int(mask.sum())}
