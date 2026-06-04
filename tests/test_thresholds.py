"""Tests for per-subreddit threshold calibration — sparse fallback, threshold
ordering tracks the score shift, application logic, and macro-F1 improvement."""
from __future__ import annotations

import numpy as np

from src.evaluation.thresholds import (
    apply_per_subreddit,
    fit_per_subreddit_thresholds,
    macro_f1_by_subreddit,
)


def _subreddit(rng, name, pos_mu, neg_mu, n=300):
    s = np.r_[rng.normal(pos_mu, 0.05, n), rng.normal(neg_mu, 0.05, n)].clip(0, 1)
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    names = np.array([name] * (2 * n))
    return s, y, names


def test_per_subreddit_thresholds_beat_global_macro_f1():
    rng = np.random.default_rng(0)
    sA, yA, nA = _subreddit(rng, "A", 0.75, 0.55)   # high-intensity community
    sB, yB, nB = _subreddit(rng, "B", 0.45, 0.25)   # low-intensity community
    scores, y, subs = np.r_[sA, sB], np.r_[yA, yB], np.r_[nA, nB]

    thr, gthr = fit_per_subreddit_thresholds(scores, y, subs, min_pos=20)
    macro_g = np.mean(list(macro_f1_by_subreddit(y, (scores >= gthr).astype(int), subs).values()))
    macro_p = np.mean(list(macro_f1_by_subreddit(y, apply_per_subreddit(scores, subs, thr, gthr), subs).values()))

    assert macro_p > macro_g            # per-subreddit thresholds win
    assert thr["A"] > thr["B"]          # higher cutoff for the higher-intensity sub


def test_sparse_subreddit_falls_back_to_global():
    rng = np.random.default_rng(1)
    sA, yA, nA = _subreddit(rng, "A", 0.7, 0.3)
    sB = np.r_[rng.normal(0.5, 0.05, 5), rng.normal(0.5, 0.05, 50)].clip(0, 1)  # only 5 positives
    yB = np.r_[np.ones(5), np.zeros(50)].astype(int)
    nB = np.array(["B"] * 55)
    scores, y, subs = np.r_[sA, sB], np.r_[yA, yB], np.r_[nA, nB]

    thr, gthr = fit_per_subreddit_thresholds(scores, y, subs, min_pos=20)
    assert thr["B"] == gthr             # too few positives -> global fallback


def test_apply_per_subreddit_uses_each_threshold_and_global_fallback():
    scores = np.array([0.4, 0.6, 0.4, 0.6])
    subs = np.array(["A", "A", "B", "B"])
    thr = {"A": 0.5, "B": 0.3}
    assert list(apply_per_subreddit(scores, subs, thr, 0.5)) == [0, 1, 1, 1]
    assert apply_per_subreddit(np.array([0.4]), np.array(["Z"]), thr, 0.5)[0] == 0  # unknown -> global
