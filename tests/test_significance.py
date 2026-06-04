"""Tests for paired significance — McNemar and paired bootstrap detect a real
difference and correctly find none when models are identical."""
from __future__ import annotations

import numpy as np

from src.evaluation.significance import mcnemar_test, paired_bootstrap


def test_mcnemar_detects_one_model_strictly_better():
    y = np.array([1] * 100 + [0] * 100)
    pred_a = y.copy()                     # A perfect
    pred_b = y.copy(); pred_b[:40] = 1 - pred_b[:40]   # B wrong on 40
    res = mcnemar_test(y, pred_a, pred_b)
    assert res["b"] == 40 and res["c"] == 0
    assert res["p_value"] < 0.001


def test_mcnemar_identical_predictions_not_significant():
    y = np.array([1, 0, 1, 0, 1, 0] * 20)
    pred = np.array([1, 1, 1, 0, 0, 0] * 20)
    res = mcnemar_test(y, pred, pred.copy())
    assert res["b"] == 0 and res["c"] == 0
    assert res["p_value"] == 1.0


def test_paired_bootstrap_detects_auroc_gap():
    rng = np.random.default_rng(0)
    n = 1000
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    score_a = np.r_[rng.normal(0.8, 0.1, n), rng.normal(0.2, 0.1, n)].clip(0, 1)  # strong
    score_b = rng.random(2 * n)                                                    # random
    res = paired_bootstrap(y, score_a, score_b, metric="auroc", n_boot=500)
    assert res["delta"] > 0.2
    assert res["ci_lo"] > 0          # CI excludes 0
    assert res["p_value"] < 0.05


def test_paired_bootstrap_identical_scores_no_difference():
    rng = np.random.default_rng(1)
    n = 800
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    s = np.r_[rng.normal(0.7, 0.2, n), rng.normal(0.3, 0.2, n)].clip(0, 1)
    res = paired_bootstrap(y, s, s.copy(), metric="auroc", n_boot=500)
    assert abs(res["delta"]) < 1e-9
    assert res["p_value"] == 1.0
