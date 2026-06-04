"""Tests for eRisk metrics — streaming decision rule and hand-computed ERDE / latency-F1."""
from __future__ import annotations

import math

import numpy as np

from src.evaluation.erisk import erde, first_crossing_decision, latency_weighted_f1


def test_first_crossing_decision():
    assert first_crossing_decision(np.array([0.1, 0.6, 0.2]), 0.5) == (1, 2)
    assert first_crossing_decision(np.array([0.1, 0.2]), 0.5) == (0, 2)   # never crosses
    assert first_crossing_decision(np.array([0.9]), 0.5) == (1, 1)        # immediate


def test_erde_rewards_early_true_positives():
    early = erde([1, 1], [1, 1], [1, 1], o=5)
    expected = 1.0 / (1.0 + math.exp(5 - 1))   # overflow-safe lc_o(k) form
    assert abs(early - expected) < 1e-9
    late = erde([1, 1], [1, 1], [20, 20], o=5)
    assert late > early          # detecting later costs more


def test_erde_penalizes_fn_zero_for_tn_and_fp_base_rate():
    assert abs(erde([1], [0], [3], o=5) - 1.0) < 1e-9          # FN -> 1
    assert erde([0], [0], [3], o=5) == 0.0                      # TN -> 0
    assert abs(erde([0], [1], [2], o=5, c_fp=0.5) - 0.5) < 1e-9  # FP -> c_fp


def test_latency_weighted_f1_perfect_early_detection():
    r = latency_weighted_f1([1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 5, 5])
    assert r["f1"] == 1.0
    assert r["speed"] > 0.99           # k=1 -> penalty ~ 0
    assert r["latency_weighted_f1"] > 0.99
    assert r["median_latency"] == 1.0
