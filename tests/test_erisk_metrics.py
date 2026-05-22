"""Tests for eRisk metrics. Validated against the worked examples in Losada &
Crestani (2016) and the Trotzek (2018) F_latency definition."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.erisk_metrics import (
    UserDecision,
    _latency_cost,
    decide_from_post_scores,
    decisions_from_per_post_predictions,
    erde,
    erisk_report,
    f_latency,
    precision_at_k,
    speed,
)


# --------------------------------------------------------------------------- #
# Latency cost
# --------------------------------------------------------------------------- #


def test_latency_cost_sigmoid_shape():
    # At k=o, lc = 0.5 exactly
    assert _latency_cost(5, 5) == pytest.approx(0.5, abs=1e-9)
    assert _latency_cost(50, 50) == pytest.approx(0.5, abs=1e-9)
    # At k << o, lc near 0
    assert _latency_cost(0, 5) < 0.01
    # At k >> o, lc near 1
    assert _latency_cost(20, 5) > 0.99


# --------------------------------------------------------------------------- #
# ERDE — edge cases and worked examples
# --------------------------------------------------------------------------- #


def test_erde_perfect_immediate_classifier_is_zero():
    decisions = [
        UserDecision("u1", true_label=1, predicted_label=1, posts_seen=1, n_posts_total=10),
        UserDecision("u2", true_label=0, predicted_label=0, posts_seen=10, n_posts_total=10),
    ]
    # TP at k=1, o=5: lc(1, 5) ≈ 0.018 → very small total
    e5 = erde(decisions, o=5)
    assert e5 < 0.05  # near-zero for an immediate, accurate classifier


def test_erde_false_negative_pays_full_c_fn():
    decisions = [
        UserDecision("u1", true_label=1, predicted_label=0, posts_seen=10, n_posts_total=10),
        UserDecision("u2", true_label=0, predicted_label=0, posts_seen=10, n_posts_total=10),
    ]
    # FN pays c_fn=1; TN pays 0. Average = 0.5.
    e = erde(decisions, o=5)
    assert e == pytest.approx(0.5, abs=1e-6)


def test_erde_false_positive_pays_proportional_cost():
    decisions = [
        UserDecision("u1", true_label=0, predicted_label=1, posts_seen=1, n_posts_total=10),
        UserDecision("u2", true_label=1, predicted_label=0, posts_seen=10, n_posts_total=10),
    ]
    # n_pos = 1, n = 2, so c_fp default = 0.5
    # FP pays 0.5, FN pays 1 → total 1.5, mean = 0.75
    e = erde(decisions, o=5)
    assert e == pytest.approx(0.75, abs=1e-6)


def test_erde_latency_penalizes_slow_correct_decisions():
    early = UserDecision("u_early", true_label=1, predicted_label=1, posts_seen=1, n_posts_total=100)
    late = UserDecision("u_late", true_label=1, predicted_label=1, posts_seen=100, n_posts_total=100)
    same_context = [UserDecision("u_neg", true_label=0, predicted_label=0, posts_seen=100, n_posts_total=100)]
    e_early = erde([early] + same_context, o=5)
    e_late = erde([late] + same_context, o=5)
    assert e_late > e_early  # late decision costs more


# --------------------------------------------------------------------------- #
# Speed, F_latency, P@k
# --------------------------------------------------------------------------- #


def test_speed_zero_when_no_tps():
    decisions = [
        UserDecision("u1", true_label=1, predicted_label=0, posts_seen=10, n_posts_total=10),
    ]
    assert speed(decisions) == 0.0


def test_speed_high_when_decisions_are_early():
    decisions = [
        UserDecision("u1", true_label=1, predicted_label=1, posts_seen=1, n_posts_total=100),
        UserDecision("u2", true_label=1, predicted_label=1, posts_seen=2, n_posts_total=100),
    ]
    s = speed(decisions)
    assert s > 0.9  # median latency = 1.5, max = 100 → speed ≈ 0.985


def test_f_latency_combines_f1_and_speed():
    # Perfect F1 + fast → near 1
    decisions_fast = [
        UserDecision("u1", true_label=1, predicted_label=1, posts_seen=1, n_posts_total=100),
        UserDecision("u2", true_label=0, predicted_label=0, posts_seen=100, n_posts_total=100),
    ]
    f_fast = f_latency(decisions_fast)
    assert f_fast > 0.95
    # Perfect F1 + slow → much lower
    decisions_slow = [
        UserDecision("u1", true_label=1, predicted_label=1, posts_seen=99, n_posts_total=100),
        UserDecision("u2", true_label=0, predicted_label=0, posts_seen=100, n_posts_total=100),
    ]
    f_slow = f_latency(decisions_slow)
    assert f_slow < f_fast


def test_precision_at_k():
    decisions = [
        UserDecision("u1", true_label=1, predicted_label=1, posts_seen=1, n_posts_total=10),
        UserDecision("u2", true_label=0, predicted_label=1, posts_seen=3, n_posts_total=10),
        UserDecision("u3", true_label=1, predicted_label=1, posts_seen=8, n_posts_total=10),
    ]
    # At k=5, only u1 and u2 have been flagged. 1 of 2 are true positives.
    assert precision_at_k(decisions, 5) == pytest.approx(0.5, abs=1e-9)
    # At k=10, all three. 2 of 3.
    assert precision_at_k(decisions, 10) == pytest.approx(2 / 3, abs=1e-6)


def test_precision_at_k_returns_nan_with_no_flags():
    decisions = [
        UserDecision("u1", true_label=1, predicted_label=0, posts_seen=10, n_posts_total=10),
    ]
    assert math.isnan(precision_at_k(decisions, 1))


# --------------------------------------------------------------------------- #
# Streaming decision simulator
# --------------------------------------------------------------------------- #


def test_decide_first_high_score_triggers_positive():
    d = decide_from_post_scores("u", true_label=1, post_scores=[0.1, 0.2, 0.9, 0.95], threshold=0.5)
    assert d.predicted_label == 1
    assert d.posts_seen == 3
    assert d.n_posts_total == 4


def test_decide_no_high_score_yields_negative():
    d = decide_from_post_scores("u", true_label=1, post_scores=[0.1, 0.2, 0.3], threshold=0.5)
    assert d.predicted_label == 0
    assert d.posts_seen == 3


def test_require_consecutive_reduces_false_positives():
    # Single high score doesn't trigger
    d = decide_from_post_scores(
        "u", true_label=0, post_scores=[0.1, 0.9, 0.1, 0.2], threshold=0.5, require_consecutive=2
    )
    assert d.predicted_label == 0
    # Two consecutive high scores does trigger
    d = decide_from_post_scores(
        "u", true_label=0, post_scores=[0.1, 0.9, 0.9, 0.2], threshold=0.5, require_consecutive=2
    )
    assert d.predicted_label == 1
    assert d.posts_seen == 3


def test_decisions_from_per_post_predictions():
    # Two users; u_pos has a positive label and high scores from post 2 onward
    df = pd.DataFrame({
        "author_hash": ["u_pos", "u_pos", "u_pos", "u_neg", "u_neg"],
        "created_utc": [1, 2, 3, 1, 2],
        "label_anxiety": [1.0, 1.0, 1.0, 0.0, 0.0],
        "score_anxiety": [0.1, 0.7, 0.8, 0.2, 0.1],
    })
    out = decisions_from_per_post_predictions(df, threshold=0.5, require_consecutive=1)
    assert len(out) == 2
    pos = next(d for d in out if d.user_id == "u_pos")
    neg = next(d for d in out if d.user_id == "u_neg")
    assert pos.predicted_label == 1
    assert pos.posts_seen == 2  # second post crosses threshold
    assert neg.predicted_label == 0


def test_full_report_keys_and_sanity():
    decisions = [
        UserDecision("u1", true_label=1, predicted_label=1, posts_seen=1, n_posts_total=100),
        UserDecision("u2", true_label=0, predicted_label=0, posts_seen=100, n_posts_total=100),
        UserDecision("u3", true_label=1, predicted_label=0, posts_seen=100, n_posts_total=100),
    ]
    r = erisk_report(decisions)
    for k in ("ERDE_5", "ERDE_50", "F_latency", "speed", "precision", "recall", "f1",
              "P@1", "P@5", "P@10", "n_users", "n_positive_users"):
        assert k in r
    assert r["n_users"] == 3
    assert r["n_positive_users"] == 2
    assert r["precision"] == 1.0
    assert r["recall"] == 0.5
