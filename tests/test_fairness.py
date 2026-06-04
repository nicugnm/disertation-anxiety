"""Tests for the fairness utilities — self-report extraction and subgroup gaps."""
from __future__ import annotations

import numpy as np

from src.evaluation.fairness import (
    extract_age,
    extract_gender,
    fairness_gaps,
    subgroup_metrics,
)


def test_extract_gender():
    assert extract_gender("23M here, been anxious for years") == "M"
    assert extract_gender("F30 and struggling") == "F"
    assert extract_gender("i'm a woman dealing with panic") == "F"
    assert extract_gender("as a guy this is hard") == "M"
    assert extract_gender("just feeling anxious today") is None
    assert extract_gender("my brother is 23M and my sister is 25F") is None  # conflicting -> None


def test_extract_age():
    assert extract_age("23M here") == 23
    assert extract_age("i'm 34 and tired") == 34
    assert extract_age("as a 19 year old this sucks") == 19
    assert extract_age("no age mentioned") is None
    assert extract_age("the year 2020 was hard") is None  # 2020 out of [13,99]


def test_subgroup_metrics_and_gaps():
    # group A: perfect; group B: misses half the positives (lower TPR)
    y_true = np.array([1] * 20 + [0] * 20 + [1] * 20 + [0] * 20)
    y_pred = np.array([1] * 20 + [0] * 20 + [1] * 10 + [0] * 10 + [0] * 20)
    groups = np.array(["A"] * 40 + ["B"] * 40)
    sub = subgroup_metrics(y_true, y_pred, groups, min_n=10, min_pos=5)
    assert set(sub["group"]) == {"A", "B"}
    a = sub.set_index("group").loc["A"]; b = sub.set_index("group").loc["B"]
    assert a["tpr"] == 1.0 and b["tpr"] == 0.5
    gaps = fairness_gaps(sub)
    assert abs(gaps["tpr_gap"] - 0.5) < 1e-9
    assert gaps["equalized_odds_diff"] >= gaps["tpr_gap"] - 1e-9


def test_subgroup_metrics_drops_sparse_groups():
    y_true = np.array([1, 0, 1, 0] * 20 + [1, 0])
    y_pred = np.array([1, 0, 1, 0] * 20 + [1, 0])
    groups = np.array(["big"] * 80 + ["tiny"] * 2)
    sub = subgroup_metrics(y_true, y_pred, groups, min_n=30, min_pos=5)
    assert list(sub["group"]) == ["big"]   # tiny dropped
