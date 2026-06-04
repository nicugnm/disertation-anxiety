"""Tests for confident-disagreement weak-label filtering."""
from __future__ import annotations

import numpy as np

from src.labeling.filtering import confident_label_issues


def test_confident_label_issues_flags_both_directions():
    y = [1, 1, 0, 0, 1]
    s = [0.95, 0.02, 0.01, 0.97, 0.5]
    mask, counts = confident_label_issues(y, s, low=0.1, high=0.9)
    # idx1: weak=1 but score 0.02 -> false positive; idx3: weak=0 but 0.97 -> false negative
    assert list(mask) == [False, True, False, True, False]
    assert counts["likely_false_pos"] == 1
    assert counts["likely_false_neg"] == 1
    assert counts["total_flagged"] == 2


def test_no_issues_when_model_agrees():
    y = [1, 1, 0, 0]
    s = [0.8, 0.7, 0.2, 0.3]
    mask, counts = confident_label_issues(y, s)
    assert not mask.any()
    assert counts["total_flagged"] == 0
