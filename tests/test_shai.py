"""Tests for SHAI symptom decomposition — the right dimensions fire on a
health-anxiety post and stay near zero on neutral text."""
from __future__ import annotations

from src.features.shai import SHAI_DIMENSIONS, score_shai, shai_dimensions


def test_health_anxiety_post_fires_expected_dimensions():
    text = ("I googled my symptoms for hours and I'm convinced I have cancer. "
            "The doctor said I'm fine but I can't accept the test results. "
            "Please reassure me, is this normal?")
    s = score_shai(text)
    assert s["symptom_checking"] > 0      # 'googled my symptoms'
    assert s["serious_illness_fear"] > 0  # 'convinced i have' / 'cancer'
    assert s["difficulty_reassured"] > 0  # "doctor said i'm fine" / "can't accept the test results"
    assert s["reassurance_seeking"] > 0   # 'please reassure me' / 'is this normal'


def test_neutral_post_scores_near_zero():
    s = score_shai("I went for a nice walk in the park and cooked dinner with friends.")
    assert sum(s.values()) == 0.0


def test_score_shai_returns_all_dimensions_and_handles_empty():
    s = score_shai("")
    assert set(s) == set(SHAI_DIMENSIONS) == set(shai_dimensions())
    assert all(v == 0.0 for v in s.values())
