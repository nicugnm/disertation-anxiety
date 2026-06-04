"""Tests for robustness perturbations — they edit text, stay meaning-adjacent,
are deterministic under a seed, and the flip-rate metric is correct."""
from __future__ import annotations

import numpy as np

from src.evaluation.robustness import (
    PERTURBATIONS,
    char_swap,
    flip_rate,
    mean_abs_score_drift,
    punct_strip,
)


def test_perturbations_change_text_but_keep_word_count_mostly():
    text = "i have been feeling really anxious about everything lately honestly"
    for name, fn in PERTURBATIONS.items():
        out = fn(text, np.random.default_rng(0), p=1.0)
        assert isinstance(out, str) and out
        if name != "punct_strip":
            assert out != text, name                         # something changed
            assert len(out.split()) == len(text.split()), name  # words not lost/merged


def test_perturbation_is_deterministic_under_seed():
    text = "persistent worry and racing thoughts every single night"
    a = char_swap(text, np.random.default_rng(7), p=0.5)
    b = char_swap(text, np.random.default_rng(7), p=0.5)
    assert a == b


def test_punct_strip_removes_punctuation():
    assert punct_strip("anxiety, panic! and... fear?", np.random.default_rng(0)) == "anxiety panic and fear"


def test_flip_rate_and_drift():
    assert flip_rate([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5
    assert abs(mean_abs_score_drift([0.2, 0.9], [0.3, 0.5]) - 0.25) < 1e-9


def test_handles_short_and_empty_text():
    for fn in PERTURBATIONS.values():
        assert isinstance(fn("", np.random.default_rng(0)), str)
        assert isinstance(fn("ok", np.random.default_rng(0)), str)
