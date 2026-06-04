"""Tests for temperature scaling — recovers a known temperature, reduces ECE,
preserves ranking, and leaves already-calibrated scores ~unchanged."""
from __future__ import annotations

import numpy as np

from src.evaluation.calibration import TemperatureScaler
from src.evaluation.metrics import expected_calibration_error


def _calibrated_sample(n: int, seed: int):
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.5, n)              # true logits
    p_true = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p_true).astype(int)
    return z, p_true, y


def test_recovers_known_overconfidence_and_reduces_ece():
    z, _, y = _calibrated_sample(20000, seed=0)
    p_over = 1.0 / (1.0 + np.exp(-z * 3.0))   # overconfident model (T_true ~ 3)
    scaler = TemperatureScaler().fit(p_over, y)
    assert 2.0 < scaler.temperature < 4.5
    ece_before = expected_calibration_error(y, p_over)
    ece_after = expected_calibration_error(y, scaler.transform(p_over))
    assert ece_after < ece_before


def test_transform_is_monotonic_preserving_ranking():
    p = np.array([0.01, 0.2, 0.5, 0.7, 0.99])
    scaler = TemperatureScaler()
    scaler.temperature = 2.5
    out = scaler.transform(p)
    assert np.array_equal(np.argsort(out), np.argsort(p))


def test_calibrated_data_yields_temperature_near_one():
    _, p_true, y = _calibrated_sample(20000, seed=1)
    scaler = TemperatureScaler().fit(p_true, y)
    assert 0.7 < scaler.temperature < 1.4
