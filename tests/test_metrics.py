import numpy as np

from src.evaluation.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    full_report,
)


def test_full_report_keys():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    score = rng.uniform(0, 1, size=200)
    report = full_report(y, score, bootstrap=True)
    for k in ("f1", "auroc", "auprc", "ece", "threshold"):
        assert k in report


def test_bootstrap_ci_runs():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=100)
    score = rng.uniform(0, 1, size=100)
    point, lo, hi = bootstrap_ci(y, score, metric="f1", n_iters=50)
    assert lo <= point <= hi or np.isnan(point)


def test_ece_well_calibrated_low():
    # Perfectly calibrated: y matches score deterministically.
    n = 1000
    rng = np.random.default_rng(2)
    score = rng.uniform(0, 1, size=n)
    y = (rng.uniform(0, 1, size=n) < score).astype(int)
    ece = expected_calibration_error(y, score, n_bins=10)
    assert ece < 0.1
