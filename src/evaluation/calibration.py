"""Post-hoc probability calibration — temperature scaling (Guo et al., 2017).

Temperature scaling learns a single scalar T and rescales logits z -> z/T before
the sigmoid. T>1 softens overconfident scores; T<1 sharpens under-confident ones.
The transform is strictly monotonic, so ranking metrics (AUROC/AUPRC) are
unchanged — only ECE/Brier and the meaning of the probability/threshold improve.

Fits on probabilities (recovers logits internally) so it can calibrate any model
that exposes `predict_proba`, including the TF-IDF baseline and the transformers.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureScaler:
    """Single-parameter temperature scaling for binary/sigmoid probabilities."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, proba, y, bounds: tuple[float, float] = (0.05, 20.0)) -> "TemperatureScaler":
        """Fit T by minimising negative log-likelihood on a held-out calibration set."""
        from scipy.optimize import minimize_scalar

        z = _to_logit(proba)
        y = np.asarray(y, dtype=float)

        def nll(t: float) -> float:
            p = np.clip(_sigmoid(z / t), _EPS, 1.0 - _EPS)
            return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        res = minimize_scalar(nll, bounds=bounds, method="bounded")
        self.temperature = float(res.x)
        return self

    def transform(self, proba) -> np.ndarray:
        return _sigmoid(_to_logit(proba) / self.temperature)

    def fit_transform(self, proba, y) -> np.ndarray:
        return self.fit(proba, y).transform(proba)
