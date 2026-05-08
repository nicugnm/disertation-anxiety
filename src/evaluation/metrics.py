"""Core metrics with bootstrap CIs and calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def basic_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Standard binary metrics + threshold-free PR/ROC scores."""
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "support_pos": int(y_true.sum()),
        "support_neg": int((1 - y_true).sum()),
    }
    # Some splits may have only one class — guard.
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1",
    n_iters: int = 500,
    threshold: float = 0.5,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Returns (point_estimate, lo, hi) for the requested metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def _eval(idx: np.ndarray) -> float:
        m = basic_metrics(y_true[idx], y_score[idx], threshold=threshold)
        return m.get(metric, float("nan"))

    point = _eval(np.arange(n))
    samples: list[float] = []
    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        v = _eval(idx)
        if not np.isnan(v):
            samples.append(v)
    if not samples:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return point, lo, hi


def calibration_curve_data(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10):
    """Returns (bin_centers, observed_freq, predicted_mean, counts) for a reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_score, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    observed = np.full(n_bins, np.nan)
    predicted = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            observed[b] = float(y_true[mask].mean())
            predicted[b] = float(y_score[mask].mean())
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, observed, predicted, counts


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    _, observed, predicted, counts = calibration_curve_data(y_true, y_score, n_bins=n_bins)
    n = counts.sum()
    if n == 0:
        return float("nan")
    ece = 0.0
    for b in range(n_bins):
        if counts[b] == 0 or np.isnan(observed[b]):
            continue
        ece += (counts[b] / n) * abs(observed[b] - predicted[b])
    return float(ece)


def best_threshold_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Pick the threshold that maximizes F1 on the held set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    # f1 has length len(thresholds)+1; align by trimming
    if len(thresholds) == 0:
        return 0.5, 0.0
    best_i = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best_i]), float(f1[best_i])


def full_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
    bootstrap: bool = True,
) -> dict:
    """Full metric bundle including bootstrap CIs and calibration."""
    if threshold is None:
        threshold, _ = best_threshold_f1(y_true, y_score)
    base = basic_metrics(y_true, y_score, threshold=threshold)
    out = {**base, "threshold": float(threshold), "ece": expected_calibration_error(y_true, y_score)}
    if bootstrap:
        for m in ("f1", "auroc", "auprc"):
            point, lo, hi = bootstrap_ci(y_true, y_score, metric=m, threshold=threshold)
            out[f"{m}_ci_lo"] = lo
            out[f"{m}_ci_hi"] = hi
    return out
