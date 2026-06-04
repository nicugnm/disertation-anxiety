"""eRisk early-detection metrics (CLEF eRisk; Losada & Crestani 2016, Sadeque 2018).

Given per-user records — true label, system decision, and the number of posts read
before that decision (latency) — these score how EARLY and ACCURATELY a system
flags at-risk users from a time-ordered post stream:

  - erde(o)              — Early Risk Detection Error; penalizes FP, FN, and TPs
                           that arrive after reading more than `o` posts. Lower better.
  - latency_weighted_f1  — F1 scaled by a speed factor from the median posts-to-flag.
  - first_crossing_decision — the streaming decision rule: flag at the first post
                           whose score crosses the threshold.
"""
from __future__ import annotations

import numpy as np


def first_crossing_decision(scores, threshold: float) -> tuple[int, int]:
    """Flag positive at the first post whose score >= threshold.

    Returns (decision, latency): latency is the 1-based index of the flagging post,
    or the full stream length if it never crosses (decision 0)."""
    scores = np.asarray(scores, dtype=float)
    crossings = np.where(scores >= threshold)[0]
    if crossings.size:
        return 1, int(crossings[0]) + 1
    return 0, int(len(scores))


def erde(y_true, decision, latency, o: int, c_fp: float | None = None) -> float:
    """Early Risk Detection Error with penalty horizon `o` (lower is better).

    TP cost = latency cost lc_o(k) = 1 - 1/(1+e^(k-o)); FP cost = c_fp
    (default = positive base rate); FN cost = 1; TN cost = 0. Mean over users."""
    y = np.asarray(y_true, dtype=int)
    d = np.asarray(decision, dtype=int)
    k = np.asarray(latency, dtype=float)
    n = len(y)
    if n == 0:
        return 0.0
    if c_fp is None:
        c_fp = float(y.sum()) / n
    costs = np.zeros(n, dtype=float)
    tp = (y == 1) & (d == 1)
    # lc_o(k) = 1 - 1/(1+e^(k-o)) == 1/(1+e^(o-k)); the latter is overflow-safe for large k
    costs[tp] = 1.0 / (1.0 + np.exp(o - k[tp]))
    costs[(y == 0) & (d == 1)] = c_fp          # FP
    costs[(y == 1) & (d == 0)] = 1.0           # FN
    return float(costs.mean())


def latency_weighted_f1(y_true, decision, latency, p: float = 0.0078) -> dict:
    """F1 × speed, where speed = 1 - median latency penalty over true positives.

    penalty(k) = -1 + 2/(1+e^(-p(k-1))) (Sadeque et al. 2018; p=0.0078 is the eRisk
    default). Also returns the raw median posts-to-detection (the interpretable
    earliness number, useful when streams are short)."""
    from sklearn.metrics import f1_score

    y = np.asarray(y_true, dtype=int)
    d = np.asarray(decision, dtype=int)
    k = np.asarray(latency, dtype=float)
    f1 = float(f1_score(y, d, zero_division=0))
    tp = (y == 1) & (d == 1)
    if tp.sum() == 0:
        return {"f1": f1, "speed": 0.0, "latency_weighted_f1": 0.0, "median_latency": None}
    penalties = -1.0 + 2.0 / (1.0 + np.exp(-p * (k[tp] - 1)))
    speed = float(1.0 - np.median(penalties))
    return {"f1": f1, "speed": speed, "latency_weighted_f1": f1 * speed,
            "median_latency": float(np.median(k[tp]))}
