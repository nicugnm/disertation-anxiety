"""eRisk-specific metrics for early-risk detection.

Implements the metrics used in CLEF eRisk shared tasks since 2017. These are
**per-user**, not per-post — they evaluate how quickly and accurately a model
can decide a user's status as their post history streams in.

References:
  - Losada, D. & Crestani, F. (2016). "A test collection for research on
    depression and language use." CLEF. *(original ERDE definition)*
  - Trotzek, M., Koitka, S., Friedrich, C.M. (2018). "Utilizing Neural Networks
    and Linguistic Metadata for Early Detection of Depression Indications in
    Text Sequences." IEEE TKDE. *(F_latency)*

Per-user decision protocol:
  At each post k (1-indexed), the system outputs a decision in {wait, positive,
  negative}. The decision is final once it commits to positive or negative.
  Metrics are computed on the final decision plus the number of posts seen
  before the decision (the *latency*).

ERDE_o (Early Risk Detection Error with parameter o):

  ERDE_o(d, k, true) =
      c_fp                if d == 1 and true == 0
      c_fn                if d == 0 and true == 1
      lc_o(k) * c_tp      if d == 1 and true == 1
      0                   if d == 0 and true == 0

  where lc_o(k) = 1 - 1 / (1 + exp(k - o))   (sigmoid that ramps up with k)

  Standard hyperparameters: c_fp = #pos/#users (proportion of positives),
  c_fn = 1, c_tp = 1. Standard o values: 5 and 50.

  Lower ERDE is better. A perfect early classifier scores 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Decision record
# --------------------------------------------------------------------------- #


@dataclass
class UserDecision:
    """One model's final decision for one user, with latency."""

    user_id: str
    true_label: int           # 0 or 1
    predicted_label: int      # 0 or 1
    posts_seen: int           # k — number of posts the model saw before deciding
    n_posts_total: int        # total posts the user actually has (for sanity / Sslat)


# --------------------------------------------------------------------------- #
# ERDE_o
# --------------------------------------------------------------------------- #


def _latency_cost(k: int, o: int) -> float:
    """Sigmoid latency-cost curve: 1 - sigmoid(k - o).

    At k = o, lc = 0.5. Below k=o it's near 0 (cheap), above it's near 1 (full TP cost).
    """
    return float(1.0 - 1.0 / (1.0 + np.exp(k - o)))


def erde(
    decisions: Iterable[UserDecision],
    o: int,
    c_fp: float | None = None,
    c_fn: float = 1.0,
    c_tp: float = 1.0,
) -> float:
    """Average ERDE_o across users.

    `c_fp` defaults to (# positive users) / (# users) per the eRisk convention,
    so the false-positive penalty scales with class imbalance.
    """
    decisions = list(decisions)
    if not decisions:
        return float("nan")
    n_pos = sum(1 for d in decisions if d.true_label == 1)
    n = len(decisions)
    if c_fp is None:
        c_fp = n_pos / n if n > 0 else 1.0

    total = 0.0
    for d in decisions:
        if d.predicted_label == 1 and d.true_label == 0:
            total += c_fp
        elif d.predicted_label == 0 and d.true_label == 1:
            total += c_fn
        elif d.predicted_label == 1 and d.true_label == 1:
            total += _latency_cost(d.posts_seen, o) * c_tp
        # else true_negative → 0
    return total / n


# --------------------------------------------------------------------------- #
# F_latency (Trotzek et al. 2018) and helper metrics
# --------------------------------------------------------------------------- #


def speed(decisions: Iterable[UserDecision], median_k: float | None = None) -> float:
    """Penalty-style speed score (per Trotzek 2018):

      speed = (1 - median_latency / max_latency)

    Where max_latency = max posts_seen across all true-positive decisions. We use
    median posts_seen across TRUE POSITIVE decisions (correctly identified) by
    default — falling back to median across all positives.

    Returns a value in [0, 1]; higher = faster.
    """
    decisions = list(decisions)
    if not decisions:
        return float("nan")
    pos_correct = [d for d in decisions if d.predicted_label == 1 and d.true_label == 1]
    if not pos_correct:
        return 0.0
    ks = [d.posts_seen for d in pos_correct]
    max_k = max(d.n_posts_total for d in decisions) or 1
    median_k = median_k if median_k is not None else float(np.median(ks))
    return float(max(0.0, 1.0 - median_k / max_k))


def f_latency(decisions: Iterable[UserDecision]) -> float:
    """F1 (positive class) * speed. Per Trotzek 2018.

    Combines correctness with timeliness; lower latency → higher F_latency.
    Range [0, 1]; higher is better.
    """
    decisions = list(decisions)
    if not decisions:
        return float("nan")
    tp = sum(1 for d in decisions if d.predicted_label == 1 and d.true_label == 1)
    fp = sum(1 for d in decisions if d.predicted_label == 1 and d.true_label == 0)
    fn = sum(1 for d in decisions if d.predicted_label == 0 and d.true_label == 1)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return f1 * speed(decisions)


# --------------------------------------------------------------------------- #
# P@k — precision at first k posts
# --------------------------------------------------------------------------- #


def precision_at_k(decisions: Iterable[UserDecision], k: int) -> float:
    """Of users whose first k posts the model classified positive, what fraction are truly positive?

    Decisions made before seeing k posts still count.
    """
    decisions = list(decisions)
    flagged = [d for d in decisions if d.predicted_label == 1 and d.posts_seen <= k]
    if not flagged:
        return float("nan")
    tp = sum(1 for d in flagged if d.true_label == 1)
    return tp / len(flagged)


# --------------------------------------------------------------------------- #
# Full report
# --------------------------------------------------------------------------- #


def erisk_report(decisions: Iterable[UserDecision]) -> dict:
    """Headline eRisk metrics. The numbers you put in the thesis table."""
    decisions = list(decisions)
    n = len(decisions)
    y_true = np.array([d.true_label for d in decisions])
    y_pred = np.array([d.predicted_label for d in decisions])
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n_users": n,
        "n_positive_users": int(y_true.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ERDE_5": float(erde(decisions, o=5)),
        "ERDE_50": float(erde(decisions, o=50)),
        "speed": float(speed(decisions)),
        "F_latency": float(f_latency(decisions)),
        "P@1": float(precision_at_k(decisions, 1)),
        "P@5": float(precision_at_k(decisions, 5)),
        "P@10": float(precision_at_k(decisions, 10)),
    }


# --------------------------------------------------------------------------- #
# Streaming-decision simulator — bridges per-post scores → per-user decisions
# --------------------------------------------------------------------------- #


def decide_from_post_scores(
    user_id: str,
    true_label: int,
    post_scores: list[float],
    threshold: float = 0.5,
    require_consecutive: int = 1,
) -> UserDecision:
    """Convert a sequence of per-post probabilities into one user decision.

    Decision rule: the first time `require_consecutive` consecutive posts have
    score ≥ threshold, decide positive at that k. If we exhaust the sequence
    without that condition, decide negative.

    `require_consecutive > 1` reduces false positives at the cost of latency.
    """
    streak = 0
    for k, score in enumerate(post_scores, start=1):
        if score >= threshold:
            streak += 1
            if streak >= require_consecutive:
                return UserDecision(
                    user_id=user_id,
                    true_label=true_label,
                    predicted_label=1,
                    posts_seen=k,
                    n_posts_total=len(post_scores),
                )
        else:
            streak = 0
    return UserDecision(
        user_id=user_id,
        true_label=true_label,
        predicted_label=0,
        posts_seen=len(post_scores),
        n_posts_total=len(post_scores),
    )


def decisions_from_per_post_predictions(
    df: pd.DataFrame,
    user_col: str = "author_hash",
    true_label_col: str = "label_anxiety",
    score_col: str = "score_anxiety",
    date_col: str = "created_utc",
    threshold: float = 0.5,
    require_consecutive: int = 1,
) -> list[UserDecision]:
    """Build per-user decisions from a DataFrame of per-post predictions.

    Posts are ordered by `date_col` within each user, then streamed.
    `true_label_col` is the user's actual label (we take the max — any positive post → user is positive).
    """
    out: list[UserDecision] = []
    grouped = df.sort_values(date_col).groupby(user_col, sort=False)
    for user_id, grp in grouped:
        true_user_label = int((grp[true_label_col].astype(float) >= 0.5).max())
        scores = grp[score_col].astype(float).tolist()
        out.append(
            decide_from_post_scores(
                user_id=str(user_id),
                true_label=true_user_label,
                post_scores=scores,
                threshold=threshold,
                require_consecutive=require_consecutive,
            )
        )
    return out
