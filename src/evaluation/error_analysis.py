"""Error analysis: confusion buckets, hardest examples, length effect."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_predictions(df: pd.DataFrame, y_score: np.ndarray, target: str, threshold: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    out[f"score_{target}"] = y_score
    out[f"pred_{target}"] = (y_score >= threshold).astype(int)
    out[f"y_{target}"] = (out[f"label_{target}"].astype(float).fillna(0.0) >= 0.5).astype(int)
    out[f"err_{target}"] = (out[f"pred_{target}"] != out[f"y_{target}"]).astype(int)
    out[f"bucket_{target}"] = np.where(
        (out[f"y_{target}"] == 1) & (out[f"pred_{target}"] == 1), "TP",
        np.where(
            (out[f"y_{target}"] == 0) & (out[f"pred_{target}"] == 0), "TN",
            np.where(
                (out[f"y_{target}"] == 0) & (out[f"pred_{target}"] == 1), "FP",
                "FN",
            ),
        ),
    )
    return out


def hardest_examples(df: pd.DataFrame, target: str, n: int = 20) -> pd.DataFrame:
    """Return n posts with the largest |score - label| (most-confidently-wrong)."""
    score_col = f"score_{target}"
    y_col = f"y_{target}"
    if score_col not in df.columns:
        raise ValueError(f"Run add_predictions first (missing {score_col}).")
    work = df.copy()
    work["_dist"] = (work[score_col] - work[y_col]).abs()
    return work.sort_values("_dist", ascending=False).head(n).drop(columns=["_dist"])


def confusion_by_subgroup(
    df: pd.DataFrame,
    target: str,
    group_col: str = "subreddit",
) -> pd.DataFrame:
    """Per-subgroup confusion counts + per-subgroup F1, precision, recall."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    rows = []
    for grp, sub in df.groupby(group_col):
        y = sub[f"y_{target}"].values
        p = sub[f"pred_{target}"].values
        rows.append({
            group_col: grp,
            "n": len(sub),
            "n_pos": int(y.sum()),
            "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)),
            "f1": float(f1_score(y, p, zero_division=0)),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


def length_effect(df: pd.DataFrame, target: str, n_bins: int = 5) -> pd.DataFrame:
    """Does accuracy depend on post length? Equal-frequency binning."""
    work = df.copy()
    work["_len"] = work["clean_text"].astype(str).str.len()
    work["_bin"] = pd.qcut(work["_len"], q=n_bins, duplicates="drop")
    return (
        work.groupby("_bin", observed=True)
        .apply(lambda s: pd.Series({
            "n": len(s),
            "mean_len": float(s["_len"].mean()),
            "f1": _safe_f1(s[f"y_{target}"].values, s[f"pred_{target}"].values),
        }))
        .reset_index()
    )


def _safe_f1(y, p) -> float:
    from sklearn.metrics import f1_score

    if len(y) == 0:
        return float("nan")
    return float(f1_score(y, p, zero_division=0))
