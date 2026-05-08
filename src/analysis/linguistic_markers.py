"""Statistical tests on linguistic features by label.

Drives the linguistic-analysis chapter (RQ2). For each feature:
  - mean ± std by group
  - effect size (Cohen's d)
  - Mann-Whitney U test (non-parametric — text features are not normal)
  - multiple-comparison correction (Benjamini-Hochberg)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.features.linguistic import extract_dataframe, feature_columns


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _bh_correct(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def compare_features_by_label(
    df: pd.DataFrame,
    target: str,
    text_col: str = "clean_text",
) -> pd.DataFrame:
    """For every linguistic feature, compare positives vs negatives on `label_<target>`."""
    feat_df = extract_dataframe(df, text_col=text_col)
    cols = feature_columns(feat_df)
    label_col = f"label_{target}"
    y = (feat_df[label_col].astype(float).fillna(0.0) >= 0.5).astype(int).values

    rows = []
    pvals: list[float] = []
    for c in cols:
        x = feat_df[c].astype(float).values
        a = x[y == 1]
        b = x[y == 0]
        if len(a) == 0 or len(b) == 0:
            rows.append({"feature": c, "n_pos": int(len(a)), "n_neg": int(len(b))})
            pvals.append(1.0)
            continue
        try:
            u_stat, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            pval = 1.0
            u_stat = float("nan")
        rows.append({
            "feature": c,
            "n_pos": int(len(a)),
            "n_neg": int(len(b)),
            "mean_pos": float(np.mean(a)),
            "mean_neg": float(np.mean(b)),
            "cohen_d": _cohens_d(a, b),
            "u_stat": float(u_stat),
            "p_raw": float(pval),
        })
        pvals.append(pval)

    out = pd.DataFrame(rows)
    out["p_bh"] = _bh_correct(np.array(pvals))
    out["significant"] = out["p_bh"] < 0.05
    out = out.sort_values("cohen_d", key=np.abs, ascending=False).reset_index(drop=True)
    return out
