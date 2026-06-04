"""Helpers for summarising SHAP values from the XGBoost linguistic model."""
from __future__ import annotations

import numpy as np
import pandas as pd


def xgb_shap_values(booster_or_model, X_df: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP via XGBoost's native `pred_contribs` (version-robust — avoids
    shap.TreeExplainer's base_score parsing bug). Returns (n_samples, n_features);
    the bias/base column is dropped."""
    import xgboost as xgb

    booster = booster_or_model.get_booster() if hasattr(booster_or_model, "get_booster") else booster_or_model
    dm = xgb.DMatrix(np.asarray(X_df, dtype=float), feature_names=list(X_df.columns))
    contribs = np.asarray(booster.predict(dm, pred_contribs=True))
    return contribs[:, :-1]


def summarize_shap(shap_values, X_df: pd.DataFrame) -> pd.DataFrame:
    """Rank features by mean|SHAP| and attach a direction.

    direction is '+' when higher feature values push the prediction toward the
    positive class (corr(feature, shap) >= 0), '-' otherwise, '0' if degenerate.
    Returns a DataFrame [feature, mean_abs_shap, direction] sorted by importance.
    """
    sv = np.asarray(shap_values, dtype=float)
    cols = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)
    if sv.shape != X.shape:
        raise ValueError(f"shap_values shape {sv.shape} != X shape {X.shape}")

    mean_abs = np.abs(sv).mean(axis=0)
    directions = []
    for j in range(sv.shape[1]):
        xj, sj = X[:, j], sv[:, j]
        if np.std(xj) < 1e-12 or np.std(sj) < 1e-12:
            directions.append("0")
        else:
            directions.append("+" if np.corrcoef(xj, sj)[0, 1] >= 0 else "-")

    return (
        pd.DataFrame({"feature": cols, "mean_abs_shap": mean_abs, "direction": directions})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
