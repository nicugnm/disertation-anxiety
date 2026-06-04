"""Tests for SHAP summarisation — ranking by importance, direction signs, and a
tiny end-to-end TreeExplainer sanity check."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.shap_utils import summarize_shap, xgb_shap_values


def test_summarize_shap_ranks_by_importance_and_signs_direction():
    rng = np.random.default_rng(0)
    n = 300
    x0 = rng.normal(0, 1, n)
    x1 = rng.normal(0, 1, n)
    sv = np.c_[2.0 * x0 + rng.normal(0, 0.01, n),   # strong, + correlated
               -1.0 * x1 + rng.normal(0, 0.01, n)]  # weaker, - correlated
    out = summarize_shap(sv, pd.DataFrame({"a": x0, "b": x1}))

    assert out.iloc[0]["feature"] == "a"            # larger mean|SHAP|
    d = out.set_index("feature")["direction"]
    assert d["a"] == "+" and d["b"] == "-"


def test_summarize_shap_constant_feature_is_degenerate():
    sv = np.c_[np.zeros(50), np.linspace(-1, 1, 50)]
    out = summarize_shap(sv, pd.DataFrame({"flat": np.ones(50), "v": np.linspace(0, 1, 50)}))
    assert out.set_index("feature").loc["flat", "direction"] == "0"


def test_xgb_native_shap_ranks_informative_feature():
    from xgboost import XGBClassifier

    rng = np.random.default_rng(1)
    n = 400
    signal = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    y = (signal + rng.normal(0, 0.3, n) > 0).astype(int)
    X = pd.DataFrame({"signal": signal, "noise": noise})
    model = XGBClassifier(n_estimators=40, max_depth=3, random_state=0, eval_metric="logloss").fit(X, y)

    sv = xgb_shap_values(model, X)          # native TreeSHAP, version-robust
    assert sv.shape == X.shape
    out = summarize_shap(sv, X)
    assert out.iloc[0]["feature"] == "signal"
    assert out.set_index("feature").loc["signal", "direction"] == "+"
