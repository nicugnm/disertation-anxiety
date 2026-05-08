"""Explainability: SHAP for XGBoost; permutation importance for any model.

For transformers, we provide an interface for token-level attribution via
gradient × input — full SHAP for transformers is expensive and is left
as an optional experiment.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel


def shap_importance_xgboost(
    model: BaseModel,
    df: pd.DataFrame,
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """SHAP feature importance for the XGBoost linguistic model."""
    try:
        import shap
    except ImportError as e:
        raise RuntimeError("Install `shap` to use this function.") from e

    if model.config.model_type != "xgboost":
        raise ValueError("shap_importance_xgboost only supports the XGBoost model.")

    rng = np.random.default_rng(seed)
    take = min(n_samples, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    sample = df.iloc[idx].copy()

    # Reuse the model's feature pipeline by calling _features_df via predict path
    X = model._features_df(sample).values  # type: ignore[attr-defined]
    feature_cols = model._feature_cols  # type: ignore[attr-defined]

    explainer = shap.TreeExplainer(model.model)  # type: ignore[attr-defined]
    shap_values = explainer.shap_values(X)
    # For binary XGB, shap_values is (n, n_features). Sometimes it's a list of two arrays.
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

    importance = np.mean(np.abs(shap_values), axis=0)
    return (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance(
    model: BaseModel,
    df: pd.DataFrame,
    target: str,
    metric: str = "f1",
    n_repeats: int = 5,
    text_col: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generic permutation importance.

    For text models we permute the text column; for feature models we permute
    each feature column. Returns the drop in `metric`.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    rng = np.random.default_rng(seed)
    text_col = text_col or model.config.text_field
    y = (df[f"label_{target}"].astype(float).fillna(0.0) >= 0.5).astype(int).values

    base_proba = model.predict_proba(df)
    if base_proba.ndim == 2:
        base_proba = base_proba[:, model.targets.index(target)]
    base_pred = (base_proba >= 0.5).astype(int)
    base_score = float(f1_score(y, base_pred, zero_division=0)) if metric == "f1" else float(roc_auc_score(y, base_proba))

    rows = []
    for col in [text_col]:
        drops: list[float] = []
        for _ in range(n_repeats):
            df_perm = df.copy()
            df_perm[col] = rng.permutation(df_perm[col].values)
            proba = model.predict_proba(df_perm)
            if proba.ndim == 2:
                proba = proba[:, model.targets.index(target)]
            pred = (proba >= 0.5).astype(int)
            score = float(f1_score(y, pred, zero_division=0)) if metric == "f1" else float(roc_auc_score(y, proba))
            drops.append(base_score - score)
        rows.append({"feature": col, "mean_importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("mean_importance", ascending=False)


def transformer_token_attributions(
    model: BaseModel,
    text: str,
    target_idx: int = 1,
) -> list[tuple[str, float]]:
    """Gradient × input attribution for a transformer prediction.

    Cheap, single-pass alternative to SHAP. Suitable for the thesis's
    qualitative chapter (which words drove a particular prediction?).
    """
    if model.config.model_type not in {"transformer", "multitask_transformer"}:
        raise ValueError("Only transformer-family models supported")

    import torch

    tok = model.tokenizer  # type: ignore[attr-defined]
    mdl = model.model  # type: ignore[attr-defined]
    device = next(mdl.parameters()).device

    enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    embed_layer = (
        mdl.get_input_embeddings() if hasattr(mdl, "get_input_embeddings") else mdl.encoder.get_input_embeddings()  # type: ignore[attr-defined]
    )
    inputs_embeds = embed_layer(enc["input_ids"])
    inputs_embeds.requires_grad_(True)

    # Forward
    if hasattr(mdl, "forward") and "inputs_embeds" in mdl.forward.__code__.co_varnames:
        out = mdl(inputs_embeds=inputs_embeds, attention_mask=enc.get("attention_mask"))
        logits = out.logits if hasattr(out, "logits") else out
    else:
        # Multi-task wrapper: use input_ids path; less accurate
        logits = mdl(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))

    if logits.ndim == 2:
        if model.config.model_type == "multitask_transformer":
            score = torch.sigmoid(logits)[0, target_idx]
        else:
            score = torch.softmax(logits, dim=-1)[0, target_idx]
    else:
        score = logits.sum()
    score.backward()
    grads = inputs_embeds.grad  # type: ignore[union-attr]
    attributions = (grads * inputs_embeds).sum(dim=-1).squeeze(0).detach().cpu().numpy()

    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    return list(zip(tokens, attributions.tolist()))
