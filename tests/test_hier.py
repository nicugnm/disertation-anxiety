"""Offline tests for the hierarchical user-model — user-sequence construction,
aggregator masking, registry, and the predict_proba broadcast contract.
(The frozen-encoder forward + training are exercised on GPU, not here.)"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.hier import HierUserModel, build_aggregator
from src.models.hier_dataset import build_user_sequences
from src.models.registry import build_model
from src.utils.config import ModelConfig


def _cfg(targets=None):
    return ModelConfig(
        name="hier_test", model_type="hier_user", text_field="clean_text",
        target=None, targets=targets or ["anxiety", "health_anxiety", "depression"], extra={},
    )


def test_build_user_sequences_order_pad_mask_label():
    df = pd.DataFrame({
        "author_hash": ["a", "a", "a", "b"],
        "created_utc": [30.0, 10.0, 20.0, 5.0],         # author a out of order
        "label_anxiety": [1, 0, 0, 0],
        "label_health_anxiety": [0, 0, 0, 1],
        "label_depression": [0, 0, 0, 0],
        "clean_text": ["p3", "p1", "p2", "q1"],
    })
    users, post_idx, post_mask, y = build_user_sequences(
        df, ["anxiety", "health_anxiety", "depression"], max_posts=4)
    ui = users.index("a")
    # author a's 3 posts ordered by time (10,20,30) -> df rows [1,2,0], then padding
    assert list(post_idx[ui][:3]) == [1, 2, 0]
    assert list(post_mask[ui]) == [True, True, True, False]
    assert list(y[ui]) == [1.0, 0.0, 0.0]               # user-positive for anxiety (max over posts)
    bi = users.index("b")
    assert list(y[bi]) == [0.0, 1.0, 0.0]               # health_anxiety positive


def test_build_user_sequences_truncates_recent():
    df = pd.DataFrame({
        "author_hash": ["a"] * 6,
        "created_utc": [1.0, 2, 3, 4, 5, 6],
        "label_anxiety": [0, 0, 0, 0, 0, 1],
        "clean_text": [f"p{i}" for i in range(6)],
    })
    _, post_idx, post_mask, _ = build_user_sequences(df, ["anxiety"], max_posts=3, order="recent")
    assert post_mask[0].sum() == 3
    assert list(post_idx[0][:3]) == [3, 4, 5]            # most recent 3, chronological


def test_aggregator_attention_respects_mask():
    import torch
    agg = build_aggregator("attention", embed_dim=4, hidden=5, dropout=0.0)
    post_emb = torch.tensor([[[1.0, 1, 1, 1], [9.0, 9, 9, 9]]])   # post 1 is padding
    mask = torch.tensor([[True, False]])
    out = agg(post_emb, mask)
    assert out.shape == (1, 5)
    assert torch.allclose(out, torch.tanh(agg.proj(post_emb[:, 0, :])), atol=1e-5)


def test_aggregator_mean_shape():
    import torch
    agg = build_aggregator("mean", embed_dim=8, hidden=6, dropout=0.0)
    out = agg(torch.randn(3, 5, 8), torch.ones(3, 5, dtype=torch.bool))
    assert out.shape == (3, 6)


def test_registry_builds_hier_user():
    m = build_model(_cfg())
    assert m.__class__.__name__ == "HierUserModel"
    assert m._aggregator == "attention" and m._max_posts == 64


def test_config_requires_targets():
    with pytest.raises(Exception):
        ModelConfig(name="x", model_type="hier_user", text_field="clean_text", target=None, targets=None)


def test_predict_proba_broadcasts_user_scores(monkeypatch):
    m = HierUserModel(_cfg())
    df = pd.DataFrame({"author_hash": ["a", "a", "b"], "clean_text": ["x", "y", "z"]})
    monkeypatch.setattr(m, "_user_scores", lambda d: {
        "a": np.array([0.9, 0.1, 0.2], np.float32), "b": np.array([0.3, 0.4, 0.5], np.float32)})
    out = m.predict_proba(df)
    assert out.shape == (3, 3)
    assert np.allclose(out[0], [0.9, 0.1, 0.2]) and np.allclose(out[1], [0.9, 0.1, 0.2])
    assert np.allclose(out[2], [0.3, 0.4, 0.5])           # broadcast per author
