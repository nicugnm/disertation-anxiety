"""Offline tests for the fusion architecture-surgery pieces — attention pooling,
focal loss, fusion MLP, feature normalisation, registry + config wiring.
(Full encoder forward is exercised on GPU, not here.)"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.fusion import (
    FusionMultiTaskModel,
    build_attention_pool,
    build_fusion_mlp,
    focal_bce,
    make_activation,
)
from src.models.registry import build_model
from src.utils.config import ModelConfig


def _cfg(extra=None, targets=None):
    return ModelConfig(
        name="fusion_test", model_type="fusion_multitask", text_field="clean_text",
        target=None, targets=targets or ["anxiety", "health_anxiety", "depression", "suicidality"],
        extra=extra or {},
    )


def test_make_activation():
    from torch import nn
    assert isinstance(make_activation("gelu"), nn.GELU)
    assert isinstance(make_activation("relu"), nn.ReLU)
    assert isinstance(make_activation("silu"), nn.SiLU)
    assert isinstance(make_activation("unknown"), nn.GELU)  # default


def test_attention_pool_respects_mask():
    import torch
    pool = build_attention_pool(hidden=4, attn_dim=3)
    last = torch.tensor([[[1.0, 1, 1, 1], [9.0, 9, 9, 9]]])   # (1, 2, 4); token1 is "padding"
    mask = torch.tensor([[1, 0]])                              # token1 masked out
    out = pool(last, mask)
    assert out.shape == (1, 4)
    assert torch.allclose(out, last[:, 0, :], atol=1e-5)       # masked token contributes ~0


def test_attention_pool_shape_no_mask():
    import torch
    pool = build_attention_pool(hidden=8, attn_dim=4)
    out = pool(torch.randn(3, 5, 8), None)
    assert out.shape == (3, 8)


def test_fusion_mlp_shape():
    import torch
    mlp = build_fusion_mlp(hidden=8, n_feats=3, activation="gelu", dropout=0.1, use_layernorm=True)
    out = mlp(torch.randn(4, 8), torch.randn(4, 3))
    assert out.shape == (4, 8)


def test_focal_equals_bce_at_gamma_zero():
    import torch
    import torch.nn.functional as F
    logits = torch.tensor([[0.4, -1.2], [2.0, -0.3]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    assert torch.allclose(focal_bce(logits, targets, 0.0),
                          F.binary_cross_entropy_with_logits(logits, targets, reduction="none"))


def test_focal_downweights_confident_examples():
    import torch
    import torch.nn.functional as F
    logits = torch.tensor([[6.0]])          # very confident, correct
    targets = torch.tensor([[1.0]])
    foc = focal_bce(logits, targets, 2.0).item()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").item()
    assert foc < bce * 0.05                  # focal heavily down-weights easy positives


def test_raw_feats_33_columns_and_normalization():
    m = FusionMultiTaskModel(_cfg(extra={"fusion": {"enabled": True}}))
    df = pd.DataFrame({"clean_text": [
        "I have been so anxious about my health and keep googling my symptoms terrified it is cancer",
        "just made a nice dinner and went for a walk in the park with friends today",
        "constant worry that something is seriously wrong with my body, checking my pulse all day",
        "the weather was pleasant and i enjoyed reading a book on the couch this afternoon",
        "panic attacks and racing heart, convinced i am dying, the doctor says i am fine but i cannot accept it",
    ]})
    raw = m._raw_feats(df)
    assert raw.shape == (5, 33)                       # 26 linguistic + 7 SHAI
    assert any(c.startswith("shai_") for c in raw.columns)
    m._fit_norm(raw)
    normed = m._apply_norm(raw)
    assert normed.shape == (5, 33)
    assert np.isfinite(normed).all()                  # no NaN/inf from constant columns
    nonconstant = raw.std(axis=0).to_numpy() > 0
    assert np.allclose(normed[:, nonconstant].mean(axis=0), 0, atol=1e-5)


def test_feature_set_no_label_lexicon_drops_label_vocab():
    m = FusionMultiTaskModel(_cfg(extra={"fusion": {"enabled": True, "feature_set": "no_label_lexicon"}}))
    df = pd.DataFrame({"clean_text": [
        "so anxious and worried about cancer, googling my symptoms",
        "made a nice dinner and read a book today",
    ]})
    cols = set(m._raw_feats(df).columns)
    # features whose vocabulary builds the weak labels must be gone
    for c in ["f_anx_term_rate", "f_anx_phrase_count", "f_health_anx_term_rate",
              "f_health_anx_phrase_count", "f_reassurance_count", "f_dep_term_rate", "f_suic_term_rate"]:
        assert c not in cols
    assert not any(c.startswith("shai_") for c in cols)     # SHAI echoes the HA label
    # style/structure features must remain
    for c in ["f_first_sing_rate", "f_sent_compound", "f_n_tokens", "f_flesch", "f_body_part_rate"]:
        assert c in cols


def test_feature_set_style_only_drops_all_clinical_vocab():
    m = FusionMultiTaskModel(_cfg(extra={"fusion": {"enabled": True, "feature_set": "style_only"}}))
    df = pd.DataFrame({"clean_text": ["worried about my heart and chest", "lovely sunny morning"]})
    cols = set(m._raw_feats(df).columns)
    assert "f_body_part_rate" not in cols                    # also drops the somatic lexicon
    assert not any(c.startswith("shai_") for c in cols)
    assert "f_sent_compound" in cols and "f_first_sing_rate" in cols


def test_raw_feats_pins_column_order():
    m = FusionMultiTaskModel(_cfg(extra={"fusion": {"enabled": True}}))
    df = pd.DataFrame({"clean_text": ["i am very anxious about my health today honestly"] * 4})
    cols1 = list(m._raw_feats(df).columns)
    cols2 = list(m._raw_feats(df.iloc[:2]).columns)   # reuses pinned order
    assert cols1 == cols2 == m._feature_cols


def test_registry_builds_fusion_multitask():
    model = build_model(_cfg(extra={"fusion": {"enabled": True}, "focal": {"enabled": True}}))
    assert model.__class__.__name__ == "FusionMultiTaskModel"
    assert model._fusion is True and model._focal is True


def test_config_requires_targets():
    with pytest.raises(Exception):
        ModelConfig(name="x", model_type="fusion_multitask", text_field="clean_text", target=None, targets=None)


def test_flags_default_off():
    m = FusionMultiTaskModel(_cfg(extra={}))
    assert m._fusion is False and m._attn_pool is False and m._focal is False
    assert m._activation == "gelu"
