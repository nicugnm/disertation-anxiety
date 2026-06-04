"""Offline tests for the DANN pieces — gradient reversal, domain-label mapping,
and registry wiring. (The full transformer training is exercised on GPU, not here.)"""
from __future__ import annotations

import pandas as pd

from src.models.dann import DannMultiTaskModel, grad_reverse
from src.models.registry import build_model
from src.utils.config import ModelConfig


def _cfg(domain: str, targets=None) -> ModelConfig:
    return ModelConfig(
        name="dann_test", model_type="dann_multitask", text_field="clean_text",
        target=None, targets=targets or ["anxiety", "health_anxiety", "depression", "suicidality"],
        extra={"domain": domain},
    )


def test_gradient_reversal_identity_forward_negated_backward():
    import torch

    x = torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
    y = grad_reverse(x, alpha=3.0)
    assert torch.allclose(y.detach(), x.detach())  # forward = identity
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -3.0))  # backward = -alpha * grad


def test_domain_series_subreddit_mode_is_identity():
    m = DannMultiTaskModel(_cfg("subreddit"))
    df = pd.DataFrame({"subreddit": ["Anxiety", "cooking", "HealthAnxiety"]})
    assert list(m._domain_series(df)) == ["Anxiety", "cooking", "HealthAnxiety"]


def test_domain_series_group_mode_maps_to_config_groups():
    m = DannMultiTaskModel(_cfg("group"))
    df = pd.DataFrame({"subreddit": ["Anxiety", "HealthAnxiety", "cooking", "depression"]})
    # Anxiety->anxiety_primary, HealthAnxiety->health_anxiety_primary,
    # cooking->baseline, depression->depression_primary (per configs/subreddits.yaml)
    assert list(m._domain_series(df)) == [
        "anxiety_primary", "health_anxiety_primary", "baseline", "depression_primary",
    ]


def test_domain_series_group_mode_unknown_subreddit_defaults_baseline():
    m = DannMultiTaskModel(_cfg("group"))
    df = pd.DataFrame({"subreddit": ["NotInConfig123"]})
    assert list(m._domain_series(df)) == ["baseline"]


def test_registry_builds_dann_multitask():
    model = build_model(_cfg("subreddit", targets=["anxiety", "depression"]))
    assert model.__class__.__name__ == "DannMultiTaskModel"
    assert model.targets == ["anxiety", "depression"]
