"""Offline tests for the causal-LM baseline — verbalizer scoring, collation,
prompt formatting, flag parsing, registry + config wiring. (The real 7-8B
forward / QLoRA fit is exercised on GPU, not here.)"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.llm_causal import (
    TARGET_DESCRIPTIONS,
    HfCausalLmModel,
)
from src.models.registry import build_model
from src.utils.config import ModelConfig


def _cfg(extra=None, target="ha"):
    return ModelConfig(
        name="llm_test", model_type="llm_causal", text_field="clean_text",
        target=target, targets=None, extra=extra or {},
    )


class _FakeTok:
    """Minimal tokenizer stub: deterministic per-word ids, no chat template."""

    chat_template = None
    pad_token_id = 0

    def __init__(self):
        self._vocab = {"yes": 11, "no": 22, " yes": 11, " no": 22, "Yes": 13, "No": 24,
                       " Yes": 13, " No": 24, "YES": 15, "NO": 26, " YES": 15, " NO": 26}

    def encode(self, w, add_special_tokens=False):  # noqa: ANN001
        return [self._vocab.get(w, 99)]


def test_registry_builds_llm_causal():
    m = build_model(_cfg())
    assert isinstance(m, HfCausalLmModel)


def test_config_requires_target():
    # single-target model: omitting target must fail validation
    with pytest.raises(ValueError):
        ModelConfig(name="x", model_type="llm_causal", target=None, targets=None)


def test_target_descriptions_cover_ha_and_core():
    for t in ["ha", "anxiety", "health_anxiety", "depression", "suicidality"]:
        assert t in TARGET_DESCRIPTIONS and TARGET_DESCRIPTIONS[t]


def test_flag_parsing_defaults_zero_shot():
    m = HfCausalLmModel(_cfg())
    assert m._lora_enabled is False           # zero-shot by default
    assert m._want_4bit is True


def test_flag_parsing_qlora():
    m = HfCausalLmModel(_cfg(extra={
        "pretrained": "meta-llama/Llama-3.1-8B-Instruct",
        "lora": {"enabled": True, "r": 8, "alpha": 16},
        "train": {"num_train_epochs": 2, "learning_rate": 2e-4, "max_train": 5000},
    }))
    assert m._lora_enabled is True
    assert m._lora_cfg["r"] == 8 and m._lora_cfg["alpha"] == 16
    assert m._epochs == 2 and m._max_train == 5000


def test_first_token_ids_dedupes_and_sorts():
    tok = _FakeTok()
    yes = HfCausalLmModel._first_token_ids(tok, ["yes", " yes", "Yes", " Yes", "YES"])
    assert yes == [11, 13, 15]                # deduped + sorted


def test_scores_from_logits_monotonic_and_bounded():
    import torch
    # vocab of 30; yes ids favoured in row0, no ids in row1
    logits = torch.full((2, 30), -10.0)
    logits[0, [11, 13, 15]] = 5.0             # row0 -> "yes"
    logits[1, [22, 24, 26]] = 5.0             # row1 -> "no"
    s = HfCausalLmModel._scores_from_logits(logits, [11, 13, 15], [22, 24, 26])
    assert s.shape == (2,)
    assert (s >= 0).all() and (s <= 1).all()
    assert s[0] > 0.9 and s[1] < 0.1          # yes-dominant vs no-dominant


def test_scores_from_logits_balanced_is_half():
    import torch
    logits = torch.full((1, 30), -10.0)
    logits[0, [11]] = 2.0
    logits[0, [22]] = 2.0
    s = HfCausalLmModel._scores_from_logits(logits, [11], [22])
    assert abs(float(s[0]) - 0.5) < 1e-4


def test_collate_pads_and_masks_labels():
    import torch
    batch = [([5, 6, 7], [-100, -100, 7]), ([8, 9], [-100, 9])]
    ids, attn, labs = HfCausalLmModel._collate(batch, pad_id=0)
    assert ids.shape == (2, 3) and attn.shape == (2, 3) and labs.shape == (2, 3)
    # second row right-padded
    assert ids[1].tolist() == [8, 9, 0]
    assert attn[1].tolist() == [1, 1, 0]
    assert labs[1].tolist() == [-100, 9, -100]    # pad position label masked


def test_format_fallback_without_chat_template():
    m = HfCausalLmModel(_cfg(target="ha"))
    m._tok = _FakeTok()
    out = m._format("I keep googling my symptoms, sure I have cancer.")
    assert "yes" in out.lower() and "no" in out.lower()
    assert "googling my symptoms" in out          # post text embedded
    assert TARGET_DESCRIPTIONS["ha"][:20] in out   # target description embedded


def test_format_truncates_to_char_cap():
    m = HfCausalLmModel(_cfg(target="anxiety", extra={"char_cap": 50}))
    m._tok = _FakeTok()
    long = "Q" * 5000                               # 'Q' absent from prompt boilerplate
    out = m._format(long)
    assert out.count("Q") == 50                    # post capped to char_cap
    assert ("Q" * 51) not in out


def test_zero_shot_fit_is_noop():
    # lora disabled -> fit must not touch the network and just marks fitted
    m = HfCausalLmModel(_cfg())
    df = pd.DataFrame({"clean_text": ["a", "b"], "label_ha": [1, 0]})
    out = m.fit(df)
    assert out is m and m._fitted is True and m._model is None
