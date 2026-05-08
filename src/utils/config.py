"""Pydantic-validated config loaders for YAML files in configs/.

Each loader returns a typed object. Add a new config type by defining a
Pydantic model and a loader function — the rest of the pipeline references
the typed object, not raw dicts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Subreddits
# --------------------------------------------------------------------------- #


class SubredditEntry(BaseModel):
    name: str
    group: str
    expected_anxiety_prior: float = 0.0
    expected_health_anxiety_prior: float = 0.0
    expected_depression_prior: float = 0.0
    expected_suicidality_prior: float = 0.0
    handle_with_care: bool = False
    notes: str = ""


class CollectionConfig(BaseModel):
    time_filter: Literal["all", "year", "month", "week", "day", "hour"] = "all"
    posts_per_subreddit: int = 5000
    min_score: int = 0
    min_body_chars: int = 50
    include_comments: bool = False
    include_self_only: bool = True


class SubredditsConfig(BaseModel):
    subreddits: list[SubredditEntry]
    collection: CollectionConfig

    def by_name(self, name: str) -> SubredditEntry | None:
        for s in self.subreddits:
            if s.name.lower() == name.lower():
                return s
        return None

    def names(self) -> list[str]:
        return [s.name for s in self.subreddits]

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.subreddits:
            out.setdefault(s.group, []).append(s.name)
        return out


# --------------------------------------------------------------------------- #
# Labeling
# --------------------------------------------------------------------------- #


class Tier1Config(BaseModel):
    subreddit_prior_weight: float = 0.5
    lexicon_weight: float = 0.5
    thresholds: dict[str, float] = Field(default_factory=dict)
    max_tokens_for_lex: int = 800


class Tier2Config(BaseModel):
    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0
    max_posts: int = 8000
    per_group_sample: dict[str, int] = Field(default_factory=dict)
    cache_path: str = ".cache/llm_labels.sqlite"
    rpm: int = 30


class Tier3Config(BaseModel):
    target_size: int = 1000
    stratify_by: str = "subreddit_group"
    min_cohen_kappa: dict[str, float] = Field(default_factory=dict)
    output_path: str = "data/processed/gold_test_set.parquet"


class AggregateConfig(BaseModel):
    precedence: list[str] = Field(default_factory=lambda: ["manual", "llm", "weak"])
    require_at_least_one_tier: bool = True
    tier_confidence: dict[str, float] = Field(
        default_factory=lambda: {"manual": 1.0, "llm": 0.7, "weak": 0.4}
    )


class LabelingConfig(BaseModel):
    tier1_weak: Tier1Config
    tier2_llm: Tier2Config
    tier3_manual: Tier3Config
    aggregate: AggregateConfig


# --------------------------------------------------------------------------- #
# Models (loose schema — each model interprets its own block)
# --------------------------------------------------------------------------- #


class ModelConfig(BaseModel):
    name: str
    model_type: Literal[
        "tfidf", "xgboost", "transformer", "multitask_transformer", "llm_zero_shot"
    ]
    text_field: str = "clean_text"
    target: str | None = None
    targets: list[str] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_targets(self) -> ModelConfig:
        if self.model_type == "multitask_transformer" and not self.targets:
            raise ValueError("multitask_transformer requires `targets` list")
        if self.model_type != "multitask_transformer" and not self.target:
            raise ValueError(f"{self.model_type} requires `target`")
        return self


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_subreddits(path: str | Path = "configs/subreddits.yaml") -> SubredditsConfig:
    return SubredditsConfig.model_validate(_load_yaml(path))


def load_labeling(path: str | Path = "configs/labeling.yaml") -> LabelingConfig:
    return LabelingConfig.model_validate(_load_yaml(path))


def load_model_config(path: str | Path) -> ModelConfig:
    """Load a model YAML.

    Unknown keys are stashed in `extra` so model implementations can read
    their own hyperparameters without us re-validating every field here.
    """
    raw = _load_yaml(path)
    known_keys = {"name", "model_type", "text_field", "target", "targets"}
    extra = {k: v for k, v in raw.items() if k not in known_keys}
    return ModelConfig(
        name=raw["name"],
        model_type=raw["model_type"],
        text_field=raw.get("text_field", "clean_text"),
        target=raw.get("target"),
        targets=raw.get("targets"),
        extra=extra,
    )


# --------------------------------------------------------------------------- #
# Paths / env
# --------------------------------------------------------------------------- #


def project_root() -> Path:
    """Resolve to the repo root regardless of where Python was launched from."""
    return Path(__file__).resolve().parents[2]


def data_dir(sub: str | None = None) -> Path:
    root = Path(os.getenv("DATA_DIR", project_root() / "data"))
    return root / sub if sub else root


def cache_dir() -> Path:
    p = Path(os.getenv("CACHE_DIR", project_root() / ".cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p
