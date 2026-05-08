"""Tier-1 weak labeling: subreddit-prior + lexicon overlap.

Outputs *probabilistic* weak labels rather than hard 0/1 — downstream
training can use them as soft targets or with confidence weighting.
"""
from __future__ import annotations

import re

import pandas as pd

from src.labeling.lexicons import (
    ANXIETY_PHRASES,
    ANXIETY_TERMS,
    DEPRESSION_TERMS,
    HEALTH_ANXIETY_PHRASES,
    HEALTH_ANXIETY_TERMS,
    REASSURANCE_PATTERNS,
    SUICIDALITY_TERMS,
)
from src.utils.config import LabelingConfig, SubredditsConfig
from src.utils.logging import get_logger

log = get_logger(__name__)

LABELS = ("anxiety", "health_anxiety", "depression", "suicidality")

RE_TOKEN = re.compile(r"\b[\w']+\b", flags=re.UNICODE)


def _tokenize(text: str, max_tokens: int) -> list[str]:
    toks = RE_TOKEN.findall((text or "").lower())
    return toks[:max_tokens]


def _phrase_hits(text: str, phrases: set[str]) -> int:
    t = (text or "").lower()
    return sum(1 for p in phrases if p in t)


def _token_hits(tokens: list[str], terms: set[str]) -> int:
    return sum(1 for t in tokens if t in terms)


def _normalize_score(hits: int, n_tokens: int) -> float:
    """Map hit count to a [0, 1] score using a saturating curve.

    Avoids letting one very long anxious post dominate; saturates around 5+ hits.
    """
    if n_tokens == 0:
        return 0.0
    raw = hits / max(1, n_tokens / 100)  # hits per ~100 tokens
    return min(1.0, raw / 3.0)  # saturate at 3 hits per 100 tokens


def lexicon_scores(text: str, max_tokens: int = 800) -> dict[str, float]:
    tokens = _tokenize(text, max_tokens)
    n = max(1, len(tokens))

    anx_hits = _token_hits(tokens, ANXIETY_TERMS) + _phrase_hits(text, ANXIETY_PHRASES)
    ha_hits = _token_hits(tokens, HEALTH_ANXIETY_TERMS) + _phrase_hits(text, HEALTH_ANXIETY_PHRASES)
    ha_hits += _phrase_hits(text, REASSURANCE_PATTERNS) // 2  # weaker signal
    dep_hits = _token_hits(tokens, DEPRESSION_TERMS)
    suic_hits = _token_hits(tokens, SUICIDALITY_TERMS)

    return {
        "anxiety": _normalize_score(anx_hits, n),
        "health_anxiety": _normalize_score(ha_hits, n),
        "depression": _normalize_score(dep_hits, n),
        "suicidality": _normalize_score(suic_hits, n),
    }


def subreddit_priors(subreddits: SubredditsConfig, name: str) -> dict[str, float]:
    s = subreddits.by_name(name)
    if s is None:
        return dict.fromkeys(LABELS, 0.0)
    return {
        "anxiety": s.expected_anxiety_prior,
        "health_anxiety": s.expected_health_anxiety_prior,
        "depression": s.expected_depression_prior,
        "suicidality": s.expected_suicidality_prior,
    }


def label_post(
    text: str,
    subreddit: str,
    subreddits: SubredditsConfig,
    cfg: LabelingConfig,
) -> dict[str, float]:
    """Return weak label probabilities for one post."""
    t1 = cfg.tier1_weak
    lex = lexicon_scores(text, max_tokens=t1.max_tokens_for_lex)
    pri = subreddit_priors(subreddits, subreddit)
    out = {}
    for k in LABELS:
        out[k] = (
            t1.subreddit_prior_weight * pri[k] + t1.lexicon_weight * lex[k]
        )
    return out


def apply_weak_labels(
    df: pd.DataFrame,
    subreddits: SubredditsConfig,
    cfg: LabelingConfig,
) -> pd.DataFrame:
    """Add `weak_<label>` (probability) and `weak_<label>_bin` columns."""
    log.info("weak_label.start", n=len(df))
    out = df.copy()

    scores = [
        label_post(t, s, subreddits, cfg)
        for t, s in zip(out.get("clean_text", out.get("body", "")), out["subreddit"])
    ]
    for k in LABELS:
        out[f"weak_{k}"] = [s[k] for s in scores]
        out[f"weak_{k}_bin"] = (out[f"weak_{k}"] >= cfg.tier1_weak.thresholds.get(k, 0.5)).astype(int)

    log.info(
        "weak_label.done",
        anxiety_pos=int(out["weak_anxiety_bin"].sum()),
        health_pos=int(out["weak_health_anxiety_bin"].sum()),
        dep_pos=int(out["weak_depression_bin"].sum()),
        suic_pos=int(out["weak_suicidality_bin"].sum()),
    )
    return out
