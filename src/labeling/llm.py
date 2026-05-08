"""Tier-2 LLM-assisted labeling using the Anthropic Claude API.

The prompt operationalizes the codebook (`docs/codebook.md`) so the LLM
acts like a trained annotator. All responses are cached on disk: re-runs
are free unless the prompt or the post changes.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.labeling.weak import LABELS
from src.utils.cache import SqliteCache
from src.utils.config import LabelingConfig
from src.utils.logging import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = """You are an expert annotator for a mental-health NLP research project. \
You apply a published codebook to short Reddit posts and return strict JSON. \
You do not diagnose individuals — you label *language*."""

USER_PROMPT_TEMPLATE = """Apply the following codebook to the post below.

CODEBOOK:
- anxiety = first-person expression of anxious affect, anxious cognition, or anxious physiological experience.
- health_anxiety = anxiety specifically about one's own (or a loved one's) physical health, illness, symptoms, fear of disease. Implies anxiety=1.
- depression = first-person expression of depressive symptoms (anhedonia, hopelessness, low mood, worthlessness).
- suicidality = first-person expression of suicidal ideation, intent, plan, or recent attempt.

For each label, also return a confidence in {{1,2,3,4,5}}.

Return ONLY a JSON object with this exact shape, no markdown, no commentary:

{{
  "anxiety": 0 or 1,
  "anxiety_conf": 1..5,
  "health_anxiety": 0 or 1,
  "health_anxiety_conf": 1..5,
  "depression": 0 or 1,
  "depression_conf": 1..5,
  "suicidality": 0 or 1,
  "suicidality_conf": 1..5,
  "rationale": "<= 30 words"
}}

POST:
\"\"\"
{post}
\"\"\"
"""


def _make_client():
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Install `anthropic` to use the LLM labeler.") from e

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env or your environment."
        )
    return anthropic.Anthropic()


def _parse_json_strict(s: str) -> dict[str, Any]:
    """Parse the LLM JSON response, tolerant to surrounding fluff."""
    import json
    import re

    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def label_one(
    client,
    post: str,
    model: str,
    max_tokens: int,
    temperature: float,
    cache: SqliteCache,
) -> dict[str, Any]:
    """Label a single post, with caching."""
    key = SqliteCache.make_key(model, USER_PROMPT_TEMPLATE, post)
    cached = cache.get(key)
    if cached is not None:
        return cached

    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT_TEMPLATE.format(post=post[:4000])}],
    )
    text = msg.content[0].text  # type: ignore[attr-defined]
    try:
        parsed = _parse_json_strict(text)
    except Exception as e:  # noqa: BLE001
        log.warning("llm_label.parse_failed", error=str(e), preview=text[:200])
        parsed = {k: 0 for k in LABELS} | {f"{k}_conf": 1 for k in LABELS} | {"rationale": ""}

    cache.set(key, parsed)
    return parsed


def _stratified_sample(
    df: pd.DataFrame,
    per_group: dict[str, int],
    group_col: str = "subreddit_group",
) -> pd.DataFrame:
    parts = []
    for grp, n in per_group.items():
        sub = df[df[group_col] == grp]
        if sub.empty:
            continue
        take = min(len(sub), n)
        parts.append(sub.sample(n=take, random_state=42))
    return pd.concat(parts, ignore_index=True) if parts else df.head(0)


def apply_llm_labels(
    df: pd.DataFrame,
    cfg: LabelingConfig,
    text_col: str = "clean_text",
) -> pd.DataFrame:
    """Run the LLM labeler on a stratified sample. Returns the labeled subset.

    Caller is expected to merge this back with `df` on `id`.
    """
    t2 = cfg.tier2_llm
    sample = _stratified_sample(df, t2.per_group_sample)
    log.info("llm_label.sample", n=len(sample))

    client = _make_client()
    cache = SqliteCache(t2.cache_path)

    interval = 60.0 / max(1, t2.rpm)
    rows: list[dict[str, Any]] = []
    last_call = 0.0
    for _, row in sample.iterrows():
        # Rate limit (cache hits skip the sleep)
        key = SqliteCache.make_key(t2.model, USER_PROMPT_TEMPLATE, row[text_col])
        if key not in cache:
            elapsed = time.time() - last_call
            if elapsed < interval:
                time.sleep(interval - elapsed)
            last_call = time.time()

        result = label_one(
            client,
            row[text_col],
            model=t2.model,
            max_tokens=t2.max_tokens,
            temperature=t2.temperature,
            cache=cache,
        )
        out = {"id": row["id"]}
        for k in LABELS:
            out[f"llm_{k}"] = int(result.get(k, 0) or 0)
            out[f"llm_{k}_conf"] = int(result.get(f"{k}_conf", 1) or 1)
        out["llm_rationale"] = result.get("rationale", "")
        rows.append(out)

    cache.close()
    out_df = pd.DataFrame(rows)
    log.info("llm_label.done", n=len(out_df))
    return out_df


def attach_subreddit_groups(df: pd.DataFrame, subreddits) -> pd.DataFrame:  # noqa: ANN001
    """Add a `subreddit_group` column for stratified sampling."""
    name_to_group = {s.name.lower(): s.group for s in subreddits.subreddits}
    df = df.copy()
    df["subreddit_group"] = df["subreddit"].str.lower().map(name_to_group).fillna("baseline")
    return df


def label_corpus(
    df: pd.DataFrame,
    subreddits,  # noqa: ANN001
    cfg: LabelingConfig,
    text_col: str = "clean_text",
) -> pd.DataFrame:
    """Public entry point. Returns df augmented with llm_* columns where available."""
    df = attach_subreddit_groups(df, subreddits)
    labels_df = apply_llm_labels(df, cfg, text_col=text_col)
    if labels_df.empty:
        for k in LABELS:
            df[f"llm_{k}"] = pd.NA
            df[f"llm_{k}_conf"] = pd.NA
        return df
    return df.merge(labels_df, on="id", how="left")


def label_corpus_iter(
    rows: Iterable[dict],
    cfg: LabelingConfig,
    text_col: str = "clean_text",
) -> Iterable[dict]:
    """Stream variant for huge corpora that don't fit in memory."""
    t2 = cfg.tier2_llm
    client = _make_client()
    cache = SqliteCache(t2.cache_path)
    interval = 60.0 / max(1, t2.rpm)
    last_call = 0.0
    for row in rows:
        text = row.get(text_col, "")
        key = SqliteCache.make_key(t2.model, USER_PROMPT_TEMPLATE, text)
        if key not in cache:
            elapsed = time.time() - last_call
            if elapsed < interval:
                time.sleep(interval - elapsed)
            last_call = time.time()
        result = label_one(
            client,
            text,
            model=t2.model,
            max_tokens=t2.max_tokens,
            temperature=t2.temperature,
            cache=cache,
        )
        for k in LABELS:
            row[f"llm_{k}"] = int(result.get(k, 0) or 0)
            row[f"llm_{k}_conf"] = int(result.get(f"{k}_conf", 1) or 1)
        row["llm_rationale"] = result.get("rationale", "")
        yield row
    cache.close()
