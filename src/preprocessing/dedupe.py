"""Duplicate / near-duplicate removal.

Reddit cross-posting and bot reposts mean the same text can appear under
multiple post IDs. We use:
  1. Exact dedup on normalized text (cheap)
  2. SimHash-based near-dedup on top of that (handles minor edits)
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

RE_NORM = re.compile(r"\s+")


def _norm_for_hash(text: str) -> str:
    return RE_NORM.sub(" ", (text or "").lower()).strip()


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _simhash(text: str, n_bits: int = 64) -> int:
    """Very small SimHash. Tokens = whitespace split; weight = 1.

    This is enough for catching near-duplicate Reddit posts; for a production
    near-dup system you'd want shingles + IDF weighting.
    """
    v = [0] * n_bits
    tokens = text.split()
    if not tokens:
        return 0
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for i in range(n_bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(n_bits):
        if v[i] > 0:
            out |= 1 << i
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def deduplicate(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    near_dup_threshold: int = 5,
) -> pd.DataFrame:
    """Drop exact and near-duplicate posts. Keeps the first occurrence by index."""
    if df.empty:
        return df

    work = df.copy()
    work["_norm"] = work[text_col].astype(str).map(_norm_for_hash)
    work["_md5"] = work["_norm"].map(_md5)

    # Exact dedup
    work = work.drop_duplicates(subset="_md5", keep="first").reset_index(drop=True)

    # Near-dup (within each subreddit — rarely cross-subreddit reposts in our domain)
    keep_mask = pd.Series(True, index=work.index)
    if "subreddit" in work.columns:
        for _, group in work.groupby("subreddit"):
            seen: list[int] = []
            for idx, row in group.iterrows():
                sh = _simhash(row["_norm"])
                duplicate = any(_hamming(sh, prev) <= near_dup_threshold for prev in seen)
                if duplicate:
                    keep_mask.loc[idx] = False
                else:
                    seen.append(sh)
    out = work.loc[keep_mask].drop(columns=["_norm", "_md5"]).reset_index(drop=True)
    return out
