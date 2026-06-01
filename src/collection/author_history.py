"""By-author full-history collection (no-auth Reddit JSON).

Recovers the real usernames of an anonymized cohort from data/raw/ (where the
`author` column survives), then fetches each user's full submission + comment
history. Output shards are keyed by `author_hash` so usernames never appear in
filenames.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.collection.base import RedditPost
from src.collection.json_scraper import BASE, JsonScraperCollector
from src.preprocessing.anonymize import _hash_username
from src.utils.config import cache_dir
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)

_BAD_AUTHORS = {"[deleted]", "[removed]", "", "none"}


def recover_author_usernames(
    users_df: pd.DataFrame, raw_dir: str | Path = "data/raw"
) -> dict[str, str]:
    """Map each cohort `author_hash` back to its real username via data/raw/.

    Recomputes the pipeline's salted hash (`_hash_username`) over the `author`
    column of every raw shard and matches against the cohort's hashes. Skips
    deleted/removed authors. Returns {author_hash: username}.
    """
    raw_dir = Path(raw_dir)
    target = {str(h) for h in users_df["author_hash"].dropna().astype(str)} - {""}
    out: dict[str, str] = {}
    for fp in sorted(raw_dir.glob("*.parquet")):
        try:
            d = read_parquet(fp)
        except Exception as e:  # noqa: BLE001
            log.warning("author.recover.read_failed", file=str(fp), error=str(e))
            continue
        if "author" not in d.columns:
            continue
        for a in d["author"].dropna().astype(str).unique():
            if a.lower() in _BAD_AUTHORS:
                continue
            h = _hash_username(a)
            if h in target and h not in out:
                out[h] = a
        if len(out) == len(target):
            break
    log.info("author.recover.done", requested=len(target), recovered=len(out))
    return out
