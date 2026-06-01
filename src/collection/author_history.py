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
            d = read_parquet(fp, columns=["author"])
        except Exception as e:  # noqa: BLE001 — unreadable shard or missing `author` column
            log.warning("author.recover.read_failed", file=str(fp), error=str(e))
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


# append to src/collection/author_history.py

class AuthorHistoryCollector(JsonScraperCollector):
    """Fetch one user's full submission + comment history via no-auth JSON.

    Reuses JsonScraperCollector's cached, backoff-aware `_get_json`. Uses a
    separate cache file so it never collides with the subreddit-listing cache.
    """

    def __init__(self, config, cache_path: str | None = None, **kwargs) -> None:  # noqa: ANN001
        super().__init__(
            config,
            cache_path=cache_path or str(cache_dir() / "author_history.sqlite"),
            **kwargs,
        )

    def _iter_user_listing(self, username: str, section: str) -> Iterator[tuple[str, dict]]:
        """Yield (kind, data) for each item in /user/<name>/<section>.json."""
        url = f"{BASE}/user/{username}/{section}.json"
        after: str | None = None
        for _ in range(self.max_pages_per_listing):
            params = {"limit": "100"}
            if after:
                params["after"] = after
            data = self._get_json(url, params)
            if not data or "data" not in data:
                return
            children = data["data"].get("children", [])
            if not children:
                return
            for c in children:
                d = c.get("data")
                if isinstance(d, dict):
                    yield str(c.get("kind", "")), d
            after = data["data"].get("after")
            if not after:
                return

    def collect_user(self, username: str) -> Iterator[RedditPost]:
        seen: set[str] = set()
        n_sub = n_cmt = 0
        for section in ("submitted", "comments"):
            for kind, raw in self._iter_user_listing(username, section):
                pid = str(raw.get("id", ""))
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                if kind == "t3":
                    post = self._submission_to_post(raw)
                elif kind == "t1":
                    post = self._comment_to_post(raw)
                else:
                    continue
                if post is None or not self.passes_filters(post):
                    continue
                if post.kind == "submission":
                    n_sub += 1
                else:
                    n_cmt += 1
                yield post
        log.info("author.collect_user.done", user_hash=_hash_username(username),
                 submissions=n_sub, comments=n_cmt)

    @staticmethod
    def _submission_to_post(raw: dict) -> RedditPost | None:
        try:
            body = raw.get("selftext") or ""
            if body.strip().lower() in ("[deleted]", "[removed]"):
                body = ""
            return RedditPost(
                id=str(raw.get("id", "")),
                subreddit=str(raw.get("subreddit", "")),
                created_utc=float(raw.get("created_utc", 0.0) or 0.0),
                title=str(raw.get("title", "") or ""),
                body=body,
                author=raw.get("author"),
                score=int(raw.get("score", 0) or 0),
                num_comments=int(raw.get("num_comments", 0) or 0),
                permalink=str(raw.get("permalink", "") or ""),
                is_self=bool(raw.get("is_self", False)),
                over_18=bool(raw.get("over_18", False)),
                source="author_history",
                collected_at=time.time(),
                kind="submission",
                parent_id=None,
            )
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _comment_to_post(raw: dict) -> RedditPost | None:
        try:
            body = str(raw.get("body") or "")
            if body.strip().lower() in ("[deleted]", "[removed]", ""):
                return None
            parent_id = str(raw.get("parent_id", "") or "")
            if parent_id.startswith(("t1_", "t3_")):
                parent_id = parent_id[3:]
            author = raw.get("author")
            return RedditPost(
                id=str(raw.get("id", "")),
                subreddit=str(raw.get("subreddit", "")),
                created_utc=float(raw.get("created_utc", 0.0) or 0.0),
                title="",
                body=body,
                author=author if author != "[deleted]" else None,
                score=int(raw.get("score", 0) or 0),
                num_comments=0,
                permalink=str(raw.get("permalink", "") or ""),
                is_self=True,
                over_18=False,
                source="author_history",
                collected_at=time.time(),
                kind="comment",
                parent_id=parent_id or None,
            )
        except (ValueError, TypeError):
            return None

    def collect_subreddit(self, name: str) -> Iterator[RedditPost]:  # type: ignore[override]
        """Not used for author-history collection; delegates to the parent."""
        return super().collect_subreddit(name)
