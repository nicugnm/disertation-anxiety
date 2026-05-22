"""Reddit-wide search scraper for self-disclosure phrases.

Instead of pulling everything from anxiety subreddits and filtering for
self-disclosure afterward, this collector queries Reddit's public search
endpoint *directly* with the phrases we care about ("I was diagnosed with
depression", "I have GAD", ...) across all of Reddit.

This is the field-standard data-source for self-disclosure-labeled corpora
(Coppersmith et al. 2014, 2015; CLEF eRisk 2017–present). Posts surfaced this
way have the highest a-priori probability of containing a verified clinical
self-disclosure.

Endpoint: `https://old.reddit.com/search.json?q=<phrase>&restrict_sr=false&...`
  - Up to 100 results per page
  - Paginates via the `after` cursor (same as listings)
  - No OAuth required
  - Returns posts (t3 only); comments aren't surfaced by the search endpoint
"""
from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from typing import Any

from src.collection.base import BaseCollector, RedditPost
from src.collection.json_scraper import BASE, DEFAULT_UA  # share UA + base URL
from src.utils.cache import SqliteCache
from src.utils.config import cache_dir
from src.utils.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Default disclosure queries — chosen to mirror our self-disclosure patterns
# while remaining short enough for the search engine to handle.
# --------------------------------------------------------------------------- #

DEFAULT_DISCLOSURE_QUERIES: list[str] = [
    # Anxiety
    "I was diagnosed with anxiety",
    "I was diagnosed with generalized anxiety disorder",
    "I have GAD",
    "diagnosed with anxiety disorder",
    "I was diagnosed with panic disorder",
    # Health anxiety
    "I have health anxiety",
    "I was diagnosed with health anxiety",
    "I was diagnosed with illness anxiety",
    "I am a hypochondriac",
    "I have hypochondria",
    # Depression
    "I was diagnosed with depression",
    "I was diagnosed with major depressive disorder",
    "I have MDD",
    "I have clinical depression",
    "diagnosed with depression",
    # Comorbid / context
    "diagnosed with OCD and anxiety",
    "diagnosed with PTSD",
]


def _slug(query: str) -> str:
    """Convert a search phrase to a filesystem-safe slug for the output filename."""
    s = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
    return s[:80] if s else "query"


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #


class SearchScraperCollector(BaseCollector):
    """Hits Reddit's search endpoint for each configured query.

    `collect_subreddit(name)` is **interpreted as a search query**, not as a
    subreddit name. `collect_all()` iterates the internal query list.

    Output `RedditPost.subreddit` is whatever Reddit returns for the matched
    post (so it can be any subreddit — that's the point of search). To keep
    the runner's "one parquet per source" convention working, the runner
    writes one parquet per *query* (named by slug), not per subreddit.
    """

    def __init__(
        self,
        config,  # noqa: ANN001
        queries: list[str] | None = None,
        request_interval: float = 1.5,
        max_pages_per_query: int = 10,  # 10 × 100 = up to 1000 hits per query
        user_agent: str | None = None,
        cache_path: str | None = None,
        retries: int = 3,
    ) -> None:
        super().__init__(config)
        self.queries = list(queries) if queries is not None else list(DEFAULT_DISCLOSURE_QUERIES)
        self.request_interval = float(request_interval)
        self.max_pages_per_query = int(max_pages_per_query)
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", DEFAULT_UA)
        self.cache = SqliteCache(cache_path or str(cache_dir() / "search_scraper.sqlite"))
        self.retries = retries
        self._last_request = 0.0
        # Lazy session import so test environments that mock requests still work.
        import requests as _r

        self._session = _r.Session()
        self._session.headers.update({"User-Agent": self.user_agent})

    # ------------------------------------------------------------------ #
    # HTTP plumbing (mirrors JsonScraperCollector — we keep them separate
    # so the search cache doesn't accidentally invalidate the listing cache)
    # ------------------------------------------------------------------ #

    def _sleep_for_rate(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)

    def _get_json(self, url: str, params: dict[str, str]) -> dict[str, Any] | None:
        import requests

        key = SqliteCache.make_key(url, params)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        backoff = 2.0
        for attempt in range(self.retries):
            self._sleep_for_rate()
            try:
                resp = self._session.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                log.warning("search.network_error", url=url, error=str(e), attempt=attempt)
                time.sleep(backoff)
                backoff *= 2
                continue
            self._last_request = time.time()

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    log.warning("search.bad_json", url=url)
                    return None
                self.cache.set(key, data)
                return data

            if resp.status_code in (403, 404):
                log.warning("search.unavailable", url=url, status=resp.status_code)
                self.cache.set(key, None)
                return None

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                wait = min(max(wait, 1.0), 600.0)
                log.warning(
                    "search.rate_limited",
                    url=url,
                    status=resp.status_code,
                    wait_seconds=round(wait, 1),
                )
                time.sleep(wait)
                backoff = min(backoff * 2, 600.0)
                continue

            log.warning("search.http_error", url=url, status=resp.status_code)
            return None

        log.error("search.give_up", url=url)
        return None

    # ------------------------------------------------------------------ #
    # Query → posts
    # ------------------------------------------------------------------ #

    def _iter_query(self, query: str) -> Iterator[dict[str, Any]]:
        url = f"{BASE}/search.json"
        after: str | None = None
        for _ in range(self.max_pages_per_query):
            # `restrict_sr=on` would limit to a subreddit; we want Reddit-wide.
            params = {
                "q": query,
                "limit": "100",
                "sort": "relevance",
                "t": "all",
                "type": "link",
                "include_over_18": "off",
            }
            if after:
                params["after"] = after
            data = self._get_json(url, params)
            if not data or "data" not in data:
                return
            children = data["data"].get("children", [])
            if not children:
                return
            for c in children:
                if c.get("kind") == "t3" and isinstance(c.get("data"), dict):
                    yield c["data"]
            after = data["data"].get("after")
            if not after:
                return

    def collect_subreddit(self, name: str) -> Iterator[RedditPost]:
        """In this collector, `name` is treated as a search query."""
        seen_ids: set[str] = set()
        for raw in self._iter_query(name):
            pid = str(raw.get("id", ""))
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            post = self._record_to_post(raw, matched_query=name)
            if post is None:
                continue
            if self.passes_filters(post):
                yield post

    def collect_all(self) -> Iterator[RedditPost]:
        """Iterate every configured query (overriding BaseCollector.collect_all)."""
        for q in self.queries:
            log.info("search.collect.start", query=q)
            n = 0
            for post in self.collect_subreddit(q):
                n += 1
                yield post
            log.info("search.collect.done", query=q, yielded=n)

    @staticmethod
    def _record_to_post(record: dict[str, Any], matched_query: str = "") -> RedditPost | None:
        try:
            body = record.get("selftext") or ""
            if body.strip().lower() in ("[deleted]", "[removed]"):
                body = ""
            return RedditPost(
                id=str(record.get("id", "")),
                subreddit=str(record.get("subreddit", "")),
                created_utc=float(record.get("created_utc", 0.0)),
                title=str(record.get("title", "") or ""),
                body=body,
                author=record.get("author"),
                score=int(record.get("score", 0) or 0),
                num_comments=int(record.get("num_comments", 0) or 0),
                permalink=str(record.get("permalink", "") or ""),
                is_self=bool(record.get("is_self", False)),
                over_18=bool(record.get("over_18", False)),
                source="search",
                collected_at=time.time(),
                kind="submission",
                parent_id=None,
                extra={"matched_query": matched_query},
            )
        except (ValueError, TypeError):
            return None

    def close(self) -> None:
        self._session.close()
        self.cache.close()
