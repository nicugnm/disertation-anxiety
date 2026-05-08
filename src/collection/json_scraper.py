"""No-credentials Reddit scraper using the public JSON endpoints.

Reddit serves a JSON view of every page when you append `.json` to the URL:
e.g., `https://old.reddit.com/r/Anxiety/top.json?t=all&limit=100`.
This works without OAuth — Reddit only requires a non-default User-Agent
and respects standard rate limits (~1 req/sec unauthenticated).

The scraper combines multiple listings (top × {all, year, month, week} + new)
to get well past the per-listing 1000-item cap. All responses are cached
on disk so re-runs are idempotent and free.

Compliance notes:
  - Polite, research-identifying User-Agent (configurable, defaults state academic intent).
  - Exponential back-off on 429.
  - Conservative default of 1 request per 1.5s.
  - Honors Reddit's "after" pagination — no scraping of HTML or undocumented endpoints.

This is intended for academic research within Reddit's terms; you are
responsible for verifying compliance with current Reddit policy and your
institution's ethics requirements.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import requests

from src.collection.base import BaseCollector, RedditPost
from src.utils.cache import SqliteCache
from src.utils.config import cache_dir
from src.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_UA = (
    "academic-research:disertation-anxiety:0.1 "
    "(non-commercial mental-health NLP research; contact via repo)"
)
BASE = "https://old.reddit.com"

# Listing combinations chosen to maximize coverage. Reddit caps each listing
# at ~1000 items; combining listings + time filters lets us pull several
# thousand distinct posts per subreddit.
LISTING_PLAN: list[tuple[str, dict[str, str]]] = [
    ("top", {"t": "all"}),
    ("top", {"t": "year"}),
    ("top", {"t": "month"}),
    ("top", {"t": "week"}),
    ("new", {}),
    ("hot", {}),
]


class JsonScraperCollector(BaseCollector):
    def __init__(
        self,
        config,  # noqa: ANN001
        request_interval: float = 1.5,
        max_pages_per_listing: int = 10,  # 10 * 100 = 1000 items per listing
        user_agent: str | None = None,
        cache_path: str | None = None,
        retries: int = 3,
    ) -> None:
        super().__init__(config)
        self.request_interval = float(request_interval)
        self.max_pages_per_listing = int(max_pages_per_listing)
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", DEFAULT_UA)
        self.cache = SqliteCache(cache_path or str(cache_dir() / "json_scraper.sqlite"))
        self.retries = retries
        self._last_request = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})

    # ------------------------------------------------------------------ #
    # HTTP plumbing
    # ------------------------------------------------------------------ #

    def _sleep_for_rate(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)

    def _get_json(self, url: str, params: dict[str, str]) -> dict[str, Any] | None:
        """Cached GET. Returns None on permanent failure (404, banned sub, etc.)."""
        # Build a stable cache key
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
                log.warning("scraper.network_error", url=url, error=str(e), attempt=attempt)
                time.sleep(backoff)
                backoff *= 2
                continue
            self._last_request = time.time()

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    log.warning("scraper.bad_json", url=url)
                    return None
                # Don't cache empty listings — Reddit sometimes returns empty
                # transient responses we'd rather retry next run.
                self.cache.set(key, data)
                return data

            if resp.status_code in (403, 404):
                log.warning("scraper.unavailable", url=url, status=resp.status_code)
                # Cache the negative result so we don't re-hit private/banned subs
                self.cache.set(key, None)
                return None

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                # Honor Retry-After when present
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                log.warning(
                    "scraper.rate_limited",
                    url=url,
                    status=resp.status_code,
                    retry_after=wait,
                    attempt=attempt,
                )
                time.sleep(wait)
                backoff *= 2
                continue

            log.warning("scraper.http_error", url=url, status=resp.status_code)
            return None

        log.error("scraper.give_up", url=url)
        return None

    # ------------------------------------------------------------------ #
    # Listing iteration
    # ------------------------------------------------------------------ #

    def _iter_listing(
        self,
        subreddit: str,
        listing: str,
        params: dict[str, str],
    ) -> Iterator[dict[str, Any]]:
        url = f"{BASE}/r/{subreddit}/{listing}.json"
        after: str | None = None
        for page in range(self.max_pages_per_listing):
            page_params = {"limit": "100", **params}
            if after:
                page_params["after"] = after
            data = self._get_json(url, page_params)
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

    # ------------------------------------------------------------------ #
    # Public API (BaseCollector)
    # ------------------------------------------------------------------ #

    def collect_subreddit(self, name: str) -> Iterator[RedditPost]:
        log.info("scraper.collect.start", subreddit=name, ua=self.user_agent)
        seen_ids: set[str] = set()
        n_yielded = 0
        for listing, params in LISTING_PLAN:
            for raw in self._iter_listing(name, listing, params):
                pid = str(raw.get("id", ""))
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)

                post = self._record_to_post(raw)
                if post is None:
                    continue
                if self.passes_filters(post):
                    n_yielded += 1
                    yield post
        log.info("scraper.collect.done", subreddit=name, unique=len(seen_ids), yielded=n_yielded)

    @staticmethod
    def _record_to_post(record: dict[str, Any]) -> RedditPost | None:
        try:
            body = record.get("selftext") or ""
            # Reddit serializes deleted/removed bodies as these literal strings.
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
                source="json_scraper",
                collected_at=time.time(),
            )
        except (ValueError, TypeError):
            return None

    def close(self) -> None:
        self._session.close()
        self.cache.close()
