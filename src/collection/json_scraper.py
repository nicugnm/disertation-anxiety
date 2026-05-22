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
import random
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
        include_comments: bool = False,
        min_submission_comments: int = 5,
        max_comments_per_post: int = 40,
        min_comment_score: int = 1,
    ) -> None:
        super().__init__(config)
        self.request_interval = float(request_interval)
        self.max_pages_per_listing = int(max_pages_per_listing)
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", DEFAULT_UA)
        self.cache = SqliteCache(cache_path or str(cache_dir() / "json_scraper.sqlite"))
        self.retries = retries
        # Comment-fetching parameters. Comments are off by default because the
        # cost is high (one HTTP fetch per submission, 5–10× total wall time).
        # When on, `min_submission_comments` skips low-engagement posts to keep
        # the runtime bounded; `max_comments_per_post` caps the per-post yield.
        self.include_comments = bool(include_comments)
        self.min_submission_comments = int(min_submission_comments)
        self.max_comments_per_post = int(max_comments_per_post)
        self.min_comment_score = int(min_comment_score)
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

    @staticmethod
    def _parse_retry_after(value: str | None, fallback: float) -> float:
        """Parse a Retry-After header. Accepts integer/float seconds or HTTP-date."""
        if not value:
            return fallback
        s = value.strip()
        try:
            return max(1.0, float(s))
        except ValueError:
            pass
        try:
            target = parsedate_to_datetime(s)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            delta = (target - datetime.now(timezone.utc)).total_seconds()
            return max(1.0, delta)
        except (TypeError, ValueError):
            return fallback

    def _get_json(self, url: str, params: dict[str, str]) -> dict[str, Any] | None:
        """Cached GET. Returns None on permanent failure (404, banned sub, etc.).

        429 (rate limit) is retried indefinitely, honoring the server's
        Retry-After header. Network errors and 5xx server errors fall back
        to the bounded `self.retries` cap.
        """
        key = SqliteCache.make_key(url, params)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        # Per-error-class counters and backoff. 429 is uncapped (we want all
        # the data); transient network/5xx errors keep the bounded retry budget.
        net_attempts = 0
        srv_attempts = 0
        rl_attempts = 0
        rl_backoff = 30.0  # default wait when Retry-After is missing
        while True:
            self._sleep_for_rate()
            try:
                resp = self._session.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                net_attempts += 1
                if net_attempts > self.retries:
                    log.error("scraper.give_up", url=url, reason="network", error=str(e))
                    return None
                wait = min(2.0 ** net_attempts, 60.0)
                log.warning(
                    "scraper.network_error",
                    url=url, error=str(e), attempt=net_attempts, wait=wait,
                )
                time.sleep(wait)
                continue
            self._last_request = time.time()

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    log.warning("scraper.bad_json", url=url)
                    return None
                self.cache.set(key, data)
                return data

            if resp.status_code in (403, 404):
                log.warning("scraper.unavailable", url=url, status=resp.status_code)
                # Cache the negative result so we don't re-hit private/banned subs
                self.cache.set(key, None)
                return None

            if resp.status_code == 429:
                rl_attempts += 1
                wait = self._parse_retry_after(
                    resp.headers.get("Retry-After"), fallback=rl_backoff
                )
                # Cap a single sleep at 10 min and add ≤5s jitter to be polite.
                wait = min(max(wait, 1.0), 600.0)
                wait += random.uniform(0, min(wait * 0.1, 5.0))
                log.warning(
                    "scraper.rate_limited",
                    url=url, status=429, attempt=rl_attempts,
                    wait_seconds=round(wait, 1),
                )
                time.sleep(wait)
                # Increase fallback for the next 429 in case the server keeps
                # not sending Retry-After (cap at 10 min).
                rl_backoff = min(rl_backoff * 2, 600.0)
                continue

            if 500 <= resp.status_code < 600:
                srv_attempts += 1
                if srv_attempts > self.retries:
                    log.error("scraper.give_up", url=url, status=resp.status_code)
                    return None
                wait = min(2.0 ** srv_attempts, 60.0)
                log.warning(
                    "scraper.server_error",
                    url=url, status=resp.status_code, attempt=srv_attempts, wait=wait,
                )
                time.sleep(wait)
                continue

            log.warning("scraper.http_error", url=url, status=resp.status_code)
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
        log.info(
            "scraper.collect.start",
            subreddit=name,
            ua=self.user_agent,
            include_comments=self.include_comments,
        )
        seen_ids: set[str] = set()
        n_submissions = 0
        n_comments = 0
        for listing, params in LISTING_PLAN:
            for raw in self._iter_listing(name, listing, params):
                pid = str(raw.get("id", ""))
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)

                post = self._record_to_post(raw)
                if post is None:
                    continue
                if not self.passes_filters(post):
                    continue
                n_submissions += 1
                yield post

                if self.include_comments and post.num_comments >= self.min_submission_comments:
                    for c in self._fetch_comments(name, post.id):
                        if self.passes_filters(c):
                            n_comments += 1
                            yield c
        log.info(
            "scraper.collect.done",
            subreddit=name,
            unique_submissions=len(seen_ids),
            yielded_submissions=n_submissions,
            yielded_comments=n_comments,
        )

    # ------------------------------------------------------------------ #
    # Comment fetching
    # ------------------------------------------------------------------ #

    def _fetch_comments(self, subreddit: str, submission_id: str) -> Iterator[RedditPost]:
        """Fetch and flatten the comment tree for one submission.

        Reddit's comments endpoint returns a 2-element JSON array:
          [0] = listing containing the submission (one item)
          [1] = listing containing the top-level comment tree
        We walk the tree depth-first, keeping only comments with score above
        the configured minimum, capped at `max_comments_per_post`.
        """
        url = f"{BASE}/r/{subreddit}/comments/{submission_id}.json"
        params = {"limit": "100", "depth": "10", "showmore": "false"}
        data = self._get_json(url, params)
        if not isinstance(data, list) or len(data) < 2:
            return
        comments_listing = data[1]
        if not isinstance(comments_listing, dict):
            return
        children = comments_listing.get("data", {}).get("children", [])
        emitted = 0
        for child in children:
            if emitted >= self.max_comments_per_post:
                return
            for c in self._walk_comment(child, subreddit, submission_id):
                if emitted >= self.max_comments_per_post:
                    return
                emitted += 1
                yield c

    def _walk_comment(
        self,
        node: dict[str, Any],
        subreddit: str,
        submission_id: str,
    ) -> Iterator[RedditPost]:
        if not isinstance(node, dict):
            return
        if node.get("kind") != "t1":
            # Skip non-comment objects (e.g., "more" load-more sentinels).
            return
        d = node.get("data", {})
        body = str(d.get("body") or "")
        # Reddit returns these literals for removed content; skip.
        if body.strip().lower() in ("[deleted]", "[removed]", ""):
            return
        score = int(d.get("score") or 0)
        if score < self.min_comment_score:
            return

        parent_id = str(d.get("parent_id", "") or "")
        # Reddit's parent_id prefixes: "t1_" = comment, "t3_" = submission. Strip.
        if parent_id.startswith("t1_") or parent_id.startswith("t3_"):
            parent_id = parent_id[3:]

        yield RedditPost(
            id=str(d.get("id", "")),
            subreddit=subreddit,
            created_utc=float(d.get("created_utc", 0.0) or 0.0),
            title="",
            body=body,
            author=d.get("author") if d.get("author") != "[deleted]" else None,
            score=score,
            num_comments=0,
            permalink=str(d.get("permalink", "") or ""),
            is_self=True,
            over_18=False,
            source="scraper",
            collected_at=time.time(),
            kind="comment",
            parent_id=parent_id or submission_id,
        )

        # Recurse into nested replies if present.
        replies = d.get("replies")
        if isinstance(replies, dict):
            for child in replies.get("data", {}).get("children", []) or []:
                yield from self._walk_comment(child, subreddit, submission_id)

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
                source="scraper",
                collected_at=time.time(),
                kind="submission",
                parent_id=None,
            )
        except (ValueError, TypeError):
            return None

    def close(self) -> None:
        self._session.close()
        self.cache.close()
