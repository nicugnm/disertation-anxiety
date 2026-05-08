"""Live Reddit collection via PRAW.

Requires Reddit API credentials in .env (see .env.example).
PRAW respects the API rate limit (~100 QPM). Listings are capped by
Reddit at ~1000 items, so this is best for *recent* posts; for years
of history use `dump_collector.py`.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterator

from src.collection.base import BaseCollector, RedditPost
from src.utils.logging import get_logger

log = get_logger(__name__)


class PrawCollector(BaseCollector):
    def __init__(self, config) -> None:  # noqa: ANN001 — circular-import-friendly
        super().__init__(config)
        self._reddit = self._build_client()

    @staticmethod
    def _build_client():
        import praw

        creds = {
            "client_id": os.getenv("REDDIT_CLIENT_ID"),
            "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
            "user_agent": os.getenv("REDDIT_USER_AGENT", "anxiety-research/0.1"),
        }
        # Username/password optional — only needed for actions that require auth.
        if os.getenv("REDDIT_USERNAME") and os.getenv("REDDIT_PASSWORD"):
            creds["username"] = os.getenv("REDDIT_USERNAME")
            creds["password"] = os.getenv("REDDIT_PASSWORD")

        missing = [k for k in ("client_id", "client_secret") if not creds.get(k)]
        if missing:
            raise RuntimeError(
                f"Missing Reddit credentials: {missing}. "
                "Copy .env.example to .env and fill in the keys."
            )

        return praw.Reddit(**creds, check_for_async=False)

    def collect_subreddit(self, name: str) -> Iterator[RedditPost]:
        c = self.config.collection
        sub = self._reddit.subreddit(name)
        log.info("praw.collect.start", subreddit=name, time_filter=c.time_filter)

        # `top` gives us the highest-scoring posts within the time window;
        # `new` is the alternative for chronologically-recent collection.
        listing = sub.top(time_filter=c.time_filter, limit=c.posts_per_subreddit)

        n_yielded = 0
        for submission in listing:
            try:
                post = RedditPost(
                    id=submission.id,
                    subreddit=str(submission.subreddit),
                    created_utc=float(submission.created_utc),
                    title=submission.title or "",
                    body=submission.selftext or "",
                    author=str(submission.author) if submission.author else None,
                    score=int(submission.score),
                    num_comments=int(submission.num_comments),
                    permalink=submission.permalink,
                    is_self=bool(submission.is_self),
                    over_18=bool(submission.over_18),
                    source="praw",
                    collected_at=time.time(),
                )
            except Exception as e:  # noqa: BLE001 — defensive; log and skip
                log.warning("praw.collect.parse_error", error=str(e))
                continue

            if self.passes_filters(post):
                n_yielded += 1
                yield post

        log.info("praw.collect.done", subreddit=name, yielded=n_yielded)
