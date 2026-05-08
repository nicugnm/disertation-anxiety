"""Common interface for Reddit collectors and the canonical post schema."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from src.utils.config import SubredditsConfig


@dataclass
class RedditPost:
    """Canonical post schema. All collectors emit records of this shape.

    Fields beyond Reddit's raw API exist for downstream pipeline use:
    `source` records which collector produced the row, `collected_at` lets us
    audit data freshness in the thesis, and `extra` keeps backend-specific
    fields without polluting the schema.
    """

    id: str
    subreddit: str
    created_utc: float
    title: str
    body: str
    author: str | None
    score: int
    num_comments: int
    permalink: str
    is_self: bool
    over_18: bool
    source: str  # "praw" | "dump" | "synthetic"
    collected_at: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subreddit": self.subreddit,
            "created_utc": self.created_utc,
            "title": self.title,
            "body": self.body,
            "author": self.author,
            "score": self.score,
            "num_comments": self.num_comments,
            "permalink": self.permalink,
            "is_self": self.is_self,
            "over_18": self.over_18,
            "source": self.source,
            "collected_at": self.collected_at,
        }


class BaseCollector(ABC):
    """All collectors implement `collect_subreddit` yielding RedditPost."""

    def __init__(self, config: SubredditsConfig) -> None:
        self.config = config

    @abstractmethod
    def collect_subreddit(self, name: str) -> Iterator[RedditPost]:
        """Yield posts from the given subreddit, respecting the config filters."""

    def collect_all(self) -> Iterator[RedditPost]:
        """Iterate over every configured subreddit."""
        for s in self.config.subreddits:
            yield from self.collect_subreddit(s.name)

    def passes_filters(self, post: RedditPost) -> bool:
        c = self.config.collection
        if c.include_self_only and not post.is_self:
            return False
        if post.score < c.min_score:
            return False
        if len(post.body or "") < c.min_body_chars:
            return False
        return True
