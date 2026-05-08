"""Reddit data collection — pluggable backends behind a common interface."""

from src.collection.base import BaseCollector, RedditPost

__all__ = ["BaseCollector", "RedditPost"]
