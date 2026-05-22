"""Test the JSON scraper without hitting Reddit. Mocks the HTTP layer.

The point of this test is twofold:
  1. The pagination + multi-listing logic correctly de-duplicates by post ID.
  2. The "after" cursor termination works (empty page or null after stops the loop).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.collection.json_scraper import JsonScraperCollector
from src.utils.config import load_subreddits


def _fake_listing_page(post_ids: list[str], after: str | None) -> dict:
    """Build a Reddit-shaped JSON payload with given post IDs."""
    return {
        "data": {
            "after": after,
            "children": [
                {
                    "kind": "t3",
                    "data": {
                        "id": pid,
                        "subreddit": "Anxiety",
                        "created_utc": 1700000000.0 + i,
                        "title": f"Title {pid}",
                        "selftext": "I'm worried about everything." * 5,  # passes min length filter
                        "author": f"user_{pid}",
                        "score": 10,
                        "num_comments": 2,
                        "permalink": f"/r/Anxiety/comments/{pid}/",
                        "is_self": True,
                        "over_18": False,
                    },
                }
                for i, pid in enumerate(post_ids)
            ],
        }
    }


def test_scraper_pagination_and_dedup(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    cache_path = tmp_path / "cache.sqlite"
    coll = JsonScraperCollector(
        cfg,
        request_interval=0.0,
        max_pages_per_listing=3,
        cache_path=str(cache_path),
    )

    # Different listings return overlapping IDs — the collector should dedup.
    page_a = _fake_listing_page(["aaa", "bbb"], after="t3_bbb")
    page_b = _fake_listing_page(["ccc"], after=None)
    overlapping = _fake_listing_page(["aaa", "ddd"], after=None)

    pages: list[dict] = [page_a, page_b]  # first listing: 3 posts, ends
    pages += [overlapping]  # second listing: aaa is dup, ddd is new
    # Remaining listings return empty
    pages += [{"data": {"after": None, "children": []}}] * 10

    pages_iter = iter(pages)

    class _Resp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload
            self.headers: dict = {}

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None):  # noqa: ARG001
        return _Resp(next(pages_iter))

    with patch.object(coll._session, "get", side_effect=fake_get):
        rows = list(coll.collect_subreddit("Anxiety"))

    coll.close()

    ids = [r.id for r in rows]
    assert ids == ["aaa", "bbb", "ccc", "ddd"]
    assert all(r.subreddit == "Anxiety" for r in rows)
    assert all(r.source == "scraper" for r in rows)


def test_scraper_handles_404(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    coll = JsonScraperCollector(
        cfg,
        request_interval=0.0,
        max_pages_per_listing=2,
        cache_path=str(tmp_path / "cache.sqlite"),
    )

    class _Resp:
        status_code = 404
        headers: dict = {}

    with patch.object(coll._session, "get", return_value=_Resp()):
        rows = list(coll.collect_subreddit("ThisSubDoesNotExist"))
    coll.close()
    assert rows == []
