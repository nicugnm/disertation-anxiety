"""Tests for the search-based scraper. HTTP layer is mocked — no real Reddit calls."""
from __future__ import annotations

from unittest.mock import patch

from src.collection.search_scraper import (
    DEFAULT_DISCLOSURE_QUERIES,
    SearchScraperCollector,
    _slug,
)
from src.utils.config import load_subreddits


def _fake_search_page(post_ids: list[str], after: str | None) -> dict:
    return {
        "data": {
            "after": after,
            "children": [
                {
                    "kind": "t3",
                    "data": {
                        "id": pid,
                        "subreddit": "HealthAnxiety",
                        "created_utc": 1700000000.0 + i,
                        "title": f"Title {pid}",
                        "selftext": (
                            "I was diagnosed with health anxiety last year. "
                            "Things have been hard but I'm getting better." * 2
                        ),
                        "author": f"user_{pid}",
                        "score": 5,
                        "num_comments": 1,
                        "permalink": f"/r/HealthAnxiety/comments/{pid}/",
                        "is_self": True,
                        "over_18": False,
                    },
                }
                for i, pid in enumerate(post_ids)
            ],
        }
    }


def test_slug_makes_safe_filenames():
    assert _slug("I was diagnosed with depression") == "i_was_diagnosed_with_depression"
    assert _slug("") == "query"
    assert _slug("!!!??? a/b\\c") == "a_b_c"


def test_default_queries_cover_all_targets():
    qs = " ".join(DEFAULT_DISCLOSURE_QUERIES).lower()
    for needle in ("anxiety", "depression", "health anxiety", "hypochondriac"):
        assert needle in qs


def test_search_collector_yields_and_dedupes(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    cache_path = tmp_path / "cache.sqlite"
    coll = SearchScraperCollector(
        cfg,
        queries=["I was diagnosed with health anxiety"],
        request_interval=0.0,
        max_pages_per_query=2,
        cache_path=str(cache_path),
    )

    page_a = _fake_search_page(["aaa", "bbb"], after="t3_bbb")
    page_b = _fake_search_page(["bbb", "ccc"], after=None)  # bbb is duplicate
    pages = iter([page_a, page_b])

    class _Resp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload
            self.headers: dict = {}

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None):  # noqa: ARG001
        return _Resp(next(pages))

    with patch.object(coll._session, "get", side_effect=fake_get):
        rows = list(coll.collect_subreddit("I was diagnosed with health anxiety"))
    coll.close()

    ids = [r.id for r in rows]
    assert ids == ["aaa", "bbb", "ccc"]
    assert all(r.source == "search" for r in rows)
    assert all(r.kind == "submission" for r in rows)
    # The matched query is preserved on the post for traceability
    assert all(r.extra.get("matched_query") == "I was diagnosed with health anxiety" for r in rows)


def test_search_collector_collect_all_iterates_queries(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    coll = SearchScraperCollector(
        cfg,
        queries=["q1", "q2", "q3"],
        request_interval=0.0,
        max_pages_per_query=1,
        cache_path=str(tmp_path / "cache.sqlite"),
    )

    # Each query returns one different post id, after=None to end pagination.
    pages = [_fake_search_page([f"id_{i}"], after=None) for i in range(3)]
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
        rows = list(coll.collect_all())
    coll.close()
    assert len(rows) == 3
    assert [r.id for r in rows] == ["id_0", "id_1", "id_2"]


def test_search_collector_handles_404(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    coll = SearchScraperCollector(
        cfg, queries=["dummy"], request_interval=0.0,
        max_pages_per_query=1, cache_path=str(tmp_path / "cache.sqlite"),
    )

    class _Resp:
        status_code = 404
        headers: dict = {}

    with patch.object(coll._session, "get", return_value=_Resp()):
        rows = list(coll.collect_subreddit("dummy"))
    coll.close()
    assert rows == []
