"""Tests for the comment-fetching path of JsonScraperCollector. HTTP mocked."""
from __future__ import annotations

from unittest.mock import patch

from src.collection.json_scraper import JsonScraperCollector
from src.utils.config import load_subreddits


def _listing_with_post(submission_id: str, num_comments: int = 10) -> dict:
    return {
        "data": {
            "after": None,
            "children": [
                {
                    "kind": "t3",
                    "data": {
                        "id": submission_id,
                        "subreddit": "HealthAnxiety",
                        "created_utc": 1700000000.0,
                        "title": "Test title",
                        "selftext": "I was diagnosed with health anxiety last year." * 3,
                        "author": "alice",
                        "score": 10,
                        "num_comments": num_comments,
                        "permalink": f"/r/HealthAnxiety/comments/{submission_id}/",
                        "is_self": True,
                        "over_18": False,
                    },
                }
            ],
        }
    }


def _comments_response(submission_id: str) -> list:
    """Mimics Reddit's /comments/<id>.json response shape."""
    return [
        # [0] = submission listing (we don't use it)
        {"data": {"children": []}},
        # [1] = comments tree
        {
            "data": {
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "id": "c1",
                            "body": "Top-level comment with enough text content here to pass the filter.",
                            "author": "bob",
                            "created_utc": 1700001000.0,
                            "score": 5,
                            "parent_id": f"t3_{submission_id}",
                            "permalink": "/r/HealthAnxiety/comments/x/y/c1/",
                            "replies": {
                                "data": {
                                    "children": [
                                        {
                                            "kind": "t1",
                                            "data": {
                                                "id": "c2",
                                                "body": (
                                                    "Nested reply with substantive content "
                                                    "to clear the length filter."
                                                ),
                                                "author": "carol",
                                                "created_utc": 1700002000.0,
                                                "score": 3,
                                                "parent_id": "t1_c1",
                                                "permalink": "/r/HealthAnxiety/comments/x/y/c2/",
                                                "replies": "",
                                            },
                                        },
                                    ],
                                },
                            },
                        },
                    },
                    {
                        "kind": "t1",
                        "data": {
                            "id": "deleted_one",
                            "body": "[deleted]",
                            "author": "[deleted]",
                            "created_utc": 1700003000.0,
                            "score": 1,
                            "parent_id": f"t3_{submission_id}",
                            "permalink": "",
                            "replies": "",
                        },
                    },
                    {
                        "kind": "more",  # "load more" sentinel — should be ignored
                        "data": {"children": [], "id": "ml", "count": 5},
                    },
                ],
            },
        },
    ]


def test_comments_disabled_by_default(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    coll = JsonScraperCollector(
        cfg, request_interval=0.0, max_pages_per_listing=1,
        cache_path=str(tmp_path / "c.sqlite"),
    )
    assert coll.include_comments is False


def test_comments_yielded_and_walk_recursively(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    coll = JsonScraperCollector(
        cfg,
        request_interval=0.0,
        max_pages_per_listing=1,
        cache_path=str(tmp_path / "c.sqlite"),
        include_comments=True,
        min_submission_comments=1,
        max_comments_per_post=10,
        min_comment_score=0,
    )

    # First HTTP call returns the listing; subsequent calls (for each of the
    # 6 listings in LISTING_PLAN) — we have to handle them all.
    # The listing call returns our 1-post fixture; the *comments* fetch URL
    # is different.
    listing_payload = _listing_with_post("abc123")
    comments_payload = _comments_response("abc123")
    empty_listing = {"data": {"after": None, "children": []}}

    class _Resp:
        status_code = 200
        headers: dict = {}

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None):  # noqa: ARG001
        if "comments/abc123" in url:
            return _Resp(comments_payload)
        if "comments/" in url:
            # Some unknown comment URL (shouldn't happen in this test)
            return _Resp(empty_listing)
        # Listing pages — first call returns our fixture, rest empty
        return _Resp(listing_payload if not fake_get.served else empty_listing)
    fake_get.served = False  # type: ignore[attr-defined]

    def fake_get_stateful(url, params=None, timeout=None):  # noqa: ARG001
        if "comments/abc123" in url:
            return _Resp(comments_payload)
        # Return listing payload exactly once; afterwards return empty
        if not fake_get_stateful.served:
            fake_get_stateful.served = True
            return _Resp(listing_payload)
        return _Resp(empty_listing)
    fake_get_stateful.served = False

    with patch.object(coll._session, "get", side_effect=fake_get_stateful):
        rows = list(coll.collect_subreddit("HealthAnxiety"))

    # Expect: 1 submission + 2 valid comments (c1 + nested c2), [deleted] skipped, "more" skipped
    kinds = [r.kind for r in rows]
    assert kinds.count("submission") == 1
    assert kinds.count("comment") == 2

    submission = next(r for r in rows if r.kind == "submission")
    assert submission.id == "abc123"
    assert submission.parent_id is None

    c1 = next(r for r in rows if r.id == "c1")
    assert c1.kind == "comment"
    assert c1.parent_id == "abc123"  # t3_ prefix stripped

    c2 = next(r for r in rows if r.id == "c2")
    assert c2.kind == "comment"
    assert c2.parent_id == "c1"  # t1_ prefix stripped — nested reply preserved


def test_comments_skipped_when_min_submission_comments_not_met(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    coll = JsonScraperCollector(
        cfg,
        request_interval=0.0,
        max_pages_per_listing=1,
        cache_path=str(tmp_path / "c.sqlite"),
        include_comments=True,
        min_submission_comments=100,  # higher than the post's num_comments
    )

    listing_payload = _listing_with_post("xyz", num_comments=3)
    empty_listing = {"data": {"after": None, "children": []}}

    class _Resp:
        status_code = 200
        headers: dict = {}

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None):  # noqa: ARG001
        if "comments/" in url:
            # Should never be called — we expect the filter to skip the fetch
            raise AssertionError("Comment fetch should have been skipped")
        if not fake_get.served:
            fake_get.served = True
            return _Resp(listing_payload)
        return _Resp(empty_listing)
    fake_get.served = False

    with patch.object(coll._session, "get", side_effect=fake_get):
        rows = list(coll.collect_subreddit("HealthAnxiety"))

    assert len(rows) == 1
    assert rows[0].kind == "submission"
