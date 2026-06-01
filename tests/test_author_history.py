from __future__ import annotations

import pandas as pd

from src.collection.author_history import recover_author_usernames
from src.preprocessing.anonymize import _hash_username
from src.utils.io import write_parquet


def test_recover_usernames_maps_hashes_and_skips_deleted(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Two raw shards with real authors + a deleted one.
    write_parquet(
        pd.DataFrame({"id": ["1", "2"], "author": ["alice", "[deleted]"]}),
        raw_dir / "SubA.parquet",
    )
    write_parquet(
        pd.DataFrame({"id": ["3"], "author": ["bob"]}),
        raw_dir / "SubB.parquet",
    )
    users = pd.DataFrame({"author_hash": [_hash_username("alice"), _hash_username("bob")]})
    mapping = recover_author_usernames(users, raw_dir=raw_dir)
    assert mapping == {_hash_username("alice"): "alice", _hash_username("bob"): "bob"}
    # Deleted author never appears.
    assert "[deleted]" not in mapping.values()


def test_recover_skips_shard_without_author_column(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_parquet(pd.DataFrame({"id": ["1"], "body": ["text"]}), raw_dir / "NoAuthor.parquet")
    users = pd.DataFrame({"author_hash": [_hash_username("alice")]})
    assert recover_author_usernames(users, raw_dir=raw_dir) == {}


from unittest.mock import patch

from src.collection.author_history import AuthorHistoryCollector
from src.utils.config import load_subreddits


class _Resp:
    status_code = 200
    headers: dict = {}

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _submitted_page(ids, after):
    return {"data": {"after": after, "children": [
        {"kind": "t3", "data": {
            "id": pid, "subreddit": "SomeSub", "created_utc": 1700000000.0,
            "title": f"Title {pid}",
            "selftext": "I was diagnosed with health anxiety. " * 3,
            "author": "alice", "score": 5, "num_comments": 1,
            "permalink": f"/r/SomeSub/comments/{pid}/", "is_self": True, "over_18": False,
        }} for pid in ids]}}


def _comments_page(ids, after):
    return {"data": {"after": after, "children": [
        {"kind": "t1", "data": {
            "id": pid, "subreddit": "OtherSub", "created_utc": 1700001000.0,
            "body": "This is a sufficiently long comment body to clear the length filter.",
            "author": "alice", "score": 3, "parent_id": "t3_xyz",
            "permalink": f"/r/OtherSub/comments/x/{pid}/",
        }} for pid in ids]}}


def test_collect_user_yields_submissions_and_comments(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    coll = AuthorHistoryCollector(
        cfg, request_interval=0.0, max_pages_per_listing=2,
        cache_path=str(tmp_path / "ah.sqlite"),
    )
    empty = {"data": {"after": None, "children": []}}

    def fake_get(url, params=None, timeout=None):  # noqa: ARG001
        if "submitted.json" in url:
            return _Resp(_submitted_page(["s1"], after=None))
        if "comments.json" in url:
            return _Resp(_comments_page(["c1"], after=None))
        return _Resp(empty)

    with patch.object(coll._session, "get", side_effect=fake_get):
        rows = list(coll.collect_user("alice"))

    kinds = [r.kind for r in rows]
    assert kinds.count("submission") == 1
    assert kinds.count("comment") == 1
    assert all(r.source == "author_history" for r in rows)
    c1 = next(r for r in rows if r.kind == "comment")
    assert c1.parent_id == "xyz"  # t3_ prefix stripped


def test_collect_user_handles_404(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    coll = AuthorHistoryCollector(
        cfg, request_interval=0.0, max_pages_per_listing=1,
        cache_path=str(tmp_path / "ah.sqlite"),
    )

    class _R404:
        status_code = 404
        headers: dict = {}

        def json(self):
            return {}

    with patch.object(coll._session, "get", return_value=_R404()):
        rows = list(coll.collect_user("ghost"))
    assert rows == []


def test_collect_user_paginates(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    coll = AuthorHistoryCollector(
        cfg, request_interval=0.0, max_pages_per_listing=3,
        cache_path=str(tmp_path / "ah.sqlite"),
    )
    calls = []

    def fake_get(url, params=None, timeout=None):  # noqa: ARG001
        calls.append(params.get("after"))
        if "submitted.json" in url:
            if params.get("after") is None:
                return _Resp(_submitted_page(["s1"], after="tok1"))
            return _Resp(_submitted_page(["s2"], after=None))
        return _Resp({"data": {"after": None, "children": []}})

    with patch.object(coll._session, "get", side_effect=fake_get):
        rows = list(coll.collect_user("alice"))
    assert len([r for r in rows if r.kind == "submission"]) == 2
    assert "tok1" in calls  # the second page's `after` cursor was requested


def test_collect_user_dedups_repeated_ids(tmp_path):
    cfg = load_subreddits("configs/subreddits.yaml")
    coll = AuthorHistoryCollector(
        cfg, request_interval=0.0, max_pages_per_listing=1,
        cache_path=str(tmp_path / "ah.sqlite"),
    )

    def fake_get(url, params=None, timeout=None):  # noqa: ARG001
        # Same id "dup1" appears as a submission AND as a comment.
        if "submitted.json" in url:
            return _Resp(_submitted_page(["dup1"], after=None))
        if "comments.json" in url:
            return _Resp(_comments_page(["dup1"], after=None))
        return _Resp({"data": {"after": None, "children": []}})

    with patch.object(coll._session, "get", side_effect=fake_get):
        rows = list(coll.collect_user("alice"))
    assert [r.id for r in rows] == ["dup1"]  # deduped across the two sections


# append to tests/test_author_history.py
from src.collection.author_history import run_author_collection
from src.collection.base import RedditPost


class _FakeCollector:
    """Yields one canned post for any username; records calls."""
    def __init__(self):
        self.calls = []

    def collect_user(self, username):
        self.calls.append(username)
        yield RedditPost(
            id=f"{username}_p1", subreddit="X", created_utc=1.0, title="t",
            body="b" * 60, author=username, score=1, num_comments=0,
            permalink="", is_self=True, over_18=False, source="author_history",
            collected_at=0.0, kind="submission",
        )


def test_run_author_collection_writes_per_hash_and_resumes(tmp_path):
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    write_parquet(pd.DataFrame({"id": ["1"], "author": ["alice"]}), raw_dir / "S.parquet")
    out_dir = tmp_path / "authors"
    users = pd.DataFrame({"author_hash": [_hash_username("alice")]})
    fake = _FakeCollector()

    stats = run_author_collection(users, raw_dir=raw_dir, out_dir=out_dir, collector=fake)
    h = _hash_username("alice")
    assert (out_dir / f"{h}.parquet").exists()
    assert stats["written"] == 1
    assert fake.calls == ["alice"]

    # Re-run: existing file → skipped, collector NOT called again.
    fake2 = _FakeCollector()
    stats2 = run_author_collection(users, raw_dir=raw_dir, out_dir=out_dir, collector=fake2)
    assert stats2["skipped_existing"] == 1
    assert fake2.calls == []


# append to tests/test_author_history.py
from src.collection.author_history import merge_and_dedupe


def test_merge_and_dedupe_removes_cross_corpus_duplicates(tmp_path):
    base = pd.DataFrame({"subreddit": ["A", "A"], "clean_text": ["hello world", "unique base post"]})
    extra = pd.DataFrame({"subreddit": ["A", "B"], "clean_text": ["hello world", "brand new post"]})
    p_base = tmp_path / "base.parquet"; p_extra = tmp_path / "extra.parquet"
    out = tmp_path / "_all.parquet"
    write_parquet(base, p_base); write_parquet(extra, p_extra)
    merged = merge_and_dedupe([p_base, p_extra], out)
    # "hello world" appears in both → collapsed to 1; 3 distinct texts total.
    assert len(merged) == 3
    assert out.exists()
