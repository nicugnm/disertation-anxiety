# By-Author Full-History Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich the full Reddit post/comment histories of the 3,943 disclosure-test users (both classes), re-detect disclosures, regroup within the cohort, and re-run the user-level evaluation.

**Architecture:** A no-auth JSON collector (`AuthorHistoryCollector`) subclasses the existing `JsonScraperCollector` to fetch `/user/<name>/submitted.json` + `/comments.json`. A new `collect-authors` CLI command recovers usernames from `data/raw/` (salted-hash → username), scrapes resumably (one parquet per `author_hash`), then existing pipeline stages re-label and a new `rebuild_groups_within_cohort` re-assigns groups before a fresh TF-IDF user-level eval.

**Tech Stack:** Python 3.12 (conda env `anxiety-disertation`), pandas, pyarrow, requests, typer, pytest. Reuses `src/collection/json_scraper.py`, `src/preprocessing/*`, `src/labeling/disclosure_dataset.py`.

**Conventions used by every task below:**
- `PY` = `C:\Users\Nicu\miniconda3\envs\anxiety-disertation\python.exe` (the only working interpreter — bare `python` is a broken Store stub).
- Run tests with: `& $PY -m pytest <path> -v`
- All new modules live under `src/`; tests under `tests/`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `.gitignore` | Exclude per-user raw histories from git | Modify |
| `src/collection/author_history.py` | Username recovery, `AuthorHistoryCollector`, orchestration, merge helper | Create |
| `src/cli.py` | `collect-authors` command | Modify |
| `src/labeling/disclosure_dataset.py` | `rebuild_groups_within_cohort` | Modify |
| `tests/test_author_history.py` | Collector + recovery + orchestration + merge tests | Create |
| `tests/test_disclosure_dataset.py` | `rebuild_groups_within_cohort` test | Modify |

---

## Task 1: Exclude per-user histories from git

**Files:**
- Modify: `.gitignore`

> ⚠️ **Tradeoff note (confirm at execution):** this repo *deliberately commits* `data/raw/` to a private repo for cross-machine workflow (see the comment in `.gitignore`). The approved spec excludes `data/raw/authors/` because full per-user mental-health histories are far more identifiable. Consequence: a second machine won't receive author histories via git (re-scrape or copy manually). `.cache/` is already ignored, so the HTTP cache needs no change.

- [ ] **Step 1: Append the exclusion under the "Project data" section**

Add these lines after line 23 (`!data/processed/.gitkeep`):

```gitignore

# Per-user full histories: real usernames + full mental-health profiles.
# Excluded even though data/raw/* is otherwise committed (see note above).
data/raw/authors/
```

- [ ] **Step 2: Verify the rule matches**

Run: `git check-ignore -v data/raw/authors/x.parquet`
Expected: a line citing `.gitignore` and the `data/raw/authors/` pattern.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore per-user author histories (data/raw/authors/)"
```

---

## Task 2: Recover usernames from raw (`recover_author_usernames`)

**Files:**
- Create: `src/collection/author_history.py`
- Test: `tests/test_author_history.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_author_history.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& $PY -m pytest tests/test_author_history.py::test_recover_usernames_maps_hashes_and_skips_deleted -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.collection.author_history'`

- [ ] **Step 3: Create the module with the recovery function**

```python
# src/collection/author_history.py
"""By-author full-history collection (no-auth Reddit JSON).

Recovers the real usernames of an anonymized cohort from data/raw/ (where the
`author` column survives), then fetches each user's full submission + comment
history. Output shards are keyed by `author_hash` so usernames never appear in
filenames.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.collection.base import RedditPost
from src.collection.json_scraper import BASE, JsonScraperCollector
from src.preprocessing.anonymize import _hash_username
from src.utils.config import cache_dir
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)

_BAD_AUTHORS = {"[deleted]", "[removed]", "", "none"}


def recover_author_usernames(
    users_df: pd.DataFrame, raw_dir: str | Path = "data/raw"
) -> dict[str, str]:
    """Map each cohort `author_hash` back to its real username via data/raw/.

    Recomputes the pipeline's salted hash (`_hash_username`) over the `author`
    column of every raw shard and matches against the cohort's hashes. Skips
    deleted/removed authors. Returns {author_hash: username}.
    """
    raw_dir = Path(raw_dir)
    target = {str(h) for h in users_df["author_hash"].dropna().astype(str)} - {""}
    out: dict[str, str] = {}
    for fp in sorted(raw_dir.glob("*.parquet")):
        try:
            d = read_parquet(fp)
        except Exception as e:  # noqa: BLE001
            log.warning("author.recover.read_failed", file=str(fp), error=str(e))
            continue
        if "author" not in d.columns:
            continue
        for a in d["author"].dropna().astype(str).unique():
            if a.lower() in _BAD_AUTHORS:
                continue
            h = _hash_username(a)
            if h in target and h not in out:
                out[h] = a
        if len(out) == len(target):
            break
    log.info("author.recover.done", requested=len(target), recovered=len(out))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& $PY -m pytest tests/test_author_history.py::test_recover_usernames_maps_hashes_and_skips_deleted -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/collection/author_history.py tests/test_author_history.py
git commit -m "feat: recover_author_usernames — map anonymized cohort back to usernames"
```

---

## Task 3: `AuthorHistoryCollector.collect_user`

**Files:**
- Modify: `src/collection/author_history.py`
- Test: `tests/test_author_history.py`

- [ ] **Step 1: Write the failing test (HTTP mocked, mirrors test_json_scraper)**

```python
# append to tests/test_author_history.py
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

    with patch.object(coll._session, "get", return_value=_R404()):
        rows = list(coll.collect_user("ghost"))
    assert rows == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& $PY -m pytest tests/test_author_history.py -k collect_user -v`
Expected: FAIL — `AttributeError: ... has no attribute 'AuthorHistoryCollector'`

- [ ] **Step 3: Add the collector class to `src/collection/author_history.py`**

```python
# append to src/collection/author_history.py

class AuthorHistoryCollector(JsonScraperCollector):
    """Fetch one user's full submission + comment history via no-auth JSON.

    Reuses JsonScraperCollector's cached, backoff-aware `_get_json`. Uses a
    separate cache file so it never collides with the subreddit-listing cache.
    """

    def __init__(self, config, cache_path: str | None = None, **kwargs) -> None:  # noqa: ANN001
        super().__init__(
            config,
            cache_path=cache_path or str(cache_dir() / "author_history.sqlite"),
            **kwargs,
        )

    def _iter_user_listing(self, username: str, section: str) -> Iterator[tuple[str, dict]]:
        """Yield (kind, data) for each item in /user/<name>/<section>.json."""
        url = f"{BASE}/user/{username}/{section}.json"
        after: str | None = None
        for _ in range(self.max_pages_per_listing):
            params = {"limit": "100"}
            if after:
                params["after"] = after
            data = self._get_json(url, params)
            if not data or "data" not in data:
                return
            children = data["data"].get("children", [])
            if not children:
                return
            for c in children:
                d = c.get("data")
                if isinstance(d, dict):
                    yield str(c.get("kind", "")), d
            after = data["data"].get("after")
            if not after:
                return

    def collect_user(self, username: str) -> Iterator[RedditPost]:
        seen: set[str] = set()
        n_sub = n_cmt = 0
        for section in ("submitted", "comments"):
            for kind, raw in self._iter_user_listing(username, section):
                pid = str(raw.get("id", ""))
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                if kind == "t3":
                    post = self._submission_to_post(raw)
                elif kind == "t1":
                    post = self._comment_to_post(raw)
                else:
                    continue
                if post is None or not self.passes_filters(post):
                    continue
                if post.kind == "submission":
                    n_sub += 1
                else:
                    n_cmt += 1
                yield post
        log.info("author.collect_user.done", user_hash=_hash_username(username),
                 submissions=n_sub, comments=n_cmt)

    @staticmethod
    def _submission_to_post(raw: dict) -> RedditPost | None:
        try:
            body = raw.get("selftext") or ""
            if body.strip().lower() in ("[deleted]", "[removed]"):
                body = ""
            return RedditPost(
                id=str(raw.get("id", "")),
                subreddit=str(raw.get("subreddit", "")),
                created_utc=float(raw.get("created_utc", 0.0) or 0.0),
                title=str(raw.get("title", "") or ""),
                body=body,
                author=raw.get("author"),
                score=int(raw.get("score", 0) or 0),
                num_comments=int(raw.get("num_comments", 0) or 0),
                permalink=str(raw.get("permalink", "") or ""),
                is_self=bool(raw.get("is_self", False)),
                over_18=bool(raw.get("over_18", False)),
                source="author_history",
                collected_at=time.time(),
                kind="submission",
                parent_id=None,
            )
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _comment_to_post(raw: dict) -> RedditPost | None:
        try:
            body = str(raw.get("body") or "")
            if body.strip().lower() in ("[deleted]", "[removed]", ""):
                return None
            parent_id = str(raw.get("parent_id", "") or "")
            if parent_id.startswith(("t1_", "t3_")):
                parent_id = parent_id[3:]
            author = raw.get("author")
            return RedditPost(
                id=str(raw.get("id", "")),
                subreddit=str(raw.get("subreddit", "")),
                created_utc=float(raw.get("created_utc", 0.0) or 0.0),
                title="",
                body=body,
                author=author if author != "[deleted]" else None,
                score=int(raw.get("score", 0) or 0),
                num_comments=0,
                permalink=str(raw.get("permalink", "") or ""),
                is_self=True,
                over_18=False,
                source="author_history",
                collected_at=time.time(),
                kind="comment",
                parent_id=parent_id or None,
            )
        except (ValueError, TypeError):
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& $PY -m pytest tests/test_author_history.py -k collect_user -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/collection/author_history.py tests/test_author_history.py
git commit -m "feat: AuthorHistoryCollector.collect_user (no-auth submitted+comments)"
```

---

## Task 4: Orchestration + `collect-authors` CLI command

**Files:**
- Modify: `src/collection/author_history.py`
- Modify: `src/cli.py`
- Test: `tests/test_author_history.py`

- [ ] **Step 1: Write the failing test (fake collector injected — no HTTP)**

```python
# append to tests/test_author_history.py
from src.collection.author_history import run_author_collection
from src.collection.base import RedditPost


class _FakeCollector:
    """Yields two canned posts for any username; records calls."""
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& $PY -m pytest tests/test_author_history.py::test_run_author_collection_writes_per_hash_and_resumes -v`
Expected: FAIL — `cannot import name 'run_author_collection'`

- [ ] **Step 3: Add the orchestration function to `src/collection/author_history.py`**

```python
# append to src/collection/author_history.py

def run_author_collection(
    users_df: pd.DataFrame,
    config=None,  # noqa: ANN001 — SubredditsConfig; required only when building the default collector
    raw_dir: str | Path = "data/raw",
    out_dir: str | Path = "data/raw/authors",
    request_interval: float = 1.5,
    max_pages: int = 10,
    cache_path: str | None = None,
    collector: "AuthorHistoryCollector | None" = None,
) -> dict[str, int]:
    """Scrape full histories for every cohort user; write one parquet per author_hash.

    Resumable: a user whose <hash>.parquet already exists is skipped (and the
    collector is never called for them). `collector` may be injected for tests.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hash_to_user = recover_author_usernames(users_df, raw_dir)
    if collector is None:
        collector = AuthorHistoryCollector(
            config, request_interval=request_interval,
            max_pages_per_listing=max_pages, cache_path=cache_path,
        )

    hashes = sorted({str(h) for h in users_df["author_hash"].dropna().astype(str)} - {""})
    written = skipped = empty = unrecoverable = 0

    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    progress = Progress(
        SpinnerColumn(), TextColumn("[bold]Author histories"),
        BarColumn(bar_width=None), MofNCompleteColumn(),
        TextColumn("•"), TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn(),
    )
    with progress:
        task = progress.add_task("...", total=len(hashes))
        for h in hashes:
            target = out / f"{h}.parquet"
            if target.exists():
                skipped += 1
                progress.advance(task)
                continue
            username = hash_to_user.get(h)
            if not username:
                unrecoverable += 1
                progress.advance(task)
                continue
            rows = [p.to_dict() for p in collector.collect_user(username)]
            if rows:
                write_parquet(pd.DataFrame(rows), target)
                written += 1
            else:
                empty += 1
            progress.advance(task)

    stats = {
        "requested": len(hashes), "written": written, "skipped_existing": skipped,
        "empty": empty, "unrecoverable": unrecoverable,
    }
    log.info("author.run.done", **stats)
    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& $PY -m pytest tests/test_author_history.py::test_run_author_collection_writes_per_hash_and_resumes -v`
Expected: PASS

- [ ] **Step 5: Add the CLI command to `src/cli.py`**

Insert this command after the existing `collect` command (after line 89, before the `preprocess` section):

```python
@app.command("collect-authors")
def collect_authors(
    users_csv: str = typer.Option(
        None, help="Cohort CSV (default data/processed/disclosure_testset__users.csv)"
    ),
    raw_dir: str = typer.Option("data/raw", help="Where to recover usernames from"),
    out_dir: str = typer.Option("data/raw/authors", help="Per-user history shards"),
    request_interval: float = typer.Option(1.5, help="Seconds between requests"),
    max_pages: int = typer.Option(10, help="Pages per listing (10×100 ≈ 1000-item cap)"),
    config: str = typer.Option("configs/subreddits.yaml"),
) -> None:
    """Fetch full Reddit histories for the disclosure-test cohort (both classes).

    Resumable: re-running skips users whose <hash>.parquet already exists.
    """
    import pandas as pd

    from src.collection.author_history import run_author_collection

    cfg = load_subreddits(config)
    csv_path = (
        Path(users_csv) if users_csv
        else data_dir("processed") / "disclosure_testset__users.csv"
    )
    if not csv_path.exists():
        raise typer.Exit(f"Cohort CSV not found: {csv_path} — run build-disclosure-testset first.")
    users_df = pd.read_csv(csv_path)
    stats = run_author_collection(
        users_df, cfg, raw_dir=raw_dir, out_dir=out_dir,
        request_interval=request_interval, max_pages=max_pages,
    )
    console.print(f"[green]Author history collection complete:[/green] {stats}")
```

- [ ] **Step 6: Verify the CLI command registers**

Run: `& $PY -m src.cli collect-authors --help`
Expected: help text listing `--users-csv`, `--out-dir`, `--max-pages`, etc.

- [ ] **Step 7: Commit**

```bash
git add src/collection/author_history.py src/cli.py tests/test_author_history.py
git commit -m "feat: collect-authors CLI + resumable run_author_collection"
```

---

## Task 5: `merge_and_dedupe` (combine author interim into the corpus)

**Files:**
- Modify: `src/collection/author_history.py`
- Test: `tests/test_author_history.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& $PY -m pytest tests/test_author_history.py::test_merge_and_dedupe_removes_cross_corpus_duplicates -v`
Expected: FAIL — `cannot import name 'merge_and_dedupe'`

- [ ] **Step 3: Add the merge helper to `src/collection/author_history.py`**

```python
# append to src/collection/author_history.py

def merge_and_dedupe(
    paths: list[str | Path], out_path: str | Path, text_col: str = "clean_text"
) -> pd.DataFrame:
    """Concat interim corpora and apply GLOBAL exact dedup (near-dedup already
    ran within each corpus). Writes the merged frame to `out_path`."""
    from src.preprocessing.dedupe import deduplicate

    frames = [read_parquet(p) for p in paths if Path(p).exists()]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = deduplicate(
        combined, text_col=text_col, near_dup_threshold=-1, show_progress=False
    )
    write_parquet(combined, out_path)
    log.info("author.merge.done", inputs=len(frames), rows=len(combined), out=str(out_path))
    return combined
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& $PY -m pytest tests/test_author_history.py::test_merge_and_dedupe_removes_cross_corpus_duplicates -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/collection/author_history.py tests/test_author_history.py
git commit -m "feat: merge_and_dedupe — fold author histories into the corpus"
```

---

## Task 6: `rebuild_groups_within_cohort` + CLI wiring

**Files:**
- Modify: `src/labeling/disclosure_dataset.py`
- Modify: `src/cli.py` (`build-disclosure-testset` gets a `--cohort-users-csv` option)
- Test: `tests/test_disclosure_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_disclosure_dataset.py
from src.labeling.disclosure_dataset import rebuild_groups_within_cohort


def test_rebuild_groups_within_cohort_promotes_disclosed_control():
    # Cohort = alice (was control, now discloses in enriched history),
    #          bob (still never discloses), carol (already a positive).
    df = pd.DataFrame({
        "author_hash": ["alice", "alice", "bob", "bob", "carol", "outsider"],
        "subreddit": ["Anxiety", "GAD", "cooking", "cooking", "depression", "Anxiety"],
        "disclosure_anxiety":      [1, 0, 0, 0, 0, 1],
        "disclosure_health_anxiety": [0, 0, 0, 0, 0, 0],
        "disclosure_depression":   [0, 0, 0, 0, 1, 0],
    })
    cohort = {"alice", "bob", "carol"}  # NB: 'outsider' must NOT appear
    users = rebuild_groups_within_cohort(df, cohort, targets=("anxiety", "depression"))
    by = users.set_index("author_hash")
    assert set(users["author_hash"]) == cohort           # no external recruitment
    assert int(by.loc["alice", "user_anxiety"]) == 1     # promoted
    assert by.loc["alice", "user_group"].startswith("disclosed_")
    assert int(by.loc["carol", "user_depression"]) == 1
    assert by.loc["bob", "user_group"] == "matched_control"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& $PY -m pytest tests/test_disclosure_dataset.py::test_rebuild_groups_within_cohort_promotes_disclosed_control -v`
Expected: FAIL — `cannot import name 'rebuild_groups_within_cohort'`

- [ ] **Step 3: Add the function to `src/labeling/disclosure_dataset.py`**

Add after `build_disclosure_test_users` (it reuses `find_disclosed_users` and `_user_subreddit_index` already defined in this module):

```python
def rebuild_groups_within_cohort(
    df: pd.DataFrame,
    cohort_hashes: Iterable[str],
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> pd.DataFrame:
    """Re-assign positive/control groups using ONLY the given cohort.

    Re-detects disclosures on the (now enriched) histories of the cohort and
    regroups them: any cohort user who discloses any target = positive; the rest
    = matched_control. No external users are recruited, so both classes keep the
    comparable history depth the enrichment gave them.
    """
    targets = list(targets)
    cohort = {str(h) for h in cohort_hashes} - {""}
    sub = df[df["author_hash"].astype(str).isin(cohort)].copy()

    disclosed_per_target = {t: (find_disclosed_users(sub, t) & cohort) for t in targets}
    all_disclosed: set[str] = set().union(*disclosed_per_target.values()) if targets else set()
    post_counts = sub.groupby("author_hash").size().to_dict()
    user_subs = _user_subreddit_index(sub)

    rows: list[dict] = []
    for u in sorted(cohort):
        if not u:
            continue
        row: dict = {"author_hash": u}
        for t in targets:
            row[f"user_{t}"] = int(u in disclosed_per_target[t])
        if u in all_disclosed:
            disc = [t for t in targets if u in disclosed_per_target[t]]
            row["user_group"] = "disclosed_" + "+".join(disc)
        else:
            row["user_group"] = "matched_control"
        row["n_posts"] = int(post_counts.get(u, 0))
        row["subreddits"] = ",".join(sorted(user_subs.get(u, set())))
        rows.append(row)

    expected_cols = (
        ["author_hash"] + [f"user_{t}" for t in targets] + ["user_group", "n_posts", "subreddits"]
    )
    out = pd.DataFrame(rows, columns=expected_cols) if rows else pd.DataFrame(columns=expected_cols)
    log.info(
        "disclosure_testset.regrouped",
        n_users=len(out),
        n_positives={t: int(out[f"user_{t}"].sum()) for t in targets} if not out.empty else {},
        n_controls=int((out["user_group"] == "matched_control").sum()) if not out.empty else 0,
    )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& $PY -m pytest tests/test_disclosure_dataset.py::test_rebuild_groups_within_cohort_promotes_disclosed_control -v`
Expected: PASS

- [ ] **Step 5: Wire it into the `build-disclosure-testset` CLI command in `src/cli.py`**

Add a new option to the `build_disclosure_testset` signature (after the `seed` option, ~line 425):

```python
    cohort_users_csv: str = typer.Option(
        None,
        help="If set, regroup ONLY these users (author_hash column) within their "
             "enriched histories instead of building a fresh corpus-wide test set.",
    ),
```

Then, in the function body, replace the `test_users = build_disclosure_test_users(...)` call (~lines 444-449) with this branch:

```python
    if cohort_users_csv:
        import pandas as pd

        from src.labeling.disclosure_dataset import rebuild_groups_within_cohort

        cohort_df = pd.read_csv(cohort_users_csv)
        cohort_hashes = set(cohort_df["author_hash"].astype(str))
        test_users = rebuild_groups_within_cohort(df, cohort_hashes)
    else:
        test_users = build_disclosure_test_users(
            df,
            controls_per_positive=controls_per_positive,
            min_posts_per_user=min_posts_per_user,
            seed=seed,
        )
```

(The rest of the command — `materialize_test_posts`, `mark_held_out`, the summary table, and writing `disclosure_testset.parquet` + `__users.csv` — is unchanged and works for both branches.)

- [ ] **Step 6: Verify the option registered + full suite still green**

Run: `& $PY -m src.cli build-disclosure-testset --help`
Expected: help text now lists `--cohort-users-csv`.
Run: `& $PY -m pytest tests/test_disclosure_dataset.py tests/test_author_history.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/labeling/disclosure_dataset.py src/cli.py tests/test_disclosure_dataset.py
git commit -m "feat: rebuild_groups_within_cohort + build-disclosure-testset --cohort-users-csv"
```

---

## Task 7: Operational enrichment run (LIVE — several hours)

> This task is **not** TDD — it runs the live scrape and re-runs the pipeline. Execute it only after Tasks 1–6 are merged and green. It is **resumable**: if `collect-authors` is interrupted, re-run the same command. Run from the repo root with `PY` defined as above.

- [ ] **Step 1: Back up the current corpus + test set (so we can compare / roll back)**

```powershell
Copy-Item data/processed/labeled.parquet data/processed/labeled.prebackup.parquet -Force
Copy-Item data/interim/_all.parquet data/interim/_all.prebackup.parquet -Force
Copy-Item data/processed/disclosure_testset__users.csv data/processed/disclosure_testset__users.prebackup.csv -Force
```

- [ ] **Step 2: Scrape full histories for the 3,943-user cohort (the multi-hour step; resumable)**

Run: `& $PY -m src.cli collect-authors`
Expected: progress bar over ~3,943 users; final line like `Author history collection complete: {'requested': 3943, 'written': N, 'skipped_existing': 0, 'empty': M, 'unrecoverable': 1}`. Per-user shards appear under `data/raw/authors/`.

- [ ] **Step 3: Preprocess the author shards into a separate interim dir**

Run: `& $PY -m src.cli preprocess --raw-dir data/raw/authors --out-dir data/interim/authors --no-ner --near-dup-threshold 5`
Expected: `data/interim/authors/_all.parquet` written. (`--no-ner` for speed; drop it to match the main corpus's NER pass if you ran NER there — slower.)

- [ ] **Step 4: Merge author interim into the combined corpus (overwrites `_all.parquet`)**

Run:
```powershell
& $PY -c "from src.collection.author_history import merge_and_dedupe; merge_and_dedupe(['data/interim/_all.prebackup.parquet','data/interim/authors/_all.parquet'],'data/interim/_all.parquet')"
```
Expected: log line `author.merge.done rows=<new total>` larger than the original 743,881.

- [ ] **Step 5: Re-label (weak + disclosure) on the enriched corpus**

```powershell
& $PY -m src.cli label --tier weak
& $PY -m src.cli label --tier disclosure
```
Expected: `data/processed/labeled.parquet` rewritten; disclosure auto-audit prints higher positive counts than before.

- [ ] **Step 6: Regroup within the cohort + rebuild the test set**

Run: `& $PY -m src.cli build-disclosure-testset --cohort-users-csv data/processed/disclosure_testset__users.prebackup.csv`
Expected: summary table where the disclosed groups have grown vs 1,323 positives / 2,620 controls (some controls promoted); `held_out_split` re-marked on `labeled.parquet`.

- [ ] **Step 7: Retrain the TF-IDF baseline (excludes held-out) and re-run the user-level eval**

```powershell
& $PY -m src.cli train configs/models/baseline.yaml
& $PY -m src.cli eval-disclosure experiments/runs/tfidf_logreg --target anxiety --aggregation max
& $PY -m src.cli eval-disclosure experiments/runs/tfidf_logreg --target health_anxiety --aggregation max
```
Expected: fresh `*__disclosure_userlevel.json` with/without-masking values reflecting the enriched histories.

- [ ] **Step 8: Sanity-check the new dataset values**

```powershell
& $PY -c "import pandas as pd; u=pd.read_csv('data/processed/disclosure_testset__users.csv'); print('users',len(u)); print(u['user_group'].value_counts().to_string()); print('median posts/user', u['n_posts'].median())"
```
Expected: median posts/user ≫ 4; positive counts up.

- [ ] **Step 9: Record results (no code commit needed — data is gitignored under authors/)**

Note the before/after user-level AUROC/F1 (with and without masking) for the project record. The `data/raw/authors/` shards are gitignored; `labeled.parquet` / `disclosure_testset.parquet` follow the repo's existing data-commit policy.

---

## Self-Review

**Spec coverage:**
- Both-classes scope → Task 7 scrapes all cohort users from `disclosure_testset__users.csv`. ✓
- Submissions + comments, uncapped (≤1000/listing) → Task 3 `collect_user` iterates both sections; `max_pages=10`. ✓
- Promote controls that disclosed → Task 6 `rebuild_groups_within_cohort` + Task 7 step 6. ✓
- Full send / resumable → Task 4 resume-by-existing-file + inherited HTTP cache; Task 7 re-runnable. ✓
- Approach A (no-auth JSON, standalone command) → Tasks 3–4. ✓
- Username recovery from raw → Task 2. ✓
- Regroup within cohort (no thin external controls) → Task 6 (cohort-restricted). ✓
- Merge + global exact dedup → Task 5. ✓
- Ethics/gitignore → Task 1 (with tradeoff note). Anonymization reused via existing preprocess (Task 7 step 3). ✓
- Error handling (404/resume/unrecoverable) → Task 3 (404), Task 4 (resume + unrecoverable). ✓
- Tests offline → Tasks 2–6 all mock HTTP / inject fakes. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type/name consistency:** `AuthorHistoryCollector`, `recover_author_usernames`, `run_author_collection(...collector=...)`, `merge_and_dedupe(paths, out_path)`, `rebuild_groups_within_cohort(df, cohort_hashes, targets)` are used identically across tasks and the operational run. `source="author_history"` consistent. CLI commands `collect-authors` / `build-disclosure-testset --cohort-users-csv` match Task 7 invocations. ✓
