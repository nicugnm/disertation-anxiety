"""Loaders for the CLEF eRisk 2025 collection.

Two formats:

  Task 1 (TREC XML, sentence-level):
    <DOC>
      <DOCNO> SENTENCE_ID </DOCNO>
      <PRE> previous sentence </PRE>
      <TEXT> sentence </TEXT>
      <POST> next sentence </POST>
    </DOC>

  Task 2 (JSON, submission + nested comments):
    [
      {
        "submissionId": "...",
        "author": "subject_...",
        "date": "ISO-8601",
        "body": "...",
        "title": "...",
        "number": 3,
        "targetSubject": "subject_...",
        "comments": [
          {"commentId": "...", "author": "...", "date": "...", "body": "...", "parent": "..."}
        ]
      }
    ]

We expose two normalized DataFrames:
  - `load_task1(path) -> DataFrame[sentence_id, pre, text, post, target_subject?]`
  - `load_task2(path) -> DataFrame[item_id, parent_id, kind, author, date, title, body, ...]`

The Task-2 loader **flattens the submission + comment hierarchy** into one row
per item, with a `kind` column ("submission" | "comment") and `parent_id` so
thread structure can be reconstructed for conversational models.

Both loaders deliberately keep eRisk-specific fields (e.g. `target_subject`,
`number`) rather than collapsing them into our canonical RedditPost schema,
because the eRisk evaluation procedure (Task 1's per-sentence relevance,
Task 2's per-conversation triage) depends on them.

References:
  - Parapar, J., Perez, A., Wang, X., & Crestani, F. (2025). eRisk 2025:
    Contextual and Conversational Approaches for Depression Challenges.
    European Conference on Information Retrieval.
  - Crestani, F., Losada, D., & Parapar, J. (2022). Early Detection of Mental
    Health Disorders by Social Media Monitoring.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Task 1 — TREC XML, sentence-level
# --------------------------------------------------------------------------- #

# eRisk's TREC files use entity-decoded text inside the tags. We parse with
# regex rather than a full XML parser because (a) the official files are not
# strictly XML-valid (no root element wrapping the DOCs), and (b) regex is
# tolerant of stray characters / partial files.

_DOC_PATTERN = re.compile(
    r"<DOC>\s*"
    r"<DOCNO>\s*(?P<docno>.*?)\s*</DOCNO>\s*"
    r"<PRE>\s*(?P<pre>.*?)\s*</PRE>\s*"
    r"<TEXT>\s*(?P<text>.*?)\s*</TEXT>\s*"
    r"<POST>\s*(?P<post>.*?)\s*</POST>\s*"
    r"</DOC>",
    re.DOTALL | re.IGNORECASE,
)


def iter_task1(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield one dict per <DOC> from a Task-1 TREC file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    content = p.read_text(encoding="utf-8", errors="replace")
    n = 0
    for m in _DOC_PATTERN.finditer(content):
        n += 1
        yield {
            "sentence_id": m.group("docno").strip(),
            "pre": m.group("pre").strip(),
            "text": m.group("text").strip(),
            "post": m.group("post").strip(),
        }
    log.info("erisk.task1.parsed", path=str(p), n_docs=n)


def load_task1(path: str | Path, show_progress: bool = True) -> pd.DataFrame:
    """Load a Task-1 TREC file into a DataFrame.

    Columns: sentence_id, pre, text, post.
    """
    p = Path(path)
    if show_progress:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        rows: list[dict[str, str]] = []
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]Parsing eRisk Task-1 ({task.fields[file]})"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        )
        with progress:
            # We don't know the doc count until we iterate; use indeterminate total.
            task = progress.add_task("...", total=None, file=p.name)
            for r in iter_task1(p):
                rows.append(r)
                progress.advance(task)
    else:
        rows = list(iter_task1(p))

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["sentence_id", "pre", "text", "post"])
    df["source_file"] = p.name
    return df


# --------------------------------------------------------------------------- #
# Task 2 — JSON submissions + comments
# --------------------------------------------------------------------------- #


def _flatten_submission(sub: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one row for the submission plus one row per comment."""
    rows: list[dict[str, Any]] = []
    rows.append({
        "item_id": str(sub.get("submissionId", "")),
        "kind": "submission",
        "parent_id": None,
        "author": str(sub.get("author", "") or ""),
        "date": str(sub.get("date", "") or ""),
        "title": str(sub.get("title", "") or ""),
        "body": str(sub.get("body", "") or ""),
        "number": int(sub.get("number", 0) or 0),
        "target_subject": str(sub.get("targetSubject", "") or ""),
        "submission_id": str(sub.get("submissionId", "")),
    })
    for c in sub.get("comments", []) or []:
        rows.append({
            "item_id": str(c.get("commentId", "")),
            "kind": "comment",
            "parent_id": str(c.get("parent", "") or "") or None,
            "author": str(c.get("author", "") or ""),
            "date": str(c.get("date", "") or ""),
            "title": "",
            "body": str(c.get("body", "") or ""),
            "number": int(sub.get("number", 0) or 0),
            "target_subject": str(sub.get("targetSubject", "") or ""),
            "submission_id": str(sub.get("submissionId", "")),
        })
    return rows


def iter_task2(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield flattened items (submissions + comments) from a Task-2 JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {p}, got {type(data).__name__}")
    for sub in data:
        yield from _flatten_submission(sub)


def load_task2(path: str | Path, show_progress: bool = True) -> pd.DataFrame:
    """Load a Task-2 JSON file into a flattened DataFrame.

    One row per item (submission OR comment). Use `parent_id` + `submission_id`
    to reconstruct the conversation tree.

    Columns:
      item_id, kind, parent_id, author, date, title, body, number,
      target_subject, submission_id, source_file
    """
    p = Path(path)
    if not show_progress:
        rows = list(iter_task2(p))
    else:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        # Pre-load to know the count, then bar over the flatten step.
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected top-level list in {p}")
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]Loading eRisk Task-2 ({task.fields[file]})"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        )
        rows = []
        with progress:
            task = progress.add_task("...", total=len(data), file=p.name)
            for sub in data:
                rows.extend(_flatten_submission(sub))
                progress.advance(task)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "item_id", "kind", "parent_id", "author", "date",
            "title", "body", "number", "target_subject", "submission_id",
        ])
    df["source_file"] = p.name
    return df


# --------------------------------------------------------------------------- #
# Directory loaders + thread reconstruction
# --------------------------------------------------------------------------- #


def load_task1_dir(directory: str | Path, pattern: str = "*.trec") -> pd.DataFrame:
    """Concat-load every Task-1 file matching `pattern` under `directory`."""
    d = Path(directory)
    files = sorted(d.glob(pattern)) + sorted(d.glob("*.xml")) + sorted(d.glob("*.txt"))
    files = list(dict.fromkeys(files))  # dedupe preserving order
    if not files:
        raise FileNotFoundError(f"No Task-1 files matched in {d}")
    parts = [load_task1(f, show_progress=False) for f in files]
    df = pd.concat(parts, ignore_index=True)
    log.info("erisk.task1.dir_loaded", n_files=len(files), n_rows=len(df))
    return df


def load_task2_dir(directory: str | Path, pattern: str = "*.json") -> pd.DataFrame:
    """Concat-load every Task-2 JSON file under `directory`."""
    d = Path(directory)
    files = sorted(d.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Task-2 files matched in {d}")
    parts = [load_task2(f, show_progress=False) for f in files]
    df = pd.concat(parts, ignore_index=True)
    log.info("erisk.task2.dir_loaded", n_files=len(files), n_rows=len(df))
    return df


def reconstruct_threads(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Return {submission_id: [items in DFS order]} for a Task-2 DataFrame.

    Useful for conversational models that need ordered thread context.
    """
    threads: dict[str, list[dict[str, Any]]] = {}
    by_sub = df.groupby("submission_id", sort=False)
    for sub_id, grp in by_sub:
        records = grp.to_dict("records")
        parent_to_children: dict[str | None, list[dict[str, Any]]] = {}
        for r in records:
            parent_to_children.setdefault(r.get("parent_id"), []).append(r)
        ordered: list[dict[str, Any]] = []

        # Submission roots have parent_id == None.
        def _visit(item):
            ordered.append(item)
            for child in parent_to_children.get(item["item_id"], []):
                _visit(child)

        for root in parent_to_children.get(None, []):
            _visit(root)
        threads[str(sub_id)] = ordered
    return threads


# --------------------------------------------------------------------------- #
# Canonical-schema bridge (so existing pipeline stages can consume eRisk data)
# --------------------------------------------------------------------------- #


def to_canonical(df: pd.DataFrame, kind_filter: Iterable[str] = ("submission",)) -> pd.DataFrame:
    """Map an eRisk DataFrame to the same column set our preprocess pipeline expects.

    By default keeps only submissions (drop comments) so it's drop-in compatible
    with the rest of the pipeline. Pass `kind_filter=('submission', 'comment')`
    to keep both — but then your preprocessing must handle items without titles.
    """
    sub = df[df["kind"].isin(list(kind_filter))].copy()
    out = pd.DataFrame({
        "id": sub["item_id"].astype(str),
        "subreddit": "erisk",  # placeholder — eRisk subreddit not exposed
        "created_utc": _iso_to_unix(sub["date"]),
        "title": sub.get("title", ""),
        "body": sub["body"].astype(str),
        "author": sub["author"].astype(str),
        "score": 0,
        "num_comments": 0,
        "permalink": "",
        "is_self": True,
        "over_18": False,
        "source": "erisk",
        "collected_at": 0.0,
    })
    return out


def _iso_to_unix(s: pd.Series) -> pd.Series:
    """Convert ISO-8601 timestamp strings to unix seconds (float). NaT → 0."""
    out = pd.to_datetime(s, utc=True, errors="coerce")
    return (out.astype("int64") // 1_000_000_000).astype(float).fillna(0.0)
