"""Tests for the eRisk loaders. Uses synthetic mini-fixtures matching the
documented formats — no real eRisk data required."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.collection.erisk_loader import (
    load_task1,
    load_task2,
    reconstruct_threads,
    to_canonical,
)


# --------------------------------------------------------------------------- #
# Task 1 — TREC XML
# --------------------------------------------------------------------------- #


def test_task1_parses_canonical_format(tmp_path: Path):
    sample = """<DOC>
  <DOCNO> S001 </DOCNO>
    <PRE> Yesterday I felt fine. </PRE>
    <TEXT> Today I'm having a panic attack. </TEXT>
    <POST> I should call my therapist. </POST>
</DOC>
<DOC>
  <DOCNO> S002 </DOCNO>
    <PRE>  </PRE>
    <TEXT> What if I have cancer? </TEXT>
    <POST> Googled my symptoms again. </POST>
</DOC>
"""
    f = tmp_path / "task1_pilot.trec"
    f.write_text(sample, encoding="utf-8")
    df = load_task1(f, show_progress=False)
    assert len(df) == 2
    assert df.iloc[0]["sentence_id"] == "S001"
    assert "panic attack" in df.iloc[0]["text"]
    assert df.iloc[1]["text"].startswith("What if I have cancer")


def test_task1_handles_missing_file(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_task1(tmp_path / "does_not_exist.trec", show_progress=False)


def test_task1_tolerates_extra_whitespace(tmp_path: Path):
    sample = "<DOC><DOCNO>X</DOCNO><PRE></PRE><TEXT>hi</TEXT><POST></POST></DOC>"
    f = tmp_path / "compact.trec"
    f.write_text(sample, encoding="utf-8")
    df = load_task1(f, show_progress=False)
    assert len(df) == 1
    assert df.iloc[0]["text"] == "hi"


# --------------------------------------------------------------------------- #
# Task 2 — JSON
# --------------------------------------------------------------------------- #


def _task2_fixture():
    return [
        {
            "submissionId": "sub_001",
            "author": "subject_AAA",
            "date": "2024-01-15T10:00:00.000+00:00",
            "title": "Asking about anxiety",
            "body": "I've been struggling.",
            "number": 1,
            "targetSubject": "subject_AAA",
            "comments": [
                {
                    "commentId": "c_001",
                    "author": "subject_BBB",
                    "date": "2024-01-15T10:30:00.000+00:00",
                    "body": "Same here.",
                    "parent": "sub_001",
                },
                {
                    "commentId": "c_002",
                    "author": "subject_CCC",
                    "date": "2024-01-15T11:00:00.000+00:00",
                    "body": "Reply to first comment.",
                    "parent": "c_001",
                },
            ],
        },
        {
            "submissionId": "sub_002",
            "author": "subject_DDD",
            "date": "2024-02-01T08:00:00.000+00:00",
            "title": "Health worry",
            "body": "Chest pain again.",
            "number": 2,
            "targetSubject": "subject_DDD",
            "comments": [],
        },
    ]


def test_task2_flattens_submissions_and_comments(tmp_path: Path):
    f = tmp_path / "task2.json"
    f.write_text(json.dumps(_task2_fixture()), encoding="utf-8")
    df = load_task2(f, show_progress=False)
    # 2 submissions + 2 comments = 4 rows
    assert len(df) == 4
    assert (df["kind"] == "submission").sum() == 2
    assert (df["kind"] == "comment").sum() == 2

    # Parent linkage preserved
    c1 = df[df["item_id"] == "c_001"].iloc[0]
    assert c1["parent_id"] == "sub_001"
    c2 = df[df["item_id"] == "c_002"].iloc[0]
    assert c2["parent_id"] == "c_001"

    # Submission has no parent
    assert df[df["kind"] == "submission"]["parent_id"].isna().all()


def test_task2_thread_reconstruction(tmp_path: Path):
    f = tmp_path / "task2.json"
    f.write_text(json.dumps(_task2_fixture()), encoding="utf-8")
    df = load_task2(f, show_progress=False)
    threads = reconstruct_threads(df)

    # Two threads, by submissionId
    assert set(threads.keys()) == {"sub_001", "sub_002"}
    # Thread sub_001 is DFS-ordered: submission → comment_1 → comment_2 (nested reply)
    order = [item["item_id"] for item in threads["sub_001"]]
    assert order == ["sub_001", "c_001", "c_002"]
    assert [item["item_id"] for item in threads["sub_002"]] == ["sub_002"]


def test_task2_to_canonical_keeps_only_submissions_by_default(tmp_path: Path):
    f = tmp_path / "task2.json"
    f.write_text(json.dumps(_task2_fixture()), encoding="utf-8")
    df = load_task2(f, show_progress=False)
    canon = to_canonical(df)
    # Only the 2 submissions are kept
    assert len(canon) == 2
    expected_cols = {
        "id", "subreddit", "created_utc", "title", "body", "author",
        "score", "num_comments", "permalink", "is_self", "over_18",
        "source", "collected_at",
    }
    assert expected_cols.issubset(set(canon.columns))
    assert (canon["source"] == "erisk").all()
    assert (canon["subreddit"] == "erisk").all()
    # ISO date converted to unix seconds (non-zero for valid date)
    assert canon["created_utc"].iloc[0] > 0


def test_task2_to_canonical_can_include_comments(tmp_path: Path):
    f = tmp_path / "task2.json"
    f.write_text(json.dumps(_task2_fixture()), encoding="utf-8")
    df = load_task2(f, show_progress=False)
    canon_all = to_canonical(df, kind_filter=("submission", "comment"))
    assert len(canon_all) == 4


def test_task2_empty_file(tmp_path: Path):
    f = tmp_path / "empty.json"
    f.write_text("[]", encoding="utf-8")
    df = load_task2(f, show_progress=False)
    assert df.empty
    # Empty DataFrames still have the expected columns
    assert "item_id" in df.columns
    assert "parent_id" in df.columns


def test_task2_rejects_non_list_root(tmp_path: Path):
    import pytest

    f = tmp_path / "wrong.json"
    f.write_text('{"oops": "not a list"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_task2(f, show_progress=False)
