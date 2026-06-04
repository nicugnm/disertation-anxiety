"""Tests for external-corpus loaders (RMHD CSV layout, ANGST absence handling)."""
from __future__ import annotations

import pandas as pd

from src.evaluation.external import load_angst, load_rmhd


def test_load_rmhd_labels_and_cleaning(tmp_path):
    long = "I have been extremely anxious and worried about my health for months now honestly"
    pd.DataFrame({"subreddit": ["anxiety"] * 3, "author": ["a", "b", "c"], "post": [long, long, "x"]}).to_csv(
        tmp_path / "anxiety_2018.csv", index=False)
    pd.DataFrame({"subreddit": ["fitness"] * 2, "author": ["d", "e"], "post": [long, long]}).to_csv(
        tmp_path / "fitness_2018.csv", index=False)

    df = load_rmhd(tmp_path, pos_subs=["anxiety"], neg_subs=["fitness"], cap_per_sub=0)
    assert set(df.columns) == {"subreddit", "author", "clean_text", "y"}
    assert (df[df.subreddit == "anxiety"]["y"] == 1).all()
    assert (df[df.subreddit == "fitness"]["y"] == 0).all()
    assert (df["clean_text"].str.len() >= 30).all()      # short "x" post dropped
    assert len(df) == 4


def test_load_rmhd_missing_files_returns_empty(tmp_path):
    df = load_rmhd(tmp_path, pos_subs=["anxiety"], neg_subs=["fitness"])
    assert df.empty


def test_load_angst_absent_returns_none(tmp_path):
    assert load_angst(tmp_path) is None   # gated data not present -> None, no crash
