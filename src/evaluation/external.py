"""Loaders for external-validation corpora (cross-corpus zero-shot evaluation).

- RMHD (Low et al. 2020, Zenodo 3941387): per-subreddit CSVs with a raw `post`
  column. Public (PDDL). Subreddit-as-label — anxiety-related subs vs controls.
- ANGST (Hengle et al. 2024, HF `ameyhengle/ANGST`): 3-expert-psychologist labels.
  GATED — requires requesting access on HF + `huggingface-cli login`; this loader
  works once the CSVs are downloaded to the given directory.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.clean import clean_text


def load_rmhd(rmhd_dir, pos_subs, neg_subs, year: str = "2018",
              cap_per_sub: int = 5000, seed: int = 42, min_len: int = 30) -> pd.DataFrame:
    """Load RMHD subreddit CSVs into [subreddit, author, clean_text, y] (y=1 for
    anxiety-related subs). Applies our clean_text to match training preprocessing."""
    rows = []
    for sub, y in [(s, 1) for s in pos_subs] + [(s, 0) for s in neg_subs]:
        p = Path(rmhd_dir) / f"{sub}_{year}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, usecols=lambda c: c in ("subreddit", "author", "post"))
        d = d[d["post"].notna()]
        if cap_per_sub and len(d) > cap_per_sub:
            d = d.sample(cap_per_sub, random_state=seed)
        d["clean_text"] = [clean_text("", str(t)) for t in d["post"]]
        d = d[d["clean_text"].str.len() >= min_len].copy()
        d["y"] = y
        rows.append(d[["subreddit", "author", "clean_text", "y"]])
    if not rows:
        return pd.DataFrame(columns=["subreddit", "author", "clean_text", "y"])
    return pd.concat(rows, ignore_index=True)


def load_angst(angst_dir, target: str = "anxiety") -> pd.DataFrame | None:
    """Load ANGST gold test set into [clean_text, y]. Uses the expert binary
    `<target>_label` column directly (anxiety_label=1 covers Anxiety + Comorbid).
    Falls back to the multiclass label if the per-target column is absent. Returns
    None if the (gated) data is not present."""
    p = Path(angst_dir) / "test.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    text_col = next((c for c in d.columns if c.lower() in ("text", "post", "body", "selftext", "sentence")), None)
    if text_col is None:
        raise ValueError(f"Could not find a text column in ANGST test.csv; got {list(d.columns)}")
    label_col = f"{target}_label"
    if label_col in d.columns:
        y = d[label_col].astype(int)
    else:
        mc = next((c for c in d.columns if "multiclass" in c.lower() or c.lower() in ("label", "class")), None)
        if mc is None:
            raise ValueError(f"No '{label_col}' or multiclass column in ANGST test.csv; got {list(d.columns)}")
        y = d[mc].astype(str).str.lower().apply(lambda v: int(target in v or "comorbid" in v))
    out = pd.DataFrame({"clean_text": d[text_col].astype(str).map(lambda t: clean_text("", t)), "y": y.values})
    return out[out["clean_text"].str.len() >= 10].reset_index(drop=True)
