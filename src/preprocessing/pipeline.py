"""End-to-end preprocessing pipeline: raw parquet -> processed parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.anonymize import _hash_username, anonymize
from src.preprocessing.clean import clean_text
from src.preprocessing.dedupe import deduplicate
from src.utils.config import data_dir
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)


def _detect_lang(text: str) -> str:
    """Best-effort language detection. Returns 'en' on failure (don't drop)."""
    try:
        from langdetect import LangDetectException, detect

        return detect(text)
    except (ImportError, LangDetectException, Exception):  # noqa: BLE001
        return "en"


def preprocess_dataframe(
    df: pd.DataFrame,
    use_ner: bool = True,
    keep_only_english: bool = True,
    min_chars_after_clean: int = 50,
) -> pd.DataFrame:
    """Apply the full preprocessing pipeline to a raw collected DataFrame."""
    if df.empty:
        return df

    log.info("preprocess.start", n=len(df))

    # 1. Clean
    df = df.copy()
    df["clean_text"] = [
        clean_text(t, b) for t, b in zip(df["title"].fillna(""), df["body"].fillna(""))
    ]

    # 2. Drop deleted/removed/short
    drop_markers = {"[deleted]", "[removed]", "", None}
    mask_keep = ~df["body"].astype(str).str.strip().str.lower().isin(drop_markers)
    df = df[mask_keep]
    df = df[df["clean_text"].str.len() >= min_chars_after_clean]
    log.info("preprocess.after_drop_short", n=len(df))

    # 3. Language filter (optional, expensive on huge sets)
    if keep_only_english and len(df) > 0:
        df["lang"] = df["clean_text"].map(_detect_lang)
        df = df[df["lang"] == "en"]
        log.info("preprocess.after_lang_filter", n=len(df))

    # 4. Anonymize
    df["clean_text"] = df["clean_text"].map(lambda t: anonymize(t, use_ner=use_ner))
    df["author_hash"] = df["author"].map(_hash_username)
    df = df.drop(columns=["author"], errors="ignore")

    # 5. Deduplicate
    df = deduplicate(df)
    log.info("preprocess.after_dedupe", n=len(df))

    # 6. Tidy
    df = df.reset_index(drop=True)
    log.info("preprocess.done", n=len(df))
    return df


def run_preprocessing(
    raw_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    use_ner: bool = True,
) -> Path:
    raw = Path(raw_dir) if raw_dir else data_dir("raw")
    out = Path(out_dir) if out_dir else data_dir("interim")
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(raw.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw parquet files in {raw}")

    all_dfs = []
    for fp in files:
        log.info("preprocess.file.start", file=str(fp))
        df = read_parquet(fp)
        df = preprocess_dataframe(df, use_ner=use_ner)
        if df.empty:
            continue
        write_parquet(df, out / fp.name)
        all_dfs.append(df)
        log.info("preprocess.file.done", file=str(fp), n=len(df))

    combined_path = out / "_all.parquet"
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        write_parquet(combined, combined_path)
        log.info("preprocess.combined", path=str(combined_path), n=len(combined))
    return out
