"""End-to-end preprocessing pipeline: raw parquet -> processed parquet.

Sized for corpora from 10k to 1M+ posts. Heavy steps (NER, language detection,
near-dedup) are batched/parallelized and surface progress bars so a long run
on a large corpus is observable, not a black box.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.anonymize import _hash_username, anonymize_batch, regex_redact
from src.preprocessing.clean import clean_text
from src.preprocessing.dedupe import deduplicate
from src.utils.config import data_dir
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)


def _detect_lang_safe(text: str) -> str:
    """Best-effort language detection. Returns 'en' on failure (don't drop)."""
    try:
        from langdetect import LangDetectException, detect

        return detect(text)
    except (ImportError, LangDetectException, Exception):  # noqa: BLE001
        return "en"


def _make_progress():
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


def preprocess_dataframe(
    df: pd.DataFrame,
    use_ner: bool = True,
    keep_only_english: bool = True,
    min_chars_after_clean: int = 50,
    n_process_ner: int = 1,
    near_dup_threshold: int = 5,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Apply the full preprocessing pipeline to a raw collected DataFrame.

    - `use_ner`: enable spaCy NER for PII redaction. Off → regex-only PII pass.
    - `n_process_ner`: spaCy worker processes. >1 ≈ linear speedup on multi-core
      machines but uses more RAM.
    - `keep_only_english`: drop non-English posts via langdetect.
    - `near_dup_threshold`: SimHash Hamming radius; set -1 to skip near-dedup
      (exact-MD5 dedup is still applied).
    """
    if df.empty:
        return df

    log.info("preprocess.start", n=len(df))

    # 1. Clean (cheap, vectorizable per row)
    df = df.copy()
    if show_progress:
        with _make_progress() as bar:
            task = bar.add_task("Cleaning text", total=len(df))
            cleaned: list[str] = []
            for t, b in zip(df["title"].fillna(""), df["body"].fillna("")):
                cleaned.append(clean_text(t, b))
                bar.advance(task)
            df["clean_text"] = cleaned
    else:
        df["clean_text"] = [clean_text(t, b) for t, b in zip(df["title"].fillna(""), df["body"].fillna(""))]

    # 2. Drop deleted/removed/short
    drop_markers = {"[deleted]", "[removed]", "", None}
    mask_keep = ~df["body"].astype(str).str.strip().str.lower().isin(drop_markers)
    df = df[mask_keep]
    df = df[df["clean_text"].str.len() >= min_chars_after_clean]
    log.info("preprocess.after_drop_short", n=len(df))

    # 3. Language filter (sample-based fast path for huge corpora)
    if keep_only_english and len(df) > 0:
        if show_progress:
            with _make_progress() as bar:
                task = bar.add_task("Language detection", total=len(df))
                langs: list[str] = []
                for t in df["clean_text"]:
                    langs.append(_detect_lang_safe(t))
                    bar.advance(task)
                df = df.assign(lang=langs)
        else:
            df["lang"] = df["clean_text"].map(_detect_lang_safe)
        df = df[df["lang"] == "en"]
        log.info("preprocess.after_lang_filter", n=len(df))

    # 4. Anonymize — batched
    if show_progress:
        with _make_progress() as bar:
            task = bar.add_task(
                f"Anonymizing (NER={'on' if use_ner else 'off'}, n_proc={n_process_ner})",
                total=len(df),
            )
            # Batch in chunks so the progress bar advances visibly.
            chunk_size = 2000
            texts = df["clean_text"].astype(str).tolist()
            redacted: list[str] = []
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i : i + chunk_size]
                redacted.extend(anonymize_batch(chunk, use_ner=use_ner, n_process=n_process_ner))
                bar.advance(task, advance=len(chunk))
            df = df.assign(clean_text=redacted)
    else:
        df = df.assign(
            clean_text=anonymize_batch(
                df["clean_text"].astype(str).tolist(),
                use_ner=use_ner,
                n_process=n_process_ner,
            )
        )

    df["author_hash"] = df["author"].map(_hash_username) if "author" in df.columns else None
    df = df.drop(columns=["author"], errors="ignore")

    # 5. Deduplicate (exact + LSH near-dedup)
    df = deduplicate(df, near_dup_threshold=near_dup_threshold, show_progress=show_progress)
    log.info("preprocess.after_dedupe", n=len(df))

    df = df.reset_index(drop=True)
    log.info("preprocess.done", n=len(df))
    return df


def run_preprocessing(
    raw_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    use_ner: bool = True,
    n_process_ner: int = 1,
    near_dup_threshold: int = 5,
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
        df = preprocess_dataframe(
            df,
            use_ner=use_ner,
            n_process_ner=n_process_ner,
            near_dup_threshold=near_dup_threshold,
        )
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
