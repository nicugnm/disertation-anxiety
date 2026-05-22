"""Duplicate / near-duplicate removal.

Reddit cross-posting and bot reposts mean the same text can appear under
multiple post IDs. We use:
  1. Exact dedup on normalized text (cheap, always-on).
  2. SimHash-based near-dedup on top of that, with two scaling tricks:
     - LSH bucket partition so we only compare posts likely to be near-dup
       (cuts the O(n²) cost to roughly O(n·k) where k is the bucket size).
     - Auto-skip near-dup for very large subreddit groups (>`max_group_for_full_neardedup`)
       since at >50k posts per subreddit the LSH worst case is still slow and
       exact dedup catches the bulk of duplicates anyway.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

RE_NORM = re.compile(r"\s+")


def _norm_for_hash(text: str) -> str:
    return RE_NORM.sub(" ", (text or "").lower()).strip()


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _simhash(text: str, n_bits: int = 64) -> int:
    """SimHash on whitespace tokens, equal weights.

    Sufficient for catching near-duplicate Reddit posts; a production system
    would use shingles + IDF weighting.
    """
    v = [0] * n_bits
    tokens = text.split()
    if not tokens:
        return 0
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for i in range(n_bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(n_bits):
        if v[i] > 0:
            out |= 1 << i
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _lsh_buckets(simhash: int, n_bands: int = 8) -> tuple[int, ...]:
    """Split a 64-bit simhash into `n_bands` equal segments.

    Two posts are near-dup candidates iff they share at least one band — a
    standard SimHash-LSH trick. With 64 bits / 8 bands = 8 bits per band,
    each band buckets posts into one of 256 keys. Posts that don't share any
    of the 8 band keys can't be near-duplicates (within reasonable Hamming
    radius) and we never compute the pairwise Hamming distance for them.
    """
    bits_per_band = 64 // n_bands
    mask = (1 << bits_per_band) - 1
    return tuple((simhash >> (i * bits_per_band)) & mask for i in range(n_bands))


def _dedup_near_in_group(
    group: pd.DataFrame,
    text_col: str,
    threshold: int,
    n_bands: int,
) -> pd.Index:
    """Return indices to drop in this group via LSH-bucketed near-dedup."""
    if len(group) <= 1:
        return pd.Index([])

    sigs: dict[int, int] = {}      # row_idx -> simhash
    bucket_map: dict[tuple[int, int], list[int]] = {}  # (band_idx, band_value) -> row_idxs
    drop: list[int] = []

    for idx, text in group[text_col].items():
        sh = _simhash(text)
        bands = _lsh_buckets(sh, n_bands=n_bands)

        # Look at every bucket this row would hash into; compare only with
        # already-kept rows that share at least one band.
        candidates: set[int] = set()
        for band_idx, band_val in enumerate(bands):
            candidates.update(bucket_map.get((band_idx, band_val), []))

        is_dup = False
        for cand_idx in candidates:
            if _hamming(sh, sigs[cand_idx]) <= threshold:
                is_dup = True
                break

        if is_dup:
            drop.append(idx)
        else:
            sigs[idx] = sh
            for band_idx, band_val in enumerate(bands):
                bucket_map.setdefault((band_idx, band_val), []).append(idx)

    return pd.Index(drop)


def deduplicate(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    near_dup_threshold: int = 5,
    n_lsh_bands: int = 8,
    max_group_for_full_neardedup: int = 100_000,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Drop exact and near-duplicate posts. Keeps the first occurrence per group.

    Pass `near_dup_threshold=-1` to disable near-dedup entirely (exact-only,
    very fast on huge corpora).
    """
    if df.empty:
        return df

    work = df.copy()
    work["_norm"] = work[text_col].astype(str).map(_norm_for_hash)
    work["_md5"] = work["_norm"].map(_md5)

    n_before = len(work)
    # 1) Exact dedup — global, fast.
    work = work.drop_duplicates(subset="_md5", keep="first").reset_index(drop=True)
    log.info("dedupe.exact_done", before=n_before, after=len(work))

    # 2) Optional near-dedup within each subreddit.
    if near_dup_threshold is not None and near_dup_threshold >= 0 and "subreddit" in work.columns:
        keep_mask = pd.Series(True, index=work.index)
        groups = list(work.groupby("subreddit"))

        if show_progress:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]Near-dedup ({task.fields[group]})"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            )
            with progress:
                task = progress.add_task("...", total=len(groups), group="")
                for sub, group in groups:
                    progress.update(task, group=str(sub))
                    if len(group) > max_group_for_full_neardedup:
                        log.warning(
                            "dedupe.near_skipped_large_group",
                            subreddit=sub,
                            n=len(group),
                            cap=max_group_for_full_neardedup,
                        )
                        progress.advance(task)
                        continue
                    drop_idx = _dedup_near_in_group(
                        group, text_col="_norm",
                        threshold=near_dup_threshold,
                        n_bands=n_lsh_bands,
                    )
                    if len(drop_idx):
                        keep_mask.loc[drop_idx] = False
                    progress.advance(task)
        else:
            for sub, group in groups:
                if len(group) > max_group_for_full_neardedup:
                    continue
                drop_idx = _dedup_near_in_group(
                    group, text_col="_norm",
                    threshold=near_dup_threshold,
                    n_bands=n_lsh_bands,
                )
                if len(drop_idx):
                    keep_mask.loc[drop_idx] = False

        work = work.loc[keep_mask]

    out = work.drop(columns=["_norm", "_md5"]).reset_index(drop=True)
    log.info("dedupe.done", before=n_before, after=len(out))
    return out
