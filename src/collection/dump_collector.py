"""Offline collection from Pushshift / arctic_shift `.zst` JSONL dumps.

These dumps are the only practical way to get years of historical data
post-2023 API restrictions. Files are large (TBs across the full archive);
this collector streams and filters as it reads.

Expected file naming: `RS_YYYY-MM.zst` (submissions) — Pushshift convention.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

from src.collection.base import BaseCollector, RedditPost
from src.utils.io import iter_zst_jsonl
from src.utils.logging import get_logger

log = get_logger(__name__)


class DumpCollector(BaseCollector):
    """Stream posts from Pushshift-style .zst dump files.

    Looks for files in `dump_dir`. Reddit submission dumps live under
    `data/external/dumps/RS_*.zst` by convention.
    """

    def __init__(self, config, dump_dir: str | Path = "data/external/dumps") -> None:  # noqa: ANN001
        super().__init__(config)
        self.dump_dir = Path(dump_dir)
        if not self.dump_dir.exists():
            log.warning("dump.dir_missing", path=str(self.dump_dir))

    def _files(self) -> list[Path]:
        if not self.dump_dir.exists():
            return []
        return sorted(self.dump_dir.glob("RS_*.zst"))

    def collect_subreddit(self, name: str) -> Iterator[RedditPost]:
        files = self._files()
        if not files:
            log.warning("dump.no_files", dir=str(self.dump_dir))
            return

        target = name.lower()
        log.info("dump.collect.start", subreddit=name, n_files=len(files))
        n_yielded = 0
        for fp in files:
            for record in iter_zst_jsonl(fp):
                if str(record.get("subreddit", "")).lower() != target:
                    continue
                post = self._record_to_post(record)
                if post and self.passes_filters(post):
                    n_yielded += 1
                    yield post
        log.info("dump.collect.done", subreddit=name, yielded=n_yielded)

    @staticmethod
    def _record_to_post(record: dict) -> RedditPost | None:
        try:
            return RedditPost(
                id=str(record.get("id", "")),
                subreddit=str(record.get("subreddit", "")),
                created_utc=float(record.get("created_utc", 0.0)),
                title=str(record.get("title", "") or ""),
                body=str(record.get("selftext", "") or ""),
                author=record.get("author"),
                score=int(record.get("score", 0) or 0),
                num_comments=int(record.get("num_comments", 0) or 0),
                permalink=str(record.get("permalink", "") or ""),
                is_self=bool(record.get("is_self", False)),
                over_18=bool(record.get("over_18", False)),
                source="dump",
                collected_at=time.time(),
            )
        except (ValueError, TypeError):
            return None
