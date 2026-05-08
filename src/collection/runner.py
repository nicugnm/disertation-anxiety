"""Run a collector and write the result to parquet under data/raw/."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.collection.base import BaseCollector
from src.collection.dump_collector import DumpCollector
from src.collection.json_scraper import JsonScraperCollector
from src.collection.praw_collector import PrawCollector
from src.collection.synthetic import SyntheticCollector
from src.utils.config import SubredditsConfig, data_dir
from src.utils.io import write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)

CollectorName = Literal["praw", "dump", "synthetic", "scraper"]


def make_collector(name: CollectorName, config: SubredditsConfig, **kwargs) -> BaseCollector:
    if name == "praw":
        return PrawCollector(config)
    if name == "dump":
        return DumpCollector(config, **kwargs)
    if name == "synthetic":
        return SyntheticCollector(config, **kwargs)
    if name == "scraper":
        return JsonScraperCollector(config, **kwargs)
    raise ValueError(f"Unknown collector: {name}")


def run_collection(
    backend: CollectorName,
    config: SubredditsConfig,
    out_dir: str | Path | None = None,
    **kwargs,
) -> Path:
    """Collect every configured subreddit and write one parquet per subreddit.

    Returns the directory containing the parquet shards.
    """
    out = Path(out_dir) if out_dir else data_dir("raw")
    out.mkdir(parents=True, exist_ok=True)

    collector = make_collector(backend, config, **kwargs)
    total = 0
    for sub in config.subreddits:
        rows = [p.to_dict() for p in collector.collect_subreddit(sub.name)]
        if not rows:
            log.warning("collection.empty", subreddit=sub.name)
            continue
        df = pd.DataFrame(rows)
        path = out / f"{sub.name}.parquet"
        write_parquet(df, path)
        total += len(df)
        log.info("collection.subreddit_done", subreddit=sub.name, n=len(df), path=str(path))

    log.info("collection.done", backend=backend, total=total, out=str(out))
    return out
