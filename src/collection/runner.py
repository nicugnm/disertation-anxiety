"""Run a collector and write the result to parquet under data/raw/."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.collection.base import BaseCollector
from src.collection.dump_collector import DumpCollector
from src.collection.json_scraper import JsonScraperCollector
from src.collection.praw_collector import PrawCollector
from src.collection.search_scraper import SearchScraperCollector, _slug
from src.collection.synthetic import SyntheticCollector
from src.utils.config import SubredditsConfig, data_dir
from src.utils.io import write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)
_console = Console()

CollectorName = Literal["praw", "dump", "synthetic", "scraper", "search"]


def make_collector(name: CollectorName, config: SubredditsConfig, **kwargs) -> BaseCollector:
    if name == "praw":
        return PrawCollector(config)
    if name == "dump":
        return DumpCollector(config, **kwargs)
    if name == "synthetic":
        return SyntheticCollector(config, **kwargs)
    if name == "scraper":
        return JsonScraperCollector(config, **kwargs)
    if name == "search":
        return SearchScraperCollector(config, **kwargs)
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

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[posts]:,} posts"),
        console=_console,
    )

    # --- Search backend has a different iteration unit (queries, not subs).
    if backend == "search":
        search_collector: SearchScraperCollector = collector  # type: ignore[assignment]
        queries = search_collector.queries
        with progress:
            task = progress.add_task(
                "Collecting (search)", total=len(queries), posts=0
            )
            for q in queries:
                progress.update(task, description=f"search: {q!r}")
                rows = [p.to_dict() for p in search_collector.collect_subreddit(q)]
                if rows:
                    df = pd.DataFrame(rows)
                    path = out / f"search__{_slug(q)}.parquet"
                    write_parquet(df, path)
                    total += len(df)
                    log.info("search.query_done", query=q, n=len(df), path=str(path))
                else:
                    log.warning("search.query_empty", query=q)
                progress.update(task, advance=1, posts=total)
        log.info("collection.done", backend=backend, total=total, out=str(out))
        return out

    # --- All other backends iterate the configured subreddit list.
    with progress:
        task = progress.add_task(
            f"Collecting ({backend})", total=len(config.subreddits), posts=0
        )
        for sub in config.subreddits:
            progress.update(task, description=f"r/{sub.name}")
            rows: list[dict] = []
            for p in collector.collect_subreddit(sub.name):
                rows.append(p.to_dict())
                progress.update(task, posts=total + len(rows))
            if not rows:
                log.warning("collection.empty", subreddit=sub.name)
                progress.advance(task)
                continue
            df = pd.DataFrame(rows)
            path = out / f"{sub.name}.parquet"
            write_parquet(df, path)
            total += len(df)
            progress.update(task, advance=1, posts=total)
            log.info(
                "collection.subreddit_done",
                subreddit=sub.name,
                n=len(df),
                path=str(path),
            )

    log.info("collection.done", backend=backend, total=total, out=str(out))
    return out
