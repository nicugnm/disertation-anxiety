"""Tier-3 manual annotation — minimal terminal UI.

Run with `anxiety annotate`. Stratifies posts across subreddit_group, presents
them one at a time with a crisis-resource banner, accepts a 4-tuple of
binary labels and a confidence score, and writes incrementally to parquet.

Designed so two annotators can label the same set in parallel by passing
different `--annotator-id` values; we then compute Cohen's kappa.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt

from src.labeling.weak import LABELS
from src.utils.config import LabelingConfig
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)
console = Console()

CRISIS_BANNER = """[bold red]CRISIS RESOURCES[/bold red]
US:    988 (call/text)        UK & ROI: Samaritans 116 123
EU:    https://www.befrienders.org/
World: https://findahelpline.com/

You can SKIP any post that distresses you (press 's')."""


def _stratified_sample(
    df: pd.DataFrame,
    target_size: int,
    stratify_by: str,
    seed: int = 42,
) -> pd.DataFrame:
    if stratify_by not in df.columns:
        return df.sample(n=min(target_size, len(df)), random_state=seed)
    groups = df.groupby(stratify_by)
    per = max(1, target_size // len(groups))
    parts = []
    for _, sub in groups:
        parts.append(sub.sample(n=min(per, len(sub)), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _ask_label(name: str) -> int:
    while True:
        ans = console.input(f"[cyan]{name}[/cyan] (0/1, ?=help): ").strip()
        if ans in ("0", "1"):
            return int(ans)
        if ans == "?":
            console.print("[dim]See docs/codebook.md for definitions.[/dim]")
        else:
            console.print("[red]Enter 0 or 1.[/red]")


def annotate(
    input_path: str | Path,
    output_path: str | Path,
    annotator_id: str,
    cfg: LabelingConfig,
) -> Path:
    """Interactive annotation loop. Resumes from existing output."""
    df = read_parquet(input_path)
    sample = _stratified_sample(df, cfg.tier3_manual.target_size, cfg.tier3_manual.stratify_by)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done_ids: set[str] = set()
    if out_path.exists():
        existing = read_parquet(out_path)
        done_ids = set(existing[existing["annotator_id"] == annotator_id]["id"].astype(str))
        console.print(f"[dim]Resuming — {len(done_ids)} posts already labeled by '{annotator_id}'.[/dim]")

    console.print(Panel(CRISIS_BANNER, border_style="red"))
    console.print(f"[bold]Annotator:[/bold] {annotator_id}")
    console.print(f"[bold]Target:[/bold] {len(sample)} posts | [bold]Done:[/bold] {len(done_ids)}\n")

    new_rows: list[dict] = []
    for _, row in sample.iterrows():
        if str(row["id"]) in done_ids:
            continue
        os.system("clear" if os.name == "posix" else "cls")
        console.print(Panel.fit(
            row["clean_text"][:2000] + ("..." if len(row["clean_text"]) > 2000 else ""),
            title=f"r/{row['subreddit']}  |  id={row['id']}",
            border_style="blue",
        ))
        console.print()

        if console.input("Label this post? (y/n/s=skip/q=quit): ").strip().lower() == "q":
            break
        # 's' or 'n' = skip
        choice = console.input("Decision (y=label, s=skip): ").strip().lower()
        if choice == "s":
            continue

        labels = {k: _ask_label(k) for k in LABELS}
        # Health anxiety implies anxiety
        if labels["health_anxiety"] and not labels["anxiety"]:
            if Confirm.ask("[yellow]health_anxiety=1 but anxiety=0 — auto-set anxiety=1?[/yellow]"):
                labels["anxiety"] = 1
        confidence = IntPrompt.ask("Confidence (1-5)", default=3)

        new_rows.append({
            "id": row["id"],
            "annotator_id": annotator_id,
            **{f"manual_{k}": v for k, v in labels.items()},
            "manual_confidence": int(confidence),
        })

        # Write incrementally so a crash doesn't lose progress.
        if len(new_rows) % 10 == 0:
            _flush(out_path, new_rows)
            new_rows = []

    if new_rows:
        _flush(out_path, new_rows)

    console.print("[green]Done.[/green]")
    return out_path


def _flush(out_path: Path, rows: list[dict]) -> None:
    new_df = pd.DataFrame(rows)
    if out_path.exists():
        existing = read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    write_parquet(combined, out_path)


def cohen_kappa(
    annotations: pd.DataFrame,
    label: str,
    annotator_a: str,
    annotator_b: str,
) -> float | None:
    """Cohen's kappa between two annotators on a given label."""
    from sklearn.metrics import cohen_kappa_score

    col = f"manual_{label}"
    a = annotations[annotations["annotator_id"] == annotator_a][["id", col]]
    b = annotations[annotations["annotator_id"] == annotator_b][["id", col]]
    merged = a.merge(b, on="id", suffixes=("_a", "_b"))
    if len(merged) < 5:
        return None
    return float(cohen_kappa_score(merged[f"{col}_a"], merged[f"{col}_b"]))
