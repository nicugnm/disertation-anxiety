"""Corpus + labeling audit: how much data, how many disclosures, and are they real?

Produces:
  - Console tables (Rich) for live inspection
  - Optional Markdown report for the thesis methodology chapter
  - A JSON dict you can paste into any other doc

The most important section is the **disclosure examples** — for each target,
N actual posts where the regex fired, with the matched span highlighted. This
is the only way to catch lexicon false-positives early (e.g. "I was diagnosed
with depression" caught in a movie review).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_TARGETS = ("anxiety", "health_anxiety", "depression", "suicidality")


# --------------------------------------------------------------------------- #
# Data class
# --------------------------------------------------------------------------- #


@dataclass
class CorpusAudit:
    """All audit numbers for the latest labeled corpus."""

    # Section 1: corpus stats
    corpus_stats: dict = field(default_factory=dict)

    # Section 2: per-tier label counts (target → tier → n_positive)
    label_counts: list[dict] = field(default_factory=list)

    # Section 3: per (subreddit, target) disclosure matrix
    subreddit_breakdown: list[dict] = field(default_factory=list)

    # Section 4: user-level disclosure stats per target
    user_stats: dict = field(default_factory=dict)

    # Section 5: example disclosure matches per target
    examples: dict = field(default_factory=dict)

    # Section 6: data-quality warnings
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Audit computation
# --------------------------------------------------------------------------- #


def _corpus_stats(df: pd.DataFrame) -> dict:
    out: dict = {"total_rows": int(len(df))}
    if "kind" in df.columns:
        kinds = df["kind"].value_counts().to_dict()
        out["submissions"] = int(kinds.get("submission", 0))
        out["comments"] = int(kinds.get("comment", 0))
    if "subreddit" in df.columns:
        out["n_subreddits"] = int(df["subreddit"].nunique())
    if "author_hash" in df.columns:
        out["n_unique_authors"] = int(df["author_hash"].nunique(dropna=True))
    if "created_utc" in df.columns and not df["created_utc"].empty:
        ts_min = float(df["created_utc"].min())
        ts_max = float(df["created_utc"].max())
        out["date_first"] = datetime.fromtimestamp(ts_min, tz=timezone.utc).date().isoformat()
        out["date_last"] = datetime.fromtimestamp(ts_max, tz=timezone.utc).date().isoformat()
    if "clean_text" in df.columns:
        lens = df["clean_text"].astype(str).str.len()
        out["avg_text_chars"] = round(float(lens.mean()), 1)
        out["median_text_chars"] = float(lens.median())
    return out


def _label_counts(df: pd.DataFrame, targets: tuple[str, ...]) -> list[dict]:
    """Per-target counts of (weak, disclosure, final) positives + LLM if present."""
    rows: list[dict] = []
    n = len(df)
    for t in targets:
        row = {"target": t}
        for tier, col in [
            ("weak", f"weak_{t}_bin"),
            ("disclosure", f"disclosure_{t}"),
            ("llm", f"llm_{t}"),
            ("manual", f"manual_{t}"),
            ("final_label", f"label_{t}"),
        ]:
            if col not in df.columns:
                row[tier] = None
                row[f"{tier}_pct"] = None
                continue
            if tier == "final_label":
                positives = int((df[col].astype(float).fillna(0) >= 0.5).sum())
            else:
                positives = int(df[col].fillna(0).astype(int).sum())
            row[tier] = positives
            row[f"{tier}_pct"] = round(100.0 * positives / max(1, n), 3)
        rows.append(row)
    return rows


def _subreddit_breakdown(df: pd.DataFrame, targets: tuple[str, ...], top_n: int = 25) -> list[dict]:
    """For each subreddit, count disclosure positives per target. Sorted by total."""
    if "subreddit" not in df.columns:
        return []
    rows: list[dict] = []
    for sub, g in df.groupby("subreddit"):
        row = {"subreddit": sub, "n_posts": int(len(g))}
        total = 0
        for t in targets:
            col = f"disclosure_{t}"
            n_d = int(g[col].fillna(0).astype(int).sum()) if col in g.columns else 0
            row[f"disclosure_{t}"] = n_d
            total += n_d
        row["total_disclosures"] = total
        rows.append(row)
    rows.sort(key=lambda r: r["total_disclosures"], reverse=True)
    return rows[:top_n]


def _user_stats(df: pd.DataFrame, targets: tuple[str, ...]) -> dict:
    """For each target: unique disclosed users + post-count distribution."""
    out: dict = {}
    if "author_hash" not in df.columns:
        return out
    post_counts = df.groupby("author_hash").size()
    for t in targets:
        col = f"disclosure_{t}"
        if col not in df.columns:
            continue
        disclosed_mask = df[col].fillna(0).astype(int) == 1
        disclosed_users = df.loc[disclosed_mask, "author_hash"].dropna().unique()
        if len(disclosed_users) == 0:
            out[t] = {
                "n_disclosed_users": 0,
                "avg_posts_per_user": None,
                "median_posts_per_user": None,
                "max_posts_per_user": None,
            }
            continue
        user_posts = post_counts.reindex(disclosed_users).fillna(0)
        out[t] = {
            "n_disclosed_users": int(len(disclosed_users)),
            "avg_posts_per_user": round(float(user_posts.mean()), 1),
            "median_posts_per_user": float(user_posts.median()),
            "max_posts_per_user": int(user_posts.max()),
        }
    return out


def _disclosure_examples(
    df: pd.DataFrame,
    targets: tuple[str, ...],
    n: int = 5,
    seed: int = 42,
) -> dict:
    """For each target, sample N disclosure-positive posts with the matched span."""
    out: dict = {}
    rng = random.Random(seed)
    for t in targets:
        col = f"disclosure_{t}"
        if col not in df.columns:
            continue
        mask = df[col].fillna(0).astype(int) == 1
        positive = df[mask]
        if positive.empty:
            out[t] = []
            continue
        take = min(n, len(positive))
        idx = rng.sample(range(len(positive)), take)
        match_col = f"disclosure_{t}_match"
        items = []
        for i in idx:
            row = positive.iloc[i]
            preview = str(row.get("clean_text") or "")[:280]
            items.append({
                "id": str(row.get("id") or ""),
                "subreddit": str(row.get("subreddit") or ""),
                "matched_span": str(row.get(match_col) or "") if match_col in df.columns else "",
                "preview": preview,
            })
        out[t] = items
    return out


def _warnings(df: pd.DataFrame, label_counts: list[dict]) -> list[str]:
    out: list[str] = []
    # Class-balance flags
    for row in label_counts:
        t = row["target"]
        d = row.get("disclosure")
        if d is not None and d == 0:
            out.append(f"⚠️  {t}: 0 disclosure positives — regex didn't match anything")
        elif d is not None and d < 30:
            out.append(
                f"⚠️  {t}: only {d} disclosure positives — consider running "
                "`anxiety collect --backend search` to find more"
            )
    # Schema flags
    if "kind" not in df.columns:
        out.append("⚠️  `kind` column missing — comment / submission split not tracked")
    if "author_hash" not in df.columns:
        out.append("⚠️  `author_hash` missing — user-level analysis won't work")
    # Held-out flag
    if "held_out_split" in df.columns:
        n_held = int(df["held_out_split"].fillna(False).astype(bool).sum())
        out.append(
            f"ℹ️  {n_held:,} posts marked held_out_split=True (disclosure test set);"
            " `anxiety train` will exclude them by default"
        )
    return out


def run_audit(
    df: pd.DataFrame,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    n_examples: int = 5,
    top_n_subreddits: int = 25,
    seed: int = 42,
) -> CorpusAudit:
    log.info("audit.start", n_rows=len(df), targets=list(targets))
    audit = CorpusAudit(
        corpus_stats=_corpus_stats(df),
        label_counts=_label_counts(df, targets),
        subreddit_breakdown=_subreddit_breakdown(df, targets, top_n=top_n_subreddits),
        user_stats=_user_stats(df, targets),
        examples=_disclosure_examples(df, targets, n=n_examples, seed=seed),
    )
    audit.warnings = _warnings(df, audit.label_counts)
    log.info("audit.done")
    return audit


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def print_audit(audit: CorpusAudit, console: Console | None = None) -> None:
    console = console or Console()

    # 1. Corpus
    cs = audit.corpus_stats
    panel_lines = []
    for k in ("total_rows", "submissions", "comments", "n_subreddits", "n_unique_authors"):
        if k in cs:
            panel_lines.append(f"[bold]{k:<20}[/bold] {cs[k]:>10,}")
    if "date_first" in cs and "date_last" in cs:
        panel_lines.append(f"[bold]{'date range':<20}[/bold] {cs['date_first']} → {cs['date_last']}")
    if "avg_text_chars" in cs:
        panel_lines.append(f"[bold]{'avg text chars':<20}[/bold] {cs['avg_text_chars']:>10}")
    console.print(Panel("\n".join(panel_lines), title="1) Corpus", border_style="cyan"))

    # 2. Per-tier counts
    t = Table(title="2) Per-target label counts (positives across the corpus)", show_lines=False)
    t.add_column("target")
    for tier_name in ("weak", "disclosure", "llm", "manual", "final_label"):
        t.add_column(tier_name, justify="right")
    for row in audit.label_counts:
        vals = []
        for tier_name in ("weak", "disclosure", "llm", "manual", "final_label"):
            v = row.get(tier_name)
            pct = row.get(f"{tier_name}_pct")
            if v is None:
                vals.append("—")
            else:
                vals.append(f"{v:,} ({pct:.2f}%)" if pct is not None else f"{v:,}")
        t.add_row(row["target"], *vals)
    console.print(t)

    # 3. Top subreddits by disclosure
    targets = [k for k in audit.label_counts and audit.label_counts[0] if k.startswith("disclosure_") is False]
    t = Table(title=f"3) Top subreddits by total disclosures (top {len(audit.subreddit_breakdown)})")
    t.add_column("subreddit")
    t.add_column("posts", justify="right")
    target_cols = [k.replace("disclosure_", "") for k in audit.subreddit_breakdown[0].keys()
                   if k.startswith("disclosure_")] if audit.subreddit_breakdown else []
    for tg in target_cols:
        t.add_column(tg, justify="right")
    t.add_column("total", justify="right", style="bold")
    for row in audit.subreddit_breakdown:
        cells = [row["subreddit"], f"{row['n_posts']:,}"]
        for tg in target_cols:
            cells.append(str(row.get(f"disclosure_{tg}", 0)))
        cells.append(str(row.get("total_disclosures", 0)))
        t.add_row(*cells)
    console.print(t)

    # 4. User-level stats
    t = Table(title="4) Disclosed users per target")
    t.add_column("target")
    t.add_column("n disclosed users", justify="right")
    t.add_column("avg posts/user", justify="right")
    t.add_column("median posts/user", justify="right")
    t.add_column("max posts/user", justify="right")
    for tg, stats in audit.user_stats.items():
        t.add_row(
            tg,
            f"{stats.get('n_disclosed_users', 0):,}",
            str(stats.get("avg_posts_per_user", "—")),
            str(stats.get("median_posts_per_user", "—")),
            str(stats.get("max_posts_per_user", "—")),
        )
    console.print(t)

    # 5. Examples (this is the most important sanity check)
    console.print()
    console.print("[bold cyan]5) Example disclosure matches[/bold cyan]  (verify the regex isn't catching nonsense)")
    for tg, items in audit.examples.items():
        if not items:
            console.print(f"  [dim]{tg}: no positives to show[/dim]")
            continue
        for i, item in enumerate(items, 1):
            console.print(
                f"  [bold]{tg}[/bold]  #{i}  [dim]r/{item['subreddit']}  id={item['id']}[/dim]"
            )
            if item["matched_span"]:
                console.print(f"    [green]matched:[/green] '{item['matched_span']}'")
            console.print(f"    [italic]{item['preview']}[/italic]")
        console.print()

    # 6. Warnings
    if audit.warnings:
        console.print(Panel("\n".join(audit.warnings), title="6) Notices", border_style="yellow"))


def write_audit_markdown(audit: CorpusAudit, path: str | Path) -> Path:
    """Write a self-contained Markdown report suitable for the thesis appendix."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Corpus + labeling audit\n")
    lines.append(
        f"_Generated {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n"
    )

    # 1. Corpus
    lines.append("## 1. Corpus stats\n")
    cs = audit.corpus_stats
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for k, v in cs.items():
        if isinstance(v, int):
            lines.append(f"| {k} | {v:,} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.append("")

    # 2. Label counts
    lines.append("## 2. Per-target positives by tier\n")
    if audit.label_counts:
        lines.append("| target | weak | disclosure | llm | manual | final_label |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in audit.label_counts:
            cells = [row["target"]]
            for tier in ("weak", "disclosure", "llm", "manual", "final_label"):
                v = row.get(tier)
                pct = row.get(f"{tier}_pct")
                cells.append(f"{v:,} ({pct:.2f}%)" if v is not None and pct is not None else "—" if v is None else f"{v:,}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # 3. Subreddit breakdown
    lines.append("## 3. Top subreddits by total disclosures\n")
    if audit.subreddit_breakdown:
        target_cols = [k.replace("disclosure_", "") for k in audit.subreddit_breakdown[0].keys()
                       if k.startswith("disclosure_")]
        header = ["subreddit", "n_posts"] + target_cols + ["total"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
        for row in audit.subreddit_breakdown:
            cells = [row["subreddit"], f"{row['n_posts']:,}"]
            for tg in target_cols:
                cells.append(str(row.get(f"disclosure_{tg}", 0)))
            cells.append(f"**{row.get('total_disclosures', 0)}**")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # 4. User-level stats
    lines.append("## 4. Disclosed users per target\n")
    if audit.user_stats:
        lines.append("| target | n disclosed users | avg posts/user | median | max |")
        lines.append("|---|---:|---:|---:|---:|")
        for tg, stats in audit.user_stats.items():
            lines.append(
                f"| {tg} | {stats.get('n_disclosed_users', 0):,} | "
                f"{stats.get('avg_posts_per_user', '—')} | "
                f"{stats.get('median_posts_per_user', '—')} | "
                f"{stats.get('max_posts_per_user', '—')} |"
            )
        lines.append("")

    # 5. Examples
    lines.append("## 5. Example disclosure matches\n")
    lines.append("These are random samples — useful for spotting regex false positives.\n")
    for tg, items in audit.examples.items():
        lines.append(f"### {tg}\n")
        if not items:
            lines.append("_No positives in the corpus._\n")
            continue
        for i, item in enumerate(items, 1):
            lines.append(f"**{i}.** `r/{item['subreddit']}` `id={item['id']}`")
            if item["matched_span"]:
                lines.append(f"  matched: `{item['matched_span']}`")
            lines.append(f"  > {item['preview']}")
            lines.append("")

    # 6. Warnings
    if audit.warnings:
        lines.append("## 6. Notices\n")
        for w in audit.warnings:
            lines.append(f"- {w}")
        lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
    log.info("audit.markdown_written", path=str(p))
    return p


def audit_to_json(audit: CorpusAudit) -> str:
    return json.dumps(asdict(audit), indent=2, default=str)
