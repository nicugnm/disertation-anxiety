"""Typer-based CLI: collect / preprocess / label / train / evaluate / analyze."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.utils.config import (
    data_dir,
    load_labeling,
    load_model_config,
    load_subreddits,
)
from src.utils.logging import configure_logging, get_logger

# Load .env early so subcommands have access to creds
load_dotenv(override=False)

app = typer.Typer(
    name="anxiety",
    help="Anxiety / health-anxiety detection pipeline (dissertation).",
    no_args_is_help=True,
)
console = Console()
log = get_logger(__name__)


@app.callback()
def _root(
    log_level: str = typer.Option("INFO", help="Log level"),
    json_logs: bool = typer.Option(False, help="JSON-formatted logs"),
) -> None:
    configure_logging(log_level, json=json_logs)


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #


@app.command()
def collect(
    backend: str = typer.Option("scraper", help="scraper | praw | dump | synthetic"),
    config: str = typer.Option("configs/subreddits.yaml"),
    out_dir: str = typer.Option(None),
    n_synthetic: int = typer.Option(200, help="Posts per subreddit (synthetic only)"),
    dump_dir: str = typer.Option("data/external/dumps", help="Path to .zst dumps"),
    request_interval: float = typer.Option(1.5, help="Seconds between requests (scraper only)"),
    max_pages: int = typer.Option(10, help="Max pages per listing — 100 posts each (scraper only)"),
) -> None:
    """Collect Reddit data into data/raw/<subreddit>.parquet.

    Default backend is `scraper` — uses Reddit's public JSON endpoints with
    no credentials. Use `praw` for OAuth-authenticated collection (more
    permissive limits) or `dump` for Pushshift `.zst` archives.
    """
    from src.collection.runner import run_collection

    cfg = load_subreddits(config)
    kwargs: dict = {}
    if backend == "synthetic":
        kwargs["n_per_subreddit"] = n_synthetic
    if backend == "dump":
        kwargs["dump_dir"] = dump_dir
    if backend == "scraper":
        kwargs["request_interval"] = request_interval
        kwargs["max_pages_per_listing"] = max_pages
    run_collection(backend, cfg, out_dir=out_dir, **kwargs)
    console.print("[green]Collection complete.[/green]")


# --------------------------------------------------------------------------- #
# preprocess
# --------------------------------------------------------------------------- #


@app.command()
def preprocess(
    raw_dir: str = typer.Option(None),
    out_dir: str = typer.Option(None),
    no_ner: bool = typer.Option(False, help="Skip spaCy NER (regex-only anonymization)"),
) -> None:
    """Clean, anonymize, dedupe — raw → interim parquet."""
    from src.preprocessing.pipeline import run_preprocessing

    run_preprocessing(raw_dir=raw_dir, out_dir=out_dir, use_ner=not no_ner)
    console.print("[green]Preprocessing complete.[/green]")


# --------------------------------------------------------------------------- #
# label
# --------------------------------------------------------------------------- #


@app.command()
def label(
    tier: str = typer.Option("weak", help="weak | llm | aggregate | all"),
    interim_dir: str = typer.Option(None),
    out_path: str = typer.Option(None, help="Output parquet (default data/processed/labeled.parquet)"),
    config: str = typer.Option("configs/subreddits.yaml"),
    labeling_config: str = typer.Option("configs/labeling.yaml"),
) -> None:
    """Apply tier-1 weak labels, tier-2 LLM labels, or aggregate all tiers."""
    import pandas as pd

    from src.labeling.aggregate import aggregate_labels
    from src.labeling.llm import label_corpus
    from src.labeling.weak import apply_weak_labels
    from src.utils.io import read_parquet, write_parquet

    interim = Path(interim_dir) if interim_dir else data_dir("interim")
    out = Path(out_path) if out_path else data_dir("processed") / "labeled.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg_subs = load_subreddits(config)
    cfg_lab = load_labeling(labeling_config)

    combined = interim / "_all.parquet"
    if not combined.exists():
        # fall back to concatenating shards
        files = sorted(interim.glob("*.parquet"))
        df = pd.concat([read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    else:
        df = read_parquet(combined)
    if df.empty:
        raise typer.Exit("No interim data — run preprocess first.")

    if tier in ("weak", "all"):
        df = apply_weak_labels(df, cfg_subs, cfg_lab)
        write_parquet(df, out)
        console.print(f"[green]Tier-1 weak labels written to {out}[/green]")

    if tier in ("llm", "all"):
        # Reload to pick up tier-1 columns
        df = read_parquet(out) if out.exists() else df
        df = label_corpus(df, cfg_subs, cfg_lab)
        write_parquet(df, out)
        console.print(f"[green]Tier-2 LLM labels written to {out}[/green]")

    if tier in ("aggregate", "all"):
        df = read_parquet(out) if out.exists() else df
        df = aggregate_labels(df, cfg_lab)
        write_parquet(df, out)
        console.print(f"[green]Aggregated labels written to {out}[/green]")


@app.command()
def annotate(
    input_path: str = typer.Option(None, help="Path to labeled parquet"),
    output_path: str = typer.Option(None, help="Where to write manual annotations"),
    annotator_id: str = typer.Option(..., help="Identifier for this annotator"),
    labeling_config: str = typer.Option("configs/labeling.yaml"),
) -> None:
    """Tier-3 manual annotation TUI."""
    from src.labeling.manual import annotate as do_annotate

    cfg = load_labeling(labeling_config)
    inp = Path(input_path) if input_path else data_dir("processed") / "labeled.parquet"
    outp = Path(output_path) if output_path else Path(cfg.tier3_manual.output_path)
    do_annotate(inp, outp, annotator_id, cfg)


@app.command()
def kappa(
    annotations_path: str = typer.Option("data/processed/gold_test_set.parquet"),
    annotator_a: str = typer.Argument(...),
    annotator_b: str = typer.Argument(...),
) -> None:
    """Compute Cohen's kappa between two annotators."""
    from src.labeling.manual import cohen_kappa
    from src.utils.io import read_parquet

    ann = read_parquet(annotations_path)
    table = Table(title=f"Cohen's κ — {annotator_a} vs {annotator_b}")
    table.add_column("label")
    table.add_column("kappa", justify="right")
    for label in ("anxiety", "health_anxiety", "depression", "suicidality"):
        k = cohen_kappa(ann, label, annotator_a, annotator_b)
        table.add_row(label, f"{k:.3f}" if k is not None else "n/a")
    console.print(table)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #


@app.command()
def train(
    config: str = typer.Argument(..., help="Path to a model config YAML"),
    data_path: str = typer.Option(None, help="Labeled parquet path"),
    output_dir: str = typer.Option(None, help="Where to save the trained model"),
) -> None:
    """Train a model from a config YAML."""
    from src.models.registry import build_model
    from src.models.splits import split
    from src.utils.io import read_parquet

    model_cfg = load_model_config(config)
    df = read_parquet(data_path or (data_dir("processed") / "labeled.parquet"))

    target = model_cfg.target or model_cfg.targets[0]  # for splitting only
    splits_cfg = model_cfg.extra.get("splits") or model_cfg.extra.get("train", {})
    train_df, val_df, test_df = split(
        df,
        target=target,
        test_size=splits_cfg.get("test_size", 0.15),
        val_size=splits_cfg.get("val_size", 0.15),
        random_state=splits_cfg.get("random_state", 42),
    )
    console.print(f"[dim]splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}[/dim]")

    model = build_model(model_cfg)
    model.fit(train_df, val=val_df)

    out = Path(output_dir) if output_dir else Path("experiments/runs") / model_cfg.name
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model")

    # Persist the test split alongside, so evaluate uses the same data
    from src.utils.io import write_parquet

    write_parquet(test_df, out / "test.parquet")
    if not val_df.empty:
        write_parquet(val_df, out / "val.parquet")
    write_parquet(train_df, out / "train.parquet")
    (out / "config.yaml").write_text(Path(config).read_text())
    console.print(f"[green]Model saved → {out}[/green]")


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #


@app.command()
def evaluate(
    run_dir: str = typer.Argument(..., help="Path to an experiments/runs/<name> directory"),
    out_dir: str = typer.Option(None),
) -> None:
    """Evaluate a saved model against its held-out test set."""
    from src.evaluation.runner import evaluate_model
    from src.models.registry import build_model
    from src.utils.io import read_parquet

    run = Path(run_dir)
    cfg = load_model_config(run / "config.yaml")
    model = build_model(cfg)
    model.load(run / "model")
    test = read_parquet(run / "test.parquet")

    out = Path(out_dir) if out_dir else run / "eval"
    targets = cfg.targets or [cfg.target]
    for t in targets:
        evaluate_model(model, test, t, out, name=cfg.name)
    console.print(f"[green]Evaluation written to {out}[/green]")


@app.command()
def report(
    eval_dir: str = typer.Argument(..., help="Directory containing *__metrics.json"),
) -> None:
    """Aggregate evaluation runs into one comparison table."""
    from src.evaluation.runner import aggregate_reports

    df = aggregate_reports(eval_dir)
    if df.empty:
        console.print("[yellow]No evaluation files found.[/yellow]")
        return
    console.print(df.to_string(index=False))


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #


@app.command()
def analyze_markers(
    target: str = typer.Option("health_anxiety"),
    data_path: str = typer.Option(None),
    out_path: str = typer.Option(None),
) -> None:
    """Statistical comparison of linguistic features between positives and negatives."""
    from src.analysis.linguistic_markers import compare_features_by_label
    from src.utils.io import read_parquet

    df = read_parquet(data_path or (data_dir("processed") / "labeled.parquet"))
    out_path = out_path or f"experiments/markers__{target}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result = compare_features_by_label(df, target)
    result.to_csv(out_path, index=False)
    console.print(result.head(20).to_string(index=False))
    console.print(f"\n[green]Full table → {out_path}[/green]")


@app.command()
def plot(
    figures_dir: str = typer.Option("docs/figures", help="Where to write PNGs"),
    labeled_path: str = typer.Option(None, help="Labeled parquet path"),
    run_dir: str = typer.Option(None, help="A trained model run dir (for performance plots)"),
) -> None:
    """Generate every standard figure from corpus + model run."""
    from src.viz.runner import run_all

    paths = run_all(figures_dir=figures_dir, labeled_path=labeled_path, run_dir=run_dir)
    for name, p in paths.items():
        console.print(f"  [dim]{name:<32}[/dim] {p}")
    console.print(f"\n[green]Wrote {len(paths)} figures → {figures_dir}[/green]")


@app.command()
def analyze_temporal(
    data_path: str = typer.Option(None),
    out_path: str = typer.Option("experiments/temporal.csv"),
) -> None:
    """COVID-window analysis (RQ4)."""
    from src.analysis.temporal import label_rates_by_period_and_subreddit
    from src.utils.io import read_parquet

    df = read_parquet(data_path or (data_dir("processed") / "labeled.parquet"))
    label_cols = [
        c for c in df.columns
        if c in {"label_anxiety", "label_health_anxiety", "label_depression", "label_suicidality"}
    ]
    res = label_rates_by_period_and_subreddit(df, label_cols)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    console.print(res.to_string(index=False))


# --------------------------------------------------------------------------- #
# end-to-end smoke run (no creds)
# --------------------------------------------------------------------------- #


@app.command("smoke-run")
def smoke_run() -> None:
    """End-to-end pipeline on synthetic data — proves the system runs."""
    from src.collection.runner import run_collection
    from src.evaluation.runner import evaluate_model
    from src.labeling.aggregate import aggregate_labels
    from src.labeling.weak import apply_weak_labels
    from src.models.registry import build_model
    from src.models.splits import split
    from src.preprocessing.pipeline import run_preprocessing
    from src.utils.io import read_parquet, write_parquet

    cfg_subs = load_subreddits()
    cfg_lab = load_labeling()

    console.rule("[1/6] Collecting synthetic data")
    run_collection("synthetic", cfg_subs, n_per_subreddit=120)

    console.rule("[2/6] Preprocessing")
    run_preprocessing(use_ner=False)  # skip spaCy in smoke for speed

    console.rule("[3/6] Tier-1 weak labeling")
    interim = data_dir("interim") / "_all.parquet"
    df = read_parquet(interim)
    df = apply_weak_labels(df, cfg_subs, cfg_lab)
    df = aggregate_labels(df, cfg_lab)
    out = data_dir("processed") / "labeled.parquet"
    write_parquet(df, out)
    console.print(f"[dim]Wrote {len(df)} labeled rows -> {out}[/dim]")

    console.rule("[4/6] Train baseline (TF-IDF + LogReg)")
    cfg = load_model_config("configs/models/baseline.yaml")
    train_df, val_df, test_df = split(df, cfg.target, test_size=0.15, val_size=0.15)
    model = build_model(cfg)
    model.fit(train_df, val=val_df)

    run_dir = Path("experiments/runs") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model")
    write_parquet(test_df, run_dir / "test.parquet")

    console.rule("[5/6] Evaluate")
    report = evaluate_model(model, test_df, cfg.target, run_dir / "eval", name=cfg.name)
    console.print(json.dumps({k: v for k, v in report.items() if not k.endswith("ci_lo") and not k.endswith("ci_hi")}, indent=2))

    console.rule("[6/6] Linguistic markers")
    from src.analysis.linguistic_markers import compare_features_by_label

    if df["label_health_anxiety"].notna().sum() > 0:
        markers = compare_features_by_label(df, "health_anxiety")
        console.print(markers.head(10).to_string(index=False))

    console.rule("[green]Smoke run complete[/green]")


if __name__ == "__main__":
    app()
