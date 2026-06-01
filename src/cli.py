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
    backend: str = typer.Option("scraper", help="scraper | search | praw | dump | synthetic"),
    config: str = typer.Option("configs/subreddits.yaml"),
    out_dir: str = typer.Option(None),
    n_synthetic: int = typer.Option(200, help="Posts per subreddit (synthetic only)"),
    dump_dir: str = typer.Option("data/external/dumps", help="Path to .zst dumps"),
    request_interval: float = typer.Option(1.5, help="Seconds between requests"),
    max_pages: int = typer.Option(10, help="Max pages per listing (scraper) or per query (search)"),
    include_comments: bool = typer.Option(False, help="Also fetch comments per submission (scraper only)"),
    min_submission_comments: int = typer.Option(5, help="Skip submissions with fewer comments (scraper + comments)"),
    max_comments_per_post: int = typer.Option(40, help="Max comments to keep per submission"),
    min_comment_score: int = typer.Option(1, help="Drop comments below this score"),
) -> None:
    """Collect Reddit data into data/raw/.

    Backends:
      - `scraper` (default): subreddit listings via public JSON. Optionally
        fetches comments per submission with --include-comments.
      - `search`: Reddit-wide search for self-disclosure phrases (e.g.
        "I was diagnosed with depression"). One parquet per query.
      - `praw`: OAuth-authenticated collection (needs Reddit credentials).
      - `dump`: Pushshift / arctic_shift .zst archives.
      - `synthetic`: reproducible synthetic data (no network).
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
        kwargs["include_comments"] = include_comments
        kwargs["min_submission_comments"] = min_submission_comments
        kwargs["max_comments_per_post"] = max_comments_per_post
        kwargs["min_comment_score"] = min_comment_score
    if backend == "search":
        kwargs["request_interval"] = request_interval
        kwargs["max_pages_per_query"] = max_pages
    run_collection(backend, cfg, out_dir=out_dir, **kwargs)
    console.print("[green]Collection complete.[/green]")


@app.command("collect-authors")
def collect_authors(
    users_csv: str = typer.Option(
        None, help="Cohort CSV (default data/processed/disclosure_testset__users.csv)"
    ),
    raw_dir: str = typer.Option("data/raw", help="Where to recover usernames from"),
    out_dir: str = typer.Option("data/raw/authors", help="Per-user history shards"),
    request_interval: float = typer.Option(1.5, help="Seconds between requests"),
    max_pages: int = typer.Option(10, help="Pages per listing (10x100 ~= 1000-item cap)"),
    config: str = typer.Option("configs/subreddits.yaml"),
) -> None:
    """Fetch full Reddit histories for the disclosure-test cohort (both classes).

    Resumable: re-running skips users whose <hash>.parquet already exists.
    """
    import pandas as pd

    from src.collection.author_history import run_author_collection

    cfg = load_subreddits(config)
    csv_path = (
        Path(users_csv) if users_csv
        else data_dir("processed") / "disclosure_testset__users.csv"
    )
    if not csv_path.exists():
        raise typer.Exit(f"Cohort CSV not found: {csv_path} — run build-disclosure-testset first.")
    users_df = pd.read_csv(csv_path)
    stats = run_author_collection(
        users_df, cfg, raw_dir=raw_dir, out_dir=out_dir,
        request_interval=request_interval, max_pages=max_pages,
    )
    console.print(f"[green]Author history collection complete:[/green] {stats}")


# --------------------------------------------------------------------------- #
# preprocess
# --------------------------------------------------------------------------- #


@app.command()
def preprocess(
    raw_dir: str = typer.Option(None),
    out_dir: str = typer.Option(None),
    no_ner: bool = typer.Option(False, help="Skip spaCy NER (regex-only PII redaction; ~10× faster)"),
    n_process_ner: int = typer.Option(
        1, help="spaCy worker processes (>1 ≈ linear speedup on multi-core)"
    ),
    near_dup_threshold: int = typer.Option(
        5, help="SimHash Hamming radius; -1 disables near-dedup (exact-only)"
    ),
) -> None:
    """Clean, anonymize, dedupe — raw → interim parquet.

    At 100k+ posts, use --n-process-ner 4 (or higher, depending on your CPU)
    to parallelize the spaCy NER pass. Or --no-ner to skip NER entirely
    (regex still redacts emails, phones, usernames, URLs).
    """
    from src.preprocessing.pipeline import run_preprocessing

    run_preprocessing(
        raw_dir=raw_dir,
        out_dir=out_dir,
        use_ner=not no_ner,
        n_process_ner=n_process_ner,
        near_dup_threshold=near_dup_threshold,
    )
    console.print("[green]Preprocessing complete.[/green]")


# --------------------------------------------------------------------------- #
# label
# --------------------------------------------------------------------------- #


@app.command()
def label(
    tier: str = typer.Option("weak", help="weak | disclosure | llm | aggregate | all"),
    interim_dir: str = typer.Option(None),
    out_path: str = typer.Option(None, help="Output parquet (default data/processed/labeled.parquet)"),
    config: str = typer.Option("configs/subreddits.yaml"),
    labeling_config: str = typer.Option("configs/labeling.yaml"),
) -> None:
    """Apply tier-1 weak labels, self-disclosure labels, tier-2 LLM labels, or aggregate all tiers."""
    import pandas as pd

    from src.labeling.aggregate import aggregate_labels
    from src.labeling.llm import label_corpus
    from src.labeling.self_disclosure import apply_disclosure_labels
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

    if tier in ("disclosure", "all"):
        # Reload to pick up prior tiers; disclosure is independent of subreddit
        # and lexicon-thresholds so it can also be run standalone.
        df = read_parquet(out) if out.exists() else df
        df = apply_disclosure_labels(df)
        write_parquet(df, out)
        positives = {
            t: int(df[f"disclosure_{t}"].sum())
            for t in ("anxiety", "health_anxiety", "depression", "suicidality")
            if f"disclosure_{t}" in df.columns
        }
        console.print(f"[green]Self-disclosure labels written to {out}[/green]")
        console.print(f"[dim]Disclosure positives: {positives}[/dim]")

        # Auto-audit: gives the user immediate visibility into how much
        # disclosure data we got, per subreddit + per user, plus sample matches
        # for sanity-checking the regex. Also written to docs/audit_report.md.
        from src.labeling.audit import print_audit, run_audit, write_audit_markdown

        console.rule("[bold cyan]Audit")
        audit = run_audit(df)
        print_audit(audit, console)
        report_path = Path("docs/audit_report.md")
        write_audit_markdown(audit, report_path)
        console.print(f"\n[green]Full report → {report_path}[/green]")

    if tier in ("llm", "all"):
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
    include_held_out: bool = typer.Option(
        False, help="Include held-out (disclosure-test-set) users in training. Off by default — keep separation."
    ),
) -> None:
    """Train a model from a config YAML.

    By default, posts marked `held_out_split=True` (members of the user-level
    disclosure test set) are excluded from training to preserve the
    noisy-train / clean-test separation. Pass --include-held-out to ignore
    that filter.
    """
    from src.models.registry import build_model
    from src.models.splits import split
    from src.utils.io import read_parquet

    model_cfg = load_model_config(config)
    df = read_parquet(data_path or (data_dir("processed") / "labeled.parquet"))

    if "held_out_split" in df.columns and not include_held_out:
        n_before = len(df)
        df = df[~df["held_out_split"].fillna(False).astype(bool)].reset_index(drop=True)
        if n_before != len(df):
            console.print(
                f"[dim]Excluded {n_before - len(df):,} held-out posts "
                f"(disclosure test users); training on {len(df):,}.[/dim]"
            )

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
def audit(
    data_path: str = typer.Option(None, help="Labeled corpus path"),
    out_path: str = typer.Option("docs/audit_report.md", help="Markdown report path; pass empty to skip"),
    n_examples: int = typer.Option(5, help="Example disclosure matches per target"),
    top_n_subreddits: int = typer.Option(25),
) -> None:
    """Comprehensive corpus + labeling audit.

    Reports corpus stats, per-tier label counts, per-subreddit × target
    matrix, user-level disclosure stats, and sample disclosure matches so
    you can sanity-check the regex isn't catching nonsense. Also writes a
    self-contained Markdown report for the thesis appendix.
    """
    from src.labeling.audit import print_audit, run_audit, write_audit_markdown
    from src.utils.io import read_parquet

    in_p = Path(data_path) if data_path else data_dir("processed") / "labeled.parquet"
    df = read_parquet(in_p)
    a = run_audit(df, n_examples=n_examples, top_n_subreddits=top_n_subreddits)
    print_audit(a, console)
    if out_path:
        path = write_audit_markdown(a, out_path)
        console.print(f"\n[green]Markdown report → {path}[/green]")


@app.command("build-disclosure-testset")
def build_disclosure_testset(
    in_path: str = typer.Option(None, help="Labeled corpus parquet"),
    test_path: str = typer.Option(None, help="Output test-set parquet"),
    controls_per_positive: int = typer.Option(2, help="Matched controls per positive user"),
    min_posts_per_user: int = typer.Option(3, help="Skip users below this post count"),
    seed: int = typer.Option(42),
) -> None:
    """Build a user-level disclosure test set with subreddit-matched controls.

    Disclosed users become positives; non-disclosed users from the same
    subreddits become controls. All their posts are held out from training
    so the train/test split is genuinely user-disjoint and label-source-disjoint.
    """
    from src.labeling.disclosure_dataset import (
        build_disclosure_test_users,
        mark_held_out,
        materialize_test_posts,
    )
    from src.utils.io import read_parquet, write_parquet

    in_p = Path(in_path) if in_path else data_dir("processed") / "labeled.parquet"
    test_p = Path(test_path) if test_path else data_dir("processed") / "disclosure_testset.parquet"

    df = read_parquet(in_p)
    test_users = build_disclosure_test_users(
        df,
        controls_per_positive=controls_per_positive,
        min_posts_per_user=min_posts_per_user,
        seed=seed,
    )
    if test_users.empty:
        console.print(
            "[yellow]No disclosure positives in the corpus — "
            "run `anxiety label --tier disclosure` first.[/yellow]"
        )
        raise typer.Exit(1)

    # Diagnostic guards — fire only on impossible-by-construction states.
    if "user_group" not in test_users.columns:
        console.print(
            "[red]test_users is missing the `user_group` column. "
            "Columns present: " + ", ".join(test_users.columns) + "[/red]"
        )
        raise typer.Exit(2)
    n_positive_users = sum(
        int((test_users[f"user_{t}"].fillna(0).astype(int) == 1).sum())
        for t in ("anxiety", "health_anxiety", "depression", "suicidality")
        if f"user_{t}" in test_users.columns
    )
    n_controls = int((test_users["user_group"] == "matched_control").sum())
    if n_positive_users == 0:
        console.print("[yellow]No disclosed positive users survived filtering. "
                      "Try lowering --min-posts-per-user.[/yellow]")
        raise typer.Exit(1)
    if n_controls == 0:
        console.print(
            "[yellow]No matched controls found. The disclosed users are in "
            "subreddits where no other users have enough posts. "
            "Try lowering --min-posts-per-user, or --controls-per-positive 1 first.[/yellow]"
        )

    test_posts = materialize_test_posts(df, test_users)
    write_parquet(test_posts, test_p)

    # Write the user-level summary alongside (useful for thesis tables).
    users_summary_path = test_p.with_name(test_p.stem + "__users.csv")
    test_users.to_csv(users_summary_path, index=False)

    # Mark the corpus so `anxiety train` automatically excludes these users.
    marked = mark_held_out(df, test_users)
    write_parquet(marked, in_p)

    # Summary table — test_posts already has the user_group column from
    # materialize_test_posts (don't re-merge or pandas renames it to *_x/_y).
    from rich.table import Table
    table = Table(title="Disclosure test set")
    table.add_column("group")
    table.add_column("n_users", justify="right")
    table.add_column("n_posts", justify="right")
    grouped = test_users.groupby("user_group").size().sort_values(ascending=False)
    posts_per_group = test_posts.groupby("user_group").size()
    for g, n in grouped.items():
        table.add_row(str(g), f"{int(n):,}", f"{int(posts_per_group.get(g, 0)):,}")
    console.print(table)
    console.print(f"[green]Test posts → {test_p}[/green]")
    console.print(f"[green]Test users → {users_summary_path}[/green]")
    console.print(f"[green]Corpus updated with held_out_split flag → {in_p}[/green]")


@app.command("eval-disclosure")
def eval_disclosure(
    run_dir: str = typer.Argument(..., help="experiments/runs/<name> directory"),
    target: str = typer.Option("anxiety", help="Target label to evaluate"),
    test_path: str = typer.Option(None, help="Disclosure test parquet"),
    aggregation: str = typer.Option("mean", help="mean | max | topk_mean"),
    mask_disclosure_posts: bool = typer.Option(
        False, help="Drop the explicit disclosure utterances before aggregating per user"
    ),
) -> None:
    """Evaluate a trained model at user-level on the disclosure test set."""
    import json

    from src.labeling.disclosure_dataset import evaluate_user_level
    from src.models.registry import build_model
    from src.utils.io import read_parquet, write_parquet

    run = Path(run_dir)
    cfg = load_model_config(run / "config.yaml")
    model = build_model(cfg).load(run / "model")

    test_p = Path(test_path) if test_path else data_dir("processed") / "disclosure_testset.parquet"
    test = read_parquet(test_p)

    # Score each post
    proba = model.predict_proba(test)
    if proba.ndim == 2:
        if target not in (cfg.targets or [cfg.target]):
            raise typer.Exit(f"Target {target} not in model targets")
        proba = proba[:, (cfg.targets or [cfg.target]).index(target)]
    score_col = f"score_{target}"
    test_with_scores = test.copy()
    test_with_scores[score_col] = proba

    # Two reports: with and without masking disclosure posts (always emit both,
    # the table tells the story).
    rep_full = evaluate_user_level(
        test_with_scores, score_col=score_col, target=target,
        aggregation=aggregation, mask_disclosure_posts=False,
    )
    rep_masked = evaluate_user_level(
        test_with_scores, score_col=score_col, target=target,
        aggregation=aggregation, mask_disclosure_posts=True,
    )

    out_dir = run / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target,
        "model": cfg.name,
        "with_disclosure_posts": rep_full,
        "without_disclosure_posts": rep_masked,
    }
    (out_dir / f"{cfg.name}__{target}__disclosure_userlevel.json").write_text(
        json.dumps(payload, indent=2)
    )
    write_parquet(
        test_with_scores[
            ["author_hash", "subreddit", "id", "kind", "user_group",
             f"user_{target}", score_col, "is_disclosure_post"]
        ],
        out_dir / f"{cfg.name}__{target}__disclosure_predictions.parquet",
    )

    # Compact terminal display
    from rich.table import Table

    table = Table(title=f"{cfg.name} — user-level disclosure eval ({target}, agg={aggregation})")
    table.add_column("mode")
    for m in ("n_users", "n_positive_users", "precision", "recall", "f1", "auroc", "auprc"):
        table.add_column(m, justify="right")
    for mode_label, rep in (("with disclosure", rep_full), ("disclosure masked", rep_masked)):
        table.add_row(
            mode_label,
            f"{rep['n_users']}",
            f"{rep['n_positive_users']}",
            f"{rep['precision']:.3f}",
            f"{rep['recall']:.3f}",
            f"{rep['f1']:.3f}",
            f"{rep.get('auroc', float('nan')):.3f}",
            f"{rep.get('auprc', float('nan')):.3f}",
        )
    console.print(table)


@app.command("erisk-load")
def erisk_load(
    path: str = typer.Argument(..., help="Path to an eRisk file or directory"),
    task: str = typer.Option("auto", help="auto | task1 | task2"),
    out_path: str = typer.Option(None, help="Output parquet path"),
) -> None:
    """Load an eRisk 2025 file (Task 1 TREC or Task 2 JSON) into a parquet."""
    from pathlib import Path as _P

    from src.collection.erisk_loader import (
        load_task1,
        load_task1_dir,
        load_task2,
        load_task2_dir,
    )
    from src.utils.io import write_parquet

    p = _P(path)

    # Auto-detect format
    if task == "auto":
        if p.is_dir():
            json_files = list(p.glob("*.json"))
            task = "task2" if json_files else "task1"
        else:
            task = "task2" if p.suffix.lower() == ".json" else "task1"

    if task == "task1":
        df = load_task1_dir(p) if p.is_dir() else load_task1(p)
    elif task == "task2":
        df = load_task2_dir(p) if p.is_dir() else load_task2(p)
    else:
        raise typer.Exit(f"Unknown task: {task}")

    out = _P(out_path) if out_path else data_dir("external") / f"erisk_{task}.parquet"
    write_parquet(df, out)
    console.print(f"[green]Loaded {len(df):,} rows → {out}[/green]")


@app.command("erisk-eval")
def erisk_eval(
    predictions: str = typer.Argument(..., help="Per-post predictions parquet"),
    user_col: str = typer.Option("author_hash"),
    label_col: str = typer.Option("label_anxiety"),
    score_col: str = typer.Option("score_anxiety"),
    threshold: float = typer.Option(0.5),
    require_consecutive: int = typer.Option(1, help="Posts above threshold required to commit"),
) -> None:
    """Compute eRisk-style per-user metrics (ERDE_5, ERDE_50, F_latency, P@k) from per-post predictions."""
    import json

    from src.evaluation.erisk_metrics import (
        decisions_from_per_post_predictions,
        erisk_report,
    )
    from src.utils.io import read_parquet

    df = read_parquet(predictions)
    decisions = decisions_from_per_post_predictions(
        df,
        user_col=user_col,
        true_label_col=label_col,
        score_col=score_col,
        threshold=threshold,
        require_consecutive=require_consecutive,
    )
    report = erisk_report(decisions)
    console.print(json.dumps(report, indent=2))


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
