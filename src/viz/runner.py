"""Run every standard plot for the corpus + a model run.

Outputs PNGs into `docs/figures/`. Idempotent — run after collection,
preprocessing, labeling, training, or any combination.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import data_dir
from src.utils.io import read_parquet
from src.utils.logging import get_logger
from src.viz.plots import (
    plot_calibration,
    plot_confusion,
    plot_corpus_overview,
    plot_label_cooccurrence,
    plot_label_distribution,
    plot_length_distribution,
    plot_linguistic_markers,
    plot_pr_roc,
    plot_subreddit_f1,
    plot_temporal,
    set_style,
)

log = get_logger(__name__)


def run_all(
    figures_dir: str | Path = "docs/figures",
    labeled_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Generate every plot we can given what's on disk."""
    set_style()
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Path] = {}

    labeled = Path(labeled_path) if labeled_path else (data_dir("processed") / "labeled.parquet")
    if labeled.exists():
        df = read_parquet(labeled)
        produced["corpus_overview"] = plot_corpus_overview(df, out / "corpus_overview.png")
        produced["length_distribution"] = plot_length_distribution(df, out / "length_distribution.png")
        produced["temporal"] = plot_temporal(df, out / "temporal.png")
        produced["label_distribution"] = plot_label_distribution(df, out / "label_distribution.png")
        produced["label_cooccurrence"] = plot_label_cooccurrence(df, out / "label_cooccurrence.png")
        log.info("viz.corpus_done", n_plots=len(produced))

    if run_dir:
        run_dir = Path(run_dir)
        eval_dir = run_dir / "eval"
        # Find first metrics file to extract the target name
        metrics_files = sorted(eval_dir.glob("*__metrics.json"))
        for mf in metrics_files:
            stem = mf.stem.replace("__metrics", "")
            try:
                model_name, target = stem.split("__")
            except ValueError:
                continue

            with mf.open() as f:
                metrics = json.load(f)
            preds_file = eval_dir / f"{model_name}__{target}__predictions.parquet"
            if not preds_file.exists():
                continue
            preds = read_parquet(preds_file)
            y = preds[f"label_{target}"].astype(float).fillna(0).values
            y_bin = (y >= 0.5).astype(int)
            score = preds[f"score_{target}"].values
            pred = preds[f"pred_{target}"].values

            produced[f"pr_roc__{target}"] = plot_pr_roc(
                y_bin, score, out / f"pr_roc__{target}.png",
                title=f"{model_name} — {target}"
            )
            produced[f"calibration__{target}"] = plot_calibration(
                y_bin, score, out / f"calibration__{target}.png"
            )
            produced[f"confusion__{target}"] = plot_confusion(
                y_bin, pred, out / f"confusion__{target}.png"
            )

            by_sub_file = eval_dir / f"{model_name}__{target}__by_subreddit.csv"
            if by_sub_file.exists():
                by_sub = pd.read_csv(by_sub_file)
                produced[f"subreddit_f1__{target}"] = plot_subreddit_f1(
                    by_sub, out / f"subreddit_f1__{target}.png",
                    title=f"{model_name} — {target} — F1 by subreddit"
                )
            log.info("viz.run_done", run=str(run_dir), target=target)

    # Linguistic markers if present
    for f in Path("experiments").glob("markers__*.csv"):
        target = f.stem.replace("markers__", "")
        markers = pd.read_csv(f)
        produced[f"markers__{target}"] = plot_linguistic_markers(
            markers, out / f"markers__{target}.png"
        )
        log.info("viz.markers_done", target=target)

    log.info("viz.all_done", n=len(produced), out=str(out))
    return produced
