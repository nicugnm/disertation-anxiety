"""Run an evaluation pass: load test set, compute metrics, write report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.error_analysis import add_predictions, confusion_by_subgroup, length_effect
from src.evaluation.metrics import full_report
from src.models.base import BaseModel
from src.utils.config import data_dir
from src.utils.io import write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)


def evaluate_model(
    model: BaseModel,
    test: pd.DataFrame,
    target: str,
    out_dir: str | Path,
    name: str | None = None,
) -> dict:
    """Run model on test, compute metrics, save predictions + per-subreddit breakdown."""
    name = name or model.name
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    proba = model.predict_proba(test)
    if proba.ndim == 2:
        # Multi-target model — pick the column for `target`
        if target not in model.targets:
            raise ValueError(f"Target {target} not in model targets {model.targets}")
        proba = proba[:, model.targets.index(target)]

    y = (test[f"label_{target}"].astype(float).fillna(0.0) >= 0.5).astype(int).values
    report = full_report(y, proba)
    log.info("eval.metrics", model=name, target=target, **{k: v for k, v in report.items() if not k.endswith("_ci_lo") and not k.endswith("_ci_hi")})

    # Save report
    (out / f"{name}__{target}__metrics.json").write_text(json.dumps(report, indent=2))

    # Save predictions for downstream analysis
    pred_df = add_predictions(test.copy(), np.array(proba), target, threshold=report["threshold"])
    write_parquet(
        pred_df[["id", "subreddit", f"label_{target}", f"score_{target}", f"pred_{target}", f"bucket_{target}"]],
        out / f"{name}__{target}__predictions.parquet",
    )

    # Per-subreddit breakdown
    by_sub = confusion_by_subgroup(pred_df, target, "subreddit")
    by_sub.to_csv(out / f"{name}__{target}__by_subreddit.csv", index=False)

    # Length effect
    by_len = length_effect(pred_df, target)
    by_len.to_csv(out / f"{name}__{target}__by_length.csv", index=False)

    return report


def aggregate_reports(out_dir: str | Path) -> pd.DataFrame:
    """Walk output directory, collect all model__target__metrics.json into one table."""
    rows: list[dict] = []
    for fp in Path(out_dir).glob("*__metrics.json"):
        name = fp.stem.replace("__metrics", "")
        try:
            model_name, target = name.split("__")
        except ValueError:
            continue
        with fp.open() as f:
            data = json.load(f)
        rows.append({"model": model_name, "target": target, **data})
    return pd.DataFrame(rows).sort_values(["target", "f1"], ascending=[True, False])


def default_eval_dir() -> Path:
    return data_dir() / ".." / "experiments" / "eval"
