"""Experiment 9 — does DANN reduce the cross-subreddit collapse?

Trains plain multi-task vs DANN(subreddit) vs DANN(group), then evaluates each at
a SINGLE operating point (threshold tuned once on the in-distribution val set and
reused everywhere) on three held sets:

  in_dist        — held-out authors from the training subreddits.
  cross_heldout  — anxiety-bearing subreddits HELD OUT of training (PanicAttack,
                   panicdisorder, agoraphobia). These have real anxiety positives,
                   so transfer F1/AUROC/AUPRC are measurable. THIS is the headline.
  neutral        — baseline subreddits (cooking, personalfinance, ...). ~0 anxiety
                   positives, so we report the predicted-positive RATE = the
                   false-positive rate. Lower is better; a subreddit-confounded
                   model over-flags these, DANN should reduce it.

DANN should (a) recover/raise cross_heldout F1 and (b) lower the neutral
false-positive rate, while keeping in_dist AUROC roughly constant.

GPU strongly recommended. Training is capped (--max-train) so 3 transformers
finish in a few hours rather than days on the full 744k corpus.

Run:
  python scripts/exp_dann_transfer.py                         # all 3 models
  python scripts/exp_dann_transfer.py --models multitask      # one at a time
  python scripts/exp_dann_transfer.py --max-train 60000 --max-eval 20000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from src.evaluation.metrics import best_threshold_f1
from src.models.registry import build_model
from src.utils.config import load_model_config, load_subreddits
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
OUT = Path("experiments")
SEED = 42
TARGETS = ["anxiety", "health_anxiety", "depression", "suicidality"]
MODELS = {
    "multitask": "configs/models/multitask.yaml",
    "dann_subreddit": "configs/models/dann_multitask_subreddit.yaml",
    "dann_group": "configs/models/dann_multitask_group.yaml",
}
# Anxiety-bearing subreddits HELD OUT of training to measure positive transfer to
# unseen subreddits (they have real anxiety positives, unlike the baseline subs).
HELDOUT_CROSS = ["panicattack", "panicdisorder", "agoraphobia"]


def _cap(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(n=n, random_state=SEED).reset_index(drop=True) if n and len(df) > n else df


def load_splits(max_train: int, max_eval: int, test_size: float = 0.15):
    subs = load_subreddits()
    baseline = {s.name.lower() for s in subs.subreddits if s.group == "baseline"}
    cross_names = set(HELDOUT_CROSS)
    df = read_parquet(DATA)
    df = df[df["clean_text"].astype(str).str.len() >= 30].copy()
    sl = df["subreddit"].astype(str).str.lower()
    train_pool = df[
        ~sl.isin(baseline) & ~sl.isin(cross_names) & df["author_hash"].notna()
    ].reset_index(drop=True)
    cross = df[sl.isin(cross_names)].reset_index(drop=True)   # held-out anxiety subs
    neutral = df[sl.isin(baseline)].reset_index(drop=True)    # neutral FP control
    # author-disjoint train/val within the training subreddits
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr, va = next(gss.split(train_pool, groups=train_pool["author_hash"]))
    train = _cap(train_pool.iloc[tr].reset_index(drop=True), max_train)
    val = _cap(train_pool.iloc[va].reset_index(drop=True), max_eval)
    cross = _cap(cross, max_eval)
    neutral = _cap(neutral, max_eval)
    return train, val, cross, neutral, sorted(cross_names), sorted(baseline)


def evaluate(model, df: pd.DataFrame, name: str, split: str, thresholds: dict | None = None):
    """Score a split. If `thresholds` is None, tune per-target threshold (best F1)
    here and return it; otherwise reuse the supplied operating points."""
    proba = model.predict_proba(df)
    rows: list[dict] = []
    thr_out: dict[str, float] = {}
    for i, t in enumerate(TARGETS):
        col = f"label_{t}"
        if col not in df.columns:
            continue
        y = (df[col].astype(float).fillna(0) >= 0.5).astype(int).values
        p = proba[:, i] if proba.ndim == 2 else proba
        npos, nneg = int(y.sum()), int((y == 0).sum())
        measurable = npos > 0 and nneg > 0
        if thresholds is not None:
            thr = thresholds.get(t, 0.5)
        elif measurable:
            thr, _ = best_threshold_f1(y, p)
        else:
            thr = 0.5
        thr_out[t] = thr
        pred = (p >= thr).astype(int)
        row = {"model": name, "split": split, "target": t, "n": len(y), "n_pos": npos,
               "thr": round(float(thr), 3), "pred_pos_rate": round(float(pred.mean()), 4)}
        if measurable:
            row |= {
                "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
                "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
                "auroc": round(float(roc_auc_score(y, p)), 4),
                "auprc": round(float(average_precision_score(y, p)), 4),
            }
        else:
            row |= {k: None for k in ("f1", "precision", "recall", "auroc", "auprc")}
        rows.append(row)
    return rows, thr_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--max-eval", type=int, default=20000)
    args = ap.parse_args()
    chosen = [m.strip() for m in args.models.split(",") if m.strip()]

    train, val, cross, neutral, cross_names, baseline = load_splits(args.max_train, args.max_eval)
    print(f"train={len(train):,}  val(in_dist)={len(val):,}  "
          f"cross_heldout({cross_names})={len(cross):,}  neutral(baseline)={len(neutral):,}")

    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for mi, name in enumerate(chosen, 1):
        cfg = load_model_config(MODELS[name])
        print(f"\n=== [{mi}/{len(chosen)}] training {name} ({cfg.name}) ===")
        model = build_model(cfg).fit(train, val=val)

        print(f"  evaluating {name}: in_dist ({len(val):,})  -> tuning thresholds")
        val_rows, thr = evaluate(model, val, name, "in_dist")
        print(f"  evaluating {name}: cross_heldout ({len(cross):,})  @ in_dist thresholds")
        cross_rows, _ = evaluate(model, cross, name, "cross_heldout", thr)
        print(f"  evaluating {name}: neutral baseline ({len(neutral):,})  @ in_dist thresholds")
        neutral_rows, _ = evaluate(model, neutral, name, "neutral", thr)
        results += val_rows + cross_rows + neutral_rows

        rd = Path("experiments/runs") / cfg.name
        rd.mkdir(parents=True, exist_ok=True)
        model.save(rd / "model")
        pd.DataFrame(results).to_csv(OUT / "exp_dann_transfer.csv", index=False)
        print(pd.DataFrame([r for r in results if r["model"] == name]).to_string(index=False))

    (OUT / "exp_dann_transfer.json").write_text(json.dumps(results, indent=2))
    df = pd.DataFrame(results)
    print("\n=== anxiety: positive transfer (in_dist vs cross_heldout) ===")
    a = df[(df.target == "anxiety") & (df.split.isin(["in_dist", "cross_heldout"]))]
    if not a.empty:
        print(a.pivot_table(index="model", columns="split", values=["f1", "auroc", "auprc"]).to_string())
    print("\n=== anxiety: false-positive rate on neutral baseline subs (lower is better) ===")
    nb = df[(df.target == "anxiety") & (df.split == "neutral")]
    if not nb.empty:
        print(nb[["model", "pred_pos_rate", "thr"]].to_string(index=False))
    print(f"\nResults -> {OUT / 'exp_dann_transfer.json'}  and  {OUT / 'exp_dann_transfer.csv'}")


if __name__ == "__main__":
    main()
