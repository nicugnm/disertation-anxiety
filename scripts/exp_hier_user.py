"""Experiment 11 — hierarchical user-level model vs naive aggregation.

Trains HierUserModel (attention | mean aggregator over a user's post stream) on
author-grouped corpus posts with weak user labels (user-positive if any post is
weak-positive), excluding the held-out disclosure-test authors, then evaluates at
the USER level on the self-disclosure test set (disclosure posts masked) — the one
place cheap TF-IDF still competes (~0.74 user-AUROC). Reports per-target user AUROC/F1
for the attention vs mean aggregator.

GPU. Run:
  python scripts/exp_hier_user.py --aggregators attention,mean --max-train-posts 150000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.evaluation.metrics import best_threshold_f1
from src.models.registry import build_model
from src.utils.config import load_model_config
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
DISC = "data/processed/disclosure_testset.parquet"
SEED = 42
TARGETS = ["anxiety", "health_anxiety", "depression"]
OUTCSV = Path("experiments/hier_user.csv")
DOC = Path("docs/hier_user.md")
FIG = Path("docs/figures/hier_user.png")


def _user_eval(model, disc, rows, agg_name):
    """User-level AUROC/F1 per target on the disclosure test set (disclosure posts masked)."""
    masked = disc[disc["is_disclosure_post"] == 0].copy()
    proba = np.asarray(model.predict_proba(masked))           # broadcast per author
    masked = masked.assign(_aid=masked["author_hash"].to_numpy())
    for i, t in enumerate(TARGETS):
        g = masked.assign(_s=proba[:, i]).groupby("_aid").agg(_s=("_s", "first"), _y=(f"user_{t}", "max"))
        y, s = g["_y"].to_numpy().astype(int), g["_s"].to_numpy()
        if y.sum() < 5 or (y == 0).sum() < 5:
            continue
        thr = best_threshold_f1(y, s)[0]
        rows.append({"aggregator": agg_name, "target": t, "n_users": int(len(y)), "n_pos": int(y.sum()),
                     "user_auroc": round(float(roc_auc_score(y, s)), 4),
                     "user_f1": round(float(f1_score(y, (s >= thr).astype(int), zero_division=0)), 4)})
        print(f"  {agg_name}/{t}: user-AUROC {rows[-1]['user_auroc']}  F1 {rows[-1]['user_f1']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregators", default="attention,mean")
    ap.add_argument("--max-train-posts", type=int, default=150000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--min-posts", type=int, default=2)
    args = ap.parse_args()

    disc = read_parquet(DISC)
    disc_authors = set(disc["author_hash"].dropna())
    df = read_parquet(DATA)
    df = df[(df["clean_text"].astype(str).str.len() >= 30) & df["author_hash"].notna()
            & ~df["author_hash"].isin(disc_authors)].reset_index(drop=True)
    # keep authors with >= min_posts (user-level modelling needs multi-post users)
    counts = df["author_hash"].value_counts()
    keep = counts[counts >= args.min_posts].index
    df = df[df["author_hash"].isin(keep)]
    if len(df) > args.max_train_posts:
        df = df.sample(args.max_train_posts, random_state=SEED)
    df = df.reset_index(drop=True)
    print(f"train pool: {len(df):,} posts / {df['author_hash'].nunique():,} authors  |  "
          f"disclosure eval users: {disc['author_hash'].nunique():,}")

    rows: list[dict] = []
    aggs = [a.strip() for a in args.aggregators.split(",") if a.strip()]
    for ai, agg in enumerate(aggs, 1):
        cfg = load_model_config("configs/models/hier_user.yaml")
        cfg.extra.setdefault("user_model", {})["aggregator"] = agg
        cfg.extra.setdefault("train", {})["num_train_epochs"] = args.epochs
        print(f"\n=== [{ai}/{len(aggs)}] training hierarchical ({agg}) ===", flush=True)
        model = build_model(cfg).fit(df, val=None)
        _user_eval(model, disc, rows, agg)
        pd.DataFrame(rows).to_csv(OUTCSV, index=False)

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    if not out.empty:
        piv = out.pivot_table(index="target", columns="aggregator", values="user_auroc")
        fig, ax = plt.subplots(figsize=(8, 5))
        piv.plot(kind="bar", ax=ax)
        ax.axhline(0.74, ls="--", color="k", lw=1, label="TF-IDF mean-pool ≈ 0.74")
        ax.set_ylabel("user-level AUROC"); ax.set_ylim(0, 1.0)
        ax.set_title("Experiment 11 — hierarchical user-model (aggregator vs naive mean)")
        ax.legend(fontsize=8)
        fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG, dpi=130); plt.close(fig)

        cols = ["aggregator", "target", "n_users", "n_pos", "user_auroc", "user_f1"]
        md = [
            "# Experiment 11 — hierarchical user-level model",
            "",
            "HierUserModel (`src/models/hier.py`): frozen MentalRoBERTa post-encoder → learned "
            "attention/mean aggregation over a user's post stream → user head. Trained on "
            "author-grouped corpus posts (weak user labels), evaluated at the USER level on the "
            "self-disclosure test set (disclosure posts masked). Reference: TF-IDF mean-pool ≈ 0.74 user-AUROC.",
            "",
            "_Regenerate: `python scripts/exp_hier_user.py`_",
            "",
            "| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|",
        ]
        for r in rows:
            md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        md += ["", "![hierarchical user model](figures/hier_user.png)"]
        DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out.to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
