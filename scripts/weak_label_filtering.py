"""Idea 13 — Whisper-style weak-label filtering.

Cleans the noisy anxiety weak labels via confident learning: get out-of-fold model
scores, flag examples where the model confidently disagrees with the weak label
(likely mislabels), remove them, and retrain. Tests whether the cleaned model
detects anxiety BETTER on the held-out self-disclosure test set (user-level AUROC) —
the disclosure users are excluded from training, so the eval is honest.

CPU, a few minutes. Run:
  python scripts/weak_label_filtering.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from src.labeling.filtering import confident_label_issues
from src.models.registry import build_model
from src.utils.config import load_model_config
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
DISC = "data/processed/disclosure_testset.parquet"
TARGET = "anxiety"
SEED = 42
FIG = Path("docs/figures/weak_label_filtering.png")
OUTCSV = Path("experiments/weak_label_filtering.csv")
DOC = Path("docs/weak_label_filtering.md")


def _user_auroc(disc, scores):
    g = disc.assign(_s=scores).groupby("author_hash").agg(_s=("_s", "mean"), _y=(f"user_{TARGET}", "max"))
    return float(roc_auc_score(g["_y"], g["_s"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pool", type=int, default=80000)
    ap.add_argument("--low", type=float, default=0.10)
    ap.add_argument("--high", type=float, default=0.90)
    args = ap.parse_args()

    disc = read_parquet(DISC)
    disc = disc[disc["is_disclosure_post"] == 0].copy()
    disc_authors = set(disc["author_hash"].dropna())

    pool = read_parquet(DATA)
    pool = pool[(pool["clean_text"].astype(str).str.len() >= 30) & ~pool["author_hash"].isin(disc_authors)].reset_index(drop=True)
    if len(pool) > args.max_pool:
        pool = pool.sample(args.max_pool, random_state=SEED).reset_index(drop=True)
    y = (pool[f"label_{TARGET}"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
    print(f"train pool={len(pool):,} (weak-positives={int(y.sum())})  disclosure eval posts={len(disc):,}")

    cfg = load_model_config("configs/models/baseline.yaml")

    # out-of-fold scores
    oof = np.zeros(len(pool))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in tqdm(list(skf.split(pool, y)), desc="OOF folds", unit="fold"):
        m = build_model(cfg).fit(pool.iloc[tr_idx].reset_index(drop=True))
        oof[va_idx] = np.asarray(m.predict_proba(pool.iloc[va_idx].reset_index(drop=True))).ravel()

    mask, counts = confident_label_issues(y, oof, low=args.low, high=args.high)
    print(f"flagged label issues: {counts}")

    # train original vs cleaned, evaluate on disclosure test (user-level AUROC)
    model_orig = build_model(cfg).fit(pool)
    auroc_orig = _user_auroc(disc, np.asarray(model_orig.predict_proba(disc)).ravel())
    cleaned = pool[~mask].reset_index(drop=True)
    model_clean = build_model(cfg).fit(cleaned)
    auroc_clean = _user_auroc(disc, np.asarray(model_clean.predict_proba(disc)).ravel())
    print(f"disclosure user-level AUROC: original {auroc_orig:.4f} -> cleaned {auroc_clean:.4f}")

    rows = [
        {"setting": "original (all weak labels)", "n_train": len(pool), "flagged_removed": 0,
         "disclosure_user_auroc": round(auroc_orig, 4)},
        {"setting": "cleaned (confident issues removed)", "n_train": len(cleaned), "flagged_removed": int(mask.sum()),
         "disclosure_user_auroc": round(auroc_clean, 4)},
    ]
    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    # example flagged posts (most confident disagreements)
    pool_f = pool.assign(oof=oof, weak=y, flagged=mask)
    ex_fp = pool_f[(pool_f.weak == 1) & pool_f.flagged].nsmallest(5, "oof")
    ex_fn = pool_f[(pool_f.weak == 0) & pool_f.flagged].nlargest(5, "oof")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([0, 1], [auroc_orig, auroc_clean], color=["#7f7f7f", "#55A868"])
    for i, v in enumerate([auroc_orig, auroc_clean]):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["original\n(all weak labels)", f"cleaned\n(−{int(mask.sum())} issues)"])
    ax.set_ylabel("disclosure user-level AUROC")
    lo = min(auroc_orig, auroc_clean); ax.set_ylim(max(0, lo - 0.03), max(auroc_orig, auroc_clean) + 0.02)
    ax.set_title("Idea 13 — weak-label filtering effect on disclosure detection")
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    md = [
        "# Whisper-style weak-label filtering",
        "",
        "Confident-learning cleanup of the anxiety weak labels: out-of-fold scores flag examples "
        f"where the model confidently disagrees with the weak label (score < {args.low} for "
        f"weak-positives; > {args.high} for weak-negatives). Remove them, retrain, and evaluate on "
        "the held-out self-disclosure test set (disclosure users excluded from training). "
        "`src/labeling/filtering.py`, `scripts/weak_label_filtering.py`.",
        "",
        f"- Flagged: **{counts['total_flagged']}** ({counts['likely_false_pos']} likely false-positives, "
        f"{counts['likely_false_neg']} likely false-negatives) of {len(pool):,}.",
        "",
        "_Regenerate: `python scripts/weak_label_filtering.py`_",
        "",
        "| " + " | ".join(out.columns) + " |",
        "|" + "|".join(["---"] * len(out.columns)) + "|",
    ]
    for _, r in out.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in out.columns) + " |")
    md += ["", "![weak-label filtering](figures/weak_label_filtering.png)", "",
           "## Example flagged weak-POSITIVES the model rejects (likely off-topic in an anxiety sub)"]
    for _, r in ex_fp.iterrows():
        md.append(f"- [{r['subreddit']}, oof={r['oof']:.3f}] {str(r['clean_text'])[:160]}")
    md += ["", "## Example flagged weak-NEGATIVES the model flags (likely anxiety in a neutral sub)"]
    for _, r in ex_fn.iterrows():
        md.append(f"- [{r['subreddit']}, oof={r['oof']:.3f}] {str(r['clean_text'])[:160]}")
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out.to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
