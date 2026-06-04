"""Idea 3 — per-subreddit decision-threshold calibration.

The saved eval predictions only cover the stale 10-subreddit snapshot, so this
trains the TF-IDF baseline on an author-disjoint split of the CURRENT corpus and
predicts anxiety on a held-out set spanning all subreddits, then compares a single
GLOBAL best-F1 threshold against PER-SUBREDDIT thresholds (global fallback for
sparse subreddits). Author-disjoint train / calibration / test (no author shared).

CPU only (~a few minutes to train TF-IDF). Run:
  python scripts/threshold_calibration.py
  python scripts/threshold_calibration.py --max-train 200000 --max-eval 80000 --min-pos 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from src.evaluation.thresholds import (
    apply_per_subreddit,
    fit_per_subreddit_thresholds,
    macro_f1_by_subreddit,
)
from src.models.registry import build_model
from src.utils.config import load_model_config
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
TARGET = "anxiety"
SEED = 42
FIG = Path("docs/figures/threshold_calibration.png")
OUTCSV = Path("experiments/threshold_calibration.csv")
DOC = Path("docs/threshold_calibration.md")


def _author_disjoint_3way(df: pd.DataFrame):
    g = df["author_hash"].values
    tr, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED).split(df, groups=g))
    rest_df = df.iloc[rest].reset_index(drop=True)
    gr = rest_df["author_hash"].values
    ca, te = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(rest_df, groups=gr))
    return (df.iloc[tr].reset_index(drop=True),
            rest_df.iloc[ca].reset_index(drop=True),
            rest_df.iloc[te].reset_index(drop=True))


def _cap(df, n):
    return df.sample(n=n, random_state=SEED).reset_index(drop=True) if n and len(df) > n else df


def _predict_bar(model, df, desc, chunk=5000):
    out = []
    for i in tqdm(range(0, len(df), chunk), desc=desc, unit="chunk"):
        out.append(np.asarray(model.predict_proba(df.iloc[i:i + chunk])).ravel())
    return np.concatenate(out) if out else np.array([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=200000)
    ap.add_argument("--max-eval", type=int, default=80000)
    ap.add_argument("--min-pos", type=int, default=25)
    args = ap.parse_args()

    df = read_parquet(DATA)
    df = df[(df["clean_text"].astype(str).str.len() >= 30) & df["author_hash"].notna()].reset_index(drop=True)
    train, calib, test = _author_disjoint_3way(df)
    train, calib, test = _cap(train, args.max_train), _cap(calib, args.max_eval), _cap(test, args.max_eval)
    print(f"author-disjoint: train={len(train):,}  calib={len(calib):,}  test={len(test):,}")

    cfg = load_model_config("configs/models/baseline.yaml")
    print("training TF-IDF baseline (CPU, ~2-4 min)...")
    model = build_model(cfg).fit(train)

    yc = (calib[f"label_{TARGET}"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
    yt = (test[f"label_{TARGET}"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
    pc = _predict_bar(model, calib, "predict:calib")
    pt = _predict_bar(model, test, "predict:test")
    sc_calib = calib["subreddit"].astype(str).to_numpy()
    sc_test = test["subreddit"].astype(str).to_numpy()

    # Fit thresholds on calibration, apply to test
    thresholds, global_thr = fit_per_subreddit_thresholds(pc, yc, sc_calib, min_pos=args.min_pos)
    preds_global = (pt >= global_thr).astype(int)
    preds_persub = apply_per_subreddit(pt, sc_test, thresholds, global_thr)

    f1_g = macro_f1_by_subreddit(yt, preds_global, sc_test, min_pos=args.min_pos)
    f1_p = macro_f1_by_subreddit(yt, preds_persub, sc_test, min_pos=args.min_pos)
    subs = sorted(f1_g)  # subreddits with enough positives in test to score
    tuned = {s for s in subs if thresholds.get(s, global_thr) != global_thr}

    rows = []
    for s in subs:
        m = sc_test == s
        rows.append({
            "subreddit": s, "n_test": int(m.sum()), "n_pos": int(yt[m].sum()),
            "prevalence": round(float(yt[m].mean()), 4),
            "global_thr": round(global_thr, 3), "subreddit_thr": round(float(thresholds.get(s, global_thr)), 3),
            "tuned": s in tuned,
            "f1_global": round(f1_g[s], 4), "f1_persub": round(f1_p[s], 4),
            "delta_f1": round(f1_p[s] - f1_g[s], 4),
        })
    out = pd.DataFrame(rows).sort_values("n_pos", ascending=False).reset_index(drop=True)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    macro_g = float(np.mean([f1_g[s] for s in subs]))
    macro_p = float(np.mean([f1_p[s] for s in subs]))
    pooled_g = f1_score(yt, preds_global, zero_division=0)
    pooled_p = f1_score(yt, preds_persub, zero_division=0)

    # Figure: (a) per-subreddit thresholds vs global, (b) per-subreddit F1 global vs per-sub
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    plot_subs = list(out["subreddit"])
    x = np.arange(len(plot_subs))
    axes[0].bar(x, [thresholds.get(s, global_thr) for s in plot_subs],
                color=["#C44E52" if s in tuned else "#B0B0B0" for s in plot_subs])
    axes[0].axhline(global_thr, ls="--", color="k", label=f"global thr = {global_thr:.3f}")
    axes[0].set_title("Per-subreddit best-F1 threshold\n(grey = fell back to global)")
    axes[0].set_ylabel("threshold"); axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_subs, rotation=60, ha="right", fontsize=8); axes[0].legend()
    w = 0.4
    axes[1].bar(x - w / 2, [f1_g[s] for s in plot_subs], w, label="global threshold", color="#4C72B0")
    axes[1].bar(x + w / 2, [f1_p[s] for s in plot_subs], w, label="per-subreddit threshold", color="#DD8452")
    axes[1].set_title(f"Anxiety F1 per subreddit\nmacro-F1: {macro_g:.3f} -> {macro_p:.3f}")
    axes[1].set_ylabel("F1"); axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_subs, rotation=60, ha="right", fontsize=8); axes[1].legend()
    fig.suptitle("Idea 3 — per-subreddit threshold calibration (TF-IDF, anxiety)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    cols = ["subreddit", "n_test", "n_pos", "prevalence", "global_thr", "subreddit_thr", "tuned",
            "f1_global", "f1_persub", "delta_f1"]
    md = [
        "# Per-subreddit threshold calibration",
        "",
        f"TF-IDF baseline (anxiety) on an **author-disjoint** train/calibration/test split of the current "
        f"corpus. A single global best-F1 threshold (**{global_thr:.3f}**) vs **per-subreddit** thresholds "
        f"(tuned where a subreddit has >= {args.min_pos} positives & negatives in calibration, else global "
        f"fallback). The model is identical; only the operating point changes.",
        "",
        f"- **Macro-F1 (mean over scored subreddits): {macro_g:.3f} -> {macro_p:.3f}** "
        f"(Δ {macro_p - macro_g:+.3f})",
        f"- Pooled F1: {pooled_g:.3f} -> {pooled_p:.3f}",
        f"- Subreddits tuned (vs global fallback): {len(tuned)} / {len(subs)}",
        "",
        "![per-subreddit thresholds](figures/threshold_calibration.png)",
        "",
        "_Regenerate: `python scripts/threshold_calibration.py`_",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    DOC.write_text("\n".join(md), encoding="utf-8")

    print(out[cols].to_string(index=False))
    print(f"\nmacro-F1: {macro_g:.3f} -> {macro_p:.3f}   pooled-F1: {pooled_g:.3f} -> {pooled_p:.3f}")
    print(f"tuned {len(tuned)}/{len(subs)} subreddits.  Wrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
