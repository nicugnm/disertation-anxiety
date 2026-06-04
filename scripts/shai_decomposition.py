"""Idea 11 — SHAI-item symptom decomposition.

Scores every post on the seven SHAI clinical dimensions (Salkovskis 2002), then:
  (1) profiles each subreddit's mean dimension rates (does r/HealthAnxiety load on
      bodily-vigilance + symptom-checking + difficulty-reassured more than r/Anxiety?),
  (2) correlates each dimension with the health-anxiety weak label (which SHAI
      constructs most discriminate health anxiety).
Connects the model/corpus to the clinical instrument's structure. CPU. Run:
  python scripts/shai_decomposition.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.features.shai import score_shai, shai_dimensions
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
SEED = 42
DIMS = shai_dimensions()
FIG = Path("docs/figures/shai_decomposition.png")
OUTCSV = Path("experiments/shai_decomposition.csv")
DOC = Path("docs/shai_decomposition.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-per-sub", type=int, default=2500)
    ap.add_argument("--min-sub-n", type=int, default=400)
    args = ap.parse_args()

    df = read_parquet(DATA)
    df = df[df["clean_text"].astype(str).str.len() >= 30].reset_index(drop=True)
    df = df.groupby("subreddit", group_keys=False).apply(
        lambda g: g.sample(min(len(g), args.cap_per_sub), random_state=SEED)
    ).reset_index(drop=True)
    print(f"scoring SHAI dimensions on {len(df):,} posts...")

    scores = [score_shai(t) for t in tqdm(df["clean_text"].astype(str).tolist(), desc="SHAI", unit="post")]
    S = pd.DataFrame(scores)
    S["subreddit"] = df["subreddit"].values
    ha = (df["label_health_anxiety"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy() if "label_health_anxiety" in df.columns else None

    # per-subreddit mean dimension rates
    prof = S.groupby("subreddit")[DIMS].mean()
    counts = S.groupby("subreddit").size()
    prof = prof[counts >= args.min_sub_n]
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    prof.round(6).to_csv(OUTCSV)

    # correlation of each dimension with the health-anxiety label
    corrs = {}
    if ha is not None and ha.sum() > 10:
        for d in DIMS:
            v = S[d].to_numpy()
            corrs[d] = float(np.corrcoef(v, ha)[0, 1]) if v.std() > 0 else 0.0

    # ---- figure: heatmap (z-scored per dimension) + correlation bar ----
    order = prof.mean(axis=1).sort_values(ascending=False).index
    H = prof.loc[order]
    Z = (H - H.mean(axis=0)) / (H.std(axis=0) + 1e-9)
    fig = plt.figure(figsize=(12, max(6, 0.32 * len(H))))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.0], wspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(Z.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(DIMS))); ax.set_xticklabels(DIMS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index, fontsize=7)
    ax.set_title("SHAI dimension profile by subreddit (z-scored per dimension)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    if corrs:
        ax2 = fig.add_subplot(gs[0, 1])
        cs = pd.Series(corrs).sort_values()
        ax2.barh(range(len(cs)), cs.values, color=["#4C72B0" if v < 0 else "#C44E52" for v in cs.values])
        ax2.set_yticks(range(len(cs))); ax2.set_yticklabels(cs.index, fontsize=8)
        ax2.yaxis.tick_right(); ax2.yaxis.set_label_position("right")
        ax2.axvline(0, color="k", lw=0.8)
        ax2.set_title("corr with\nhealth_anxiety label", fontsize=9)
    fig.suptitle("Idea 11 — SHAI-item symptom decomposition", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    # ---- docs ----
    md = [
        "# SHAI-item symptom decomposition",
        "",
        "Each post scored on the seven SHAI clinical dimensions (Salkovskis 2002), then "
        "averaged per subreddit and correlated with the health-anxiety weak label. "
        "`src/features/shai.py`. Rates = matched terms/phrases per token.",
        "",
        "_Regenerate: `python scripts/shai_decomposition.py`_",
        "",
    ]
    if corrs:
        md += ["## SHAI dimension ↔ health-anxiety label (point-biserial r)", "",
               "| dimension | r |", "|---|---:|"]
        for d, r in sorted(corrs.items(), key=lambda kv: -kv[1]):
            md.append(f"| {d} | {r:+.4f} |")
        md.append("")
    key = [s for s in ("HealthAnxiety", "Anxiety", "socialanxiety", "PanicAttack",
                       "depression", "COVID19positive", "cooking", "personalfinance") if s in prof.index]
    if key:
        md += ["## Mean dimension rate — selected subreddits (×1000)", "",
               "| subreddit | " + " | ".join(DIMS) + " |", "|" + "|".join(["---"] * (len(DIMS) + 1)) + "|"]
        for s in key:
            md.append(f"| {s} | " + " | ".join(f"{prof.loc[s, d] * 1000:.2f}" for d in DIMS) + " |")
    md += ["", "![SHAI decomposition](figures/shai_decomposition.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    if corrs:
        print("\nSHAI dimension correlation with health_anxiety label:")
        for d, r in sorted(corrs.items(), key=lambda kv: -kv[1]):
            print(f"  {d:24s} {r:+.4f}")
    if key:
        print("\nMean dimension rate (x1000), HealthAnxiety vs Anxiety:")
        print((prof.loc[[s for s in ('HealthAnxiety', 'Anxiety') if s in prof.index]] * 1000).round(2).to_string())
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
