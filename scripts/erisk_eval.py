"""Idea 6 — eRisk early-detection metrics on the disclosure test set.

Reframes the per-post classifier as an early-detection system over user timelines:
for each model's disclosure predictions, MASK the disclosure post (detect from the
rest), order each user's posts by time, flag at the first post crossing a fixed
threshold, and score ERDE5 / ERDE50 / latency-weighted-F1 / median latency against
the user-level disclosure label.

CPU only, seconds. Run:
  python scripts/erisk_eval.py
  python scripts/erisk_eval.py --threshold 0.5
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm

from src.evaluation.erisk import erde, first_crossing_decision, latency_weighted_f1
from src.utils.io import read_parquet

TESTSET = "data/processed/disclosure_testset.parquet"
GLOB = "experiments/runs/*/eval/*__disclosure_predictions.parquet"
FIG = Path("docs/figures/erisk.png")
OUTCSV = Path("experiments/erisk.csv")
DOC = Path("docs/erisk.md")


def _parse(path: str) -> tuple[str, str]:
    parts = Path(path).stem.split("__")
    return "__".join(parts[:-2]), parts[-2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-posts", type=int, default=1)
    args = ap.parse_args()

    ts = read_parquet(TESTSET)[["id", "created_utc"]]
    rows: list[dict] = []
    for path in sorted(glob.glob(GLOB)):
        model, target = _parse(path)
        df = read_parquet(path)
        score_c, user_c = f"score_{target}", f"user_{target}"
        if score_c not in df.columns or user_c not in df.columns:
            continue
        df = df[df["is_disclosure_post"] == 0].merge(ts, on="id", how="left")
        df = df.sort_values("created_utc")

        y_true, decision, latency = [], [], []
        groups = list(df.groupby("author_hash", sort=False))
        for _, g in tqdm(groups, desc=f"{model}:{target}", unit="user", leave=False):
            if len(g) < args.min_posts:
                continue
            d, lat = first_crossing_decision(g[score_c].to_numpy(), args.threshold)
            y_true.append(int(g[user_c].iloc[0]))
            decision.append(d)
            latency.append(lat)
        if not y_true or sum(y_true) == 0:
            continue
        y_true, decision, latency = np.array(y_true), np.array(decision), np.array(latency)
        lwf = latency_weighted_f1(y_true, decision, latency)
        rows.append({
            "model": model, "target": target, "threshold": args.threshold,
            "n_users": int(len(y_true)), "n_pos": int(y_true.sum()),
            "erde_5": round(erde(y_true, decision, latency, o=5), 4),
            "erde_50": round(erde(y_true, decision, latency, o=50), 4),
            "precision": round(float(precision_score(y_true, decision, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, decision, zero_division=0)), 4),
            "f1": round(lwf["f1"], 4),
            "latency_weighted_f1": round(lwf["latency_weighted_f1"], 4),
            "median_latency": lwf["median_latency"],
        })

    if not rows:
        print("No disclosure predictions found.")
        return

    out = pd.DataFrame(rows).sort_values(["target", "model"]).reset_index(drop=True)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    # figure: ERDE_5 / ERDE_50 per model-target (lower is better)
    anx = out[out["target"] == "anxiety"]
    plot_df = anx if not anx.empty else out
    labels = [f"{r.model}\n({r.target})" for r in plot_df.itertuples()]
    x = np.arange(len(plot_df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(plot_df)), 5))
    ax.bar(x - w / 2, plot_df["erde_5"], w, label="ERDE₅ (strict)", color="#C44E52")
    ax.bar(x + w / 2, plot_df["erde_50"], w, label="ERDE₅₀ (lenient)", color="#4C72B0")
    for i, (e5, e50) in enumerate(zip(plot_df["erde_5"], plot_df["erde_50"])):
        ax.text(i - w / 2, e5 + 0.002, f"{e5:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, e50 + 0.002, f"{e50:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ERDE (lower is better)")
    ax.set_title(f"Idea 6 — eRisk early detection (threshold={args.threshold})")
    ax.legend()
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    cols = ["model", "target", "n_users", "n_pos", "erde_5", "erde_50",
            "precision", "recall", "f1", "latency_weighted_f1", "median_latency"]
    md = [
        "# eRisk early-detection metrics (disclosure test set)",
        "",
        f"Per-post classifier reframed as an early-detection system over user timelines "
        f"(threshold={args.threshold}). The disclosure post is **masked** — the system must "
        f"detect from the rest of the user's stream. Decision = flag at the first post crossing "
        f"the threshold. **ERDE₅/₅₀ lower is better**; `median_latency` = posts-to-detection on "
        f"true positives.",
        "",
        "_Regenerate: `python scripts/erisk_eval.py`_",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md += ["", "![eRisk ERDE](figures/erisk.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print(out[cols].to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
