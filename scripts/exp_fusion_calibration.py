"""Phase 1C — recalibration of the winning fusion model.

Trains the fusion+focal model on an author-disjoint train split, then extends our
two calibration extensions to it: (1) temperature scaling per target (ECE before vs
after, on a held-out calibration/test split) and (2) per-subreddit thresholds for
anxiety (macro-F1 global vs per-subreddit). Confirms whether the new SOTA
architecture inherits the transformers' good calibration and benefits from
population-specific operating points.

GPU (~15-20 min, trains one model). Run:
  python scripts/exp_fusion_calibration.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.evaluation.calibration import TemperatureScaler
from src.evaluation.metrics import expected_calibration_error
from src.evaluation.thresholds import (
    apply_per_subreddit,
    fit_per_subreddit_thresholds,
    macro_f1_by_subreddit,
)
from src.models.registry import build_model
from src.utils.config import load_model_config
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
SEED = 42
TARGETS = ["anxiety", "health_anxiety", "depression", "suicidality"]
OUTCSV = Path("experiments/fusion_calibration.csv")
DOC = Path("docs/fusion_calibration.md")
FIG = Path("docs/figures/fusion_calibration.png")


def _cap(df, n):
    return df.sample(n=n, random_state=SEED).reset_index(drop=True) if n and len(df) > n else df


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--max-eval", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    df = read_parquet(DATA)
    df = df[(df["clean_text"].astype(str).str.len() >= 30) & df["author_hash"].notna()].reset_index(drop=True)
    tr, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED).split(df, groups=df["author_hash"].values))
    rest_df = df.iloc[rest].reset_index(drop=True)
    ca, te = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(rest_df, groups=rest_df["author_hash"].values))
    train = _cap(df.iloc[tr].reset_index(drop=True), args.max_train)
    calib = _cap(rest_df.iloc[ca].reset_index(drop=True), args.max_eval)
    test = _cap(rest_df.iloc[te].reset_index(drop=True), args.max_eval)
    print(f"author-disjoint: train={len(train):,} calib={len(calib):,} test={len(test):,}")

    cfg = load_model_config("configs/models/fusion_multitask.yaml")  # fusion + focal on by default
    cfg.extra.setdefault("train", {})["num_train_epochs"] = args.epochs
    print("training fusion+focal model...")
    model = build_model(cfg).fit(train, val=None)
    pc = np.asarray(model.predict_proba(calib))
    pt = np.asarray(model.predict_proba(test))

    rows = []
    for i, t in enumerate(TARGETS):
        yc = (calib[f"label_{t}"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
        yt = (test[f"label_{t}"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
        if yc.sum() < 5 or yt.sum() < 5:
            continue
        scaler = TemperatureScaler().fit(pc[:, i], yc)
        ece_b = expected_calibration_error(yt, pt[:, i])
        ece_a = expected_calibration_error(yt, scaler.transform(pt[:, i]))
        rows.append({"target": t, "temperature": round(scaler.temperature, 3),
                     "ece_before": round(ece_b, 4), "ece_after": round(ece_a, 4)})
        print(f"  {t}: T={scaler.temperature:.3f}  ECE {ece_b:.4f} -> {ece_a:.4f}")

    # per-subreddit thresholds for anxiety on the test split
    ai = TARGETS.index("anxiety")
    yc = (calib["label_anxiety"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
    yt = (test["label_anxiety"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
    thr, gthr = fit_per_subreddit_thresholds(pc[:, ai], yc, calib["subreddit"].to_numpy(), min_pos=25)
    pred_g = (pt[:, ai] >= gthr).astype(int)
    pred_p = apply_per_subreddit(pt[:, ai], test["subreddit"].to_numpy(), thr, gthr)
    f1g = macro_f1_by_subreddit(yt, pred_g, test["subreddit"].to_numpy(), min_pos=25)
    f1p = macro_f1_by_subreddit(yt, pred_p, test["subreddit"].to_numpy(), min_pos=25)
    macro_g = float(np.mean(list(f1g.values()))) if f1g else float("nan")
    macro_p = float(np.mean(list(f1p.values()))) if f1p else float("nan")
    print(f"\nanxiety per-subreddit macro-F1: global {macro_g:.3f} -> per-sub {macro_p:.3f}")

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(out)); w = 0.38
    ax.bar(x - w / 2, out["ece_before"], w, label="ECE before", color="#DD8452")
    ax.bar(x + w / 2, out["ece_after"], w, label="ECE after (temp. scaling)", color="#55A868")
    for i, (b, a) in enumerate(zip(out["ece_before"], out["ece_after"])):
        ax.text(i - w / 2, b + 0.001, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, a + 0.001, f"{a:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(out["target"], fontsize=9)
    ax.set_ylabel("ECE"); ax.set_title("Phase 1C — fusion+focal calibration (temperature scaling)")
    ax.legend()
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    md = [
        "# Phase 1C — recalibration of the fusion+focal model",
        "",
        "Temperature scaling (per target) + per-subreddit thresholds applied to the winning "
        "`fusion+focal` model (`src/models/fusion.py`), author-disjoint train/calib/test. "
        "`scripts/exp_fusion_calibration.py`.",
        "",
        f"- **anxiety per-subreddit macro-F1: {macro_g:.3f} -> {macro_p:.3f}** (global vs per-subreddit thresholds)",
        "",
        "| target | temperature | ECE before | ECE after |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['target']} | {r['temperature']} | {r['ece_before']} | {r['ece_after']} |")
    md += ["", "![fusion calibration](figures/fusion_calibration.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out.to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
