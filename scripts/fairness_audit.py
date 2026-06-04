"""Idea 8 — subgroup fairness audit.

Protected demographics are unavailable (anonymized corpus) and inferring them with
a classifier would be unreliable/biased — so we audit performance equity across
(a) post-length tertiles (well-powered, always available), and (b) SELF-REPORTED
gender / age extracted by regex (exploratory; ~3% coverage). TF-IDF anxiety model,
author-disjoint train/calibration/test; threshold tuned on calibration.

Reports per-group TPR/FPR/F1/selection-rate and fairness gaps (equal-opportunity,
equalized-odds, demographic-parity). CPU, a few minutes. Run:
  python scripts/fairness_audit.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from src.evaluation.fairness import extract_age, extract_gender, fairness_gaps, subgroup_metrics
from src.evaluation.metrics import best_threshold_f1
from src.models.registry import build_model
from src.utils.config import load_model_config
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
TARGET = "anxiety"
SEED = 42
FIG = Path("docs/figures/fairness.png")
OUTCSV = Path("experiments/fairness.csv")
DOC = Path("docs/fairness.md")


def _age_band(a):
    if a is None:
        return None
    return "13-24" if a < 25 else ("25-34" if a < 35 else "35+")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=150000)
    ap.add_argument("--max-eval", type=int, default=60000)
    args = ap.parse_args()

    df = read_parquet(DATA)
    df = df[(df["clean_text"].astype(str).str.len() >= 30) & df["author_hash"].notna()].reset_index(drop=True)
    tr, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED).split(df, groups=df["author_hash"].values))
    rest_df = df.iloc[rest].reset_index(drop=True)
    ca, te = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(rest_df, groups=rest_df["author_hash"].values))
    train, calib, test = df.iloc[tr].reset_index(drop=True), rest_df.iloc[ca].reset_index(drop=True), rest_df.iloc[te].reset_index(drop=True)
    if len(train) > args.max_train:
        train = train.sample(args.max_train, random_state=SEED).reset_index(drop=True)
    if len(calib) > args.max_eval:
        calib = calib.sample(args.max_eval, random_state=SEED).reset_index(drop=True)
    if len(test) > args.max_eval:
        test = test.sample(args.max_eval, random_state=SEED).reset_index(drop=True)
    print(f"author-disjoint: train={len(train):,} calib={len(calib):,} test={len(test):,}")

    print("training TF-IDF (CPU)...")
    model = build_model(load_model_config("configs/models/baseline.yaml")).fit(train)
    yc = (calib[f"label_{TARGET}"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
    thr = best_threshold_f1(yc, np.asarray(model.predict_proba(calib)).ravel())[0]
    print(f"tuned threshold = {thr:.3f}")

    y = (test[f"label_{TARGET}"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
    pred = (np.asarray(model.predict_proba(test)).ravel() >= thr).astype(int)

    print("extracting strata (length / self-reported gender / age)...")
    texts = test["clean_text"].astype(str).tolist()
    length = pd.qcut(test["clean_text"].astype(str).str.len(), 3, labels=["short", "medium", "long"]).astype(object).to_numpy()
    gender = np.array([extract_gender(t) for t in tqdm(texts, desc="gender", unit="post", leave=False)], dtype=object)
    age = np.array([_age_band(extract_age(t)) for t in tqdm(texts, desc="age", unit="post", leave=False)], dtype=object)

    strata = {"post_length": length, "self_reported_gender": gender, "self_reported_age": age}
    all_rows, summary = [], []
    for sname, groups in strata.items():
        sub = subgroup_metrics(y, pred, groups)
        if sub.empty:
            print(f"  {sname}: no group with enough data, skipping")
            continue
        sub.insert(0, "stratum", sname)
        all_rows.append(sub)
        gaps = fairness_gaps(sub)
        coverage = float(np.mean([g is not None for g in groups]))
        summary.append({"stratum": sname, "groups": len(sub), "coverage": round(coverage, 4), **{k: round(v, 4) for k, v in gaps.items()}})

    full = pd.concat(all_rows, ignore_index=True)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    full.round(4).to_csv(OUTCSV, index=False)
    summ = pd.DataFrame(summary)

    # figure: TPR & selection rate per group, one panel per stratum
    fig, axes = plt.subplots(1, len(all_rows), figsize=(5.2 * len(all_rows), 4.6), squeeze=False)
    for ax, sub in zip(axes[0], all_rows):
        g = list(sub["group"]); x = np.arange(len(g)); w = 0.38
        ax.bar(x - w / 2, sub["tpr"], w, label="TPR (recall)", color="#4C72B0")
        ax.bar(x + w / 2, sub["selection_rate"], w, label="selection rate", color="#DD8452")
        ax.set_xticks(x); ax.set_xticklabels([str(v) for v in g], fontsize=8)
        ax.set_title(sub["stratum"].iloc[0]); ax.set_ylim(0, 1.0); ax.legend(fontsize=8)
    fig.suptitle(f"Idea 8 — subgroup fairness (TF-IDF anxiety, threshold={thr:.3f})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    md = [
        "# Subgroup fairness audit",
        "",
        "Protected demographics are unavailable (anonymized corpus); inferring them with a "
        "classifier would be unreliable and biased. We therefore audit performance equity across "
        "**post-length tertiles** (well-powered) and **self-reported** gender/age (regex, exploratory — "
        "low coverage, self-report bias). TF-IDF anxiety model, author-disjoint split, "
        f"threshold tuned on calibration (={thr:.3f}). Gaps are max−min across subgroups.",
        "",
        "_Regenerate: `python scripts/fairness_audit.py`_",
        "",
        "## Fairness gaps per stratum",
        "| " + " | ".join(summ.columns) + " |",
        "|" + "|".join(["---"] * len(summ.columns)) + "|",
    ]
    for _, r in summ.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in summ.columns) + " |")
    md += ["", "## Per-group metrics", "",
           "| " + " | ".join(full.columns) + " |", "|" + "|".join(["---"] * len(full.columns)) + "|"]
    for _, r in full.round(4).iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in full.columns) + " |")
    md += ["", "![fairness](figures/fairness.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n=== fairness gaps ===")
    print(summ.to_string(index=False))
    print("\n=== per-group ===")
    print(full.round(4).to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
