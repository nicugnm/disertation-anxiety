"""Ideas 9 & 10 — external (cross-corpus) validation of the anxiety model.

Trains the TF-IDF anxiety model on OUR corpus, then applies it ZERO-SHOT to
independent external datasets:
  - RMHD (Low et al. 2020): anxiety-related subreddits vs control subreddits.
    Tests whether the learned anxiety signal transfers to a different corpus /
    time period. (Public; CSVs in data/external/rmhd/.)
  - ANGST (Hengle et al. 2024): 3-expert-psychologist anxiety labels — the gold
    external validation. GATED on HuggingFace; this runs automatically once the
    CSVs are present in data/external/angst/ (request access + huggingface-cli login).

CPU, a few minutes. Run:
  python scripts/external_validation.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

from src.evaluation.external import load_angst, load_rmhd
from src.models.registry import build_model
from src.utils.config import load_model_config
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
RMHD_DIR = "data/external/rmhd"
ANGST_DIR = "data/external/angst"
TARGET = "anxiety"
SEED = 42
POS_SUBS = ["anxiety", "healthanxiety", "socialanxiety"]
NEG_SUBS = ["fitness", "parenting", "meditation", "conspiracy"]
DOC = Path("docs/external_validation.md")
OUTCSV = Path("experiments/external_validation.csv")
FIG = Path("docs/figures/external_validation.png")


def _predictor(max_train: int):
    df = read_parquet(DATA)
    df = df[df["clean_text"].astype(str).str.len() >= 30].reset_index(drop=True)
    if len(df) > max_train:
        df = df.sample(max_train, random_state=SEED).reset_index(drop=True)
    print(f"training TF-IDF on our corpus (n={len(df):,})...")
    model = build_model(load_model_config("configs/models/baseline.yaml")).fit(df)

    def predict(texts):
        out = []
        for i in tqdm(range(0, len(texts), 5000), desc="predict", unit="chunk", leave=False):
            out.append(np.asarray(model.predict_proba(pd.DataFrame({"clean_text": texts[i:i + 5000]}))).ravel())
        return np.concatenate(out) if out else np.array([])
    return predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=200000)
    ap.add_argument("--cap-per-sub", type=int, default=5000)
    args = ap.parse_args()

    predict = _predictor(args.max_train)
    rows: list[dict] = []
    sub_rows: list[dict] = []

    # ---- RMHD ----
    rmhd = load_rmhd(RMHD_DIR, POS_SUBS, NEG_SUBS, cap_per_sub=args.cap_per_sub, seed=SEED)
    if rmhd.empty:
        print(f"RMHD CSVs not found in {RMHD_DIR}/ — download per-subreddit CSVs from "
              "https://zenodo.org/records/3941387 (e.g. anxiety_2018_features_tfidf_256.csv).")
    else:
        scores = predict(rmhd["clean_text"].tolist())
        y = rmhd["y"].to_numpy()
        rows.append({"dataset": "RMHD (Low 2020)", "n": len(y), "n_pos": int(y.sum()),
                     "auroc": round(roc_auc_score(y, scores), 4),
                     "auprc": round(average_precision_score(y, scores), 4),
                     "f1@0.5": round(f1_score(y, (scores >= 0.5).astype(int), zero_division=0), 4)})
        for sub in POS_SUBS + NEG_SUBS:
            m = rmhd["subreddit"].to_numpy() == sub
            if m.sum() == 0:
                continue
            sub_rows.append({"dataset": "RMHD", "subreddit": sub, "label": int(rmhd["y"][m].iloc[0]),
                             "n": int(m.sum()), "mean_anxiety_score": round(float(scores[m].mean()), 4),
                             "pred_pos_rate@0.5": round(float((scores[m] >= 0.5).mean()), 4)})
        print(f"RMHD: AUROC {rows[-1]['auroc']}  AUPRC {rows[-1]['auprc']}  F1@0.5 {rows[-1]['f1@0.5']}")

    # ---- ANGST (gated) ----
    angst = None
    try:
        angst = load_angst(ANGST_DIR)
    except Exception as ex:  # noqa: BLE001
        print(f"ANGST present but unreadable: {ex}")
    if angst is None:
        print(f"ANGST not available (gated). Request access at "
              "https://huggingface.co/datasets/ameyhengle/ANGST , run `huggingface-cli login`, "
              f"download test.csv to {ANGST_DIR}/, then re-run.")
    else:
        scores = predict(angst["clean_text"].tolist())
        y = angst["y"].to_numpy()
        rows.append({"dataset": "ANGST (Hengle 2024, experts)", "n": len(y), "n_pos": int(y.sum()),
                     "auroc": round(roc_auc_score(y, scores), 4),
                     "auprc": round(average_precision_score(y, scores), 4),
                     "f1@0.5": round(f1_score(y, (scores >= 0.5).astype(int), zero_division=0), 4)})
        print(f"ANGST: AUROC {rows[-1]['auroc']}  AUPRC {rows[-1]['auprc']}")

    if not rows:
        print("No external datasets available; nothing evaluated.")
        return

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)
    subdf = pd.DataFrame(sub_rows)

    if not subdf.empty:
        s = subdf[subdf.dataset == "RMHD"].sort_values("mean_anxiety_score")
        colors = ["#C44E52" if l == 1 else "#4C72B0" for l in s["label"]]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(len(s)), s["mean_anxiety_score"], color=colors)
        ax.set_xticks(range(len(s))); ax.set_xticklabels(s["subreddit"], rotation=30, ha="right")
        ax.set_ylabel("mean predicted P(anxiety)")
        auroc = out[out.dataset.str.startswith("RMHD")]["auroc"].iloc[0]
        ax.set_title(f"RMHD zero-shot transfer — our model's anxiety score per subreddit\n"
                     f"(red = anxiety-related, blue = control; AUROC={auroc})")
        fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG, dpi=130); plt.close(fig)

    md = [
        "# External (cross-corpus) validation",
        "",
        "TF-IDF anxiety model trained on OUR corpus, applied ZERO-SHOT to independent corpora. "
        "RMHD = anxiety-related subreddits (anxiety, healthanxiety, socialanxiety) vs controls "
        "(fitness, parenting, meditation, conspiracy); subreddit-as-label. "
        "ANGST = 3-expert-psychologist labels (gated; run when available).",
        "",
        "_Regenerate: `python scripts/external_validation.py`_",
        "",
        "| " + " | ".join(out.columns) + " |",
        "|" + "|".join(["---"] * len(out.columns)) + "|",
    ]
    for _, r in out.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in out.columns) + " |")
    if not subdf.empty:
        md += ["", "## RMHD per-subreddit (mean predicted anxiety score)", "",
               "| " + " | ".join(subdf.columns) + " |", "|" + "|".join(["---"] * len(subdf.columns)) + "|"]
        for _, r in subdf.iterrows():
            md.append("| " + " | ".join(str(r[c]) for c in subdf.columns) + " |")
        md += ["", "![external](figures/external_validation.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out.to_string(index=False))
    if not subdf.empty:
        print("\n", subdf.to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}" + (f", {FIG}" if not subdf.empty else ""))


if __name__ == "__main__":
    main()
