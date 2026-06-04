"""Experiment 6 (full corpus) — multitask transformer per-target evaluation.

Evaluates the full-corpus multitask MentalRoBERTa checkpoint
(experiments/runs/multitask_fullcorpus/model, trained on a 205k sample with seed 42)
on a HELD-OUT test set drawn from the rows NOT used in training. Reports per-target
F1 (best threshold) / AUROC / AUPRC / ECE — the full-corpus replacement for the
stale 16k Experiment-6 numbers.

GPU recommended. Run:
  python scripts/exp6_transformer_fullcorpus.py --test-n 30000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import full_report
from src.models.registry import build_model
from src.utils.config import load_model_config
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
MODEL_DIR = "experiments/runs/multitask_fullcorpus/model"
SEED = 42
TRAIN_N = 205000  # must match scripts/train_multitask_fullcorpus.py (n + val_size)
FIG = Path("docs/figures/exp6_transformer_fullcorpus.png")
OUTCSV = Path("experiments/exp6_transformer_fullcorpus.csv")
DOC = Path("docs/exp6_transformer_fullcorpus.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-n", type=int, default=30000)
    args = ap.parse_args()

    df = read_parquet(DATA)
    df = df[df["clean_text"].astype(str).str.len() >= 30].reset_index(drop=True)
    train_idx = df.sample(min(len(df), TRAIN_N), random_state=SEED).index   # rows the model trained on
    heldout = df.drop(train_idx).reset_index(drop=True)
    test = heldout.sample(min(len(heldout), args.test_n), random_state=123).reset_index(drop=True)
    print(f"held-out test = {len(test):,} posts (disjoint from the {len(train_idx):,} training rows)")

    cfg = load_model_config("configs/models/multitask.yaml")
    model = build_model(cfg).load(MODEL_DIR)
    proba = np.asarray(model.predict_proba(test))

    rows = []
    for i, t in enumerate(cfg.targets):
        y = (test[f"label_{t}"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
        p = proba[:, i]
        if y.sum() < 5 or (y == 0).sum() < 5:
            rows.append({"target": t, "n": len(y), "n_pos": int(y.sum()),
                         "f1": None, "auroc": None, "auprc": None, "ece": None, "threshold": None})
            continue
        rep = full_report(y, p, bootstrap=False)
        rows.append({"target": t, "n": len(y), "n_pos": int(y.sum()),
                     "f1": round(rep["f1"], 4), "auroc": round(rep["auroc"], 4),
                     "auprc": round(rep["auprc"], 4), "ece": round(rep["ece"], 4),
                     "threshold": round(rep["threshold"], 3)})

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    plot = out[out["f1"].notna()]
    x = np.arange(len(plot)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, plot["f1"], w, label="F1 (best thr)", color="#4C72B0")
    ax.bar(x + w / 2, plot["auroc"], w, label="AUROC", color="#DD8452")
    for i, (f, a) in enumerate(zip(plot["f1"], plot["auroc"])):
        ax.text(i - w / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(plot["target"], fontsize=9); ax.set_ylim(0, 1.05)
    ax.set_title("Experiment 6 — multitask transformer (full corpus, held-out test)")
    ax.legend()
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    cols = ["target", "n", "n_pos", "f1", "auroc", "auprc", "ece", "threshold"]
    md = [
        "# Experiment 6 (full corpus) — multitask transformer per-target",
        "",
        f"Full-corpus multitask MentalRoBERTa ({MODEL_DIR}, trained on a 205k sample) evaluated on a "
        f"held-out test set of {len(test):,} posts disjoint from training. Best-threshold F1 / AUROC / "
        "AUPRC / ECE. `scripts/exp6_transformer_fullcorpus.py`.",
        "",
        "_Regenerate: `python scripts/exp6_transformer_fullcorpus.py`_",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md += ["", "![exp6 transformer](figures/exp6_transformer_fullcorpus.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print(out.to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
