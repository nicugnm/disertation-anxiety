"""Idea 7 — robustness audit under meaning-preserving perturbations.

Compares how often each model's anxiety decision FLIPS when test posts are lightly
corrupted (char swaps, keyboard typos, deletions, case flips, punctuation removal,
social-media elongation). Hypothesis: TF-IDF (exact-token features) is far more
fragile than the subword transformer. Reports flip rate, score drift, and accuracy
drop per perturbation per model. Lightweight (no TextAttack), seeded, CPU/GPU.

Run:
  python scripts/robustness_audit.py
  python scripts/robustness_audit.py --models tfidf --n 3000 --p 0.15
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

from src.evaluation.robustness import PERTURBATIONS, flip_rate, mean_abs_score_drift
from src.models.registry import build_model
from src.utils.config import load_model_config
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
TARGET = "anxiety"
SEED = 42
THR = 0.5
TRANSFORMER_DIR = "experiments/runs/multitask_anxiety_health_dep_suic/model"
FIG = Path("docs/figures/robustness.png")
OUTCSV = Path("experiments/robustness.csv")
DOC = Path("docs/robustness.md")


def _tfidf_predictor(train: pd.DataFrame):
    model = build_model(load_model_config("configs/models/baseline.yaml")).fit(train)
    return lambda texts: np.asarray(model.predict_proba(pd.DataFrame({"clean_text": texts}))).ravel()


def _transformer_predictor():
    cfg = load_model_config("configs/models/multitask.yaml")
    model = build_model(cfg).load(TRANSFORMER_DIR)
    ti = cfg.targets.index(TARGET)
    return lambda texts: np.asarray(model.predict_proba(pd.DataFrame({"clean_text": texts})))[:, ti]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tfidf,transformer")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--max-train", type=int, default=120000)
    ap.add_argument("--p", type=float, default=0.15)
    args = ap.parse_args()
    want = [m.strip() for m in args.models.split(",") if m.strip()]

    df = read_parquet(DATA)
    df = df[(df["clean_text"].astype(str).str.len() >= 30) & df["author_hash"].notna()].reset_index(drop=True)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(df, groups=df["author_hash"].values))
    train = df.iloc[tr].reset_index(drop=True)
    test = df.iloc[te].reset_index(drop=True)
    if len(train) > args.max_train:
        train = train.sample(args.max_train, random_state=SEED).reset_index(drop=True)
    test = test.sample(min(args.n, len(test)), random_state=SEED).reset_index(drop=True)
    clean_texts = test["clean_text"].astype(str).tolist()
    y = (test[f"label_{TARGET}"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
    print(f"train={len(train):,} test(sample)={len(test):,}  positives={int(y.sum())}")

    predictors = {}
    if "tfidf" in want:
        print("training TF-IDF (CPU)...")
        predictors["tfidf"] = _tfidf_predictor(train)
    if "transformer" in want:
        try:
            print("loading transformer...")
            predictors["transformer"] = _transformer_predictor()
        except Exception as ex:  # noqa: BLE001
            print(f"  transformer unavailable ({ex}); skipping.")

    rows: list[dict] = []
    for mname, predict in predictors.items():
        clean_score = predict(clean_texts)
        clean_pred = (clean_score >= THR).astype(int)
        acc_clean = float((clean_pred == y).mean())
        for pname, fn in tqdm(PERTURBATIONS.items(), desc=f"{mname}", unit="perturb"):
            rng = np.random.default_rng(SEED)
            pert_texts = [fn(t, rng, args.p) for t in clean_texts]
            pert_score = predict(pert_texts)
            pert_pred = (pert_score >= THR).astype(int)
            rows.append({
                "model": mname, "perturbation": pname,
                "flip_rate": round(flip_rate(clean_pred, pert_pred), 4),
                "score_drift": round(mean_abs_score_drift(clean_score, pert_score), 4),
                "acc_clean": round(acc_clean, 4),
                "acc_perturbed": round(float((pert_pred == y).mean()), 4),
                "acc_drop": round(acc_clean - float((pert_pred == y).mean()), 4),
            })

    if not rows:
        print("No models available to audit.")
        return
    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    # grouped bar: flip rate per perturbation per model
    perts = list(PERTURBATIONS)
    models = list(predictors)
    x = np.arange(len(perts))
    w = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"tfidf": "#C44E52", "transformer": "#4C72B0"}
    for i, m in enumerate(models):
        vals = [out[(out.model == m) & (out.perturbation == pp)]["flip_rate"].iloc[0] for pp in perts]
        ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w, label=m, color=colors.get(m, None))
    ax.set_xticks(x); ax.set_xticklabels(perts, rotation=25, ha="right")
    ax.set_ylabel("decision flip rate (lower = more robust)")
    ax.set_title(f"Idea 7 — robustness to meaning-preserving perturbations (p={args.p})")
    ax.legend()
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    cols = ["model", "perturbation", "flip_rate", "score_drift", "acc_clean", "acc_perturbed", "acc_drop"]
    md = [
        "# Robustness audit — meaning-preserving perturbations",
        "",
        f"How often each model's anxiety decision flips when test posts are lightly corrupted "
        f"(fraction p={args.p} of words edited; punctuation stripping applies to all). "
        f"Lightweight TextBugger-style perturbations (no TextAttack), seeded. "
        f"**Lower flip rate / score drift = more robust.** Threshold {THR}.",
        "",
        "_Regenerate: `python scripts/robustness_audit.py`_",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md += ["", "![robustness](figures/robustness.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print(out.to_string(index=False))
    if len(models) == 2:
        piv = out.pivot_table(index="perturbation", columns="model", values="flip_rate")
        print("\nflip-rate by perturbation:\n", piv.to_string())
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
