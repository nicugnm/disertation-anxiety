"""Ideas 9 & 10 — external (cross-corpus) validation of the anxiety model.

Applies our anxiety models ZERO-SHOT to independent corpora and compares them:
  - TF-IDF + LogReg (trained on our corpus, CPU)
  - MentalRoBERTa multi-task transformer (the saved checkpoint)
against
  - RMHD (Low et al. 2020): anxiety-related subreddits vs controls (subreddit labels)
  - ANGST (Hengle et al. 2024): anxiety vs not, 3 expert-psychologist labels (gold)

ANGST is gated on HuggingFace — request access + `huggingface-cli login`, then it
runs automatically. RMHD/ANGST data live under data/external/ (gitignored).

Run:
  python scripts/external_validation.py
  python scripts/external_validation.py --models tfidf            # skip transformer
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
TRANSFORMER_DIR = "experiments/runs/multitask_anxiety_health_dep_suic/model"
TARGET = "anxiety"
SEED = 42
POS_SUBS = ["anxiety", "healthanxiety", "socialanxiety"]
NEG_SUBS = ["fitness", "parenting", "meditation", "conspiracy"]
DOC = Path("docs/external_validation.md")
OUTCSV = Path("experiments/external_validation.csv")
FIG = Path("docs/figures/external_validation.png")
FIG_SUB = Path("docs/figures/external_validation_rmhd.png")


def _chunked(predict_one, texts, desc):
    out = []
    for i in tqdm(range(0, len(texts), 4000), desc=desc, unit="chunk", leave=False):
        out.append(np.asarray(predict_one(texts[i:i + 4000])).ravel())
    return np.concatenate(out) if out else np.array([])


def tfidf_predictor(max_train: int):
    df = read_parquet(DATA)
    df = df[df["clean_text"].astype(str).str.len() >= 30].reset_index(drop=True)
    if len(df) > max_train:
        df = df.sample(max_train, random_state=SEED).reset_index(drop=True)
    print(f"training TF-IDF on our corpus (n={len(df):,})...")
    model = build_model(load_model_config("configs/models/baseline.yaml")).fit(df)
    return lambda texts: _chunked(
        lambda t: model.predict_proba(pd.DataFrame({"clean_text": t})), texts, "tfidf")


def transformer_predictor():
    cfg = load_model_config("configs/models/multitask.yaml")
    model = build_model(cfg).load(TRANSFORMER_DIR)
    ti = cfg.targets.index(TARGET)
    print(f"loaded multitask transformer (anxiety = target {ti})")
    return lambda texts: _chunked(
        lambda t: np.asarray(model.predict_proba(pd.DataFrame({"clean_text": t})))[:, ti], texts, "transformer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tfidf,transformer")
    ap.add_argument("--max-train", type=int, default=200000)
    ap.add_argument("--cap-per-sub", type=int, default=5000)
    args = ap.parse_args()
    want = [m.strip() for m in args.models.split(",") if m.strip()]

    # datasets
    datasets: dict[str, pd.DataFrame] = {}
    rmhd = load_rmhd(RMHD_DIR, POS_SUBS, NEG_SUBS, cap_per_sub=args.cap_per_sub, seed=SEED)
    if not rmhd.empty:
        datasets["RMHD (Low 2020)"] = rmhd
    else:
        print(f"RMHD CSVs not found in {RMHD_DIR}/ — download from https://zenodo.org/records/3941387")
    try:
        angst = load_angst(ANGST_DIR, target=TARGET)
    except Exception as ex:  # noqa: BLE001
        angst = None; print(f"ANGST unreadable: {ex}")
    if angst is not None:
        datasets["ANGST (experts)"] = angst
    else:
        print("ANGST not available (gated): request access at "
              "https://huggingface.co/datasets/ameyhengle/ANGST , `huggingface-cli login`, "
              f"download test.csv to {ANGST_DIR}/.")
    if not datasets:
        print("No external datasets available."); return

    # predictors
    predictors: dict[str, object] = {}
    if "tfidf" in want:
        predictors["TF-IDF"] = tfidf_predictor(args.max_train)
    if "transformer" in want:
        if Path(TRANSFORMER_DIR).exists():
            try:
                predictors["MentalRoBERTa-MT"] = transformer_predictor()
            except Exception as ex:  # noqa: BLE001
                print(f"transformer unavailable ({ex}); skipping.")
        else:
            print(f"transformer checkpoint not found at {TRANSFORMER_DIR}; skipping.")

    rows: list[dict] = []
    sub_rows: list[dict] = []
    for mname, predict in predictors.items():
        for dname, df in datasets.items():
            scores = predict(df["clean_text"].tolist())
            y = df["y"].to_numpy()
            rows.append({"model": mname, "dataset": dname, "n": len(y), "n_pos": int(y.sum()),
                         "auroc": round(roc_auc_score(y, scores), 4),
                         "auprc": round(average_precision_score(y, scores), 4),
                         "f1@0.5": round(f1_score(y, (scores >= 0.5).astype(int), zero_division=0), 4)})
            print(f"  {mname} on {dname}: AUROC {rows[-1]['auroc']}")
            if dname.startswith("RMHD"):
                for sub in POS_SUBS + NEG_SUBS:
                    m = df["subreddit"].to_numpy() == sub
                    if m.sum():
                        sub_rows.append({"model": mname, "subreddit": sub, "label": int(df["y"][m].iloc[0]),
                                         "mean_anxiety_score": round(float(scores[m].mean()), 4)})

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)
    subdf = pd.DataFrame(sub_rows)

    # ---- figure 1: AUROC by dataset x model ----
    dsets = list(dict.fromkeys(out["dataset"]))
    models = list(dict.fromkeys(out["model"]))
    x = np.arange(len(dsets)); w = 0.8 / max(1, len(models))
    colors = {"TF-IDF": "#C44E52", "MentalRoBERTa-MT": "#4C72B0"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        vals = [out[(out.model == m) & (out.dataset == ds)]["auroc"].iloc[0] for ds in dsets]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w, label=m, color=colors.get(m))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(dsets); ax.set_ylim(0, 1.02)
    ax.set_ylabel("AUROC (zero-shot)"); ax.legend()
    ax.set_title("External validation — TF-IDF vs transformer (zero-shot)")
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    # ---- figure 2: RMHD per-subreddit mean score by model ----
    if not subdf.empty:
        subs = POS_SUBS + NEG_SUBS
        xs = np.arange(len(subs)); w2 = 0.8 / max(1, len(models))
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for i, m in enumerate(models):
            vals = [subdf[(subdf.model == m) & (subdf.subreddit == s)]["mean_anxiety_score"].iloc[0]
                    if not subdf[(subdf.model == m) & (subdf.subreddit == s)].empty else 0 for s in subs]
            ax2.bar(xs + (i - (len(models) - 1) / 2) * w2, vals, w2, label=m, color=colors.get(m))
        ax2.set_xticks(xs); ax2.set_xticklabels(subs, rotation=25, ha="right")
        ax2.set_ylabel("mean predicted P(anxiety)"); ax2.legend()
        ax2.set_title("RMHD per-subreddit anxiety score (anxiety subs first, then controls)")
        fig2.tight_layout(); fig2.savefig(FIG_SUB, dpi=130); plt.close(fig2)

    md = [
        "# External (cross-corpus) validation",
        "",
        "Our anxiety models applied ZERO-SHOT to independent corpora: **RMHD** (Low 2020, "
        "subreddit labels) and **ANGST** (Hengle 2024, 3 expert-psychologist labels). "
        "TF-IDF trained on our corpus; transformer = saved MentalRoBERTa multi-task checkpoint. "
        "`src/evaluation/external.py`, `scripts/external_validation.py`.",
        "",
        "_Regenerate: `python scripts/external_validation.py`_",
        "",
        "| " + " | ".join(out.columns) + " |",
        "|" + "|".join(["---"] * len(out.columns)) + "|",
    ]
    for _, r in out.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in out.columns) + " |")
    md += ["", "![external AUROC](figures/external_validation.png)"]
    if not subdf.empty:
        md += ["", "## RMHD per-subreddit mean predicted P(anxiety)", "",
               "| " + " | ".join(subdf.columns) + " |", "|" + "|".join(["---"] * len(subdf.columns)) + "|"]
        for _, r in subdf.iterrows():
            md.append("| " + " | ".join(str(r[c]) for c in subdf.columns) + " |")
        md += ["", "![RMHD per-subreddit](figures/external_validation_rmhd.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out.to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
