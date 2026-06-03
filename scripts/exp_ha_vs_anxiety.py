"""Experiment A — r/HealthAnxiety vs r/Anxiety head-to-head (ideas-1 #1 novelty).

Binary, POST-LEVEL classifier with subreddit-membership labels
(HealthAnxiety = 1, Anxiety = 0) and an AUTHOR-DISJOINT split (no author appears
in both train and test — Harrigian et al.). Question: is health-anxiety language
separable from general-anxiety language? Baseline to beat: Low et al. 2020
SGD-L1 weighted F1 = 0.851 (same subreddit-as-proxy setup, linear model).

Models:
  - tfidf       : TF-IDF + LogReg (fast, CPU) + top discriminative n-grams
  - transformer : MentalRoBERTa fine-tune (needs GPU + HF access to mental-roberta;
                  falls back to roberta-base if the gated repo is unavailable)

Run:
  python scripts/exp_ha_vs_anxiety.py --models tfidf         # baseline only (CPU, seconds)
  python scripts/exp_ha_vs_anxiety.py                        # baseline + transformer (GPU)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import full_report
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
LOW2020_F1 = 0.851
OUT = Path("experiments")
FIG = Path("docs/figures")
SEED = 42


def load_head_to_head(submissions_only: bool = False) -> tuple[pd.DataFrame, dict]:
    df = read_parquet(DATA)
    sub = df["subreddit"].astype(str).str.lower()
    d = df[sub.isin(["healthanxiety", "anxiety"])].copy()
    if submissions_only and "kind" in d.columns:
        d = d[d["kind"].astype(str) == "submission"]
    d["clean_text"] = d["clean_text"].astype(str)
    d["y"] = (d["subreddit"].astype(str).str.lower() == "healthanxiety").astype(int)
    d = d[d["clean_text"].str.len() >= 30]
    n_before = len(d)
    # author-disjoint integrity: rows with no author_hash can't be attributed; drop them.
    d = d[d["author_hash"].notna() & (d["author_hash"].astype(str) != "")].reset_index(drop=True)
    d["label_ha"] = d["y"]  # transformer reads label_<target>
    stats = {
        "n_posts": int(len(d)),
        "n_dropped_no_author": int(n_before - len(d)),
        "n_health_anxiety": int(d["y"].sum()),
        "n_anxiety": int((d["y"] == 0).sum()),
        "n_authors": int(d["author_hash"].nunique()),
        "pct_comments": round(100 * (d["kind"].astype(str) == "comment").mean(), 1)
        if "kind" in d.columns else None,
    }
    return d, stats


def author_disjoint_split(d: pd.DataFrame, test_size: float, seed: int = SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(d, d["y"], groups=d["author_hash"]))
    return d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)


def _weighted_f1(y, proba, threshold) -> float:
    """Weighted-average F1 across both classes — the metric Low 2020 reported."""
    from sklearn.metrics import f1_score

    pred = (np.asarray(proba) >= threshold).astype(int)
    return float(f1_score(y, pred, average="weighted"))


def run_tfidf(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.95,
                                  sublinear_tf=True, max_features=80000, lowercase=True)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                   solver="liblinear", random_state=SEED)),
    ])
    pipe.fit(train["clean_text"].tolist(), train["y"].values)
    proba = pipe.predict_proba(test["clean_text"].tolist())[:, 1]
    rep = full_report(test["y"].values, proba, bootstrap=True)
    rep["f1_weighted"] = _weighted_f1(test["y"].values, proba, rep["threshold"])
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    coefs, names = clf.coef_[0], np.array(vec.get_feature_names_out())
    markers = {
        "health_anxiety": list(names[np.argsort(coefs)[-20:]][::-1]),
        "anxiety": list(names[np.argsort(coefs)[:20]]),
    }
    return rep, markers


def run_transformer(train: pd.DataFrame, test: pd.DataFrame, suffix: str = "") -> dict:
    from src.models.transformer import TransformerModel
    from src.utils.config import ModelConfig

    cfg = ModelConfig(
        name="ha_vs_anxiety_mentalroberta", model_type="transformer",
        text_field="clean_text", target="ha", targets=None,
        extra={
            "pretrained": "mental/mental-roberta-base",
            "fallback_pretrained": "roberta-base",
            "tokenizer": {"max_length": 256},
            "train": {
                "per_device_train_batch_size": 16, "num_train_epochs": 3,
                "learning_rate": 2.0e-5, "evaluation_strategy": "epoch",
                "save_strategy": "no", "load_best_model_at_end": False,
            },
        },
    )
    tr2, val = author_disjoint_split(train, test_size=0.15)
    model = TransformerModel(cfg).fit(tr2, val=val)
    proba = model.predict_proba(test)
    out = Path(f"experiments/runs/ha_vs_anxiety_mentalroberta{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model")
    rep = full_report(test["label_ha"].values, proba, bootstrap=True)
    rep["f1_weighted"] = _weighted_f1(test["label_ha"].values, proba, rep["threshold"])
    return rep


def plot_markers(markers: dict, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    ha = markers["health_anxiety"][:15][::-1]
    anx = markers["anxiety"][:15][::-1]
    y_ha = np.arange(len(ha))
    y_anx = np.arange(len(anx)) + len(ha) + 1
    ax.barh(y_ha, [1] * len(ha), color="#4c72b0")
    ax.barh(y_anx, [1] * len(anx), color="#c44e52")
    ax.set_yticks(list(y_ha) + list(y_anx))
    ax.set_yticklabels(ha + anx, fontsize=9)
    ax.set_xticks([])
    ax.set_title("Top discriminative n-grams\n(blue = r/HealthAnxiety, red = r/Anxiety)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tfidf,transformer",
                    help="comma list: tfidf,transformer")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--submissions-only", action="store_true",
                    help="restrict to submissions (matches Low 2020's unit; drops comments)")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    suffix = "_submissions" if args.submissions_only else ""

    d, stats = load_head_to_head(submissions_only=args.submissions_only)
    if args.submissions_only:
        print("[submissions-only mode — comments excluded, matches Low 2020's unit]")
    print(f"Head-to-head corpus: {stats}")
    train, test = author_disjoint_split(d, test_size=args.test_size)
    # sanity: author-disjoint
    assert set(train["author_hash"]).isdisjoint(set(test["author_hash"])), "author overlap!"
    print(f"train={len(train):,}  test={len(test):,}  (author-disjoint)")

    results: dict = {"baseline_low2020_f1": LOW2020_F1, "corpus": stats,
                     "n_train": int(len(train)), "n_test": int(len(test))}

    if "tfidf" in models:
        rep, markers = run_tfidf(train, test)
        results["tfidf_logreg"] = rep
        results["markers"] = markers
        fig = plot_markers(markers, FIG / f"ha_vs_anxiety_markers{suffix}.png")
        print(f"\nTF-IDF + LogReg:  F1(bin)={rep['f1']:.3f} "
              f"[{rep.get('f1_ci_lo', float('nan')):.3f}, {rep.get('f1_ci_hi', float('nan')):.3f}]  "
              f"F1(weighted)={rep['f1_weighted']:.3f}  AUROC={rep['auroc']:.3f}  AUPRC={rep['auprc']:.3f}")
        print(f"   weighted-F1 vs Low 2020={LOW2020_F1}  ->  delta {rep['f1_weighted'] - LOW2020_F1:+.3f}")
        print(f"   top HealthAnxiety n-grams: {markers['health_anxiety'][:10]}")
        print(f"   top Anxiety n-grams:       {markers['anxiety'][:10]}")
        print(f"   markers figure -> {fig}")

    if "transformer" in models:
        print("\nTraining MentalRoBERTa (GPU strongly recommended)...")
        rep = run_transformer(train, test, suffix=suffix)
        results["mentalroberta"] = rep
        print(f"MentalRoBERTa:  F1(bin)={rep['f1']:.3f} "
              f"[{rep.get('f1_ci_lo', float('nan')):.3f}, {rep.get('f1_ci_hi', float('nan')):.3f}]  "
              f"F1(weighted)={rep['f1_weighted']:.3f}  AUROC={rep['auroc']:.3f}  AUPRC={rep['auprc']:.3f}")
        print(f"   weighted-F1 vs Low 2020={LOW2020_F1}  ->  delta {rep['f1_weighted'] - LOW2020_F1:+.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / f"exp_ha_vs_anxiety{suffix}.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults -> {out_json}")


if __name__ == "__main__":
    main()
