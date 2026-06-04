"""Idea 5 — SHAP on the XGBoost linguistic model.

Bridges to the linguistic-analysis chapter: exact TreeExplainer SHAP values on the
26 hand-crafted features tell us which markers drive each target and in which
direction. Linguistic features are extracted ONCE (the slow step) and reused to
train one XGBoost per target on an author-disjoint split. Saves a beeswarm plot, a
mean-|SHAP| bar plot, and a ranked feature table per target.

CPU only (~feature extraction is the bottleneck; a few minutes). Run:
  python scripts/shap_linguistic.py
  python scripts/shap_linguistic.py --targets anxiety,depression --max-train 60000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from src.evaluation.shap_utils import summarize_shap, xgb_shap_values
from src.features.linguistic import extract_dataframe, feature_columns
from src.utils.config import load_model_config
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
SEED = 42
FIGDIR = Path("docs/figures")
OUTCSV = Path("experiments/shap_linguistic.csv")
DOC = Path("docs/shap.md")
DIR_ARROW = {"+": "↑", "-": "↓", "0": "·"}


def _xgb(cls_kwargs: dict, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=cls_kwargs.get("n_estimators", 500),
        max_depth=cls_kwargs.get("max_depth", 6),
        learning_rate=cls_kwargs.get("learning_rate", 0.05),
        subsample=cls_kwargs.get("subsample", 0.8),
        colsample_bytree=cls_kwargs.get("colsample_bytree", 0.8),
        scale_pos_weight=scale_pos_weight,
        random_state=cls_kwargs.get("random_state", 42),
        eval_metric="logloss",
        tree_method="hist",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="anxiety,depression,health_anxiety")
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--max-eval", type=int, default=6000)
    ap.add_argument("--min-pos", type=int, default=50)
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    df = read_parquet(DATA)
    df = df[(df["clean_text"].astype(str).str.len() >= 30) & df["author_hash"].notna()].reset_index(drop=True)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(df, groups=df["author_hash"].values))
    train = df.iloc[tr].reset_index(drop=True)
    test = df.iloc[te].reset_index(drop=True)
    if len(train) > args.max_train:
        train = train.sample(args.max_train, random_state=SEED).reset_index(drop=True)
    if len(test) > args.max_eval:
        test = test.sample(args.max_eval, random_state=SEED).reset_index(drop=True)
    print(f"author-disjoint: train={len(train):,}  test={len(test):,}")

    print("extracting linguistic features once (the slow step)...")
    feat_tr = extract_dataframe(train, text_col="clean_text")
    feat_cols = feature_columns(feat_tr)
    X_tr = feat_tr[feat_cols]
    X_te = extract_dataframe(test, text_col="clean_text")[feat_cols]

    cls_kwargs = load_model_config("configs/models/xgboost.yaml").extra.get("classifier", {})
    FIGDIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    md = [
        "# SHAP — XGBoost linguistic model",
        "",
        "Exact `TreeExplainer` SHAP values on the 26 hand-crafted linguistic features. "
        "**mean|SHAP|** ranks importance; **direction** is `↑` when a higher feature value "
        "pushes toward the positive class, `↓` otherwise. Author-disjoint split.",
        "",
        "_Regenerate: `python scripts/shap_linguistic.py`_",
        "",
    ]

    for target in tqdm(targets, desc="targets", unit="target"):
        col = f"label_{target}"
        if col not in train.columns:
            continue
        y_tr = (train[col].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
        npos = int(y_tr.sum())
        if npos < args.min_pos:
            print(f"skip {target}: only {npos} positives (< {args.min_pos})")
            continue
        spw = (len(y_tr) - npos) / max(1, npos)
        model = _xgb(cls_kwargs, spw)
        sw = train[f"{col}_weight"].astype(float).fillna(1.0).to_numpy() if f"{col}_weight" in train.columns else None
        model.fit(X_tr.to_numpy(), y_tr, sample_weight=sw)

        sv = xgb_shap_values(model, X_te)
        summ = summarize_shap(sv, X_te)
        summ.insert(0, "target", target)
        all_rows.append(summ)

        shap.summary_plot(sv, X_te, show=False, max_display=18)
        plt.title(f"SHAP — {target} (XGBoost linguistic)")
        bee = FIGDIR / f"shap_{target}_beeswarm.png"
        plt.savefig(bee, dpi=130, bbox_inches="tight"); plt.close()
        shap.summary_plot(sv, X_te, plot_type="bar", show=False, max_display=18)
        plt.title(f"mean|SHAP| — {target}")
        bar = FIGDIR / f"shap_{target}_bar.png"
        plt.savefig(bar, dpi=130, bbox_inches="tight"); plt.close()

        md.append(f"## {target}  (train positives: {npos:,})")
        md.append(f"![beeswarm](figures/{bee.name})")
        md.append("")
        md.append("| rank | feature | mean&#124;SHAP&#124; | direction |")
        md.append("|---:|---|---:|:--:|")
        for i, r in summ.head(12).iterrows():
            md.append(f"| {i + 1} | {r['feature']} | {r['mean_abs_shap']:.4f} | {DIR_ARROW[r['direction']]} |")
        md.append("")

    if not all_rows:
        print("No targets had enough positives to explain.")
        return

    full = pd.concat(all_rows, ignore_index=True)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUTCSV, index=False)
    DOC.write_text("\n".join(md), encoding="utf-8")
    for r in all_rows:
        t = r["target"].iloc[0]
        print(f"\nTop features — {t}:")
        print(r[["feature", "mean_abs_shap", "direction"]].head(8).to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, and beeswarm/bar figures under {FIGDIR}/")


if __name__ == "__main__":
    main()
