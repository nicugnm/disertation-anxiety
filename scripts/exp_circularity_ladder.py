"""The circularity ladder: how much reported performance survives moving away
from the researcher's own heuristic.

A single TF-IDF + LogReg model is evaluated at four rungs of decreasing
label-circularity:

  1. weak label, in-domain   -- label = subreddit prior + our lexicon (most circular)
  2. subreddit proxy         -- r/HealthAnxiety vs r/Anxiety membership
  3. self-disclosure, masked -- user-level, diagnosis sentence hidden (independent label)
  4. expert ANGST            -- 3 psychologists (fully independent of our heuristics)

The gap between rung 1 (~0.99) and rungs 3--4 is the "circularity tax": the part
of the in-domain score that comes from the model recovering our own labelling
heuristic rather than a clinical construct. Rung 3 is read from the existing
disclosure user-level evaluation (TF-IDF, masked); the others are recomputed here.

CPU, minutes. Run:
  python scripts/exp_circularity_ladder.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_ha_vs_anxiety import author_disjoint_split, load_head_to_head, _weighted_f1  # noqa: E402

from src.evaluation.external import load_angst  # noqa: E402
from src.utils.io import read_parquet  # noqa: E402

DATA = "data/processed/labeled.parquet"
ANGST_DIR = "data/external/angst"
DISC_SUMMARY = "experiments/disclosure_userlevel_summary.csv"
SEED = 42
OUTCSV = Path("experiments/circularity_ladder.csv")
DOC = Path("docs/circularity_ladder.md")
FIG = Path("docs/figures/circularity_ladder.png")


def _tfidf():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.95,
                                  sublinear_tf=True, max_features=80000, lowercase=True)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                   solver="liblinear", random_state=SEED)),
    ])


def main() -> None:
    rows = []

    # ---- rungs 1 & 4: one TF-IDF trained on the corpus anxiety weak label ----
    df = read_parquet(DATA)
    df = df[(df["clean_text"].astype(str).str.len() >= 30) & df["author_hash"].notna()].reset_index(drop=True)
    if len(df) > 200000:
        df = df.sample(200000, random_state=SEED).reset_index(drop=True)
    y = (df["label_anxiety"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(df, groups=df["author_hash"].values))
    model = _tfidf().fit(df.iloc[tr]["clean_text"].tolist(), y[tr])
    # rung 1: weak label, in-domain
    p_te = model.predict_proba(df.iloc[te]["clean_text"].tolist())[:, 1]
    rows.append({"rung": "1. weak label, in-domain", "circularity": "most circular",
                 "test_label": "subreddit prior + our lexicon", "auroc": round(float(roc_auc_score(y[te], p_te)), 4)})
    # rung 4: expert ANGST (zero-shot transfer of the same kind of model)
    try:
        angst = load_angst(ANGST_DIR, target="anxiety")
        p_an = model.predict_proba(angst["clean_text"].tolist())[:, 1]
        rows.append({"rung": "4. expert ANGST", "circularity": "fully independent",
                     "test_label": "3 expert psychologists", "auroc": round(float(roc_auc_score(angst["y"].to_numpy(), p_an)), 4)})
    except Exception as ex:  # noqa: BLE001
        print(f"  ANGST unavailable: {ex}")

    # ---- rung 2: subreddit proxy (HA vs Anxiety, submissions) ----
    d, _ = load_head_to_head(submissions_only=True)
    train, test = author_disjoint_split(d, test_size=0.2)
    ha = _tfidf().fit(train["clean_text"].tolist(), train["y"].to_numpy())
    p_ha = ha.predict_proba(test["clean_text"].tolist())[:, 1]
    rows.append({"rung": "2. subreddit proxy (HA vs Anxiety)", "circularity": "researcher-chosen proxy",
                 "test_label": "subreddit membership", "auroc": round(float(roc_auc_score(test["y"].to_numpy(), p_ha)), 4)})

    # ---- rung 3: masked self-disclosure (read from existing user-level eval) ----
    if Path(DISC_SUMMARY).exists():
        s = pd.read_csv(DISC_SUMMARY)
        m = s[(s["model"] == "tfidf_logreg") & (s["target"] == "anxiety") & (s["mode"] == "masked")]
        if not m.empty:
            rows.append({"rung": "3. self-disclosure, masked", "circularity": "independent label",
                         "test_label": "self-reported diagnosis (post hidden)", "auroc": round(float(m["auroc"].iloc[0]), 4)})

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    # order by circularity (most -> least), keep a fixed sensible order
    order = ["1. weak label, in-domain", "2. subreddit proxy (HA vs Anxiety)",
             "3. self-disclosure, masked", "4. expert ANGST"]
    out["__o"] = out["rung"].apply(lambda r: order.index(r) if r in order else 99)
    out = out.sort_values("__o")

    in_domain = out[out["rung"].str.startswith("1.")]["auroc"]
    base = float(in_domain.iloc[0]) if not in_domain.empty else None
    colors = {"most circular": "#C44E52", "researcher-chosen proxy": "#DD8452",
              "independent label": "#4C72B0", "fully independent": "#2CA02C"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(range(len(out)), out["auroc"], color=[colors.get(c, "#999") for c in out["circularity"]])
    for i, v in enumerate(out["auroc"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    if base is not None:
        ax.axhline(base, ls="--", color="#C44E52", lw=1)
        ax.text(len(out) - 0.5, base + 0.006, "in-domain (circular)", ha="right", fontsize=8, color="#C44E52")
    ax.set_xticks(range(len(out)))
    ax.set_xticklabels(out["rung"], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("anxiety AUROC (TF-IDF)"); ax.set_ylim(0.5, 1.0)
    ax.set_title("The circularity tax: performance vs how independent the test label is")
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    md = [
        "# The circularity ladder",
        "",
        "A single TF-IDF + LogReg model, evaluated against test labels of decreasing dependence on the "
        "researcher's own heuristic. `scripts/exp_circularity_ladder.py`.",
        "",
        "| rung | test label | independence | anxiety AUROC |",
        "|---|---|---|---:|",
    ]
    for _, r in out.iterrows():
        md.append(f"| {r['rung']} | {r['test_label']} | {r['circularity']} | {r['auroc']} |")
    if base is not None:
        indep = out[out["circularity"].isin(["independent label", "fully independent"])]["auroc"]
        if not indep.empty:
            tax = base - float(indep.min())
            md += ["", f"**Circularity tax.** In-domain weak-label AUROC is {base:.3f}, but against labels "
                   f"the lexicon cannot have produced it falls to {float(indep.min()):.3f}--{float(indep.max()):.3f} "
                   f"(a drop of up to {tax:.3f} AUROC). That gap is the share of the headline number that reflects "
                   "the model recovering our labelling heuristic rather than a clinical construct."]
    md += ["", "![circularity ladder](figures/circularity_ladder.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out.drop(columns="__o").to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
