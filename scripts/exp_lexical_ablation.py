"""Lexical-ablation probe: how much of the headline result is keyword spotting?

Train on the r/HealthAnxiety-vs-r/Anxiety task, then evaluate twice: on the
normal test text, and on test text with every clinical-lexicon word/phrase
removed (the same lexicons that build the weak labels). The performance DROP is
the share of the score that comes from matching our own keywords rather than
from the surrounding language.

TF-IDF is run over several seeds (mean +/- std); one transformer is run once.

GPU recommended (for the transformer; the TF-IDF part is CPU). Run:
  python scripts/exp_lexical_ablation.py
  python scripts/exp_lexical_ablation.py --no-transformer   # TF-IDF only, fast
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_ha_vs_anxiety import author_disjoint_split, load_head_to_head, _weighted_f1  # noqa: E402

from src.evaluation.metrics import full_report  # noqa: E402
from src.labeling.lexicons import (  # noqa: E402
    ANXIETY_PHRASES,
    ANXIETY_TERMS,
    BODY_PARTS,
    DEPRESSION_TERMS,
    HEALTH_ANXIETY_PHRASES,
    HEALTH_ANXIETY_TERMS,
    REASSURANCE_PATTERNS,
    SUICIDALITY_TERMS,
)

OUTCSV = Path("experiments/lexical_ablation.csv")
DOC = Path("docs/lexical_ablation.md")
FIG = Path("docs/figures/lexical_ablation.png")

_LEX = set().union(ANXIETY_TERMS, ANXIETY_PHRASES, HEALTH_ANXIETY_TERMS, HEALTH_ANXIETY_PHRASES,
                   REASSURANCE_PATTERNS, DEPRESSION_TERMS, SUICIDALITY_TERMS, BODY_PARTS)
_terms = sorted((t.lower() for t in _LEX), key=len, reverse=True)
_PATTERN = re.compile(
    "|".join((re.escape(t) if " " in t else r"\b" + re.escape(t) + r"\b") for t in _terms),
    re.IGNORECASE,
)


def mask_lexicon(text: str) -> str:
    return _PATTERN.sub(" ", str(text or ""))


def _tfidf(seed):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.95,
                                  sublinear_tf=True, max_features=80000, lowercase=True)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                   solver="liblinear", random_state=seed)),
    ])


def _eval(y, proba):
    rep = full_report(y, proba, bootstrap=False)
    return round(_weighted_f1(y, proba, rep["threshold"]), 4), round(rep["auroc"], 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--no-transformer", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    d, stats = load_head_to_head(submissions_only=True)
    print(f"[submissions-only] {stats}", flush=True)

    rows = []
    # ---- TF-IDF over seeds ----
    tf_clean_f1, tf_mask_f1, tf_clean_au, tf_mask_au = [], [], [], []
    for s in seeds:
        train, test = author_disjoint_split(d, test_size=0.2, seed=s)
        model = _tfidf(s).fit(train["clean_text"].tolist(), train["y"].to_numpy())
        y = test["y"].to_numpy()
        cf, ca = _eval(y, model.predict_proba(test["clean_text"].tolist())[:, 1])
        masked = [mask_lexicon(t) for t in test["clean_text"].tolist()]
        mf, ma = _eval(y, model.predict_proba(masked)[:, 1])
        tf_clean_f1.append(cf); tf_mask_f1.append(mf); tf_clean_au.append(ca); tf_mask_au.append(ma)
        print(f"  TF-IDF seed={s}: clean F1 {cf} / masked F1 {mf}  |  clean AUROC {ca} / masked AUROC {ma}", flush=True)
    rows.append({"model": f"TF-IDF (n={len(seeds)} seeds)",
                 "clean_f1": round(np.mean(tf_clean_f1), 4), "clean_f1_std": round(np.std(tf_clean_f1), 4),
                 "masked_f1": round(np.mean(tf_mask_f1), 4), "masked_f1_std": round(np.std(tf_mask_f1), 4),
                 "clean_auroc": round(np.mean(tf_clean_au), 4), "masked_auroc": round(np.mean(tf_mask_au), 4),
                 "f1_drop": round(np.mean(tf_clean_f1) - np.mean(tf_mask_f1), 4),
                 "auroc_drop": round(np.mean(tf_clean_au) - np.mean(tf_mask_au), 4)})

    # ---- transformer (one run) ----
    if not args.no_transformer:
        from src.models.transformer import TransformerModel
        from src.utils.config import ModelConfig

        train, test = author_disjoint_split(d, test_size=0.2, seed=42)
        tr2, val = author_disjoint_split(train, test_size=0.15, seed=42)
        cfg = ModelConfig(name="lexabl_mr", model_type="transformer", text_field="clean_text",
                          target="ha", targets=None, extra={
                              "pretrained": "mental/mental-roberta-base", "fallback_pretrained": "roberta-base",
                              "tokenizer": {"max_length": 256},
                              "train": {"per_device_train_batch_size": 16, "num_train_epochs": 3,
                                        "learning_rate": 2e-5, "save_strategy": "no", "load_best_model_at_end": False}})
        print("\n=== training transformer (MentalRoBERTa) ===", flush=True)
        model = TransformerModel(cfg).fit(tr2, val=val)
        y = test["label_ha"].to_numpy()
        cf, ca = _eval(y, np.asarray(model.predict_proba(test)))
        masked_df = test.copy(); masked_df["clean_text"] = [mask_lexicon(t) for t in test["clean_text"]]
        mf, ma = _eval(y, np.asarray(model.predict_proba(masked_df)))
        print(f"  transformer: clean F1 {cf} / masked F1 {mf}  |  clean AUROC {ca} / masked AUROC {ma}", flush=True)
        rows.append({"model": "MentalRoBERTa (1 seed)", "clean_f1": cf, "clean_f1_std": 0.0,
                     "masked_f1": mf, "masked_f1_std": 0.0, "clean_auroc": ca, "masked_auroc": ma,
                     "f1_drop": round(cf - mf, 4), "auroc_drop": round(ca - ma, 4)})

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    # figure: clean vs masked weighted-F1 per model
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(out)); w = 0.38
    ax.bar(x - w / 2, out["clean_f1"], w, yerr=out["clean_f1_std"], label="clean text", color="#4C72B0", capsize=3)
    ax.bar(x + w / 2, out["masked_f1"], w, yerr=out["masked_f1_std"], label="lexicon removed", color="#C44E52", capsize=3)
    for i, (c, m) in enumerate(zip(out["clean_f1"], out["masked_f1"])):
        ax.text(i - w / 2, c + 0.006, f"{c:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, m + 0.006, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(out["model"], fontsize=9)
    ax.set_ylabel("weighted-F1 (HA vs Anxiety)"); ax.set_ylim(0, 1.0)
    ax.set_title("Lexical-ablation probe: F1 when the clinical lexicon is removed from the test text")
    ax.legend()
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    cols = ["model", "clean_f1", "masked_f1", "f1_drop", "clean_auroc", "masked_auroc", "auroc_drop"]
    md = [
        "# Lexical-ablation probe (keyword reliance)",
        "",
        "Models trained on r/HealthAnxiety vs r/Anxiety, then evaluated on normal test text and on test "
        "text with every clinical-lexicon word/phrase removed (the same lexicons that build the weak "
        "labels). The drop measures how much of the score is keyword spotting. `scripts/exp_lexical_ablation.py`.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    md += ["", "![lexical ablation](figures/lexical_ablation.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out[cols].to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
