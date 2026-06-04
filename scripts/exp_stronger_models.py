"""Idea 12 — stronger encoders on the r/HealthAnxiety-vs-r/Anxiety task.

Reuses the Experiment-8 pipeline (submissions-only, author-disjoint, weighted-F1)
and swaps in larger / newer encoders, comparing to the published baseline
(Low 2020 SGD-L1 weighted-F1 0.851) and our MentalRoBERTa result (0.906).

Encoders run here (no extra heavy deps): RoBERTa-large, DeBERTa-v3-base.
Heavier variants (domain-adaptive MLM, Llama-3.1-8B QLoRA) are documented as
recipes in docs/stronger_models.md — they need extra deps (peft, bitsandbytes)
and multi-hour GPU runs.

GPU recommended. Run:
  python scripts/exp_stronger_models.py
  python scripts/exp_stronger_models.py --models roberta-large
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_ha_vs_anxiety import author_disjoint_split, load_head_to_head, _weighted_f1  # noqa: E402

from src.evaluation.metrics import full_report  # noqa: E402
from src.models.transformer import TransformerModel  # noqa: E402
from src.utils.config import ModelConfig  # noqa: E402

SEED = 42
LOW_2020 = 0.851
ENCODERS = {
    "roberta-large": "roberta-large",
    "deberta-v3-base": "microsoft/deberta-v3-base",
}
# reference rows from Experiment 8 (already in the repo)
REFERENCE = [
    {"model": "Low 2020 (SGD-L1)", "pretrained": "-", "weighted_f1": LOW_2020, "auroc": None, "note": "published baseline"},
    {"model": "TF-IDF + LogReg", "pretrained": "-", "weighted_f1": 0.886, "auroc": 0.944, "note": "Exp 8"},
    {"model": "MentalRoBERTa", "pretrained": "mental/mental-roberta-base", "weighted_f1": 0.906, "auroc": 0.955, "note": "Exp 8"},
]
DOC = Path("docs/stronger_models.md")
OUTCSV = Path("experiments/stronger_models.csv")
FIG = Path("docs/figures/stronger_models.png")


def train_eval(pretrained: str, train, test, val) -> dict:
    cfg = ModelConfig(
        name=f"ha_{pretrained.split('/')[-1]}", model_type="transformer",
        text_field="clean_text", target="ha", targets=None,
        extra={
            "pretrained": pretrained,
            "fallback_pretrained": pretrained,   # no silent cross-fallback -> failures are explicit
            "tokenizer": {"max_length": 256},
            "train": {"per_device_train_batch_size": 16, "num_train_epochs": 3,
                      "learning_rate": 2.0e-5, "evaluation_strategy": "epoch",
                      "save_strategy": "no", "load_best_model_at_end": False},
        },
    )
    model = TransformerModel(cfg).fit(train, val=val)
    proba = model.predict_proba(test)
    y = test["label_ha"].values
    rep = full_report(y, proba, bootstrap=False)
    return {"weighted_f1": round(_weighted_f1(y, proba, rep["threshold"]), 4),
            "auroc": round(rep["auroc"], 4), "f1": round(rep["f1"], 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ENCODERS))
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()
    chosen = [m.strip() for m in args.models.split(",") if m.strip()]

    d, stats = load_head_to_head(submissions_only=True)
    print(f"[submissions-only] head-to-head: {stats}")
    train, test = author_disjoint_split(d, test_size=args.test_size)
    tr2, val = author_disjoint_split(train, test_size=0.15)
    print(f"train={len(tr2):,} val={len(val):,} test={len(test):,}")

    rows = list(REFERENCE)
    for name in chosen:
        hf = ENCODERS.get(name, name)
        print(f"\n=== training {name} ({hf}) ===")
        try:
            r = train_eval(hf, tr2, test, val)
            rows.append({"model": name, "pretrained": hf, **r, "note": "this run"})
            print(f"  {name}: weighted-F1 {r['weighted_f1']}  AUROC {r['auroc']}")
        except Exception as ex:  # noqa: BLE001
            print(f"  {name} FAILED: {ex}")
            rows.append({"model": name, "pretrained": hf, "weighted_f1": None, "auroc": None,
                         "f1": None, "note": f"FAILED: {str(ex)[:80]}"})

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    # figure: weighted-F1 per model with the Low 2020 baseline line
    plot = out[out["weighted_f1"].notna()].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#7f7f7f" if n in ("Low 2020 (SGD-L1)", "TF-IDF + LogReg") else
              ("#4C72B0" if n == "MentalRoBERTa" else "#C44E52") for n in plot["model"]]
    ax.bar(range(len(plot)), plot["weighted_f1"], color=colors)
    for i, v in enumerate(plot["weighted_f1"]):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(LOW_2020, ls="--", color="k", lw=1, label=f"Low 2020 = {LOW_2020}")
    ax.set_xticks(range(len(plot))); ax.set_xticklabels(plot["model"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("weighted-F1 (submissions)"); ax.set_ylim(0.78, 0.95)
    ax.set_title("Idea 12 — stronger encoders on r/HealthAnxiety vs r/Anxiety")
    ax.legend()
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    cols = ["model", "pretrained", "weighted_f1", "auroc", "f1", "note"]
    md = [
        "# Stronger encoders — r/HealthAnxiety vs r/Anxiety (submissions, author-disjoint)",
        "",
        "Experiment-8 pipeline with larger/newer encoders. Baseline: Low 2020 SGD-L1 "
        f"weighted-F1 = {LOW_2020}. `scripts/exp_stronger_models.py`.",
        "",
        "_Regenerate: `python scripts/exp_stronger_models.py`_",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    md += [
        "", "![stronger models](figures/stronger_models.png)", "",
        "## Heavier variants (recipes — not run here)",
        "",
        "- **Domain-adaptive MLM**: continue masked-LM pretraining of RoBERTa on the 744k-post "
        "corpus (`run_mlm.py`, ~1–2 GPU-hours), then fine-tune. Expected small gain on the noisy "
        "disclosure task; larger on in-domain classification.",
        "- **Llama-3.1-8B QLoRA**: 4-bit `bitsandbytes` + `peft` LoRA adapters, instruction-framed "
        "binary classification. Needs `pip install peft bitsandbytes` and ~2–4 GPU-hours; fits in "
        "24 GB at 4-bit. Decoder-only LLMs rarely beat a fine-tuned encoder on short-text binary "
        "classification, so this is a completeness check, not an expected SOTA.",
    ]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out[cols].to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
