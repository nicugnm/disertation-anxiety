"""Multi-source label model vs the single biased heuristic, anchored on expert labels.

The weak label in this project is one researcher heuristic: 0.5*subreddit_prior +
0.5*our_lexicon. This experiment asks whether COMBINING several diverse, more
independent labelling functions (LFs) agrees with expert clinicians better than
the lexicon alone -- i.e. whether the researcher bias can be reduced.

On the ANGST test set (3 expert psychologists, the only label independent of our
heuristics) we compute several LFs on the raw text:
  - lexicon      : our anxiety word-list hit rate         (the biased signal)
  - sentiment    : VADER negativity                       (independent of our lexicon)
  - uncertainty  : LIWC-style uncertainty markers         (independent)
  - llm          : Qwen2.5-7B zero-shot yes/no            (independent of our lexicon)
then fit an unsupervised Dawid--Skene label model over the LF votes and compare
every source -- and the combination -- against the expert label (Cohen kappa,
F1, AUROC). It also yields the LLM-vs-expert agreement directly.

GPU (for the Qwen LF). Run:
  python scripts/exp_label_model.py
  python scripts/exp_label_model.py --no-llm   # skip the LLM LF (CPU only)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.evaluation.external import load_angst
from src.features.linguistic import extract_dataframe
from src.labeling.weak import lexicon_scores

ANGST_DIR = "data/external/angst"
SEED = 42
OUTCSV = Path("experiments/label_model.csv")
DOC = Path("docs/label_model.md")
FIG = Path("docs/figures/label_model.png")

ABSTAIN = -1


def dawid_skene(votes: np.ndarray, n_iter: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unsupervised binary Dawid--Skene. votes: (n, K) in {0,1,ABSTAIN}.
    Returns posterior P(y=1) per item, per-LF sensitivity, per-LF specificity."""
    n, K = votes.shape
    voted = votes != ABSTAIN
    is1 = votes == 1
    is0 = votes == 0
    pi = 0.5
    sens = np.full(K, 0.7)  # P(vote=1 | y=1)
    spec = np.full(K, 0.7)  # P(vote=0 | y=0)
    r = np.full(n, 0.5)
    for _ in range(n_iter):
        # E-step (log-space)
        log1 = np.log(pi) * np.ones(n)
        log0 = np.log(1 - pi) * np.ones(n)
        for k in range(K):
            a, b = np.clip(sens[k], 1e-3, 1 - 1e-3), np.clip(spec[k], 1e-3, 1 - 1e-3)
            log1 += np.where(is1[:, k], np.log(a), 0) + np.where(is0[:, k], np.log(1 - a), 0)
            log0 += np.where(is0[:, k], np.log(b), 0) + np.where(is1[:, k], np.log(1 - b), 0)
        m = np.maximum(log1, log0)
        r = np.exp(log1 - m) / (np.exp(log1 - m) + np.exp(log0 - m))
        # M-step
        pi = float(r.mean())
        for k in range(K):
            vk = voted[:, k]
            sens[k] = (r[vk] * is1[vk, k]).sum() / max(1e-9, (r[vk]).sum())
            spec[k] = ((1 - r[vk]) * is0[vk, k]).sum() / max(1e-9, (1 - r[vk]).sum())
    return r, sens, spec


def _vote(cont, hi, lo):
    """Map a continuous LF signal to {1, 0, ABSTAIN}: >=hi ->1, <=lo ->0, else abstain."""
    v = np.full(len(cont), ABSTAIN)
    v[cont >= hi] = 1
    v[cont <= lo] = 0
    return v


def _metrics(name, y, hard, cont=None):
    row = {"source": name, "coverage": round(float(np.mean(hard != ABSTAIN)), 3)}
    voted = hard != ABSTAIN
    if voted.sum() > 0:
        row["kappa"] = round(float(cohen_kappa_score(y[voted], hard[voted])), 4)
        row["f1"] = round(float(f1_score(y[voted], (hard[voted] == 1).astype(int), zero_division=0)), 4)
    if cont is not None and len(np.unique(y)) > 1:
        row["auroc"] = round(float(roc_auc_score(y, cont)), 4)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    angst = load_angst(ANGST_DIR, target="anxiety")
    texts = angst["clean_text"].astype(str).fillna("").tolist()
    y = angst["y"].to_numpy().astype(int)
    print(f"ANGST: n={len(y)} pos={int(y.sum())}", flush=True)

    feats = extract_dataframe(angst, text_col="clean_text")
    lex_rate = np.array([lexicon_scores(t).get("anxiety", 0.0) for t in texts])
    sent_neg = -feats["f_sent_compound"].to_numpy()          # higher = more negative
    uncert = feats["f_uncertainty_rate"].to_numpy()

    # labelling functions -> votes in {1,0,ABSTAIN}
    lf = {
        "lexicon (ours, biased)": (_vote(lex_rate, 0.10, 0.0), lex_rate),
        "sentiment (VADER)":      (_vote(sent_neg, 0.50, 0.0), sent_neg),
        "uncertainty (LIWC)":     (_vote(uncert, 0.03, 0.0), uncert),
    }
    if not args.no_llm:
        from src.models.registry import build_model
        from src.utils.config import ModelConfig
        cfg = ModelConfig(name="lf_qwen", model_type="llm_causal", text_field="clean_text",
                          target="anxiety", targets=None,
                          extra={"pretrained": "Qwen/Qwen2.5-7B-Instruct", "load_in_4bit": True,
                                 "lora": {"enabled": False}, "batch_size": 8})
        print("scoring ANGST with Qwen2.5-7B (LF)...", flush=True)
        p_llm = np.asarray(build_model(cfg).predict_proba(angst)).reshape(-1)
        lf["llm (Qwen2.5-7B)"] = ((p_llm >= 0.5).astype(int), p_llm)

    names = list(lf.keys())
    votes = np.stack([lf[n][0] for n in names], axis=1)
    r, sens, spec = dawid_skene(votes)
    maj_cont = np.array([np.mean([v for v in row if v != ABSTAIN]) if any(row != ABSTAIN) else 0.5 for row in votes])

    rows = [_metrics(n, y, lf[n][0], lf[n][1]) for n in names]
    rows.append(_metrics("majority vote", y, (maj_cont >= 0.5).astype(int), maj_cont))
    rows.append(_metrics("label model (Dawid-Skene)", y, (r >= 0.5).astype(int), r))

    # supervised combination: a LITTLE expert ground truth (5-fold over the LF continuous scores).
    # This is the principled fix the unsupervised model cannot match without knowing LF quality.
    short = {n: n.split(" ")[0] for n in names}
    X = np.nan_to_num(np.column_stack([lf[n][1] for n in names]))
    Xs = StandardScaler().fit_transform(X)
    oof = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        Xs, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba",
    )[:, 1]
    rows.append(_metrics("supervised combo (5-fold)", y, (oof >= 0.5).astype(int), oof))
    sdf = pd.DataFrame({short[n]: lf[n][1] for n in names}); sdf["y"] = y
    sdf.to_parquet("experiments/label_model_scores.parquet")

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    # figure: kappa-with-experts per source (label model + LLM vs the lexicon)
    plot = out.dropna(subset=["kappa"]).copy()
    pal = []
    for n in plot["source"]:
        if "label model" in n:
            pal.append("#2CA02C")
        elif "lexicon" in n:
            pal.append("#C44E52")
        elif "llm" in n:
            pal.append("#4C72B0")
        else:
            pal.append("#8C8C8C")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(plot)), plot["kappa"], color=pal)
    for i, v in enumerate(plot["kappa"]):
        ax.text(i, v + 0.004 * np.sign(v + 1e-9), f"{v:.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(plot))); ax.set_xticklabels(plot["source"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Cohen's kappa vs expert labels (ANGST)")
    ax.set_title("Agreement with expert clinicians: a multi-source label model vs the single biased lexicon")
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    cols = ["source", "coverage", "kappa", "f1", "auroc"]
    cols = [c for c in cols if c in out.columns]
    md = [
        "# Multi-source label model vs the single biased heuristic (anchored on expert labels)",
        "",
        "On ANGST (3 expert psychologists), several diverse labelling functions and an unsupervised "
        "Dawid--Skene combination are compared against the expert label. The question: does combining "
        "diverse, more-independent signals agree with clinicians better than our own lexicon alone? "
        "`scripts/exp_label_model.py`.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r_ in rows:
        md.append("| " + " | ".join(str(r_.get(c, "")) for c in cols) + " |")
    md += ["", "![label model](figures/label_model.png)", "",
           "## What this shows",
           "",
           "- The **lexicon** — the *biased* source — is actually the single signal **most** aligned with the "
           "expert clinicians (highest kappa and AUROC), because it is built from clinical instruments "
           "(GAD-7/SHAI). The weak label is not noise about the construct.",
           "- The other weak signals (sentiment, uncertainty, **zero-shot LLM**) agree with experts near "
           "chance, so an **unsupervised** combination (majority / Dawid--Skene) **cannot beat the lexicon**: "
           "with no ground truth it cannot tell which labelling function is reliable and is pulled toward the "
           "noisy ones.",
           "- A **supervised** combination (a little expert data, 5-fold) is what actually improves on the best "
           "single source. The route to less bias is a small amount of expert ground truth, not more "
           "unsupervised researcher heuristics.",
           "- The zero-shot LLM is a **weak anxiety annotator** here, so it cannot simply replace the lexicon.",
           "- Implication: the circularity is in the **evaluation** (testing against lexicon-derived labels "
           "inflates the in-domain score to ~0.99, see the circularity ladder), not in the lexicon's construct "
           "validity.",
           "",
           "Per-LF learned reliability (Dawid--Skene): "
           + ", ".join(f"{n} sens={sens[i]:.2f}/spec={spec[i]:.2f}" for i, n in enumerate(names)) + "."]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out.to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
