"""Experiment 10 — architecture-surgery ablation of the fusion multi-task model.

Trains the ablation matrix (baseline == plain multitask, +fusion, +attn, +focal,
+fusion+focal, all) on an author-disjoint split of the full corpus and evaluates
each on (a) the held-out in-domain test set (per-target best-threshold F1 / AUROC /
AUPRC) and (b) zero-shot external transfer to RMHD + ANGST (anxiety). Tests whether
clinical-feature fusion improves transfer and whether focal loss lifts rare-class F1.

GPU. Each variant trains one transformer (~tens of min). Run:
  python scripts/exp_fusion_ablation.py
  python scripts/exp_fusion_ablation.py --variants baseline,fusion_focal --max-train 60000 --epochs 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.evaluation.external import load_angst, load_rmhd
from src.evaluation.metrics import full_report
from src.models.registry import build_model
from src.utils.config import load_model_config
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
RMHD_DIR = "data/external/rmhd"
ANGST_DIR = "data/external/angst"
SEED = 42
TARGETS = ["anxiety", "health_anxiety", "depression", "suicidality"]
POS_SUBS = ["anxiety", "healthanxiety", "socialanxiety"]
NEG_SUBS = ["fitness", "parenting", "meditation", "conspiracy"]
OUTCSV = Path("experiments/fusion_ablation.csv")
DOC = Path("docs/fusion_ablation.md")
FIG = Path("docs/figures/fusion_ablation.png")

VARIANTS = {
    "baseline":     {"fusion": {"enabled": False}, "attn_pool": {"enabled": False}, "focal": {"enabled": False}},
    "fusion":       {"fusion": {"enabled": True},  "attn_pool": {"enabled": False}, "focal": {"enabled": False}},
    "attn":         {"fusion": {"enabled": False}, "attn_pool": {"enabled": True},  "focal": {"enabled": False}},
    "focal":        {"fusion": {"enabled": False}, "attn_pool": {"enabled": False}, "focal": {"enabled": True, "gamma": 2.0}},
    "fusion_focal": {"fusion": {"enabled": True},  "attn_pool": {"enabled": False}, "focal": {"enabled": True, "gamma": 2.0}},
    "all":          {"fusion": {"enabled": True},  "attn_pool": {"enabled": True},  "focal": {"enabled": True, "gamma": 2.0}},
}


def _cap(df, n):
    return df.sample(n=n, random_state=SEED).reset_index(drop=True) if n and len(df) > n else df


def _external_auroc(predict, df, ti):
    if df is None or df.empty:
        return None
    scores = np.asarray(predict(df))[:, ti]
    return round(float(roc_auc_score(df["y"].to_numpy(), scores)), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--max-eval", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    chosen = [v.strip() for v in args.variants.split(",") if v.strip() in VARIANTS]

    df = read_parquet(DATA)
    df = df[(df["clean_text"].astype(str).str.len() >= 30) & df["author_hash"].notna()].reset_index(drop=True)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(df, groups=df["author_hash"].values))
    train = _cap(df.iloc[tr].reset_index(drop=True), args.max_train)
    test = _cap(df.iloc[te].reset_index(drop=True), args.max_eval)
    print(f"author-disjoint: train={len(train):,} test={len(test):,}")

    rmhd = load_rmhd(RMHD_DIR, POS_SUBS, NEG_SUBS, cap_per_sub=5000, seed=SEED)
    try:
        angst = load_angst(ANGST_DIR, target="anxiety")
    except Exception:  # noqa: BLE001
        angst = None
    ti = TARGETS.index("anxiety")

    rows: list[dict] = []
    for name in chosen:
        cfg = load_model_config("configs/models/fusion_multitask.yaml")
        for k, v in VARIANTS[name].items():
            cfg.extra[k] = v
        cfg.extra.setdefault("train", {})["num_train_epochs"] = args.epochs
        print(f"\n=== training variant: {name} ({VARIANTS[name]}) ===")
        model = build_model(cfg).fit(train, val=None)
        proba = np.asarray(model.predict_proba(test))
        row = {"variant": name}
        for i, t in enumerate(TARGETS):
            y = (test[f"label_{t}"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
            if y.sum() < 5 or (y == 0).sum() < 5:
                continue
            rep = full_report(y, proba[:, i], bootstrap=False)
            row[f"{t}_f1"] = round(rep["f1"], 4)
            row[f"{t}_auroc"] = round(rep["auroc"], 4)
        row["rmhd_auroc"] = _external_auroc(model.predict_proba, rmhd if not rmhd.empty else None, ti)
        row["angst_auroc"] = _external_auroc(model.predict_proba, angst, ti)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTCSV, index=False)
        print(f"  {name}: anxiety_f1={row.get('anxiety_f1')} health_anx_f1={row.get('health_anxiety_f1')} "
              f"suic_f1={row.get('suicidality_f1')} RMHD={row['rmhd_auroc']} ANGST={row['angst_auroc']}")

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    # figure: rare-class F1 + transfer AUROC vs baseline
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(out))
    for col, lab in [("health_anxiety_f1", "health_anx F1"), ("suicidality_f1", "suicidality F1"), ("anxiety_f1", "anxiety F1")]:
        if col in out:
            axes[0].plot(x, out[col], "o-", label=lab)
    axes[0].set_xticks(x); axes[0].set_xticklabels(out["variant"], rotation=30, ha="right")
    axes[0].set_title("In-domain F1 by variant"); axes[0].legend(fontsize=8); axes[0].set_ylabel("F1")
    for col, lab in [("rmhd_auroc", "RMHD AUROC"), ("angst_auroc", "ANGST AUROC")]:
        if col in out:
            axes[1].plot(x, out[col], "o-", label=lab)
    axes[1].set_xticks(x); axes[1].set_xticklabels(out["variant"], rotation=30, ha="right")
    axes[1].set_title("Zero-shot transfer AUROC by variant"); axes[1].legend(fontsize=8); axes[1].set_ylabel("AUROC")
    fig.suptitle("Experiment 10 — fusion architecture-surgery ablation", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    cols = ["variant", "anxiety_f1", "health_anxiety_f1", "depression_f1", "suicidality_f1",
            "anxiety_auroc", "rmhd_auroc", "angst_auroc"]
    cols = [c for c in cols if c in out.columns]
    md = [
        "# Experiment 10 — fusion architecture-surgery ablation",
        "",
        "Ablation of `FusionMultiTaskModel` (`src/models/fusion.py`): clinical-feature fusion "
        "(26 linguistic + 7 SHAI), attention pooling, focal loss, on an author-disjoint split of the "
        "full corpus. In-domain held-out test + zero-shot RMHD/ANGST transfer (anxiety). "
        "`scripts/exp_fusion_ablation.py`. baseline == plain multitask.",
        "",
        "_Regenerate: `python scripts/exp_fusion_ablation.py`_",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    md += ["", "![fusion ablation](figures/fusion_ablation.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out.to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
