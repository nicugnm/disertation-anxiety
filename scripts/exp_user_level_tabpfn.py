"""Final lever: TabPFN (tabular foundation model) on the user-level feature matrix.

Research (Hollmann et al., Nature 2025) reports TabPFN beats tuned GBDT on small
tabular data with no tuning. This reuses the EXACT features from
exp_user_level_push.py and runs TabPFN (5-seed CV) for all three targets, to see
whether it beats the tree ensembles (anxiety RF 0.842 / health-anxiety ExtraTrees
0.891 / depression stacking 0.827).

NOTE: as of 2026 TabPFN's weights are license-gated by Prior Labs -- running this
requires registering at https://ux.priorlabs.ai and setting TABPFN_TOKEN before
.fit() will download the model. Without the token it raises TabPFNLicenseError.
The script is otherwise ready and reuses the exact exp_user_level_push features.

Run:  TABPFN_TOKEN=... python scripts/exp_user_level_tabpfn.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_user_level_push import DISC, base_features, build_scorer, score_features  # noqa: E402

from src.utils.io import read_parquet  # noqa: E402

OUTCSV = Path("experiments/user_level_tabpfn.csv")


def main() -> None:
    disc = read_parquet(DISC)
    masked = disc[disc["is_disclosure_post"] == 0].copy()
    masked = masked[masked["clean_text"].astype(str).str.len() >= 1].reset_index(drop=True)
    disc_authors = set(disc["author_hash"].dropna())
    print(f"cohort: {masked['author_hash'].nunique()} users", flush=True)
    base = base_features(masked)

    import torch
    from tabpfn import TabPFNClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TabPFN device: {dev}", flush=True)

    rows = []
    for target in ("anxiety", "health_anxiety", "depression"):
        scorer = build_scorer(target, disc_authors)
        ps = scorer.predict_proba(masked["clean_text"].tolist())[:, 1]
        udf = score_features(masked, ps, target).join(base, how="left").fillna(0.0)
        y = udf["label"].to_numpy().astype(int)
        X = udf[[c for c in udf.columns if c != "label"]].to_numpy(dtype=float)
        aus, aps = [], []
        for sd in (42, 1, 2, 3, 4):
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
                clf = TabPFNClassifier(device=dev, ignore_pretraining_limits=True)
                clf.fit(X[tr], y[tr])
                oof[te] = clf.predict_proba(X[te])[:, 1]
            aus.append(roc_auc_score(y, oof)); aps.append(average_precision_score(y, oof))
        au, sd_, ap = float(np.mean(aus)), float(np.std(aus)), float(np.mean(aps))
        print(f"  {target}: TabPFN AUROC {au:.4f} +/- {sd_:.4f}  AP {ap:.4f}", flush=True)
        rows.append({"target": target, "model": "tabpfn", "auroc": round(au, 4),
                     "auroc_std": round(sd_, 4), "ap": round(ap, 4)})

    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTCSV, index=False)
    print(f"\nWrote {OUTCSV}")


if __name__ == "__main__":
    main()
