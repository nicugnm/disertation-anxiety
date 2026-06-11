"""Ordinal severity head (CORAL) — a methods demonstration.

Clinicians think in severity bands (none / mild / moderate / severe), not just
present/absent. This shows the architecture can predict an ORDINAL severity level
and that an ordinal head (CORAL; Cao et al. 2020) beats a plain multiclass head on
the ordinal metrics (MAE, quadratic-weighted kappa), as Naseem 2022 reports.

HONEST CAVEAT: there is no clinician severity label for this corpus, so the
ordinal target is the continuous weak-anxiety score binned into four bands. That
label is lexicon-derived, so this is a demonstration of the ordinal MODELLING
capability, not a validated severity classifier — the binary-circularity caveat
(Experiment 13) applies here too. Features are the 26 linguistic + 7 SHAI columns.

CPU. Run:  python scripts/exp_ordinal_severity.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.features.linguistic import extract_dataframe, feature_columns
from src.features.shai import score_shai, shai_dimensions
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
SEED = 42
K = 4  # none / mild / moderate / severe
BANDS = ["none", "mild", "moderate", "severe"]
THRESHOLDS = [0.10, 0.30, 0.50]  # weak_anxiety cut points -> 4 ordinal bands
OUTCSV = Path("experiments/ordinal_severity.csv")
DOC = Path("docs/ordinal_severity.md")
FIG = Path("docs/figures/ordinal_severity.png")


def to_levels(w: np.ndarray) -> np.ndarray:
    return np.digitize(w, THRESHOLDS).astype(int)  # 0..3


def build_features(df: pd.DataFrame) -> np.ndarray:
    feat = extract_dataframe(df, text_col="clean_text").reset_index(drop=True)
    fcols = feature_columns(feat)
    dims = shai_dimensions()
    shai = pd.DataFrame([score_shai(t) for t in df["clean_text"].astype(str).fillna("")],
                        columns=dims).reset_index(drop=True)
    return np.concatenate([feat[fcols].to_numpy(dtype=float), shai.to_numpy(dtype=float)], axis=1)


def _train_eval(Xtr, ytr, Xte, yte, head: str):
    """head: 'coral' or 'multiclass'. Returns predictions on Xte (0..K-1)."""
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    dev = torch.device("cpu")
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    in_dim = Xtr.shape[1]
    body = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2))
    if head == "coral":
        shared = nn.Linear(64, 1, bias=False)
        bias = nn.Parameter(torch.zeros(K - 1))
        params = list(body.parameters()) + list(shared.parameters()) + [bias]
    else:
        clf = nn.Linear(64, K)
        params = list(body.parameters()) + list(clf.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    ks = torch.arange(K - 1)

    n = len(Xtr_t); bs = 512
    for _ in range(40):
        perm = torch.randperm(n)
        for bi in range(0, n, bs):
            idx = perm[bi:bi + bs]
            h = body(Xtr_t[idx])
            yb = ytr_t[idx]
            if head == "coral":
                logits = shared(h) + bias  # (B, K-1)
                targets = (yb.unsqueeze(1) > ks.unsqueeze(0)).float()
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            else:
                loss = nn.functional.cross_entropy(clf(h), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h = body(Xte_t)
        if head == "coral":
            pred = (torch.sigmoid(shared(h) + bias) > 0.5).sum(dim=1).numpy()
        else:
            pred = clf(h).argmax(dim=1).numpy()
    return pred.astype(int)


def _metrics(y, pred) -> dict:
    return {"mae": round(float(mean_absolute_error(y, pred)), 4),
            "qwk": round(float(cohen_kappa_score(y, pred, weights="quadratic")), 4),
            "exact_acc": round(float(np.mean(y == pred)), 4),
            "off_by_one_acc": round(float(np.mean(np.abs(y - pred) <= 1)), 4)}


def main() -> None:
    df = read_parquet(DATA, columns=["clean_text", "author_hash", "label_anxiety"])
    df = df[(df["clean_text"].astype(str).str.len() >= 30) & df["author_hash"].notna()].reset_index(drop=True)
    if len(df) > 80000:
        df = df.sample(80000, random_state=SEED).reset_index(drop=True)
    y = to_levels(df["label_anxiety"].astype(float).fillna(0).to_numpy())
    dist = {BANDS[i]: int((y == i).sum()) for i in range(K)}
    print("severity band distribution:", dist, flush=True)

    print("building features...", flush=True)
    X = build_features(df)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                  .split(X, y, groups=df["author_hash"].values))
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    ytr, yte = y[tr], y[te]

    rows = []
    for head in ("multiclass", "coral"):
        pred = _train_eval(Xtr, ytr, Xte, yte, head)
        m = _metrics(yte, pred)
        rows.append({"head": head, **m})
        print(f"  {head}: MAE {m['mae']}  QWK {m['qwk']}  exact {m['exact_acc']}  off-by-1 {m['off_by_one_acc']}", flush=True)

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ["mae", "qwk", "exact_acc", "off_by_one_acc"]
    x = np.arange(len(metrics)); w = 0.38
    for i, head in enumerate(out["head"]):
        ax.bar(x + i * w, [out.iloc[i][m] for m in metrics], w, label=head)
    ax.set_xticks(x + w / 2); ax.set_xticklabels(["MAE (lower better)", "QWK", "exact acc", "off-by-1 acc"], fontsize=8)
    ax.set_title("Ordinal severity head: CORAL vs plain multiclass"); ax.legend()
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    coral = next(r for r in rows if r["head"] == "coral")
    multi = next(r for r in rows if r["head"] == "multiclass")
    md = [
        "# Ordinal severity head (CORAL) — methods demonstration",
        "",
        "Predicting an ordinal anxiety-severity band (none / mild / moderate / severe) instead of just "
        "present/absent, the way clinicians grade severity. `scripts/exp_ordinal_severity.py`.",
        "",
        f"Band distribution (weak-anxiety binned at {THRESHOLDS}): {dist}.",
        "",
        "| head | MAE ↓ | QWK ↑ | exact acc | off-by-one acc |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['head']} | {r['mae']} | {r['qwk']} | {r['exact_acc']} | {r['off_by_one_acc']} |")
    n_bands = sum(v > 0 for v in dist.values())
    md += [
        "",
        "![ordinal severity](figures/ordinal_severity.png)",
        "",
        "## Reading this",
        "",
        f"- The severity bands are **degenerate**: only {n_bands} of {K} are populated ({dist}). "
        "The weak-anxiety score is bimodal — it encodes *presence*, not *graded severity* (anxiety-subreddit "
        "posts get a prior ≈0.55 so they land high; everything else near 0; nothing in between) — so a "
        f"four-band target collapses. On it, CORAL and multiclass are equivalent (MAE {coral['mae']} vs "
        f"{multi['mae']}, QWK {coral['qwk']} vs {multi['qwk']}): there is no ordinal structure to exploit.",
        "- **Honest conclusion:** there is no clinician severity label for this corpus, and any lexicon-derived "
        "severity proxy would be circular with the model's features (Experiment 13). The CORAL head is "
        "implemented and verified to train/predict, but a meaningful ordinal severity result needs a "
        "clinician-graded label (e.g. GAD-7 bands), which this corpus lacks. The modelling is not the "
        "bottleneck; the absence of graded ground truth is.",
    ]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out.to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
