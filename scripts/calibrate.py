"""Idea 2 — post-hoc calibration via temperature scaling.

Runs on the saved held-out test predictions (experiments/runs/*/eval/*__predictions.parquet)
— CPU only, no GPU, no retraining. For each model x target it:
  1. splits the test predictions 50/50 (stratified) into a calibration and a test half,
  2. fits a single temperature T on the calibration half (NLL),
  3. reports ECE / Brier BEFORE vs AFTER on the test half (AUROC is unchanged — the
     transform is monotonic),
  4. saves a before/after reliability diagram.
Writes experiments/calibration.csv and docs/calibration.md.

Run:
  python scripts/calibrate.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.evaluation.calibration import TemperatureScaler
from src.evaluation.metrics import calibration_curve_data, expected_calibration_error
from src.utils.io import read_parquet

EVAL_GLOB = "experiments/runs/*/eval/*__predictions.parquet"
FIGDIR = Path("docs/figures")
OUTCSV = Path("experiments/calibration.csv")
DOC = Path("docs/calibration.md")
SEED = 42
MIN_CLASS = 10  # need at least this many pos & neg to calibrate/measure


def _parse(path: str) -> tuple[str, str]:
    """`<run>/eval/<name>__<target>__predictions.parquet` -> (name, target)."""
    parts = Path(path).stem.split("__")
    return "__".join(parts[:-2]), parts[-2]


def _plot(y, p_before, p_after, name, target, ece_b, ece_a) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 5.3))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for p, lab, col, ece in [(p_before, "before", "#DD8452", ece_b), (p_after, "after", "#55A868", ece_a)]:
        _, observed, predicted, counts = calibration_curve_data(y, p, n_bins=10)
        m = counts > 0
        ax.plot(predicted[m], observed[m], "o-", color=col, label=f"{lab} (ECE={ece:.3f})")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title(f"{name} — {target}"); ax.legend(loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"calibration_{name}__{target}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def main() -> None:
    paths = sorted(glob.glob(EVAL_GLOB))
    if not paths:
        print(f"No prediction artifacts found at {EVAL_GLOB}. Run the eval pipeline first.")
        return

    rows: list[dict] = []
    for path in tqdm(paths, desc="calibrating", unit="model"):
        name, target = _parse(path)
        df = read_parquet(path)
        lc, sc = f"label_{target}", f"score_{target}"
        if lc not in df.columns or sc not in df.columns:
            continue
        y = (df[lc].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
        p = df[sc].astype(float).clip(0, 1).to_numpy()
        if int(y.sum()) < MIN_CLASS or int((y == 0).sum()) < MIN_CLASS:
            continue
        p_cal, p_test, y_cal, y_test = train_test_split(
            p, y, test_size=0.5, random_state=SEED, stratify=y
        )
        scaler = TemperatureScaler().fit(p_cal, y_cal)
        p_test_cal = scaler.transform(p_test)
        ece_b = expected_calibration_error(y_test, p_test)
        ece_a = expected_calibration_error(y_test, p_test_cal)
        rows.append({
            "model": name, "target": target, "n_test": int(len(y_test)), "n_pos": int(y_test.sum()),
            "temperature": round(scaler.temperature, 3),
            "ece_before": round(ece_b, 4), "ece_after": round(ece_a, 4),
            "ece_reduction_%": round(100 * (ece_b - ece_a) / ece_b, 1) if ece_b > 0 else 0.0,
            "brier_before": round(brier_score_loss(y_test, p_test), 4),
            "brier_after": round(brier_score_loss(y_test, p_test_cal), 4),
            "auroc": round(roc_auc_score(y_test, p_test), 4),
            "figure": _plot(y_test, p_test, p_test_cal, name, target, ece_b, ece_a).as_posix(),
        })

    if not rows:
        print("No artifacts had enough positives/negatives to calibrate.")
        return

    out = pd.DataFrame(rows).sort_values(["target", "model"]).reset_index(drop=True)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    cols = ["model", "target", "n_test", "n_pos", "temperature",
            "ece_before", "ece_after", "ece_reduction_%", "brier_before", "brier_after", "auroc"]
    md = [
        "# Probability calibration — temperature scaling",
        "",
        "Post-hoc calibration (Guo et al., 2017): a single scalar **T** rescales the logits "
        "(`p' = sigmoid(logit(p)/T)`), fit by NLL on a held-out half of each model's test "
        "predictions and evaluated on the other half. The transform is monotonic, so **AUROC is "
        "unchanged** — only ECE/Brier improve. **T > 1 ⇒ the model was overconfident.**",
        "",
        "_Regenerate: `python scripts/calibrate.py`_",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md += ["", "## Reliability diagrams (before vs after)", ""]
    for r in sorted(rows, key=lambda r: (r["target"], r["model"])):
        md.append(f"### {r['model']} — {r['target']}")
        md.append(f"![calibration]({Path(r['figure']).relative_to('docs').as_posix()})")
        md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")

    print(out[cols].to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, and {len(rows)} reliability diagrams under {FIGDIR}/")


if __name__ == "__main__":
    main()
