"""Idea 4 — statistical significance for paired model comparisons.

Loads the saved per-post predictions, aligns every model pair on shared post `id`s
(same target), and reports:
  - McNemar's test on each model's saved decisions (`pred_<target>`),
  - paired-bootstrap ΔAUROC with a 95% CI and p-value (`score_<target>`).
Writes experiments/significance.csv, docs/significance.md, and a forest plot.

CPU only, seconds. Run:
  python scripts/significance.py
  python scripts/significance.py --n-boot 5000 --min-common 100
"""
from __future__ import annotations

import argparse
import glob
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.evaluation.significance import mcnemar_test, paired_bootstrap
from src.utils.io import read_parquet

EVAL_GLOB = "experiments/runs/*/eval/*__predictions.parquet"
FIG = Path("docs/figures/significance.png")
OUTCSV = Path("experiments/significance.csv")
DOC = Path("docs/significance.md")


def _parse(path: str) -> tuple[str, str]:
    parts = Path(path).stem.split("__")
    return "__".join(parts[:-2]), parts[-2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-common", type=int, default=100)
    args = ap.parse_args()

    # index predictions by target -> {model: dataframe}
    by_target: dict[str, dict[str, pd.DataFrame]] = {}
    for path in sorted(glob.glob(EVAL_GLOB)):
        name, target = _parse(path)
        cols = [c for c in ("id", f"label_{target}", f"score_{target}", f"pred_{target}")]
        df = read_parquet(path)
        if not all(c in df.columns for c in cols):
            continue
        by_target.setdefault(target, {})[name] = df[cols]

    comparisons = [(t, a, b) for t, models in by_target.items()
                   for a, b in combinations(sorted(models), 2)]
    if not comparisons:
        print("No model pairs to compare.")
        return

    rows: list[dict] = []
    for target, a, b in tqdm(comparisons, desc="comparisons", unit="pair"):
        da, db = by_target[target][a], by_target[target][b]
        m = da.merge(db, on="id", suffixes=("_a", "_b"))
        if len(m) < args.min_common:
            continue
        y = (m[f"label_{target}_a"].astype(float).fillna(0) >= 0.5).astype(int).to_numpy()
        if y.sum() == 0 or (y == 0).sum() == 0:
            continue
        mc = mcnemar_test(y, m[f"pred_{target}_a"].to_numpy(), m[f"pred_{target}_b"].to_numpy())
        bs = paired_bootstrap(y, m[f"score_{target}_a"].to_numpy(), m[f"score_{target}_b"].to_numpy(),
                              metric="auroc", n_boot=args.n_boot, progress=True)
        rows.append({
            "target": target, "model_a": a, "model_b": b, "n_common": int(len(m)), "n_pos": int(y.sum()),
            "mcnemar_b": mc["b"], "mcnemar_c": mc["c"], "mcnemar_p": round(mc["p_value"], 5), "mcnemar_method": mc["method"],
            "auroc_a": round(bs["metric_a"], 4), "auroc_b": round(bs["metric_b"], 4),
            "delta_auroc": round(bs["delta"], 4), "ci_lo": round(bs["ci_lo"], 4), "ci_hi": round(bs["ci_hi"], 4),
            "boot_p": round(bs["p_value"], 5),
            "significant_95": bool(bs["ci_lo"] > 0 or bs["ci_hi"] < 0),
        })

    if not rows:
        print("No pairs had enough shared examples / both classes to test.")
        return

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    # forest plot of delta AUROC with 95% CI
    labels = [f"{r['model_a']}\nvs {r['model_b']} ({r['target']}, n={r['n_common']})" for r in rows]
    deltas = [r["delta_auroc"] for r in rows]
    los = [r["delta_auroc"] - r["ci_lo"] for r in rows]
    his = [r["ci_hi"] - r["delta_auroc"] for r in rows]
    yidx = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 1.1 + 0.9 * len(rows)))
    colors = ["#C44E52" if r["significant_95"] else "#7F7F7F" for r in rows]
    ax.errorbar(deltas, yidx, xerr=[los, his], fmt="o", capsize=4,
                ecolor="#999", mfc="none", ms=8, linestyle="none")
    for i, (d, c) in enumerate(zip(deltas, colors)):
        ax.plot(d, i, "o", color=c, ms=9)
    ax.axvline(0, ls="--", color="k", lw=1)
    ax.set_yticks(yidx); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Δ AUROC (model_a − model_b), 95% bootstrap CI")
    ax.set_title("Idea 4 — paired significance (red = CI excludes 0)")
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    cols = ["target", "model_a", "model_b", "n_common", "n_pos", "mcnemar_b", "mcnemar_c",
            "mcnemar_p", "auroc_a", "auroc_b", "delta_auroc", "ci_lo", "ci_hi", "boot_p", "significant_95"]
    md = [
        "# Statistical significance of paired model comparisons",
        "",
        "McNemar's test on each model's saved decisions + paired-bootstrap ΔAUROC "
        "(95% CI, two-sided p). Models aligned on shared post `id`s. "
        "**`significant_95 = True` means the AUROC difference is real at the 0.05 level.**",
        "",
        "_Regenerate: `python scripts/significance.py`_",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md += ["", "![forest plot](figures/significance.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print(out[cols].to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
