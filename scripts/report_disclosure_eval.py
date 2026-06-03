"""Aggregate every user-level disclosure eval into one comparison table + figure.

Scans experiments/runs/*/eval/*__disclosure_userlevel.json (written by
`anxiety eval-disclosure`), builds a tidy comparison across models × targets ×
mode (with-disclosure vs disclosure-masked), and emits:

  - experiments/disclosure_userlevel_summary.csv   (machine-readable)
  - docs/figures/disclosure_userlevel.png          (AUROC + F1, with vs masked)
  - docs/disclosure_eval.md                         (self-contained results page)

Run:  python -m scripts.report_disclosure_eval     (or python scripts/report_disclosure_eval.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; never start a GUI event loop
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path("experiments/runs")
FIG = Path("docs/figures")
SUMMARY_CSV = Path("experiments/disclosure_userlevel_summary.csv")
REPORT_MD = Path("docs/disclosure_eval.md")

MODEL_SHORT = {
    "tfidf_logreg": "tfidf",
    "mentalbert_anxiety": "mentalbert",
    "multitask_anxiety_health_dep_suic": "multitask",
}
TARGET_SHORT = {"anxiety": "anx", "health_anxiety": "HA", "depression": "dep", "suicidality": "suic"}
TARGET_ORDER = {"anxiety": 0, "health_anxiety": 1, "depression": 2, "suicidality": 3}
METRICS = ("precision", "recall", "f1", "auroc", "auprc")


def collect_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for fp in sorted(RUNS.glob("*/eval/*__disclosure_userlevel.json")):
        payload = json.loads(fp.read_text())
        model = payload.get("model", fp.parent.parent.name)
        target = payload.get("target", "?")
        for mode_key, mode_label in (
            ("with_disclosure_posts", "with"),
            ("without_disclosure_posts", "masked"),
        ):
            rep = payload.get(mode_key)
            if not rep:
                continue
            row = {
                "model": model,
                "target": target,
                "mode": mode_label,
                "n_users": rep.get("n_users"),
                "n_positive_users": rep.get("n_positive_users"),
                "aggregation": rep.get("aggregation"),
            }
            for m in METRICS:
                row[m] = rep.get(m)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_t"] = df["target"].map(lambda t: TARGET_ORDER.get(t, 9))
    df = df.sort_values(["_t", "model", "mode"]).drop(columns="_t").reset_index(drop=True)
    return df


def _eval_label(model: str, target: str) -> str:
    return f"{MODEL_SHORT.get(model, model)}/{TARGET_SHORT.get(target, target)}"


def plot_comparison(df: pd.DataFrame, out_path: Path) -> Path:
    # One column per (model, target); two bars (with vs masked) per metric panel.
    combos = (
        df[["model", "target"]].drop_duplicates()
        .assign(_t=lambda d: d["target"].map(lambda t: TARGET_ORDER.get(t, 9)))
        .sort_values(["_t", "model"]).drop(columns="_t")
    )
    labels = [_eval_label(m, t) for m, t in zip(combos["model"], combos["target"])]
    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(max(8, 1.4 * len(labels)), 8))
    for ax, metric in zip(axes, ("auroc", "f1")):
        for i, (mode, off, color) in enumerate(
            (("with", -width / 2, "#4c72b0"), ("masked", width / 2, "#c44e52"))
        ):
            vals = []
            for m, t in zip(combos["model"], combos["target"]):
                sel = df[(df["model"] == m) & (df["target"] == t) & (df["mode"] == mode)]
                vals.append(float(sel[metric].iloc[0]) if len(sel) else np.nan)
            bars = ax.bar(x + off, vals, width, label=f"{mode} disclosure", color=color)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=8)
        if metric == "auroc":
            ax.axhline(0.5, ls="--", lw=1, color="gray", label="chance (AUROC=0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"User-level disclosure {metric.upper()} (mean aggregation)")
        ax.legend(loc="upper right", frameon=True)
    fig.suptitle("User-level self-disclosure evaluation — model × target", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_markdown(df: pd.DataFrame, fig_path: Path, out_path: Path) -> Path:
    lines: list[str] = []
    lines.append("# User-level self-disclosure evaluation\n")
    lines.append(
        "Each trained model scores every post by the held-out disclosure-test users, the "
        "per-post scores are aggregated per user (mean), and the per-user score is compared "
        "against the user's self-disclosure label. **with** = the explicit “I was diagnosed…” "
        "posts are included; **masked** = those posts are removed first, so the score reflects only "
        "the *implicit* signal in the user's other posts (the honest generalization number).\n"
    )
    lines.append(f"![disclosure eval](figures/{fig_path.name})\n")
    cols = ["model", "target", "mode", "n_users", "n_positive_users",
            "precision", "recall", "f1", "auroc", "auprc"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in METRICS and pd.notna(v):
                cells.append(f"{float(v):.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("_Generated by `scripts/report_disclosure_eval.py` from "
                 "`experiments/runs/*/eval/*__disclosure_userlevel.json`._")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    df = collect_rows()
    if df.empty:
        print("No *__disclosure_userlevel.json files found under experiments/runs/*/eval/.")
        return
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    fig = plot_comparison(df, FIG / "disclosure_userlevel.png")
    md = write_markdown(df, fig, REPORT_MD)
    print(df.to_string(index=False))
    print(f"\nCSV    -> {SUMMARY_CSV}")
    print(f"Figure -> {fig}")
    print(f"Report -> {md}")


if __name__ == "__main__":
    main()
