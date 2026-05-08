"""Reusable plotting functions. Each takes data + an output path and writes a PNG.

All plots use a single style configured via `set_style()`. Colors are
colorblind-friendly (the seaborn 'colorblind' palette).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Global style
# --------------------------------------------------------------------------- #


def set_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="colorblind",
        font_scale=1.05,
        rc={
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def _save(fig: plt.Figure, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Corpus-level
# --------------------------------------------------------------------------- #


def plot_corpus_overview(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Posts per subreddit + average body length per subreddit (two panels)."""
    counts = df["subreddit"].value_counts().sort_values()
    avg_len = df.groupby("subreddit")["clean_text"].apply(lambda s: s.str.len().mean()).reindex(counts.index)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.barplot(x=counts.values, y=counts.index, ax=axes[0], color=sns.color_palette()[0])
    axes[0].set_xlabel("Posts (after preprocessing)")
    axes[0].set_ylabel("")
    axes[0].set_title("Corpus size per subreddit")
    for i, v in enumerate(counts.values):
        axes[0].text(v, i, f" {v:,}", va="center")

    sns.barplot(x=avg_len.values, y=avg_len.index, ax=axes[1], color=sns.color_palette()[2])
    axes[1].set_xlabel("Mean post length (chars)")
    axes[1].set_ylabel("")
    axes[1].set_title("Average post length")
    axes[1].set_yticklabels([])  # already shown on the left panel

    fig.suptitle("Corpus overview", fontweight="bold")
    return _save(fig, out_path)


def plot_length_distribution(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Box plot of post length by subreddit."""
    work = df.assign(length=df["clean_text"].str.len())
    order = work.groupby("subreddit")["length"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(data=work, x="length", y="subreddit", order=order, ax=ax, showfliers=False)
    ax.set_xlim(0, work["length"].quantile(0.99))
    ax.set_xlabel("Post length (chars, x-axis truncated at 99th percentile)")
    ax.set_ylabel("")
    ax.set_title("Post-length distribution by subreddit")
    return _save(fig, out_path)


def plot_temporal(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Posts per month, stacked by subreddit (top 6 most-frequent for clarity)."""
    work = df.copy()
    work["month"] = pd.to_datetime(work["created_utc"], unit="s").dt.to_period("M").astype(str)
    top_subs = work["subreddit"].value_counts().head(6).index.tolist()
    work["sub_grp"] = work["subreddit"].where(work["subreddit"].isin(top_subs), "other")
    pivot = (
        work.groupby(["month", "sub_grp"]).size().unstack(fill_value=0).sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot.area(ax=ax, alpha=0.85, linewidth=0)
    # Show every ~12th tick to avoid label overlap
    ticks = list(range(0, len(pivot), max(1, len(pivot) // 12)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([pivot.index[i] for i in ticks], rotation=45, ha="right")
    ax.set_xlabel("Month")
    ax.set_ylabel("Posts")
    ax.set_title("Posts over time (stacked by subreddit)")
    ax.legend(title="", loc="upper left", ncol=2, frameon=True)
    return _save(fig, out_path)


def plot_label_distribution(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Heatmap of weak-positive rate per (subreddit × label)."""
    label_cols = [c for c in ("label_anxiety", "label_health_anxiety", "label_depression", "label_suicidality") if c in df.columns]
    pretty = {c: c.replace("label_", "").replace("_", " ") for c in label_cols}
    work = df.copy()
    for c in label_cols:
        work[c] = (work[c].astype(float).fillna(0) >= 0.5).astype(int)
    pivot = work.groupby("subreddit")[label_cols].mean().sort_values(label_cols[0], ascending=False)
    pivot.columns = [pretty[c] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        cbar_kws={"label": "fraction of positives"},
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Weak-label positive rate by subreddit × label")
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Model performance
# --------------------------------------------------------------------------- #


def plot_pr_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: str | Path, title: str = "") -> Path:
    """ROC and Precision-Recall curves side by side."""
    from sklearn.metrics import (
        auc,
        precision_recall_curve,
        roc_curve,
    )

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(fpr, tpr, lw=2.2, color=sns.color_palette()[0], label=f"AUROC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, lw=2.2, color=sns.color_palette()[3], label=f"AUPRC = {pr_auc:.3f}")
    axes[1].axhline(float(np.mean(y_true)), ls="--", color="gray", lw=1, label=f"baseline = {np.mean(y_true):.2f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend(loc="lower left")

    if title:
        fig.suptitle(title, fontweight="bold")
    return _save(fig, out_path)


def plot_calibration(y_true: np.ndarray, y_score: np.ndarray, out_path: str | Path, n_bins: int = 10) -> Path:
    """Reliability diagram + score histogram."""
    from src.evaluation.metrics import calibration_curve_data, expected_calibration_error

    centers, observed, predicted, counts = calibration_curve_data(y_true, y_score, n_bins=n_bins)
    ece = expected_calibration_error(y_true, y_score, n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    mask = ~np.isnan(observed)
    axes[0].plot(predicted[mask], observed[mask], "o-", color=sns.color_palette()[0], lw=2, ms=8)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed positive rate")
    axes[0].set_title(f"Reliability diagram  (ECE = {ece:.3f})")

    axes[1].bar(centers, counts, width=1 / n_bins * 0.9, color=sns.color_palette()[2], alpha=0.85)
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("# samples")
    axes[1].set_title("Score histogram")
    return _save(fig, out_path)


def plot_subreddit_f1(by_sub: pd.DataFrame, out_path: str | Path, title: str = "") -> Path:
    """Per-subreddit F1 bar chart, sorted descending."""
    work = by_sub.sort_values("f1", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(work["subreddit"], work["f1"], color=sns.color_palette()[0])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1")
    ax.set_ylabel("")
    ax.set_title(title or "Per-subreddit F1")
    for bar, n, npos in zip(bars, work["n"], work["n_pos"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f" n={int(n)}, pos={int(npos)}", va="center", fontsize=9, color="dimgray")
    return _save(fig, out_path)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path) -> Path:
    """2x2 confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title("Confusion matrix")
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Linguistic markers
# --------------------------------------------------------------------------- #


def plot_linguistic_markers(markers: pd.DataFrame, out_path: str | Path, top_n: int = 15) -> Path:
    """Cohen's-d barplot for the top-N strongest features (signed)."""
    df = markers.copy()
    df["abs_d"] = df["cohen_d"].abs()
    df = df.sort_values("abs_d", ascending=False).head(top_n)
    df = df.sort_values("cohen_d")  # ascending for nicer barplot

    colors = ["#4c72b0" if v > 0 else "#c44e52" for v in df["cohen_d"]]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(df["feature"], df["cohen_d"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Cohen's d   (negative = lower in positives, positive = higher in positives)")
    ax.set_ylabel("")
    ax.set_title("Top discriminative linguistic markers")
    # Significance star
    for i, (_, row) in enumerate(df.iterrows()):
        sig = "***" if row.get("p_bh", 1) < 0.001 else ("**" if row.get("p_bh", 1) < 0.01 else ("*" if row.get("p_bh", 1) < 0.05 else ""))
        if sig:
            ax.text(row["cohen_d"] * 1.02 if row["cohen_d"] >= 0 else row["cohen_d"] * 1.02 - 0.01,
                    i, sig, va="center", fontsize=11, fontweight="bold")
    return _save(fig, out_path)


def plot_label_cooccurrence(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Phi-coefficient matrix between binarized labels."""
    cols = [c for c in ("label_anxiety", "label_health_anxiety", "label_depression", "label_suicidality") if c in df.columns]
    bin_df = pd.DataFrame({c.replace("label_", ""): (df[c].astype(float).fillna(0) >= 0.5).astype(int) for c in cols})
    corr = bin_df.corr()  # phi-coefficient is Pearson on binary
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "phi (= Pearson on binary)"})
    ax.set_title("Label co-occurrence (phi coefficient)")
    return _save(fig, out_path)
