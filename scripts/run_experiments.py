"""Multi-classifier experiment suite — runs without LLM/transformer/GPU.

Produces five distinct studies + plots + a JSON summary you can paste into
the thesis.

  1. Per-target model comparison: TF-IDF + LogReg vs XGBoost-on-features,
     for each of {anxiety, health_anxiety, depression, suicidality}.
  2. Cross-subreddit transfer (RQ3): train on anxiety-primary, test on baseline.
  3. Subreddit classifier: 9-way; confusion matrix.
  4. Per-target linguistic-marker heatmap.
  5. Health-anxiety severity ranking — top 20 highest-scoring posts.

All results are written under experiments/ and figures under docs/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.analysis.linguistic_markers import compare_features_by_label
from src.evaluation.metrics import basic_metrics, full_report
from src.features.linguistic import extract_dataframe, extract_one, feature_columns
from src.models.base import BaseModel
from src.models.registry import build_model
from src.utils.config import ModelConfig
from src.viz.plots import (
    plot_marker_heatmap,
    plot_per_target_comparison,
    plot_subreddit_confusion,
    plot_transfer_drop,
    set_style,
)

console = Console()
set_style()


def make_progress() -> Progress:
    """Rich Progress with a useful column set."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def status(msg: str):
    """Indeterminate spinner for opaque steps (sklearn .fit, etc.)."""
    return console.status(f"[cyan]{msg}", spinner="dots")

OUT = Path("experiments")
FIG = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TARGETS = ("anxiety", "health_anxiety", "depression", "suicidality")
SEED = 42


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
df = pd.read_parquet("data/processed/labeled.parquet")
df = df[df["clean_text"].astype(str).str.len() >= 30].reset_index(drop=True)
console.print(f"[bold]Corpus:[/bold] {len(df)} posts across {df['subreddit'].nunique()} subreddits\n")


def _y(df_, target):
    return (df_[f"label_{target}"].astype(float).fillna(0.0) >= 0.5).astype(int).values


# --------------------------------------------------------------------------- #
# 1. Per-target: TF-IDF + LogReg vs XGBoost on linguistic features
# --------------------------------------------------------------------------- #
console.rule("[1/5] Per-target model comparison")

# Pre-compute linguistic features once (used by XGBoost). This is the slowest
# bulk step — ~13k posts × ~25 features each — so it gets the biggest bar.
texts = df["clean_text"].fillna("").tolist()
feat_rows: list[dict] = []
with make_progress() as bar:
    task = bar.add_task("Extracting linguistic features", total=len(texts))
    for t in texts:
        feat_rows.append(extract_one(t))
        bar.advance(task)
feat_df = pd.concat(
    [df.reset_index(drop=True), pd.DataFrame(feat_rows)], axis=1
)
feat_cols = feature_columns(feat_df)

results_rows: list[dict] = []
with make_progress() as bar:
    task = bar.add_task("Training per-target classifiers", total=len(TARGETS) * 2)
    for target in TARGETS:
        y = _y(df, target)
        n_pos = int(y.sum())
        if n_pos < 5:
            console.print(f"  [yellow]{target}: n_pos={n_pos} — skipping ML, will evaluate as ranking only[/yellow]")
            bar.advance(task, advance=2)
            continue

        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=SEED,
            stratify=y if len(set(y)) > 1 else None,
        )
        y_train, y_test = y[train_idx], y[test_idx]

        # ---- TF-IDF + LogReg
        bar.update(task, description=f"[cyan]{target}[/]: TF-IDF + LogReg")
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.95,
                                      sublinear_tf=True, lowercase=True, max_features=80000)),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                       max_iter=1000, solver="liblinear", random_state=SEED)),
        ])
        pipe.fit(df.iloc[train_idx]["clean_text"].tolist(), y_train)
        proba_tfidf = pipe.predict_proba(df.iloc[test_idx]["clean_text"].tolist())[:, 1]
        rep_tfidf = full_report(y_test, proba_tfidf, bootstrap=False)
        rep_tfidf.update({"target": target, "model": "TF-IDF + LogReg",
                          "support_pos": n_pos, "support_neg": int((y == 0).sum())})
        results_rows.append(rep_tfidf)
        console.print(f"  {target:<16} TF-IDF + LogReg : F1={rep_tfidf['f1']:.3f}  AUROC={rep_tfidf['auroc']:.3f}")
        bar.advance(task)

        # ---- XGBoost on linguistic features
        bar.update(task, description=f"[cyan]{target}[/]: XGBoost (linguistic)")
        from xgboost import XGBClassifier

        X_train = feat_df.iloc[train_idx][feat_cols].values
        X_test = feat_df.iloc[test_idx][feat_cols].values
        spw = max(1.0, (len(y_train) - y_train.sum()) / max(1, y_train.sum()))
        xgb = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            random_state=SEED, eval_metric="logloss", tree_method="hist",
        )
        xgb.fit(X_train, y_train)
        proba_xgb = xgb.predict_proba(X_test)[:, 1]
        rep_xgb = full_report(y_test, proba_xgb, bootstrap=False)
        rep_xgb.update({"target": target, "model": "XGBoost (linguistic)",
                        "support_pos": n_pos, "support_neg": int((y == 0).sum())})
        results_rows.append(rep_xgb)
        console.print(f"  {target:<16} XGBoost (ling.) : F1={rep_xgb['f1']:.3f}  AUROC={rep_xgb['auroc']:.3f}")
        bar.advance(task)

results_df = pd.DataFrame(results_rows)
results_df.to_csv(OUT / "exp1_per_target.csv", index=False)
plot_per_target_comparison(results_df, FIG / "exp1__per_target_f1.png", metric="f1")
plot_per_target_comparison(results_df, FIG / "exp1__per_target_auroc.png", metric="auroc")
console.print(f"  → {FIG / 'exp1__per_target_f1.png'}")


# --------------------------------------------------------------------------- #
# 2. Cross-subreddit transfer (RQ3)
# --------------------------------------------------------------------------- #
console.rule("[2/5] Cross-subreddit transfer (RQ3)")

train_subs = ["Anxiety", "socialanxiety", "AnxietyDepression"]
test_subs = ["COVID19_support", "LivingAlone", "relationship_advice"]
df_train_xs = df[df["subreddit"].isin(train_subs)].reset_index(drop=True)
df_test_xs = df[df["subreddit"].isin(test_subs)].reset_index(drop=True)
console.print(f"  train on {train_subs} (n={len(df_train_xs)})")
console.print(f"  test  on {test_subs} (n={len(df_test_xs)})")

transfer_rows: list[dict] = []
for target in ("anxiety",):  # only target with enough cross-distribution signal
    y_tr = _y(df_train_xs, target)
    y_te = _y(df_test_xs, target)
    if y_tr.sum() < 10 or y_te.sum() < 5:
        console.print(f"  [yellow]{target}: insufficient positives — skipping[/yellow]")
        continue

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.95,
                                  sublinear_tf=True, lowercase=True, max_features=80000)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=1000, solver="liblinear", random_state=SEED)),
    ])
    with status(f"Fitting transfer model on {len(df_train_xs):,} posts ({target})"):
        pipe.fit(df_train_xs["clean_text"].tolist(), y_tr)

    # In-distribution split (held-out within training subs)
    train_idx, val_idx = train_test_split(
        np.arange(len(df_train_xs)), test_size=0.2, random_state=SEED, stratify=y_tr,
    )
    proba_id = pipe.predict_proba(df_train_xs.iloc[val_idx]["clean_text"].tolist())[:, 1]
    rep_id = basic_metrics(y_tr[val_idx], proba_id)
    transfer_rows.append({"target": target, "split": "in-distribution",
                          **{k: v for k, v in rep_id.items() if k in ("f1", "precision", "recall", "auroc")}})

    proba_xs = pipe.predict_proba(df_test_xs["clean_text"].tolist())[:, 1]
    rep_xs = basic_metrics(y_te, proba_xs)
    transfer_rows.append({"target": target, "split": "cross-subreddit",
                          **{k: v for k, v in rep_xs.items() if k in ("f1", "precision", "recall", "auroc")}})
    console.print(f"  {target:<16} in-dist  F1={rep_id['f1']:.3f}  AUROC={rep_id['auroc']:.3f}")
    console.print(f"  {target:<16} cross-sub F1={rep_xs['f1']:.3f}  AUROC={rep_xs['auroc']:.3f}  (drop = {rep_id['f1']-rep_xs['f1']:+.3f})")

transfer_df = pd.DataFrame(transfer_rows)
transfer_df.to_csv(OUT / "exp2_transfer.csv", index=False)
if not transfer_df.empty:
    plot_transfer_drop(transfer_df, FIG / "exp2__transfer.png", metric="f1")
    console.print(f"  → {FIG / 'exp2__transfer.png'}")


# --------------------------------------------------------------------------- #
# 3. Subreddit classifier (9-way)
# --------------------------------------------------------------------------- #
console.rule("[3/5] 9-way subreddit classifier")

sub_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.95,
                              sublinear_tf=True, max_features=60000)),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                               max_iter=2000, solver="lbfgs", random_state=SEED, n_jobs=-1)),
])
y_sub = df["subreddit"].astype(str).values
labels = sorted(df["subreddit"].unique())

train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y_sub,
)
with status(f"Fitting 9-way subreddit classifier on {len(train_idx):,} posts"):
    sub_pipe.fit(df.iloc[train_idx]["clean_text"].tolist(), y_sub[train_idx])
y_pred = sub_pipe.predict(df.iloc[test_idx]["clean_text"].tolist())
y_true = y_sub[test_idx]

sub_f1 = f1_score(y_true, y_pred, average="macro")
console.print(f"  macro-F1 = {sub_f1:.3f}")
cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
cm.to_csv(OUT / "exp3_subreddit_confusion.csv")
plot_subreddit_confusion(cm, FIG / "exp3__subreddit_confusion.png",
                         title=f"Subreddit classification — macro-F1 = {sub_f1:.3f}")
console.print(f"  → {FIG / 'exp3__subreddit_confusion.png'}")


# --------------------------------------------------------------------------- #
# 4. Per-target linguistic-marker heatmap
# --------------------------------------------------------------------------- #
console.rule("[4/5] Per-target linguistic markers")

markers_by_target: dict[str, pd.DataFrame] = {}
with make_progress() as bar:
    task = bar.add_task("Comparing features per target", total=len(TARGETS))
    for target in TARGETS:
        bar.update(task, description=f"Markers — [cyan]{target}[/]")
        y = _y(df, target)
        if y.sum() < 5:
            console.print(f"  [yellow]{target}: n_pos={y.sum()} — skipping[/yellow]")
            bar.advance(task)
            continue
        m = compare_features_by_label(df, target)
        m.to_csv(OUT / f"exp4_markers__{target}.csv", index=False)
        markers_by_target[target] = m
        console.print(f"  {target:<16} top feature: {m.iloc[0]['feature']:<28} d={m.iloc[0]['cohen_d']:+.2f}")
        bar.advance(task)

if markers_by_target:
    plot_marker_heatmap(markers_by_target, FIG / "exp4__marker_heatmap.png", top_n=8)
    console.print(f"  → {FIG / 'exp4__marker_heatmap.png'}")


# --------------------------------------------------------------------------- #
# 5. Health-anxiety severity ranking
# --------------------------------------------------------------------------- #
console.rule("[5/5] Health-anxiety severity ranking (continuous score)")

severity = df[["id", "subreddit", "weak_health_anxiety", "clean_text"]].copy()
top20 = severity.sort_values("weak_health_anxiety", ascending=False).head(20)
top20["preview"] = top20["clean_text"].str.slice(0, 220)
top20[["id", "subreddit", "weak_health_anxiety", "preview"]].to_csv(
    OUT / "exp5_health_anxiety_top20.csv", index=False
)
console.print(f"  Top 20 highest-scoring posts written to {OUT / 'exp5_health_anxiety_top20.csv'}")
console.print(f"  Score range in top 20: [{top20['weak_health_anxiety'].min():.3f}, "
              f"{top20['weak_health_anxiety'].max():.3f}]")

# Per-subreddit distribution of severity score
sev_by_sub = (
    df.groupby("subreddit")["weak_health_anxiety"]
      .agg(["mean", "median", "std", "count"])
      .sort_values("mean", ascending=False)
)
sev_by_sub.to_csv(OUT / "exp5_severity_by_subreddit.csv")
console.print(sev_by_sub.to_string())


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
console.rule("[bold green]Summary")
summary = {
    "corpus_size": int(len(df)),
    "subreddits": int(df["subreddit"].nunique()),
    "experiment_1": {
        "results": results_df.to_dict(orient="records"),
    },
    "experiment_2_transfer": transfer_df.to_dict(orient="records") if not transfer_df.empty else [],
    "experiment_3_subreddit_classifier": {"macro_f1": float(sub_f1)},
    "experiment_4_markers": {
        t: {
            "top_feature": m.iloc[0]["feature"],
            "top_cohen_d": float(m.iloc[0]["cohen_d"]),
            "n_significant_bh": int((m["p_bh"] < 0.05).sum()),
        }
        for t, m in markers_by_target.items()
    },
    "experiment_5_severity": {
        "max_health_anxiety_score": float(severity["weak_health_anxiety"].max()),
        "mean_by_subreddit_top3": sev_by_sub["mean"].head(3).to_dict(),
    },
}
(OUT / "experiments_summary.json").write_text(json.dumps(summary, indent=2, default=str))
console.print(f"  Summary → {OUT / 'experiments_summary.json'}")
console.print("\n[green]All experiments complete.[/green]")
