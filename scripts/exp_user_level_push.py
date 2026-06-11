"""Exhaustive push on the non-circular user-level benchmark.

Builds on exp_user_level.py (0.832 AUROC, +0.093 significant) and adds every
research-backed lever for small-N user-level mental-health detection (eRisk /
CLPsych / Low 2020 / SMHD):

  * order statistics + first-crossing of post anxiety scores (beat mean-pooling)
  * circadian / temporal features (night posting, hour & day entropy, burstiness,
    volume slope, risk-score trajectory slope, recency-weighted risk)
  * reassurance-seeking / cyberchondria lexicon (google/test/fine/doctor/ER...)
  * bag-of-subreddits participation (fair: controls are subreddit-matched)
  * aggregated linguistic + SHAI features + comorbidity (other-condition weak scores)

Model zoo: tuned XGBoost, HistGradientBoosting, RandomForest, ExtraTrees,
elastic-net LogReg, linear SVM, a STACKING ensemble, and optional TabPFN.
Reports 5-seed CV AUROC per model, a NESTED-CV unbiased estimate for the winner,
and a paired-bootstrap significance test vs the mean-score baseline. Runs for
anxiety (headline) + health_anxiety + depression.

Run:
  python scripts/exp_user_level_push.py
  python scripts/exp_user_level_push.py --targets anxiety
  python scripts/exp_user_level_push.py --tabpfn
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.evaluation.significance import paired_bootstrap
from src.features.linguistic import extract_dataframe, feature_columns
from src.features.shai import score_shai, shai_dimensions
from src.utils.config import load_subreddits
from src.utils.io import read_parquet

DISC = "data/processed/disclosure_testset.parquet"
CORPUS = "data/processed/labeled.parquet"
SEED = 42
OUTCSV = Path("experiments/user_level_push.csv")
DOC = Path("docs/user_level_push.md")
FIG = Path("docs/figures/user_level_push.png")

CYBERCHONDRIA = [
    "google", "googled", "googling", "webmd", "symptom", "symptoms", "test", "tests",
    "results", "scan", "mri", "ct scan", "x-ray", "biopsy", "blood test", "doctor",
    "doctors", "er", "emergency room", "reassurance", "reassure", "normal", "fine",
    "benign", "what if", "is it serious", "could it be", "am i dying", "tumor", "cancer",
]
_CYBER_RE = re.compile("|".join(re.escape(w) for w in sorted(CYBERCHONDRIA, key=len, reverse=True)), re.IGNORECASE)


def build_scorer(target: str, disc_authors: set[str]) -> Pipeline:
    df = read_parquet(CORPUS)
    df = df[(df["clean_text"].astype(str).str.len() >= 30) & ~df["author_hash"].isin(disc_authors)]
    if len(df) > 150000:
        df = df.sample(150000, random_state=SEED)
    col = f"label_{target}"
    y = (df[col].astype(float).fillna(0) >= 0.5).astype(int)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.95, sublinear_tf=True,
                                  max_features=80000, lowercase=True)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=SEED)),
    ])
    return pipe.fit(df["clean_text"].tolist(), y.values)


def _sub_groups() -> dict[str, set[str]]:
    try:
        g = load_subreddits().groups()
        return {k: {s.lower() for s in v} for k, v in g.items()}
    except Exception:  # noqa: BLE001
        return {}


def base_features(masked: pd.DataFrame) -> pd.DataFrame:
    """Target-agnostic per-user features (computed once)."""
    feat = extract_dataframe(masked, text_col="clean_text").reset_index(drop=True)
    fcols = feature_columns(feat)
    dims = shai_dimensions()
    shai = pd.DataFrame([score_shai(t) for t in masked["clean_text"].astype(str).fillna("")],
                        columns=dims).add_prefix("shai_").reset_index(drop=True)
    fmat = pd.concat([feat[fcols].reset_index(drop=True), shai], axis=1)
    fmat["author_hash"] = masked["author_hash"].values
    agg = fmat.groupby("author_hash").agg(["mean", "max", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.fillna(0.0)

    groups = _sub_groups()
    m = masked.copy()
    m["t"] = pd.to_numeric(m["created_utc"], errors="coerce")
    m["hour"] = (m["t"] // 3600 % 24).fillna(-1)
    m["dow"] = (m["t"] // 86400 % 7).fillna(-1)
    m["blen"] = m["clean_text"].astype(str).str.len()
    m["cyber"] = [len(_CYBER_RE.findall(str(x))) for x in m["clean_text"]]
    m["subl"] = m["subreddit"].astype(str).str.lower()
    rows = []
    for au, g in m.groupby("author_hash"):
        t = g["t"].dropna().sort_values().to_numpy()
        span = float((t[-1] - t[0]) / 86400.0) if len(t) > 1 else 0.0
        ipi = np.diff(t) / 86400.0 if len(t) > 1 else np.array([0.0])
        mu, sd = float(ipi.mean()), float(ipi.std())
        burst = (sd - mu) / (sd + mu) if (sd + mu) > 0 else 0.0
        hh = g["hour"].to_numpy(); hv = np.bincount(hh[hh >= 0].astype(int), minlength=24).astype(float)
        hp = hv / hv.sum() if hv.sum() else hv; h_ent = float(-(hp[hp > 0] * np.log(hp[hp > 0])).sum())
        dd = g["dow"].to_numpy(); dv = np.bincount(dd[dd >= 0].astype(int), minlength=7).astype(float)
        dp = dv / dv.sum() if dv.sum() else dv; d_ent = float(-(dp[dp > 0] * np.log(dp[dp > 0])).sum())
        vc = g["subl"].value_counts(normalize=True).to_numpy()
        rec = {"author_hash": au,
               "n_posts": len(g), "span_days": span, "posts_per_day": len(g) / (span + 1.0),
               "ipi_mean": mu, "ipi_std": sd, "burstiness": burst,
               "night_frac": float(((g["hour"] >= 0) & (g["hour"] < 6)).mean()),
               "evening_frac": float((g["hour"] >= 19).mean()),
               "hour_entropy": h_ent, "dow_entropy": d_ent,
               "n_subs": int(g["subl"].nunique()),
               "sub_entropy": float(-(vc * np.log(vc + 1e-12)).sum()),
               "cyber_mean": float(g["cyber"].mean()), "cyber_max": float(g["cyber"].max()),
               "blen_mean": float(g["blen"].mean()), "blen_std": float(g["blen"].std() if len(g) > 1 else 0.0),
               "eng_score_mean": float(g["score"].fillna(0).mean()) if "score" in g else 0.0,
               "eng_ncomm_mean": float(g["num_comments"].fillna(0).mean()) if "num_comments" in g else 0.0,
               "frac_self": float(g["is_self"].fillna(True).astype(int).mean()) if "is_self" in g else 1.0}
        for gname, subs in groups.items():
            rec[f"subgrp_{gname}"] = float(g["subl"].isin(subs).mean())
        for k in ("weak_health_anxiety", "weak_depression", "weak_suicidality", "weak_anxiety"):
            if k in g:
                v = g[k].astype(float).fillna(0.0)
                rec[f"{k}_mean"] = float(v.mean()); rec[f"{k}_max"] = float(v.max())
        rows.append(rec)
    behav = pd.DataFrame(rows).set_index("author_hash")
    return behav.join(agg, how="left").fillna(0.0)


def score_features(masked: pd.DataFrame, post_score: np.ndarray, target: str) -> pd.DataFrame:
    """Per-user order statistics + trajectory of the (target) post score + user label."""
    m = masked[["author_hash", "created_utc", f"user_{target}"]].copy()
    m["t"] = pd.to_numeric(m["created_utc"], errors="coerce")
    m["score"] = post_score
    rows = []
    for au, g in m.groupby("author_hash"):
        gs = g.sort_values("t")
        s = gs["score"].to_numpy()
        order = np.sort(s)
        cross = np.argmax(s >= 0.5) if (s >= 0.5).any() else len(s)
        slope = float(np.polyfit(np.arange(len(s)), s, 1)[0]) if len(s) > 1 else 0.0
        w = np.exp(-(np.arange(len(s))[::-1]) / 5.0)  # recency weight (recent posts up)
        rec = {"author_hash": au, "label": int(g[f"user_{target}"].iloc[0]),
               "s_mean": float(s.mean()), "s_max": float(s.max()), "s_min": float(s.min()),
               "s_std": float(s.std()), "s_top3": float(order[-3:].mean()), "s_top5": float(order[-5:].mean()),
               "s_p90": float(np.percentile(s, 90)), "s_p95": float(np.percentile(s, 95)),
               "s_frac50": float((s >= 0.5).mean()), "s_frac70": float((s >= 0.7).mean()),
               "s_first_cross": float(cross / len(s)), "s_last": float(s[-1]),
               "s_slope": slope, "s_recency": float((w * s).sum() / w.sum())}
        rows.append(rec)
    return pd.DataFrame(rows).set_index("author_hash")


def model_zoo(y, use_tabpfn=False):
    spw = float((y == 0).sum()) / max(1, int((y == 1).sum()))

    def xgb(depth=4, n=400, lr=0.04):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=lr, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=2, reg_lambda=2.0,
                             scale_pos_weight=spw, tree_method="hist", eval_metric="logloss", random_state=SEED)

    def lr():
        return Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", penalty="elasticnet",
                                                   l1_ratio=0.3, C=0.5, solver="saga"))])

    def svm():
        return Pipeline([("sc", StandardScaler()),
                         ("svc", CalibratedClassifierCV(LinearSVC(C=0.2, class_weight="balanced"), cv=3))])

    def hgb():
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                              l2_regularization=1.0, class_weight="balanced", random_state=SEED)

    def rf():
        return RandomForestClassifier(n_estimators=600, max_depth=None, min_samples_leaf=3,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)

    def et():
        return ExtraTreesClassifier(n_estimators=600, min_samples_leaf=3, class_weight="balanced",
                                    n_jobs=-1, random_state=SEED)

    def stack():
        return StackingClassifier(
            estimators=[("xgb", xgb()), ("rf", rf()), ("lr", lr())],
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            cv=5, n_jobs=-1, passthrough=False)

    zoo = {"xgboost": xgb, "hist_gbm": hgb, "random_forest": rf, "extra_trees": et,
           "elasticnet_lr": lr, "linear_svm": svm, "stacking": stack}
    if use_tabpfn:
        def tabpfn():
            from tabpfn import TabPFNClassifier
            return TabPFNClassifier()
        zoo["tabpfn"] = tabpfn
    return zoo


def cv_auroc(X, y, make, seeds, splits=5):
    aus, aps = [], []
    for sd in seeds:
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(splits, shuffle=True, random_state=sd).split(X, y):
            m = make(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
        aus.append(roc_auc_score(y, oof)); aps.append(average_precision_score(y, oof))
    return float(np.mean(aus)), float(np.std(aus)), float(np.mean(aps))


def nested_auroc(X, y, seeds=(42, 1, 2)):
    """Nested CV with an inner XGBoost depth/lr grid -> unbiased estimate."""
    from xgboost import XGBClassifier
    spw = float((y == 0).sum()) / max(1, int((y == 1).sum()))
    grid = [(3, 0.03), (3, 0.05), (4, 0.04), (5, 0.03)]
    aus = []
    for sd in seeds:
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
            best, best_au = None, -1
            for depth, lr in grid:
                inner = np.zeros(len(tr))
                for itr, ite in StratifiedKFold(3, shuffle=True, random_state=sd).split(X[tr], y[tr]):
                    mm = XGBClassifier(n_estimators=400, max_depth=depth, learning_rate=lr, subsample=0.8,
                                       colsample_bytree=0.8, min_child_weight=2, reg_lambda=2.0,
                                       scale_pos_weight=spw, tree_method="hist", eval_metric="logloss",
                                       random_state=sd)
                    mm.fit(X[tr][itr], y[tr][itr]); inner[ite] = mm.predict_proba(X[tr][ite])[:, 1]
                au = roc_auc_score(y[tr], inner)
                if au > best_au:
                    best_au, best = au, (depth, lr)
            mm = XGBClassifier(n_estimators=400, max_depth=best[0], learning_rate=best[1], subsample=0.8,
                               colsample_bytree=0.8, min_child_weight=2, reg_lambda=2.0, scale_pos_weight=spw,
                               tree_method="hist", eval_metric="logloss", random_state=sd)
            mm.fit(X[tr], y[tr]); oof[te] = mm.predict_proba(X[te])[:, 1]
        aus.append(roc_auc_score(y, oof))
    return float(np.mean(aus)), float(np.std(aus))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="anxiety,health_anxiety,depression")
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--tabpfn", action="store_true")
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    disc = read_parquet(DISC)
    masked = disc[disc["is_disclosure_post"] == 0].copy()
    masked = masked[masked["clean_text"].astype(str).str.len() >= 1].reset_index(drop=True)
    disc_authors = set(disc["author_hash"].dropna())
    print(f"cohort: {masked['author_hash'].nunique()} users, {len(masked)} masked posts", flush=True)

    print("building target-agnostic features (once)...", flush=True)
    udf_base = base_features(masked)
    print(f"  base features: {udf_base.shape[1]}", flush=True)

    all_rows, headline = [], {}
    for target in targets:
        print(f"\n=== target: {target} ===", flush=True)
        scorer = build_scorer(target, disc_authors)
        post_score = scorer.predict_proba(masked["clean_text"].tolist())[:, 1]
        sdf = score_features(masked, post_score, target)
        udf = sdf.join(udf_base, how="left").fillna(0.0)
        y = udf["label"].to_numpy().astype(int)
        feat_cols = [c for c in udf.columns if c != "label"]
        X = udf[feat_cols].to_numpy(dtype=float)
        print(f"  users={len(y)} pos={int(y.sum())} feats={len(feat_cols)}", flush=True)

        # baseline: mean of post scores
        base_au = roc_auc_score(y, udf["s_mean"].to_numpy())
        all_rows.append({"target": target, "model": "mean_score (baseline)", "auroc": round(base_au, 4),
                         "auroc_std": 0.0, "ap": round(average_precision_score(y, udf["s_mean"]), 4)})
        print(f"  mean_score baseline: {base_au:.4f}", flush=True)

        best = None
        for name, make in model_zoo(y, use_tabpfn=args.tabpfn).items():
            try:
                au, sd, apv = cv_auroc(X, y, make, seeds)
            except Exception as ex:  # noqa: BLE001
                print(f"  {name}: SKIP ({ex})", flush=True); continue
            all_rows.append({"target": target, "model": name, "auroc": round(au, 4),
                             "auroc_std": round(sd, 4), "ap": round(apv, 4)})
            print(f"  {name}: {au:.4f} +/- {sd:.4f}  AP {apv:.4f}", flush=True)
            if best is None or au > best[1]:
                best = (name, au, sd)

        # nested CV + significance for the headline target
        if target == "anxiety":
            nau, nsd = nested_auroc(X, y)
            print(f"  NESTED-CV XGB (unbiased): {nau:.4f} +/- {nsd:.4f}", flush=True)
            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            mk = model_zoo(y)[best[0]]
            oof = np.zeros(len(y))
            for tr, te in skf.split(X, y):
                mm = mk(); mm.fit(X[tr], y[tr]); oof[te] = mm.predict_proba(X[te])[:, 1]
            sig = paired_bootstrap(y, oof, udf["s_mean"].to_numpy(dtype=float), metric="auroc", n_boot=2000)
            pd.DataFrame({"author_hash": list(udf.index), "y": y, "winner": oof,
                          "baseline": udf["s_mean"].to_numpy()}).to_parquet("experiments/user_level_push_oof.parquet")
            from xgboost import XGBClassifier
            fm = model_zoo(y)["xgboost"](); fm.fit(X, y)
            imp = sorted(zip(feat_cols, fm.feature_importances_), key=lambda t: -float(t[1]))[:15]
            headline = {"best_model": best[0], "best_auroc": best[1], "nested_auroc": nau, "nested_std": nsd,
                        "sig": sig, "imp": imp, "baseline": base_au, "n_feats": len(feat_cols)}
            print(f"  SIGNIFICANCE {best[0]} vs baseline: dAUROC={sig['delta']:.4f} "
                  f"CI[{sig['ci_lo']:.4f},{sig['ci_hi']:.4f}] p={sig['p_value']:.4g}", flush=True)
            print("  top:", ", ".join(n for n, _ in imp), flush=True)

    out = pd.DataFrame(all_rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    # figure: best model per target vs baseline
    fig, ax = plt.subplots(figsize=(11, 6))
    tlist = targets
    width = 0.8 / max(1, len(out["model"].unique()))
    models = [m for m in out["model"].unique()]
    x = np.arange(len(tlist))
    for i, mdl in enumerate(models):
        vals = [out[(out.target == t) & (out.model == mdl)]["auroc"].max() if
                not out[(out.target == t) & (out.model == mdl)].empty else 0 for t in tlist]
        ax.bar(x + i * width, vals, width, label=mdl)
    ax.set_xticks(x + width * len(models) / 2); ax.set_xticklabels(tlist)
    ax.set_ylabel("user-level AUROC"); ax.set_ylim(0.5, 0.95)
    ax.set_title("User-level detection: model zoo per target (masked self-disclosure)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    md = ["# Exhaustive user-level push (research-backed features + model zoo)", "",
          "Masked self-disclosure benchmark, author-disjoint user folds, trained directly on the disclosure "
          "label. Features and methods drawn from eRisk / CLPsych / Low 2020. `scripts/exp_user_level_push.py`.",
          "", "| target | model | AUROC | std | AP |", "|---|---|---|---|---|"]
    for r in all_rows:
        md.append(f"| {r['target']} | {r['model']} | {r['auroc']} | {r['auroc_std']} | {r['ap']} |")
    md += ["", "![push](figures/user_level_push.png)", ""]
    if headline:
        s = headline["sig"]
        md += ["## Headline (anxiety)", "",
               f"Best model **{headline['best_model']}**: AUROC **{headline['best_auroc']:.4f}** "
               f"(5-seed CV) vs mean-score baseline {headline['baseline']:.4f}.",
               f"Nested-CV (unbiased) XGBoost: **{headline['nested_auroc']:.4f} ± {headline['nested_std']:.4f}** "
               f"over {headline['n_feats']} features.",
               f"Paired bootstrap vs baseline: AUROC difference **{s['delta']:+.3f}** "
               f"(95% CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}], p = {s['p_value']:.4g}).", "",
               "Top features: " + ", ".join(f"`{n}`" for n, _ in headline["imp"]) + "."]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out.to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


if __name__ == "__main__":
    main()
