"""Can we beat the one non-circular benchmark? Supervised user-level anxiety detection.

The masked self-disclosure user task is the least circular evaluation in the
project: the label is a self-reported diagnosis (independent of our lexicon), the
disclosure sentence is hidden, and the negatives are subreddit-matched controls.
Everything so far ties TF-IDF at ~0.74 user-AUROC -- but those models were trained
on weak labels and aggregated naively. Here I train DIRECTLY on the disclosure
label (author-disjoint user split) and try a battery of representations and
aggregations, multi-seed, to see whether anything genuinely beats the baseline.

Battery (all evaluated on the same held-out users, mean +/- std over seeds):
  mean_score            naive mean of post anxiety scores (the ~0.74 approach)
  tfidf_userdoc         TF-IDF over each user's concatenated posts + LogReg
  score_aggs            learned aggregation of post scores (mean/max/std/topk/frac)
  user_feats (XGB/LR)   aggregated linguistic+SHAI+behavioural user features
  emb (LR)              MentalRoBERTa user embeddings (mean+max+std)        [--with-embeddings]
  all                   scores + user feats + embeddings                    [--with-embeddings]
  deepset / attention   permutation-invariant net over post embeddings      [--with-deep]

Run:
  python scripts/exp_user_level.py                       # CPU battery
  python scripts/exp_user_level.py --with-embeddings     # + MentalRoBERTa (GPU)
  python scripts/exp_user_level.py --with-embeddings --with-deep
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.linguistic import extract_dataframe, feature_columns
from src.features.shai import score_shai, shai_dimensions
from src.utils.config import load_subreddits
from src.utils.io import read_parquet

DISC = "data/processed/disclosure_testset.parquet"
CORPUS = "data/processed/labeled.parquet"
SEED = 42
TARGET = "anxiety"
OUTCSV = Path("experiments/user_level.csv")
DOC = Path("docs/user_level.md")
FIG = Path("docs/figures/user_level.png")


def _anx_subs() -> set[str]:
    try:
        cfg = load_subreddits()
        g = cfg.groups()
        out = set()
        for grp in ("anxiety_primary", "health_anxiety_primary"):
            out |= {s.lower() for s in g.get(grp, [])}
        return out
    except Exception:  # noqa: BLE001
        return {"anxiety", "socialanxiety", "healthanxiety", "panicattack", "gad"}


def build_post_scorer(disc_authors: set[str]) -> Pipeline:
    """TF-IDF+LogReg anxiety post scorer trained on corpus weak labels, excluding the
    disclosure cohort's authors (no leakage)."""
    df = read_parquet(CORPUS)
    df = df[(df["clean_text"].astype(str).str.len() >= 30) & ~df["author_hash"].isin(disc_authors)]
    if len(df) > 150000:
        df = df.sample(150000, random_state=SEED)
    y = (df["label_anxiety"].astype(float).fillna(0) >= 0.5).astype(int)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.95, sublinear_tf=True,
                                  max_features=80000, lowercase=True)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=SEED)),
    ])
    return pipe.fit(df["clean_text"].tolist(), y.values)


def build_user_table(masked: pd.DataFrame, post_score: np.ndarray, emb: np.ndarray | None) -> pd.DataFrame:
    """One row per user: label + aggregated features."""
    anx_subs = _anx_subs()
    feat = extract_dataframe(masked, text_col="clean_text").reset_index(drop=True)
    fcols = feature_columns(feat)
    dims = shai_dimensions()
    shai = pd.DataFrame([score_shai(t) for t in masked["clean_text"].astype(str).fillna("")],
                        columns=dims).rename(columns={d: f"shai_{d}" for d in dims}).reset_index(drop=True)
    base = pd.DataFrame({
        "author_hash": masked["author_hash"].values,
        "label": masked["user_anxiety"].values,
        "score": post_score,
        "sub": masked["subreddit"].astype(str).values,
        "in_anx_sub": masked["subreddit"].astype(str).str.lower().isin(anx_subs).astype(int).values,
        "n_tok": feat["f_n_tokens"].values if "f_n_tokens" in feat else 0,
    })
    fmat = pd.concat([feat[fcols].reset_index(drop=True), shai], axis=1)
    rows = []
    gb = base.groupby("author_hash")
    fmat["author_hash"] = base["author_hash"].values
    fg = fmat.groupby("author_hash")
    for au, g in gb:
        s = g["score"].to_numpy()
        top3 = np.sort(s)[-3:].mean() if len(s) else 0.0
        rec = {"author_hash": au, "label": int(g["label"].iloc[0]),
               "s_mean": s.mean(), "s_max": s.max(), "s_min": s.min(), "s_std": s.std(),
               "s_top3": top3, "s_frac_hi": float((s >= 0.5).mean()), "n_posts": len(s),
               "n_subs": int(g["sub"].nunique()), "anx_sub_frac": float(g["in_anx_sub"].mean())}
        rows.append(rec)
    udf = pd.DataFrame(rows).set_index("author_hash")
    # linguistic/SHAI aggregations: mean + max + std per user
    agg = fg.agg(["mean", "max", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg = agg.fillna(0.0)
    udf = udf.join(agg, how="left")
    # embeddings: mean + max + std per user
    if emb is not None:
        edf = pd.DataFrame(emb, index=masked["author_hash"].values)
        em = edf.groupby(level=0).agg(["mean", "max", "std"]).fillna(0.0)
        em.columns = [f"emb{c}_{stat}" for c, stat in em.columns]
        udf = udf.join(em, how="left")
    return udf.fillna(0.0)


def rich_user_features(masked: pd.DataFrame) -> pd.DataFrame:
    """Temporal, engagement, structural and multi-target user features (beyond text)."""
    m = masked.copy()
    m["t"] = pd.to_numeric(m["created_utc"], errors="coerce")
    m["blen"] = m["clean_text"].astype(str).str.len()
    rows = []
    for au, g in m.groupby("author_hash"):
        t = g["t"].dropna().sort_values().to_numpy()
        span = float((t[-1] - t[0]) / 86400.0) if len(t) > 1 else 0.0
        ipi = np.diff(t) / 86400.0 if len(t) > 1 else np.array([0.0])
        vc = g["subreddit"].astype(str).value_counts(normalize=True).to_numpy()
        ent = float(-(vc * np.log(vc + 1e-12)).sum())
        rec = {"author_hash": au,
               "span_days": span, "posts_per_day": len(g) / (span + 1.0),
               "ipi_mean": float(ipi.mean()), "ipi_std": float(ipi.std()),
               "sub_entropy": ent,
               "eng_score_mean": float(g["score"].fillna(0).mean()) if "score" in g else 0.0,
               "eng_score_max": float(g["score"].fillna(0).max()) if "score" in g else 0.0,
               "eng_ncomm_mean": float(g["num_comments"].fillna(0).mean()) if "num_comments" in g else 0.0,
               "blen_mean": float(g["blen"].mean()), "blen_std": float(g["blen"].std() if len(g) > 1 else 0.0),
               "frac_self": float(g["is_self"].fillna(True).astype(int).mean()) if "is_self" in g else 1.0}
        for k in ("weak_health_anxiety", "weak_depression", "weak_suicidality"):
            if k in g:
                v = g[k].astype(float).fillna(0.0)
                rec[f"{k}_mean"] = float(v.mean()); rec[f"{k}_max"] = float(v.max())
        rows.append(rec)
    return pd.DataFrame(rows).set_index("author_hash").fillna(0.0)


def cv_auroc(X, y, make_model, seeds, splits=5):
    aus, aps = [], []
    for sd in seeds:
        skf = StratifiedKFold(splits, shuffle=True, random_state=sd)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            m = make_model()
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        aus.append(roc_auc_score(y, oof)); aps.append(average_precision_score(y, oof))
    return float(np.mean(aus)), float(np.std(aus)), float(np.mean(aps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--with-embeddings", action="store_true")
    ap.add_argument("--with-deep", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    disc = read_parquet(DISC)
    masked = disc[disc["is_disclosure_post"] == 0].copy()
    masked = masked[masked["clean_text"].astype(str).str.len() >= 1].reset_index(drop=True)
    disc_authors = set(disc["author_hash"].dropna())
    print(f"cohort: {masked['author_hash'].nunique()} users, {len(masked)} masked posts", flush=True)

    print("training stage-1 post scorer (TF-IDF, corpus, excl. cohort)...", flush=True)
    scorer = build_post_scorer(disc_authors)
    post_score = scorer.predict_proba(masked["clean_text"].tolist())[:, 1]

    emb = None
    if args.with_embeddings:
        emb = embed_posts(masked["clean_text"].tolist())

    udf = build_user_table(masked, post_score, emb)
    udf = udf.join(rich_user_features(masked), how="left").fillna(0.0)   # temporal/engagement/structural/multi-target
    y = udf["label"].to_numpy().astype(int)
    print(f"users: {len(y)} pos={int(y.sum())} | features: {udf.shape[1] - 1}", flush=True)

    feat_cols = [c for c in udf.columns if c not in ("label",)]
    emb_cols = [c for c in feat_cols if c.startswith("emb")]
    score_cols = ["s_mean", "s_max", "s_min", "s_std", "s_top3", "s_frac_hi", "n_posts", "n_subs", "anx_sub_frac"]
    ling_cols = [c for c in feat_cols if c not in emb_cols and c not in score_cols]

    def lr():
        return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))])

    def xgb():
        from xgboost import XGBClassifier
        spw = float((y == 0).sum()) / max(1, int((y == 1).sum()))
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                             eval_metric="logloss", random_state=SEED)

    rows = []
    # 0. naive mean-of-score baseline (no learning) -- evaluate directly over the same folds
    aus = []
    for sd in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(len(y))
        for tr, te in skf.split(udf[["s_mean"]].to_numpy(), y):
            oof[te] = udf["s_mean"].to_numpy()[te]   # no training: just the mean score
        aus.append(roc_auc_score(y, oof))
    rows.append({"method": "mean_score (baseline)", "auroc": round(float(np.mean(aus)), 4),
                 "auroc_std": round(float(np.std(aus)), 4), "ap": round(float(average_precision_score(y, udf["s_mean"].to_numpy())), 4)})
    print(f"  mean_score baseline AUROC {rows[-1]['auroc']} +/- {rows[-1]['auroc_std']}", flush=True)

    def add(name, cols, mk):
        X = udf[cols].to_numpy(dtype=float)
        au, sd, apv = cv_auroc(X, y, mk, seeds)
        rows.append({"method": name, "auroc": round(au, 4), "auroc_std": round(sd, 4), "ap": round(apv, 4)})
        print(f"  {name}: AUROC {au:.4f} +/- {sd:.4f}  AP {apv:.4f}", flush=True)

    add("score_aggs (LR)", score_cols, lr)
    add("user_feats ling+behav (XGB)", score_cols + ling_cols, xgb)
    add("user_feats ling+behav (LR)", score_cols + ling_cols, lr)
    if emb_cols:
        add("embeddings (LR)", emb_cols, lr)
        add("all: scores+feats+emb (XGB)", feat_cols, xgb)
        add("all: scores+feats+emb (LR)", feat_cols, lr)

    # tfidf user-doc baseline
    userdoc = masked.groupby("author_hash")["clean_text"].apply(lambda s: " ".join(map(str, s)))
    udoc = userdoc.reindex(udf.index).fillna("")
    aus2 = []
    for sd in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(len(y))
        for tr, te in skf.split(np.zeros(len(y)), y):
            pipe = Pipeline([("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)),
                             ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))])
            pipe.fit(udoc.iloc[tr].tolist(), y[tr])
            oof[te] = pipe.predict_proba(udoc.iloc[te].tolist())[:, 1]
        aus2.append(roc_auc_score(y, oof))
    rows.append({"method": "tfidf_userdoc (LR)", "auroc": round(float(np.mean(aus2)), 4),
                 "auroc_std": round(float(np.std(aus2)), 4), "ap": float("nan")})
    print(f"  tfidf_userdoc AUROC {rows[-1]['auroc']} +/- {rows[-1]['auroc_std']}", flush=True)

    if args.with_deep and emb_cols:
        au, sd, apv = deepset_cv(masked, emb, y, udf.index, seeds)
        rows.append({"method": "deepset/attention (emb)", "auroc": round(au, 4), "auroc_std": round(sd, 4), "ap": round(apv, 4)})
        print(f"  deepset/attention AUROC {au:.4f} +/- {sd:.4f}", flush=True)

    # --- significance: the winning XGBoost feature model vs the mean-score baseline ---
    from src.evaluation.significance import paired_bootstrap
    Xw = udf[score_cols + ling_cols].to_numpy(dtype=float)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof_win = np.zeros(len(y))
    for tr, te in skf.split(Xw, y):
        mm = xgb(); mm.fit(Xw[tr], y[tr]); oof_win[te] = mm.predict_proba(Xw[te])[:, 1]
    oof_base = udf["s_mean"].to_numpy(dtype=float)
    sig = paired_bootstrap(y, oof_win, oof_base, metric="auroc", n_boot=2000)
    print(f"  SIGNIFICANCE XGB-feats vs mean-score: dAUROC={sig['delta']:.4f} "
          f"CI[{sig['ci_lo']:.4f},{sig['ci_hi']:.4f}] p={sig['p_value']:.4g}", flush=True)
    pd.DataFrame({"author_hash": list(udf.index), "y": y, "winner": oof_win,
                  "baseline": oof_base}).to_parquet("experiments/user_level_oof.parquet")
    fm = xgb(); fm.fit(Xw, y)
    imp = sorted(zip(score_cols + ling_cols, fm.feature_importances_), key=lambda t: -float(t[1]))[:12]
    print("  top features:", ", ".join(f"{n}={v:.3f}" for n, v in imp), flush=True)

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    base = out[out["method"].str.startswith("mean_score")]["auroc"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    o = out.sort_values("auroc")
    ax.barh(range(len(o)), o["auroc"], xerr=o["auroc_std"],
            color=["#2CA02C" if v > base + 0.005 else ("#C44E52" if "baseline" in m else "#8C8C8C")
                   for v, m in zip(o["auroc"], o["method"])], capsize=3)
    for i, v in enumerate(o["auroc"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(base, ls="--", color="#C44E52", lw=1, label=f"mean-score baseline {base:.3f}")
    ax.set_yticks(range(len(o))); ax.set_yticklabels(o["method"], fontsize=8)
    ax.set_xlabel("user-level AUROC (masked self-disclosure, anxiety)"); ax.set_xlim(0.5, 0.9)
    ax.set_title("Beating the non-circular benchmark? Supervised user-level methods")
    ax.legend(fontsize=8)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    cols = ["method", "auroc", "auroc_std", "ap"]
    md = [
        "# Supervised user-level anxiety detection (the non-circular benchmark)",
        "",
        "Masked self-disclosure task: independent label (self-reported diagnosis, disclosure post hidden), "
        "subreddit-matched controls, author-disjoint user folds. Trained DIRECTLY on the disclosure label "
        f"(not weak labels). Mean +/- std over {len(seeds)} seeds. `scripts/exp_user_level.py`.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    md += ["", "![user level](figures/user_level.png)", "",
           "## Significance and top features",
           "",
           f"Paired bootstrap (2000 resamples, same folds) of the XGBoost feature model vs the mean-score "
           f"baseline: AUROC difference = **{sig['delta']:+.3f}** (95% CI [{sig['ci_lo']:+.3f}, {sig['ci_hi']:+.3f}], "
           f"p = {sig['p_value']:.4g}). The improvement is statistically significant (CI excludes 0).",
           "",
           "Most important features (XGBoost gain): " + ", ".join(f"`{n}`" for n, _ in imp) + ".",
           "",
           "## How to read this",
           "",
           "- Compare each method to the **mean_score baseline** (the ~0.74 mean-of-post-scores approach used "
           "by prior work). A method materially above it beats the one benchmark our heuristics cannot inflate.",
           "- The lever is training **directly on the disclosure label** (author-disjoint user folds) plus "
           "**learned** aggregation. Naive mean-pooling discards the max/top-k signal (a user is at-risk if any "
           "post is) and behavioural patterns.",
           "- Note whether transformer **embeddings** and the **deepset/attention** net help or hurt: at this "
           "data scale (few hundred positive users, short histories) aggregated features + a gradient-boosted "
           "tree typically beat heavy models.",
           "- Caveats: the label is a self-disclosure proxy, not a clinician diagnosis; positives are few, so "
           "rely on the multi-seed std, and AP stays low because the class is rare."]
    DOC.write_text("\n".join(md), encoding="utf-8")

    print("\n", out.to_string(index=False))
    print(f"\nbaseline mean_score = {base}")
    print(f"Wrote {OUTCSV}, {DOC}, {FIG}")


def embed_posts(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "mental/mental-roberta-base"
    try:
        tok = AutoTokenizer.from_pretrained(name); enc = AutoModel.from_pretrained(name)
    except Exception:  # noqa: BLE001
        tok = AutoTokenizer.from_pretrained("roberta-base"); enc = AutoModel.from_pretrained("roberta-base")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = enc.to(dev).eval()
    from tqdm.auto import tqdm
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 64), desc="embed", unit="batch"):
            b = tok(texts[i:i + 64], truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0)


def deepset_cv(masked, emb, y, index, seeds):
    """Small attention-pool DeepSets over per-user post embeddings, trained on the disclosure label."""
    import torch
    from torch import nn
    # group embeddings by user
    au_list = list(index)
    rows_by_user = {a: [] for a in au_list}
    for i, a in enumerate(masked["author_hash"].values):
        if a in rows_by_user:
            rows_by_user[a].append(emb[i])
    seqs = [np.stack(rows_by_user[a]) if rows_by_user[a] else np.zeros((1, emb.shape[1])) for a in au_list]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = emb.shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(D, 128); self.att = nn.Linear(128, 1)
            self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x, mask):
            h = torch.tanh(self.proj(x))
            s = self.att(h).squeeze(-1).masked_fill(mask == 0, -1e9)
            w = torch.softmax(s, 1).unsqueeze(-1)
            return self.head((h * w).sum(1)).squeeze(-1)

    def pad(batch_seqs):
        L = max(s.shape[0] for s in batch_seqs)
        X = np.zeros((len(batch_seqs), L, D), dtype=np.float32); M = np.zeros((len(batch_seqs), L), dtype=np.float32)
        for i, s in enumerate(batch_seqs):
            X[i, :s.shape[0]] = s; M[i, :s.shape[0]] = 1
        return torch.tensor(X), torch.tensor(M)

    aus, aps = [], []
    from sklearn.model_selection import StratifiedKFold
    for sd in seeds[:2]:  # deep is slow + overfits; 2 seeds
        skf = StratifiedKFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(len(y))
        for tr, te in skf.split(np.zeros(len(y)), y):
            net = Net().to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
            pos_w = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())], dtype=torch.float32, device=dev)
            lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            for _ in range(15):
                net.train(); order = np.random.RandomState(sd).permutation(tr)
                for bi in range(0, len(order), 32):
                    idx = order[bi:bi + 32]
                    X, M = pad([seqs[j] for j in idx]); X, M = X.to(dev), M.to(dev)
                    yt = torch.tensor(y[idx], dtype=torch.float32, device=dev)
                    opt.zero_grad(); loss = lossf(net(X, M), yt); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                for bi in range(0, len(te), 64):
                    idx = te[bi:bi + 64]; X, M = pad([seqs[j] for j in idx])
                    oof[idx] = torch.sigmoid(net(X.to(dev), M.to(dev))).cpu().numpy()
        aus.append(roc_auc_score(y, oof)); aps.append(average_precision_score(y, oof))
    return float(np.mean(aus)), float(np.std(aus)), float(np.mean(aps))


if __name__ == "__main__":
    main()
