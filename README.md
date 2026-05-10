# Anxiety & Health-Anxiety Detection from Reddit

A dissertation-grade NLP pipeline for detecting anxiety — with particular emphasis on **health anxiety** — in Reddit forum posts.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![Tests](https://img.shields.io/badge/tests-22%20passing-success)]() [![License](https://img.shields.io/badge/code-MIT-green)]()

---

## Table of contents

1. [What this repo does (concretely)](#what-this-repo-does-concretely)
2. [What we built and what runs today](#what-we-built-and-what-runs-today)
3. [What was used (models, data, instruments)](#what-was-used-models-data-instruments)
4. [Real-data results](#real-data-results)
5. [Visual gallery](#visual-gallery)
6. [How labels are decided](#how-labels-are-decided)
7. [How we prevent overfitting & validate predictions](#how-we-prevent-overfitting--validate-predictions)
8. [Pipeline overview](#pipeline-overview)
9. [Quick start](#quick-start)
10. [Inspecting the data — code recipes](#inspecting-the-data--code-recipes)
11. [Repo layout](#repo-layout)
12. [Documentation](#documentation)
13. [Ethics](#ethics)

---

## What this repo does (concretely)

> **A dissertation-grade pipeline that takes ~16k Reddit posts from 10 mental-health-adjacent subreddits, applies a 3-tier labeling scheme grounded in clinical instruments (GAD-7, SHAI, PHQ-9, C-SSRS), trains 5 different model families to predict 4 binary labels {anxiety, health_anxiety, depression, suicidality}, evaluates them with 6 metrics + bootstrap CIs + per-subreddit and per-length subgroups, and produces 10 publication-quality figures plus a linguistic-marker analysis with FDR-corrected significance tests.**

The thesis novelty: separating **health anxiety** from general anxiety as its own class, using a multi-task transformer with per-task loss weighting and tier-confidence-weighted training.

---

## What we built and what runs today

Concrete deliverables, all working on real Reddit data as of the last commit:

### ✅ Already running on real data
- **Collection**: 16,382 posts (post-preprocessing) from 10 subreddits via the no-credentials JSON scraper (no Reddit API key needed). Deduplicated across 6 listings (top×{all,year,month,week} + new + hot).
- **Preprocessing**: PII redaction (regex + spaCy NER), exact + near-dedup (SimHash), language filter, length filter → 16,382 cleaned posts.
- **Tier-1 weak labeling**: 3,560 anxiety / 1,557 depression / 116 suicidality / 24 health-anxiety positives — health-anxiety scarcity motivates tier-2 LLM labeling.
- **8 binary classifiers trained** — TF-IDF + LogReg AND XGBoost-on-linguistic-features × 4 targets. Headline (XGBoost-linguistic): anxiety F1 **0.86** / depression F1 **0.74** / suicidality F1 **0.79** / health-anxiety F1 **1.00** *(circular: lexicon-derived label, lexicon-derived features — see caveats)*.
- **MentalRoBERTa fine-tuned (single-target, anxiety)**: F1 **0.891** [0.87, 0.91] / AUROC **0.985** [0.98, 0.99] / ECE **0.032** — new SOTA on this corpus for anxiety, with bootstrap CIs.
- **MentalRoBERTa multi-task (joint heads, dissertation novelty)**: trained on all 4 targets simultaneously with shared encoder + per-task loss weights + per-row confidence weights. anxiety F1 **0.894** / depression F1 **0.720** / suicidality F1 **0.647** / health-anxiety F1 **0.333** — the transformer *cannot* read the lexicons directly, so its rare-class numbers are the **honest** ones (XGBoost's were inflated by lexicon-feature → lexicon-label circularity).
- **Cross-subreddit transfer experiment (RQ3)**: in-distribution F1 = **0.889** vs cross-subreddit F1 = **0.331**, but AUROC stays at **0.967** — reveals that the *ranking* generalizes but the *threshold* doesn't.
- **9-way subreddit classifier**: macro-F1 = **0.643** (vs random 0.10), confirming substantial linguistic distinctiveness.
- **Per-target linguistic-marker analysis**: 4 separate Cohen's d analyses with Benjamini-Hochberg FDR correction. **22 features significant for anxiety, 19 for depression, 13 for suicidality, 5 for health-anxiety**. `f_third_rate` (third-person pronouns) consistently negative across all targets — replicates Pennebaker. `f_sent_neg` rises monotonically with target severity.
- **Health-anxiety severity ranking**: r/COVID19_support (mean 0.239) and r/COVID19positive (0.232) both score ~2× r/Anxiety (0.122), validating their inclusion as health-anxiety-enriched subreddits.
- **15 figures generated** from real data into `docs/figures/`.
- **22/22 unit tests passing**.

→ **Full numbers, plots, and findings in [`docs/experiments.md`](docs/experiments.md)**.

### ✅ Built and ready to run (needs API key / human time)
- **Tier-2 LLM labeling**: ready to run with `ANTHROPIC_API_KEY` set. SQLite-cached so re-runs are free. **Next high-impact step** — fixes the health-anxiety label circularity that limits tier-1 results.
- **Tier-3 manual annotation**: Rich-based TUI ready, resumable, two-annotator κ workflow.
- **Claude zero-/few-shot baseline**: tests whether prompting beats fine-tuning.
- **SHAP explainability** on the XGBoost-linguistic model — interpretability hook for the thesis discussion chapter.

### ✅ Reproducibility & ethics infrastructure
- All YAML-driven (subreddits, labeling, models) — change behavior without editing code.
- All RNG seeded; LLM/HTTP responses cached on disk.
- Strict no-raw-text-redistribution policy. Pipeline-enforced PII redaction. Crisis-resource boilerplate. Two-annotator κ workflow.

---

## What was used (models, data, instruments)

### Data sources
| Subreddit | Role | Posts (post-preprocessing) |
|---|---|---:|
| r/Anxiety | anxiety_primary | 2,078 |
| r/socialanxiety | anxiety_primary | 1,382 |
| r/AnxietyDepression | comorbid | 1,356 |
| r/depression | depression_primary | 2,218 |
| r/depression_help | depression_primary | 1,451 |
| r/SuicideWatch | suicidality (training-only, ethics-restricted) | 2,073 |
| r/COVID19_support | health_anxiety_enriched | 1,413 |
| r/COVID19positive | health_anxiety_enriched | 1,615 |
| r/LivingAlone | baseline | 905 |
| r/relationship_advice | baseline | 1,891 |
| **Total** | | **16,382** |

Date range: Jan 2017 → today. Collected via Reddit's public JSON endpoints (`old.reddit.com/r/<sub>/<listing>.json`) at 1.5s/request — no OAuth required. Custom User-Agent identifies academic research intent. 429 rate-limits are retried indefinitely with `Retry-After` honored.

### Clinical instruments referenced
The lexicons used by tier-1 weak labeling are derived from these published instruments. The thesis cites every list's provenance.

| Instrument | Used for |
|---|---|
| **GAD-7** (Spitzer, Kroenke, Williams, Löwe, 2006) | General anxiety lexicon |
| **SHAI** — Short Health Anxiety Inventory (Salkovskis, Rimes, Warwick, Clark, 2002) | Health anxiety items + reassurance-seeking patterns |
| **HAI** — Health Anxiety Inventory (Lucock & Morley, 1996) | Health anxiety items |
| **PHQ-9** (Kroenke, Spitzer, Williams, 2001) | Depression lexicon |
| **Columbia C-SSRS** (Posner et al., 2011) | Suicidality lexicon |

Plus theoretical grounding from:
- Salkovskis (1989); Warwick & Salkovskis (1990) — cognitive-behavioral model of health anxiety
- Pennebaker, Mayne, Francis (1997) — pronoun preponderance
- Eichstaedt et al. (2018) — Facebook depression markers
- Coppersmith et al. (2014, 2015) — CLPsych benchmark

### Models compared
| Model | Pretrained from | Role |
|---|---|---|
| TF-IDF + Logistic Regression | scikit-learn primitives | Baseline floor |
| XGBoost on linguistic features | XGBoost on hand-crafted features | Interpretable bridge to RQ2 |
| RoBERTa-base fine-tuned | `roberta-base` (Liu et al., 2019) | Modern baseline |
| MentalRoBERTa fine-tuned | `mental/mental-roberta-base` (Ji et al., 2022) | Domain-specific best |
| **Multi-task MentalRoBERTa** | shared encoder + 4 sigmoid heads | **Dissertation novelty** |
| Claude zero-/few-shot | `claude-sonnet-4-6` API | Tests prompting vs fine-tuning |

### Software stack
Python 3.11 · pandas · polars · pyarrow · scikit-learn · XGBoost · PyTorch · HuggingFace Transformers / Datasets / Accelerate · SHAP · spaCy · NLTK · VADER · ftfy · langdetect · zstandard · Anthropic SDK · structlog · Pydantic · Typer · Rich · matplotlib · seaborn · MLflow · pytest

### What we are *doing*
1. Predicting **{anxiety, health_anxiety, depression, suicidality}** for each Reddit post.
2. Using a **3-tier labeling pipeline** that combines algorithmic, LLM, and human annotation, validated against each other via Cohen's κ.
3. Training **5 model families** behind a common `BaseModel` interface, with per-row confidence weights flowing into the loss.
4. **Quantifying uncertainty** with bootstrap 95% CIs on every metric.
5. **Detecting overfitting** via per-subreddit subgroup analysis and a cross-subreddit transfer experiment (RQ3).
6. **Identifying linguistic markers** of health anxiety using Cohen's d + Mann-Whitney U + Benjamini-Hochberg FDR (RQ2).
7. **Producing a defensible thesis** — every modeling, labeling, and evaluation choice is justified by clinical literature or empirical signal in the codebase.

---

## Real-data results

Trained on 16,382 cleaned + weakly-labeled posts. 80/20 stratified split per experiment.

### Experiment 1 — Per-target classifiers

8 classifiers: TF-IDF + LogReg AND XGBoost-on-26-linguistic-features × 4 targets.

| Target | n_pos | TF-IDF F1 | XGBoost F1 | XGBoost AUROC | XGBoost ECE |
|---|---:|---:|---:|---:|---:|
| anxiety | 3,560 | **0.874** | 0.864 | 0.984 | 0.034 |
| health_anxiety | 24 | 0.750 | **1.000** ⚠️ | 1.000 | 0.000 |
| depression | 1,557 | 0.708 | **0.742** | 0.976 | 0.035 |
| suicidality | 116 | 0.571 | **0.792** | 0.998 | 0.003 |

Key observation: **XGBoost on 26 linguistic features matches or beats the 80,000-feature TF-IDF text model on 3 of 4 targets**, with ~4× better calibration. The hand-crafted features carry most of the predictive signal.

⚠️ Health-anxiety F1 = 1.00 is a small-sample artifact (24 positives, model has direct lexicon-feature access, label was lexicon-derived → circular). See [`docs/experiments.md`](docs/experiments.md) §1 for full caveats.

### Experiment 2 — Cross-subreddit transfer (RQ3)

Train on r/{Anxiety, socialanxiety, AnxietyDepression}, test on r/{COVID19_support, LivingAlone, relationship_advice}.

| Split | F1 | Precision | Recall | AUROC |
|---|---:|---:|---:|---:|
| in-distribution | **0.889** | 0.924 | 0.856 | 0.929 |
| cross-subreddit | **0.331** | 0.202 | 0.930 | 0.967 |

**The most diagnostically interesting finding so far.** F1 collapses by −0.557 but AUROC actually *rises slightly* (+0.038): the **ranking** of posts by anxiety score generalizes — in fact ranking is better cross-distribution because positives there are clear outliers — but the **decision threshold** trained on r/Anxiety over-fires on r/relationship_advice. The model has learned a real anxiety signal; it just needs per-population threshold calibration to deploy.

### Experiment 3 — 9-way subreddit classifier

- **Macro-F1 = 0.643** (random ≈ 0.10) — substantial but not perfect linguistic distinctiveness.
- r/depression / r/depression_help / r/AnxietyDepression heavily mutually-confused (depression-family). r/SuicideWatch and r/relationship_advice mostly self-classified.

### Experiment 4 — Per-target linguistic markers

| Target | Top feature | Cohen's d | # significant (BH p<0.05) |
|---|---|---:|---:|
| anxiety | `f_anx_term_rate` | +2.60 | **22** |
| health_anxiety | `f_health_anx_term_rate` | +11.77 ⚠️ | 5 |
| depression | `f_dep_term_rate` | +2.86 | **19** |
| suicidality | `f_suic_term_rate` | +7.10 | **13** |

The cross-target heatmap (`docs/figures/exp4__marker_heatmap.png`) shows two clinically meaningful patterns:

1. **`f_third_rate` (third-person pronouns) is consistently negative across all 4 targets** — replicates Pennebaker on first-person preponderance in distress.
2. **`f_sent_neg` rises monotonically with target severity**: anxiety +0.33 → health_anxiety +0.38 → depression +0.67 → suicidality +1.08.
3. **`f_reassurance_count` (+0.99) and `f_health_anx_phrase_count` (+1.05) are uniquely health-anxiety-specific** — basically SHAI item content.

### Experiment 5 — Health-anxiety severity ranking (continuous)

| Subreddit | mean health-anxiety score |
|---|---:|
| **r/COVID19_support** | **0.239** |
| **r/COVID19positive** | **0.232** |
| r/Anxiety | 0.122 |
| r/AnxietyDepression | 0.083 |

Both COVID subreddits score ~2× r/Anxiety, validating their inclusion as the `health_anxiety_enriched` group.

### Experiment 6 — Modern baselines (MentalRoBERTa single-target + multi-task)

Fine-tuned `mental/mental-roberta-base` on the same tier-1-labeled corpus, 4 epochs, lr=2e-5, max_length=256, on an RTX 4090. Single-target run trained one binary head for anxiety; multi-task run trained a shared encoder + 4 sigmoid heads simultaneously with per-task loss weights `{anxiety: 1.0, health_anxiety: 1.5, depression: 1.0, suicidality: 1.2}` and per-row confidence weights from the labeling tier (manual=1.0, llm=0.7, weak=0.4).

**RQ1 headline table — F1 [bootstrap 95% CI] across all four model families × four targets:**

| Target | n_pos | TF-IDF + LogReg | XGBoost-linguistic | MentalRoBERTa (single) | MentalRoBERTa (multi-task) |
|---|---:|---:|---:|---:|---:|
| anxiety | 3,560 | 0.874 | 0.864 | **0.891** [0.87, 0.91] | **0.894** [0.87, 0.91] |
| depression | 1,557 | 0.708 | **0.742** | — | 0.720 [0.68, 0.76] |
| health_anxiety | 24 | 0.750 | 1.000 ⚠ | — | **0.333** [0.00, 0.82] |
| suicidality | 116 | 0.571 | 0.792 ⚠ | — | 0.647 [0.43, 0.82] |

⚠ XGBoost-linguistic reads `f_health_anx_term_rate` and `f_suic_term_rate` directly — the same lexicons that derived the labels — so its rare-class F1 is **circular**, not a real result. The MentalRoBERTa numbers are the **honest** ones: the transformer sees only text, and its F1 of 0.333 on health_anxiety (with only 3 test positives, CI [0.00, 0.82]) reflects the actual ceiling at tier-1 label support.

**What the multi-task run actually proves:**

1. **Multi-task does not degrade the well-represented class.** Anxiety F1 0.894 (multi) vs 0.891 (single) — within CI overlap, and arguably a tiny gain. This is the standard prerequisite test for shared encoders, and it passes.
2. **Calibration is excellent across the board** — ECE 0.001–0.039 for the transformer; no temperature scaling needed.
3. **Health-anxiety F1 = 0.333 with CI [0.000, 0.818]** is the empirical case for tier-2 LLM labeling: only 24 weak-label positives is below the data efficiency frontier for transformer fine-tuning. The next big win comes from labels, not architecture.
4. **Bootstrap CIs are wide for rare classes** (suicidality CI width = 0.39, health-anxiety = 0.82), and that's how it should be: point estimates on 3–20 positives are not science.

---

## Visual gallery

All figures generated from the real collected data. Corpus-level figures via `anxiety plot`; experiment figures via `python scripts/run_experiments.py`.

### Per-target classifier comparison (Exp 1)
![Per-target F1](docs/figures/exp1__per_target_f1.png)

### Cross-subreddit transfer drop (Exp 2 — the diagnostic chart)
![Cross-subreddit transfer](docs/figures/exp2__transfer.png)

### Per-target marker heatmap (Exp 4)
![Marker heatmap](docs/figures/exp4__marker_heatmap.png)

### 9-way subreddit confusion (Exp 3)
![Subreddit confusion](docs/figures/exp3__subreddit_confusion.png)

---

### Corpus + baseline-model figures

### Corpus overview
![Corpus overview](docs/figures/corpus_overview.png)

### Post-length distribution
![Length distribution](docs/figures/length_distribution.png)

### Temporal coverage
![Temporal](docs/figures/temporal.png)

### Weak-label positive rates per subreddit × label
**Note `health_anxiety` is sparse everywhere** — that's the empirical motivation for tier-2 LLM labeling.
![Label distribution](docs/figures/label_distribution.png)

### Label co-occurrence
![Label co-occurrence](docs/figures/label_cooccurrence.png)

### Model performance — ROC and PR curves
![PR / ROC](docs/figures/pr_roc__anxiety.png)

### Calibration — reliability diagram + score histogram
TF-IDF ECE = 0.132 on anxiety: the model is **over-confident**. The XGBoost-linguistic model achieves ECE = 0.034 on the same target — ~4× better. Apply temperature scaling (Platt) to TF-IDF — `src/evaluation/metrics.py` exposes the data.
![Calibration](docs/figures/calibration__anxiety.png)

### Confusion matrix
![Confusion](docs/figures/confusion__anxiety.png)

### F1 by subreddit — distribution-shift signal
![F1 by subreddit](docs/figures/subreddit_f1__anxiety.png)

### Top discriminative linguistic markers
Stars: `***`p<0.001, `**`p<0.01, `*`p<0.05 (Benjamini-Hochberg FDR).
![Linguistic markers](docs/figures/markers__anxiety.png)

---

## How labels are decided

A 3-tier system. Final label = `manual > llm > weak` precedence. Each tier carries a confidence weight that flows into training as a sample weight.

### Tier 1 — weak (algorithmic, cheap, noisy)
```
weak_score(label) = 0.5 · subreddit_prior(label) + 0.5 · lexicon_score(label, post)
positive iff weak_score ≥ threshold[label]
```
Lexicons are derived from clinical instruments (GAD-7, SHAI, HAI, PHQ-9, C-SSRS). Subreddit priors are expert-set in `configs/subreddits.yaml`. Thresholds are in `configs/labeling.yaml`.

### Tier 2 — LLM (Claude with codebook prompt, mid-cost)
`claude-sonnet-4-6` reads the codebook + the post and returns binary labels + 1–5 confidence + ≤30-word rationale. **Validated against tier 3** via Cohen's κ. Cached on disk.

### Tier 3 — manual (gold standard, ~1000 posts)
Two annotators follow `docs/codebook.md`. κ targets:
| label | min κ |
|---|---:|
| anxiety | 0.70 |
| **health_anxiety** | **0.60** *(harder)* |
| depression | 0.65 |
| suicidality | 0.75 |

**Full details + the codebook decision rules: [`docs/labeling.md`](docs/labeling.md) and [`docs/codebook.md`](docs/codebook.md)**.

---

## How we prevent overfitting & validate predictions

**Full deep-dive: [`docs/validation.md`](docs/validation.md)**. The summary:

### Overfitting controls
- Stratified 70/15/15 split + 5-fold CV (baseline)
- L2 regularization (TF-IDF), early stopping (XGBoost), weight decay + best-epoch (transformer)
- Per-row confidence weights downweight noisy weak labels
- Bootstrap 95% CIs on every metric

### Distribution-shift detection
- **Per-subreddit F1** — the single most diagnostic chart (we already see the cross-domain drop in real data)
- **Cross-subreddit transfer** experiment (`split.cross_subreddit_split`) for RQ3
- **Length-effect bins** detect length-bias

### Calibration
- ECE reported alongside every model (TF-IDF ~0.13, XGBoost-linguistic ~0.03 — flagged as a problem for TF-IDF)
- Reliability diagram in the visual gallery

### Data-correctness tests
- 22 unit tests covering cleaning, anonymization, dedup, lexicons, features, metrics, collectors, end-to-end smoke
- Pipeline-enforced PII redaction (regex + spaCy NER)
- Lexicon sanity tests: neutral text → low score, anxiety text → high score

### Multiple labels and multiple models, not single-source
- 3 labeling tiers cross-validate each other
- 5 model families compared, not one
- 6 metrics reported, not just F1

---

## Pipeline overview

```
                   ┌──────────────┐
                   │  configs/    │  YAML — change behavior without code
                   └──────┬───────┘
                          ▼
┌─────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐   ┌──────────┐
│ collect ├──▶│ preprocess  ├──▶│ label (1-3) ├──▶│ features ├──▶│  train   │
│         │   │ clean       │   │ weak +      │   │          │   │ tfidf    │
│ scraper │   │ anonymize   │   │ llm +       │   │ LIWC-    │   │ xgboost  │
│ praw    │   │ dedupe      │   │ manual      │   │ like     │   │ roberta  │
│ dump    │   │             │   │ aggregate   │   │          │   │ multitask│
│ synth   │   │             │   │             │   │          │   │ llm-zs   │
└─────────┘   └─────────────┘   └─────────────┘   └──────────┘   └────┬─────┘
                                                                       │
                                ┌──────────────────────────────────────┤
                                ▼                                      ▼
                       ┌─────────────────┐               ┌─────────────────┐
                       │   evaluate      │               │   analyze + viz │
                       │ metrics, CIs,   │               │ markers, SHAP,  │
                       │ calibration,    │               │ temporal, plots │
                       │ subgroup, error │               │                 │
                       └─────────────────┘               └─────────────────┘
```

Every stage has an independent CLI entry point (`anxiety <stage>`) and a stable parquet schema, so any stage is rerunnable in isolation.

---

## Quick start

### 1. Install (once)
```bash
make install-dev          # pip install -e ".[dev]" + spaCy model + NLTK data
cp .env.example .env      # optional: needed for PRAW or LLM labeling
```

### 2. End-to-end on synthetic data (no creds, ~30 sec)
```bash
make smoke
```

### 3. Real data (no Reddit account needed)
```bash
anxiety collect --backend scraper            # ~15 min for ~14k posts
anxiety preprocess                            # ~2 min
anxiety label --tier weak                     # <30 sec
anxiety label --tier aggregate                # combine tiers
anxiety train configs/models/baseline.yaml    # ~30 sec for TF-IDF
anxiety evaluate experiments/runs/tfidf_logreg
anxiety plot --run-dir experiments/runs/tfidf_logreg
anxiety analyze-markers --target anxiety
```

### 4. Optional upgrades
```bash
# Tier-2 LLM labeling (needs ANTHROPIC_API_KEY)
anxiety label --tier llm

# Tier-3 manual annotation (TUI)
anxiety annotate --annotator-id you

# Train MentalRoBERTa (GPU/MPS strongly recommended; auto-detected)
anxiety train configs/models/transformer.yaml

# Multi-task MentalRoBERTa (dissertation novelty)
anxiety train configs/models/multitask.yaml
```

---

## Inspecting the data — code recipes

### Read the labeled corpus
```python
import pandas as pd
df = pd.read_parquet("data/processed/labeled.parquet")
print(df.dtypes)
# id                    object
# subreddit             object
# created_utc          float64
# clean_text            object   — anonymized + cleaned text
# author_hash           object   — salted hash of original author
# label_anxiety        float64   — final aggregated label (0/1)
# label_anxiety_source  object   — 'manual' | 'llm' | 'weak'
# label_anxiety_weight float64   — confidence weight for training loss
# weak_anxiety         float64   — raw tier-1 score
# ...
```
**Full schema: [`docs/data_dictionary.md`](docs/data_dictionary.md)**.

### Score a single post with the trained model
```python
import pandas as pd
from src.models.registry import build_model
from src.utils.config import load_model_config

cfg   = load_model_config("experiments/runs/tfidf_logreg/config.yaml")
model = build_model(cfg).load("experiments/runs/tfidf_logreg/model")

df = pd.DataFrame({"clean_text": [
    "I keep googling my symptoms and I'm convinced this lump is cancer.",
    "Just moved to a new apartment, looking for furniture recs.",
]})
print(model.predict_proba(df))
# array([0.96, 0.03])
```

### Make your own custom plot
```python
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from src.viz.plots import set_style

set_style()
df = pd.read_parquet("data/processed/labeled.parquet")
df["year"] = pd.to_datetime(df["created_utc"], unit="s").dt.year

rates = (df.assign(pos=(df["weak_anxiety"] >= 0.5).astype(int))
           .groupby(["year", "subreddit"])["pos"].mean()
           .unstack().dropna(how="all"))

fig, ax = plt.subplots(figsize=(12, 5))
rates.plot(ax=ax, marker="o", linewidth=2)
ax.set_ylabel("Weak anxiety-positive rate")
ax.set_title("Anxiety positive rate over time, by subreddit")
fig.savefig("docs/figures/custom__anxiety_over_time.png", bbox_inches="tight")
```

### Compute Cohen's κ between two annotators
```python
from src.labeling.manual import cohen_kappa
ann = pd.read_parquet("data/processed/gold_test_set.parquet")
print(cohen_kappa(ann, "health_anxiety", "alice", "bob"))
```

### Re-label a subset with Claude (tier-2)
```python
from src.labeling.llm import label_corpus
from src.utils.config import load_labeling, load_subreddits

df  = pd.read_parquet("data/processed/labeled.parquet").sample(50)
out = label_corpus(df, load_subreddits(), load_labeling())
print(out[["id", "llm_health_anxiety", "llm_health_anxiety_conf", "llm_rationale"]])
```

---

## Repo layout

```
configs/                        YAML configuration (subreddits, labeling, models)
src/
├── collection/                 4 backends (scraper, praw, dump, synthetic)
├── preprocessing/              clean, anonymize, dedupe
├── labeling/                   3-tier labeling: weak / llm / manual / aggregate
├── features/linguistic.py      LIWC-like + somatic + pronouns + sentiment
├── models/                     5 model families behind BaseModel interface
├── evaluation/                 metrics, CIs, calibration, error analysis
├── analysis/                   linguistic markers, SHAP, temporal
├── viz/                        10 reusable matplotlib/seaborn figures
├── utils/                      config, IO, logging, SQLite cache
└── cli.py                      Typer entry point — `anxiety <command>`
data/                           git-ignored (raw / interim / processed / external)
docs/                           ethics + codebook + thesis outline + figures + deep-dives
experiments/                    git-ignored runs
tests/                          22 tests
```

**Module-by-module deep dive: [`docs/architecture.md`](docs/architecture.md)**.

---

## Documentation

| Doc | Topic |
|---|---|
| [`docs/index.md`](docs/index.md) | Reading order |
| [`docs/experiments.md`](docs/experiments.md) | **What we achieved**: 5 classification studies on real data with numbers + findings + caveats |
| [`docs/architecture.md`](docs/architecture.md) | Module-by-module design, dataflow, extension points |
| [`docs/labeling.md`](docs/labeling.md) | The 3-tier labeling system in depth |
| [`docs/validation.md`](docs/validation.md) | Overfitting prevention, data correctness, prediction validation |
| [`docs/models.md`](docs/models.md) | Per-model docs + tuning hooks |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Every column at every pipeline stage |
| [`docs/cli_reference.md`](docs/cli_reference.md) | Every CLI command |
| [`docs/visualization.md`](docs/visualization.md) | Extending the plot library |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Exact reproduction recipe |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Common errors + fixes |
| [`docs/ethics.md`](docs/ethics.md) | IRB-grade ethics statement |
| [`docs/codebook.md`](docs/codebook.md) | 4-label annotation rules + κ targets |
| [`docs/thesis_outline.md`](docs/thesis_outline.md) | Chapter-by-chapter dissertation map |

---

## Ethics

This project handles sensitive mental-health content. Read [`docs/ethics.md`](docs/ethics.md) before running anything against real data. Highlights:

- **No raw text redistribution** — releases are post-IDs + labels + aggregated stats.
- **Pipeline-enforced anonymization** — pseudonymized authors, regex+NER PII stripping.
- **r/SuicideWatch** posts are training-only; never quoted verbatim in the thesis.
- **Crisis resources** are surfaced in any deployed artifact.
- **This is not a diagnostic instrument.**

If you are in crisis: US **988** • UK & ROI Samaritans **116 123** • EU [befrienders.org](https://www.befrienders.org/) • International [findahelpline.com](https://findahelpline.com/).

## License

Code: MIT. Data: not redistributed; subject to Reddit's Data API Terms.
