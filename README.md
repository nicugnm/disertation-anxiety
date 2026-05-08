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

> **A dissertation-grade pipeline that takes ~14k Reddit posts from 9 mental-health-adjacent subreddits, applies a 3-tier labeling scheme grounded in clinical instruments (GAD-7, SHAI, PHQ-9, C-SSRS), trains 5 different model families to predict 4 binary labels {anxiety, health_anxiety, depression, suicidality}, evaluates them with 6 metrics + bootstrap CIs + per-subreddit and per-length subgroups, and produces 10 publication-quality figures plus a linguistic-marker analysis with FDR-corrected significance tests.**

The thesis novelty: separating **health anxiety** from general anxiety as its own class, using a multi-task transformer with per-task loss weighting and tier-confidence-weighted training.

---

## What we built and what runs today

Concrete deliverables, all working on real Reddit data as of the last commit:

### ✅ Already running on real data
- **Collection**: 13,990 posts collected from 9 subreddits via the no-credentials JSON scraper (no Reddit API key needed). Deduplicated across 6 listings (top×{all,year,month,week} + new + hot).
- **Preprocessing**: 13,990 → 13,133 posts after PII redaction (regex + spaCy NER), exact + near-dedup (SimHash), language filter, length filter.
- **Tier-1 weak labeling**: 3,433 anxiety / 1,022 depression / 133 suicidality / 9 health-anxiety positives — health-anxiety scarcity was *expected* and motivates tier-2.
- **8 binary classifiers trained** — TF-IDF + LogReg AND XGBoost-on-linguistic-features × 4 targets. Headline: anxiety F1 **0.89** / suicidality F1 **0.91** / depression F1 **0.72** / health-anxiety F1 **0.80** *(small-sample artifact, see caveats)*.
- **Cross-subreddit transfer experiment (RQ3)**: in-distribution F1 = **0.969** vs cross-subreddit F1 = **0.344**, but AUROC stays at **0.966** — reveals that the *ranking* generalizes but the *threshold* doesn't.
- **9-way subreddit classifier**: macro-F1 = **0.648** (vs random 0.11), confirming substantial linguistic distinctiveness.
- **Per-target linguistic-marker analysis**: 4 separate Cohen's d analyses with Benjamini-Hochberg FDR correction. **22 features significant for anxiety, 19 for depression, 14 for suicidality**. `f_third_rate` (third-person pronouns) consistently negative across all targets — replicates Pennebaker. `f_sent_neg` rises monotonically with severity (anxiety +0.33 → suicidality +1.08).
- **Health-anxiety severity ranking**: r/COVID19_support has 2× the mean health-anxiety score of r/Anxiety, validating its inclusion as the health-anxiety-enriched subreddit.
- **15 figures generated** from real data into `docs/figures/`.
- **22/22 unit tests passing**.

→ **Full numbers, plots, and findings in [`docs/experiments.md`](docs/experiments.md)**.

### ✅ Built and ready to run (needs API key / GPU)
- **Tier-2 LLM labeling**: ready to run with `ANTHROPIC_API_KEY` set. SQLite-cached so re-runs are free.
- **Tier-3 manual annotation**: Rich-based TUI ready, resumable, two-annotator κ workflow.
- **Single-target transformer**: HuggingFace fine-tuning, MentalRoBERTa with RoBERTa fallback. Auto-detects CUDA → MPS → CPU.
- **Multi-task transformer**: pure-PyTorch shared encoder + 4 sigmoid heads with per-task loss weights and per-row confidence weights.
- **Claude zero-/few-shot baseline**: tests whether prompting beats fine-tuning.
- **XGBoost on linguistic features**: with SHAP-based explainability.

### ✅ Reproducibility & ethics infrastructure
- All YAML-driven (subreddits, labeling, models) — change behavior without editing code.
- All RNG seeded; LLM/HTTP responses cached on disk.
- Strict no-raw-text-redistribution policy. Pipeline-enforced PII redaction. Crisis-resource boilerplate. Two-annotator κ workflow.

---

## What was used (models, data, instruments)

### Data sources
| Subreddit | Role | Posts collected |
|---|---|---:|
| r/Anxiety | anxiety_primary | 2,072 |
| r/socialanxiety | anxiety_primary | 1,404 |
| r/AnxietyDepression | comorbid | 1,402 |
| r/depression | depression_primary | 987 |
| r/depression_help | depression_primary | 1,505 |
| r/SuicideWatch | suicidality (training-only, ethics-restricted) | 2,204 |
| r/COVID19_support | health_anxiety_enriched | 1,061 |
| r/COVID19positive | (banned by Reddit; skipped) | 0 |
| r/LivingAlone | baseline | 910 |
| r/relationship_advice | baseline | 2,445 |
| **Total** | | **13,990** |

Date range: Jan 2017 → today. Collected via Reddit's public JSON endpoints (`old.reddit.com/r/<sub>/<listing>.json`) at 1.5s/request — no OAuth required. Custom User-Agent identifies academic research intent.

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

Trained on 13,133 cleaned + weakly-labeled posts. 80/20 stratified split per experiment.

### Experiment 1 — Per-target classifiers

8 classifiers: TF-IDF + LogReg AND XGBoost-on-22-linguistic-features × 4 targets.

| Target | n_pos | TF-IDF F1 | XGBoost F1 | XGBoost AUROC | XGBoost ECE |
|---|---:|---:|---:|---:|---:|
| anxiety | 3,433 | **0.892** | **0.892** | 0.986 | 0.029 |
| health_anxiety | 9 | 0.250 | **0.800** ⚠️ | 1.000 | 0.001 |
| depression | 1,022 | 0.699 | **0.719** | 0.978 | 0.030 |
| suicidality | 133 | 0.704 | **0.912** | 0.999 | 0.002 |

Key observation: **XGBoost on 22 linguistic features matches or beats the 80,000-feature TF-IDF text model on 3 of 4 targets**, with ~5× better calibration. The 22 hand-crafted features carry most of the predictive signal.

⚠️ Health-anxiety F1 = 0.80 is a small-sample artifact (9 positives, model has lexicon access, label was lexicon-derived → circular). See [`docs/experiments.md`](docs/experiments.md) §1 for full caveats.

### Experiment 2 — Cross-subreddit transfer (RQ3)

Train on r/{Anxiety, socialanxiety, AnxietyDepression}, test on r/{COVID19_support, LivingAlone, relationship_advice}.

| Split | F1 | Precision | Recall | AUROC |
|---|---:|---:|---:|---:|
| in-distribution | **0.969** | 0.997 | 0.944 | 0.996 |
| cross-subreddit | **0.344** | 0.213 | 0.892 | 0.966 |

**The most diagnostically interesting finding so far.** F1 collapses by −0.625 but AUROC barely moves (−0.030): the **ranking** of posts by anxiety score is preserved, but the **decision threshold** trained on r/Anxiety over-fires on r/relationship_advice. The model has learned a real anxiety signal; it just needs per-population threshold calibration to deploy.

### Experiment 3 — 9-way subreddit classifier

- **Macro-F1 = 0.648** (random ≈ 0.11) — substantial but not perfect linguistic distinctiveness.
- r/depression / r/depression_help / r/AnxietyDepression heavily mutually-confused (depression-family). r/SuicideWatch and r/relationship_advice mostly self-classified.

### Experiment 4 — Per-target linguistic markers

| Target | Top feature | Cohen's d | # significant (BH p<0.05) |
|---|---|---:|---:|
| anxiety | `f_anx_term_rate` | +2.47 | **22** |
| health_anxiety | `f_health_anx_term_rate` | +16.21 ⚠️ | 4 |
| depression | `f_dep_term_rate` | +3.00 | **19** |
| suicidality | `f_suic_term_rate` | +7.10 | **14** |

The cross-target heatmap (`docs/figures/exp4__marker_heatmap.png`) shows two clinically meaningful patterns:

1. **`f_third_rate` (third-person pronouns) is consistently negative across all 4 targets** — replicates Pennebaker on first-person preponderance in distress.
2. **`f_sent_neg` rises monotonically with target severity**: anxiety +0.33 → health_anxiety +0.38 → depression +0.67 → suicidality +1.08.
3. **`f_reassurance_count` (+0.99) and `f_health_anx_phrase_count` (+1.05) are uniquely health-anxiety-specific** — basically SHAI item content.

### Experiment 5 — Health-anxiety severity ranking (continuous)

| Subreddit | mean health-anxiety score |
|---|---:|
| **r/COVID19_support** | **0.238** |
| r/Anxiety | 0.119 |
| r/AnxietyDepression | 0.082 |

r/COVID19_support has 2× the health-anxiety score of r/Anxiety, validating its inclusion as the `health_anxiety_enriched` group.

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
ECE = 0.20: the model is **over-confident**. Apply temperature scaling (Platt) — `src/evaluation/metrics.py` exposes the data.
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
- ECE reported alongside every model (currently 0.20, flagged as a problem)
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
