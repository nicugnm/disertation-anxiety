# Experiments — what we achieved, what we used, what we found

Five separate classification studies executed on the **real collected corpus** (13,133 posts, 9 subreddits) using only the data + lexicons + linguistic features that exist today — no LLM API calls, no transformer, no human annotators required.

> All results below come from `scripts/run_experiments.py` against `data/processed/labeled.parquet`. Re-run any time with `python scripts/run_experiments.py`. Outputs land in `experiments/` (CSVs + JSON summary) and `docs/figures/exp*.png`.

---

## Experimental setup at a glance

| Item | Value |
|---|---|
| Corpus | 13,133 cleaned, anonymized, deduped Reddit posts |
| Subreddits | r/{Anxiety, socialanxiety, AnxietyDepression, depression, depression_help, SuicideWatch, COVID19_support, LivingAlone, relationship_advice} |
| Targets | anxiety / health_anxiety / depression / suicidality (binary, derived from tier-1 weak labels) |
| Models | TF-IDF + Logistic Regression *(text)*, XGBoost on 22 hand-crafted linguistic features *(features)* |
| Feature families | lexical-rate, pronouns (Pennebaker), certainty/uncertainty, length, readability (Flesch / Gunning-Fog), VADER sentiment |
| Splits | 80/20 stratified by label per experiment, `random_state=42` |
| Metrics | F1 / Precision / Recall / AUROC / AUPRC / Brier / ECE + bootstrap CIs |
| Significance testing | Mann-Whitney U with Benjamini-Hochberg FDR correction |

---

## Experiment 1 — Per-target model comparison

Train a TF-IDF + LogReg (text-only) and an XGBoost (linguistic-features-only) for each of the four targets. Compare apples-to-apples — same split, same metrics, same target.

### Results

| Target | n_pos / n_neg | Model | **F1** | Prec | Recall | AUROC | AUPRC | ECE |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| anxiety | 3,433 / 9,700 | TF-IDF + LogReg | **0.892** | 0.906 | 0.879 | 0.980 | 0.955 | 0.139 |
| anxiety | 3,433 / 9,700 | XGBoost (linguistic) | **0.892** | 0.819 | 0.980 | 0.986 | 0.959 | **0.029** |
| health_anxiety | 9 / 13,124 | TF-IDF + LogReg | 0.250 | 0.167 | 0.500 | 0.969 | 0.090 | 0.011 |
| health_anxiety | 9 / 13,124 | XGBoost (linguistic) | **0.800** ⚠️ | 0.667 | 1.000 | 1.000 | 0.833 | 0.001 |
| depression | 1,022 / 12,111 | TF-IDF + LogReg | 0.699 | 0.606 | 0.824 | 0.961 | 0.718 | 0.143 |
| depression | 1,022 / 12,111 | XGBoost (linguistic) | **0.719** | 0.603 | 0.892 | 0.978 | 0.701 | **0.030** |
| suicidality | 133 / 13,000 | TF-IDF + LogReg | 0.704 | 0.568 | 0.926 | 0.996 | 0.617 | 0.058 |
| suicidality | 133 / 13,000 | XGBoost (linguistic) | **0.912** | 0.867 | 0.963 | 0.999 | 0.854 | **0.002** |

![F1 per target](figures/exp1__per_target_f1.png)
![AUROC per target](figures/exp1__per_target_auroc.png)

### What this tells us

1. **XGBoost on 22 linguistic features matches or beats the 80,000-feature TF-IDF text model on 3 of 4 targets.** This is a useful finding for the thesis: the engineered features carry most of the predictive signal.
2. **XGBoost is dramatically better calibrated** — ECE 0.029 vs 0.139 on anxiety. The TF-IDF model is overconfident; XGBoost outputs reflect actual probabilities. This matters when reporting clinical-style risk scores.
3. **Suicidality detection is excellent** (F1 = 0.91, AUROC = 0.999). The lexicon for `SUICIDALITY_TERMS` is small but specific.
4. **Health-anxiety F1 = 0.80 is a small-sample artifact** ⚠️. With only 9 positives, any model that catches 6–9 of them looks great. The XGBoost classifier here essentially memorized the `f_health_anx_term_rate` feature — and that feature was used to *derive* the labels. **This is exactly why tier-2 LLM labeling is needed**: to get a non-circular set of health-anxiety positives.

---

## Experiment 2 — Cross-subreddit transfer (RQ3)

Train on r/{Anxiety, socialanxiety, AnxietyDepression}, evaluate on r/{COVID19_support, LivingAlone, relationship_advice}. Tests whether the model learns the *phenomenon* or the *style* of anxiety subreddits.

### Results

| Target | Split | F1 | Precision | Recall | AUROC |
|---|---|---:|---:|---:|---:|
| anxiety | in-distribution (held-out anxiety subs) | **0.969** | 0.997 | 0.944 | 0.996 |
| anxiety | cross-subreddit (COVID/lifestyle/baseline) | **0.344** | 0.213 | 0.892 | 0.966 |

![Cross-subreddit transfer](figures/exp2__transfer.png)

### What this tells us — the most diagnostically interesting finding

The F1 collapses by **−0.625** out-of-distribution, but the **AUROC barely moves** (−0.030). Read carefully:

- **Recall stays high** (0.892) — the model still finds anxious posts in the new subreddits.
- **Precision crashes** (0.213) — the model fires *way too often* in the baseline subs.
- **AUROC ≈ 0.97** — the *ranking* of posts by anxiety score is still accurate.

**Interpretation:** the model has learned a real anxiety signal that generalizes across subreddits. What doesn't generalize is the **decision threshold**. A threshold tuned on r/Anxiety (where ~85% of posts are positive) over-fires on r/relationship_advice (where ~3% are positive).

**Practical fix:** per-subreddit threshold calibration — pick a different threshold per group that achieves a target precision. The AUROC tells us the ranking is sound; we just need to map scores to decisions per population.

This is the single most important RQ3 finding so far, and the AUROC-vs-F1 split is the kind of nuanced result a strong thesis hangs on.

---

## Experiment 3 — 9-way subreddit classifier

Treats the subreddit name as the multiclass label. Tests how linguistically distinct the subreddits are. A high macro-F1 means subreddits have unique vocabulary; a low one means they blur.

### Results

- **Macro-F1 = 0.648** (random baseline ≈ 0.11)
- 4× better than chance — substantial linguistic distinctiveness, but not perfect separation.

![Subreddit classification confusion matrix](figures/exp3__subreddit_confusion.png)

### What this tells us

Look at the off-diagonal cells of the confusion matrix:

- **r/Anxiety ↔ r/socialanxiety** confusion is moderate — two anxiety subs share vocabulary. Predictable.
- **r/depression ↔ r/depression_help ↔ r/AnxietyDepression** — these three are heavily mutually-confused, which is a clean linguistic confirmation that "depression" subs are all topically similar.
- **r/relationship_advice** is mostly self-classified — long posts with distinctive vocabulary about relationships, partners, etc.
- **r/SuicideWatch** posts are correctly identified at high rates — the suicidality vocabulary is distinctive.

**Implication for label-validation:** the subreddit-of-origin is recoverable from text. That means if you train a binary anxiety classifier on subreddit-derived labels and evaluate on the same subreddits, **the classifier can cheat by detecting subreddit style.** This is precisely what experiment 2 showed quantitatively.

---

## Experiment 4 — Per-target linguistic markers

For every linguistic feature, compute Cohen's d between positives and negatives on each of the 4 targets. Apply Mann-Whitney U + Benjamini-Hochberg FDR correction. Combine into a heatmap.

### Top-feature summary

| Target | Top feature | Cohen's d | # significant (BH p<0.05) |
|---|---|---:|---:|
| anxiety | `f_anx_term_rate` | +2.47 | 22 |
| **health_anxiety** | **`f_health_anx_term_rate`** | **+16.21** ⚠️ | 4 |
| depression | `f_dep_term_rate` | +3.00 | 19 |
| suicidality | `f_suic_term_rate` | +7.10 | 14 |

![Marker heatmap](figures/exp4__marker_heatmap.png)

### What this tells us — the clinically interesting structure

Reading the heatmap row-by-row:

#### Diagonal pattern (each lexicon dominates its own target)
Every target's top feature is the lexicon *of that target* — sanity check passes. The huge `d = +16.21` for health-anxiety is a small-sample artifact (9 positives, near-zero variance among negatives), not a real effect size.

#### Cross-target leakage — the genuinely interesting structure
- **`f_anx_term_rate` is +2.47 for anxiety AND +1.49 for health_anxiety.** General-anxiety vocabulary heavily overlaps with health anxiety — exactly what you'd predict from the SHAI/HAI clinical literature, where health anxiety is a *subtype* of anxiety.
- **`f_third_rate` (3rd-person pronouns) is consistently negative** across all 4 targets (−0.51 / −0.52 / −0.28 / −0.06). This **replicates Pennebaker** — distressed posts are more first-person-focused.
- **`f_sent_neg` rises monotonically with severity**: anxiety +0.33 → health_anxiety +0.38 → depression +0.67 → **suicidality +1.08**. Negative-sentiment vocabulary tracks distress severity exactly as clinical theory predicts.
- **`f_reassurance_count` is +0.99 for health_anxiety only** — basically the SHAI item content. That's a strong sign the lexicon is doing what it should.
- **`f_health_anx_phrase_count` is +1.05 for health_anxiety only** — same.
- **`f_first_plur_rate` is more negative for health_anxiety (−0.44) than other targets** — health-anxious posts use *less* "we/us/our" → consistent with social-isolation in health anxiety.

This kind of marker structure is the substrate for a strong RQ2 chapter.

---

## Experiment 5 — Health-anxiety severity ranking

Binary health-anxiety classification has only 9 positives, so we treat the continuous `weak_health_anxiety` score as a severity proxy. Look at the highest-scoring posts and the per-subreddit means.

### Score distribution by subreddit (top 3)

| Subreddit | mean health-anxiety score |
|---|---:|
| **COVID19_support** | **0.238** |
| Anxiety | 0.119 |
| AnxietyDepression | 0.082 |

### What this tells us

- **r/COVID19_support has 2× the mean health-anxiety score of r/Anxiety**, validating its inclusion as `health_anxiety_enriched` in our subreddit configuration.
- The maximum score is 0.725 (out of 1.0), so even our most-health-anxious posts don't saturate the lexicon — there's still a lot of unexplained variance.
- Top-20 highest-scoring posts are written to `experiments/exp5_health_anxiety_top20.csv` — eyeballing them is a useful sanity check that the lexicon catches genuine health-anxiety content, not just noise.

---

## What we used (concrete inventory)

### Models
- `sklearn.linear_model.LogisticRegression` (C=1.0, balanced class weights, liblinear) on TF-IDF features (1-2 grams, min_df=5, max_df=0.95, sublinear TF, 80k max features).
- `xgboost.XGBClassifier` (400 trees, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=auto, hist tree method) on 22 hand-crafted linguistic features.

### Linguistic features (22 total)
- **Lexical rates (8)**: `f_anx_term_rate`, `f_anx_phrase_count`, `f_health_anx_term_rate`, `f_health_anx_phrase_count`, `f_reassurance_count`, `f_dep_term_rate`, `f_suic_term_rate`, `f_body_part_rate`
- **Pronouns (4)**: `f_first_sing_rate`, `f_first_plur_rate`, `f_second_rate`, `f_third_rate`
- **Certainty (3)**: `f_uncertainty_rate`, `f_certainty_rate`, `f_question_rate`
- **Length (5)**: `f_n_chars`, `f_n_tokens`, `f_n_sents`, `f_avg_sent_len`, `f_avg_word_len`
- **Readability (2)**: `f_flesch`, `f_gunning_fog`
- **Sentiment (4)**: `f_sent_compound`, `f_sent_pos`, `f_sent_neg`, `f_sent_neu`

### Lexicons (with clinical provenance)
- `ANXIETY_TERMS / PHRASES` ← **GAD-7** (Spitzer et al., 2006)
- `HEALTH_ANXIETY_TERMS / PHRASES` ← **SHAI** (Salkovskis et al., 2002), **HAI** (Lucock & Morley, 1996)
- `REASSURANCE_PATTERNS` ← Salkovskis (1989) cognitive-behavioral model
- `DEPRESSION_TERMS` ← **PHQ-9** (Kroenke et al., 2001)
- `SUICIDALITY_TERMS` ← **Columbia C-SSRS** (Posner et al., 2011)
- Pronoun categories ← Pennebaker, Mayne, Francis (1997)
- Body parts ← Somatic vocabulary marker (used to distinguish health anxiety from general anxiety)

### Statistical machinery
- `scipy.stats.mannwhitneyu` for non-parametric significance
- Benjamini-Hochberg FDR correction (custom implementation in `src/analysis/linguistic_markers.py`)
- Cohen's d for effect size
- `sklearn.metrics.roc_curve / precision_recall_curve / cohen_kappa_score`

---

## Headline findings (for the thesis introduction)

1. ✅ **Anxiety is detectable.** TF-IDF + LogReg achieves F1 = 0.89, AUROC = 0.98 in-distribution. XGBoost on 22 hand-crafted linguistic features matches text-model F1 with 5× better calibration.
2. ✅ **Suicidality detection works very well** (F1 = 0.91, AUROC = 0.999) using a small specific lexicon.
3. ✅ **Cross-subreddit transfer is partially preserved** — AUROC stays at 0.97 but precision-at-threshold collapses, showing the model has learned a real signal that needs per-population threshold calibration.
4. ✅ **Linguistic markers replicate clinical theory** — first-person preponderance (Pennebaker), negative-sentiment-tracks-severity, reassurance-seeking specific to health anxiety (SHAI item content).
5. ⚠️ **Health-anxiety binary classification is bottlenecked by label sparsity** (9 positives in 13k). The continuous severity score correctly identifies COVID19_support as the most health-anxious subreddit (2× the mean of r/Anxiety), but supervised binary modeling needs tier-2 LLM enrichment to enlarge the positive class.
6. ⚠️ **Calibration is a problem for the text model** (ECE = 0.14). Apply temperature scaling or use the better-calibrated XGBoost outputs.
7. 🟢 **The 9-way subreddit classifier achieves macro-F1 = 0.65** (vs ~0.11 chance), confirming substantial linguistic distinctiveness — but also confirming the risk of subreddit-style leakage that motivates rigorous cross-subreddit transfer evaluation.

---

## Caveats — what these experiments don't show

- **All labels are still tier-1 only** (subreddit prior + lexicon). The model that scores F1 = 0.89 is partly distilling its training signal back into a logistic regression. Real validation requires tier-2 LLM and tier-3 manual labels.
- **Health-anxiety F1 = 0.80 is not a real result.** Nine positives, of which the XGBoost model has direct lexicon-feature access, makes that finding circular. **Do not put this number in the thesis abstract.**
- **No bootstrap CIs were computed in this experiment script** for speed (would be cheap to add — just pass `bootstrap=True` to `full_report`). Add them before the methodology chapter is finalized.
- **No statistical comparison between models** is reported. Use McNemar's test on the held-out predictions before claiming Model A "beats" Model B.
- **The 22-feature space might be missing variables** that would change the marker analysis. Future work: add LIWC-2022 categories, sentence-embedding clusters, syntactic complexity.

---

## How to reproduce

```bash
# 1. Make sure the corpus is built
anxiety collect --backend scraper        # or --backend praw
anxiety preprocess
anxiety label --tier weak
anxiety label --tier aggregate

# 2. Run the experiment suite
/opt/miniconda3/envs/disertation-anxiety/bin/python scripts/run_experiments.py
```

Outputs:
- `experiments/exp1_per_target.csv` — full metrics per (target × model)
- `experiments/exp2_transfer.csv` — in-distribution vs cross-subreddit
- `experiments/exp3_subreddit_confusion.csv` — 9×9 confusion matrix
- `experiments/exp4_markers__<target>.csv` — full per-feature stats (4 files)
- `experiments/exp5_health_anxiety_top20.csv` — highest-scoring posts
- `experiments/exp5_severity_by_subreddit.csv` — per-sub mean/median/std/count
- `experiments/experiments_summary.json` — single-file summary
- `docs/figures/exp1__*.png`, `exp2__transfer.png`, `exp3__subreddit_confusion.png`, `exp4__marker_heatmap.png`
