# Experiments — what we achieved, what we used, what we found

Nine classification/evaluation studies using only the data + lexicons + linguistic features that exist today — no LLM API calls, no human annotators required. Experiments 1–6 run on the **stale 16,382-post / 10-subreddit snapshot** (weak labels; Experiment 6 adds transformer models); Experiments 7–9 run on the **current 744k-post corpus** — Experiment 7 evaluates against the self-disclosure user-level test set, Experiment 8 is a head-to-head on r/HealthAnxiety vs r/Anxiety, and Experiment 9 tests domain-adversarial (DANN) training (a negative result).

> All results below come from `scripts/run_experiments.py` against `data/processed/labeled.parquet`. Re-run any time with `python scripts/run_experiments.py`. Outputs land in `experiments/` (CSVs + JSON summary) and `docs/figures/exp*.png`.

---

## Experimental setup at a glance

| Item | Value |
|---|---|
| Corpus | 16,382 cleaned, anonymized, deduped Reddit posts |
| Subreddits | r/{Anxiety, socialanxiety, AnxietyDepression, depression, depression_help, SuicideWatch, COVID19_support, COVID19positive, LivingAlone, relationship_advice} |
| Targets | anxiety / health_anxiety / depression / suicidality (binary, derived from tier-1 weak labels) |
| Models | TF-IDF + Logistic Regression *(text)*, XGBoost on 26 hand-crafted linguistic features *(features)* |
| Feature families | lexical-rate (8), pronouns (4, Pennebaker), certainty/uncertainty (3), length (5), readability (2, Flesch / Gunning-Fog), VADER sentiment (4) |
| Splits | 80/20 stratified by label per experiment, `random_state=42` |
| Metrics | F1 / Precision / Recall / AUROC / AUPRC / Brier / ECE + bootstrap CIs |
| Significance testing | Mann-Whitney U with Benjamini-Hochberg FDR correction |

---

## Experiment 1 — Per-target model comparison

Train a TF-IDF + LogReg (text-only) and an XGBoost (linguistic-features-only) for each of the four targets. Compare apples-to-apples — same split, same metrics, same target.

### Results

| Target | n_pos / n_neg | Model | **F1** | Prec | Recall | AUROC | AUPRC | ECE |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| anxiety | 3,560 / 12,822 | TF-IDF + LogReg | **0.874** | 0.892 | 0.857 | 0.982 | 0.949 | 0.132 |
| anxiety | 3,560 / 12,822 | XGBoost (linguistic) | 0.864 | 0.790 | 0.952 | 0.984 | 0.942 | **0.034** |
| health_anxiety | 24 / 16,358 | TF-IDF + LogReg | 0.750 | 1.000 | 0.600 | 0.992 | 0.638 | 0.019 |
| health_anxiety | 24 / 16,358 | XGBoost (linguistic) | **1.000** ⚠️ | 1.000 | 1.000 | 1.000 | 1.000 | **0.000** |
| depression | 1,557 / 14,825 | TF-IDF + LogReg | 0.708 | 0.696 | 0.720 | 0.964 | 0.766 | 0.144 |
| depression | 1,557 / 14,825 | XGBoost (linguistic) | **0.742** | 0.655 | 0.855 | 0.976 | 0.760 | **0.035** |
| suicidality | 116 / 16,266 | TF-IDF + LogReg | 0.571 | 0.485 | 0.696 | 0.995 | 0.538 | 0.044 |
| suicidality | 116 / 16,266 | XGBoost (linguistic) | **0.792** | 0.760 | 0.826 | 0.998 | 0.717 | **0.003** |

![F1 per target](figures/exp1__per_target_f1.png)
![AUROC per target](figures/exp1__per_target_auroc.png)

### What this tells us

1. **XGBoost on 26 linguistic features matches or beats the 80,000-feature TF-IDF text model on 3 of 4 targets.** This is a useful finding for the thesis: the engineered features carry most of the predictive signal.
2. **XGBoost is dramatically better calibrated** — ECE 0.034 vs 0.132 on anxiety. The TF-IDF model is overconfident; XGBoost outputs reflect actual probabilities. This matters when reporting clinical-style risk scores.
3. **Suicidality detection is excellent** (F1 = 0.79, AUROC = 0.998). The lexicon for `SUICIDALITY_TERMS` is small but specific. The F1 dropped from a prior run (0.91 → 0.79) when the negative pool grew from 13k → 16k posts (added r/depression, r/depression_help, r/COVID19positive); the class is more imbalanced now and the threshold-tuned F1 is more pessimistic.
4. **Health-anxiety F1 = 1.00 is a small-sample artifact** ⚠️. With only 24 positives — and labels derived from the same lexicon the XGBoost model directly reads via `f_health_anx_term_rate` — perfect F1 is exactly what circularity predicts. The clean evaluation for health anxiety is the self-disclosure test set (Experiment 7) or the head-to-head with r/HealthAnxiety (Experiment 8), not weak-label F1.

---

## Experiment 2 — Cross-subreddit transfer (RQ3)

Train on r/{Anxiety, socialanxiety, AnxietyDepression}, evaluate on r/{COVID19_support, LivingAlone, relationship_advice}. Tests whether the model learns the *phenomenon* or the *style* of anxiety subreddits.

### Results

| Target | Split | F1 | Precision | Recall | AUROC |
|---|---|---:|---:|---:|---:|
| anxiety | in-distribution (held-out anxiety subs) | **0.889** | 0.924 | 0.856 | 0.929 |
| anxiety | cross-subreddit (COVID/lifestyle/baseline) | **0.331** | 0.202 | 0.930 | 0.967 |

![Cross-subreddit transfer](figures/exp2__transfer.png)

### What this tells us — the most diagnostically interesting finding

The F1 collapses by **−0.557** out-of-distribution, but the **AUROC actually rises slightly** (+0.038). Read carefully:

- **Recall stays high** (0.930) — the model still finds anxious posts in the new subreddits.
- **Precision crashes** (0.202) — the model fires *way too often* in the baseline subs.
- **AUROC = 0.97** — the *ranking* of posts by anxiety score is still accurate; in fact slightly *better* cross-distribution because the (few) real positives there are clear outliers in a mostly-non-anxious pool.

**Interpretation:** the model has learned a real anxiety signal that generalizes across subreddits. What doesn't generalize is the **decision threshold**. A threshold tuned on r/Anxiety (where ~85% of posts are positive) over-fires on r/relationship_advice (where ~3% are positive).

**Practical fix:** per-subreddit threshold calibration — pick a different threshold per group that achieves a target precision. The AUROC tells us the ranking is sound; we just need to map scores to decisions per population.

This is the single most important RQ3 finding so far, and the AUROC-vs-F1 split is the kind of nuanced result a strong thesis hangs on.

> **Methodology note**: a previous version of `scripts/run_experiments.py` evaluated the in-distribution F1 on data the pipeline had already been fitted on (the `train_test_split` was applied *after* `pipe.fit(df_train_xs[...])`). The in-distribution F1 was reported as 0.969 / AUROC 0.996. Fixed by splitting train/val first, fitting only on train. The corrected numbers above are 0.889 / 0.929 — the cross-distribution drop is real, just 8 points smaller than the leaky version claimed.

---

## Experiment 3 — 9-way subreddit classifier

Treats the subreddit name as the multiclass label. Tests how linguistically distinct the subreddits are. A high macro-F1 means subreddits have unique vocabulary; a low one means they blur.

### Results

- **Macro-F1 = 0.643** (random baseline ≈ 0.10)
- ~6× better than chance — substantial linguistic distinctiveness, but not perfect separation.

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
| anxiety | `f_anx_term_rate` | +2.60 | 22 |
| **health_anxiety** | **`f_health_anx_term_rate`** | **+11.77** ⚠️ | 5 |
| depression | `f_dep_term_rate` | +2.86 | 19 |
| suicidality | `f_suic_term_rate` | +7.10 | 13 |

![Marker heatmap](figures/exp4__marker_heatmap.png)

### What this tells us — the clinically interesting structure

Reading the heatmap row-by-row:

#### Diagonal pattern (each lexicon dominates its own target)
Every target's top feature is the lexicon *of that target* — sanity check passes. The huge `d = +11.77` for health-anxiety is a small-sample artifact (24 positives, near-zero variance among negatives), not a real effect size.

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

Binary health-anxiety classification has only 24 positives, so we treat the continuous `weak_health_anxiety` score as a severity proxy. Look at the highest-scoring posts and the per-subreddit means.

### Score distribution by subreddit (top 4)

| Subreddit | mean health-anxiety score |
|---|---:|
| **COVID19_support** | **0.239** |
| **COVID19positive** | **0.232** |
| Anxiety | 0.122 |
| AnxietyDepression | 0.083 |

### What this tells us

- **Both COVID subreddits score ~2× the mean health-anxiety of r/Anxiety**, validating their inclusion as `health_anxiety_enriched` in our subreddit configuration. r/COVID19positive — which we'd previously been unable to scrape — turns out to be almost as health-anxiety-rich as r/COVID19_support.
- The maximum score is 0.725 (out of 1.0), so even our most-health-anxious posts don't saturate the lexicon — there's still a lot of unexplained variance.
- Top-20 highest-scoring posts are written to `experiments/exp5_health_anxiety_top20.csv` — eyeballing them is a useful sanity check that the lexicon catches genuine health-anxiety content, not just noise.

---

## Experiment 6 — Modern baselines (MentalRoBERTa, single-target + multi-task)

Run independently of `scripts/run_experiments.py` via the CLI:

```bash
anxiety train configs/models/transformer.yaml           # single-target on anxiety
anxiety evaluate experiments/runs/mentalbert_anxiety

anxiety train configs/models/multitask.yaml             # 4-target joint heads
anxiety evaluate experiments/runs/multitask_anxiety_health_dep_suic
```

`mental/mental-roberta-base` (Ji et al., 2022) fine-tuned on the same tier-1-labeled corpus, 4 epochs, lr=2e-5, max_length=256, on an RTX 4090 (training time ~12 min single-target, ~25 min multi-task). The multi-task variant trains a shared encoder + 4 sigmoid heads simultaneously with per-task loss weights `{anxiety: 1.0, health_anxiety: 1.5, depression: 1.0, suicidality: 1.2}` and per-row confidence weights propagated from `label_<k>_weight`.

### Results

**Single-target (anxiety only):**

| Metric | Value | 95% CI |
|---|---:|---|
| F1 | **0.891** | [0.869, 0.909] |
| AUROC | **0.985** | [0.980, 0.989] |
| AUPRC | 0.957 | [0.945, 0.968] |
| ECE | 0.032 | — |
| Precision / Recall | 0.887 / 0.895 | — |

**Multi-task (joint over 4 targets):**

| Target | n_pos / n_neg (test) | F1 [95% CI] | AUROC [95% CI] | ECE |
|---|---:|---:|---:|---:|
| anxiety | 534 / 1,924 | **0.894** [0.875, 0.912] | 0.987 [0.982, 0.991] | 0.039 |
| depression | 216 / 2,242 | 0.720 [0.675, 0.759] | 0.973 [0.965, 0.979] | 0.033 |
| health_anxiety | 3 / 2,455 | 0.333 [0.000, 0.818] | 0.994 [0.989, 1.000] | 0.001 |
| suicidality | 20 / 2,438 | 0.647 [0.429, 0.817] | 0.995 [0.991, 0.998] | 0.004 |

#### PR / ROC curves (per target)

| anxiety | depression |
|---|---|
| ![PR/ROC anxiety](figures/pr_roc__anxiety.png) | ![PR/ROC depression](figures/pr_roc__depression.png) |

| health_anxiety | suicidality |
|---|---|
| ![PR/ROC health_anxiety](figures/pr_roc__health_anxiety.png) | ![PR/ROC suicidality](figures/pr_roc__suicidality.png) |

#### Calibration (reliability diagrams)

| anxiety | depression |
|---|---|
| ![Calibration anxiety](figures/calibration__anxiety.png) | ![Calibration depression](figures/calibration__depression.png) |

| health_anxiety | suicidality |
|---|---|
| ![Calibration health_anxiety](figures/calibration__health_anxiety.png) | ![Calibration suicidality](figures/calibration__suicidality.png) |

#### Confusion matrices

| anxiety | depression |
|---|---|
| ![Confusion anxiety](figures/confusion__anxiety.png) | ![Confusion depression](figures/confusion__depression.png) |

| health_anxiety | suicidality |
|---|---|
| ![Confusion health_anxiety](figures/confusion__health_anxiety.png) | ![Confusion suicidality](figures/confusion__suicidality.png) |

#### Per-subreddit F1 — distribution-shift diagnostic per target

| anxiety | depression |
|---|---|
| ![F1-by-sub anxiety](figures/subreddit_f1__anxiety.png) | ![F1-by-sub depression](figures/subreddit_f1__depression.png) |

| health_anxiety | suicidality |
|---|---|
| ![F1-by-sub health_anxiety](figures/subreddit_f1__health_anxiety.png) | ![F1-by-sub suicidality](figures/subreddit_f1__suicidality.png) |

### RQ1 headline table (cross-model, cross-target F1)

| Target | n_pos (full corpus) | TF-IDF + LogReg | XGBoost-linguistic | MentalRoBERTa (single) | MentalRoBERTa (multi-task) |
|---|---:|---:|---:|---:|---:|
| **anxiety** | 3,560 | 0.874 | 0.864 | 0.891 | **0.894** |
| **depression** | 1,557 | 0.708 | **0.742** | — | 0.720 |
| **health_anxiety** | 24 | 0.750 | 1.000 ⚠ | — | **0.333** |
| **suicidality** | 116 | 0.571 | 0.792 ⚠ | — | 0.647 |

⚠ XGBoost-linguistic reads `f_health_anx_term_rate` and `f_suic_term_rate` directly — the same lexicons that derived the labels — so its rare-class F1 is **circular**, not a real result. **The MentalRoBERTa numbers are the honest ones** because the transformer has no lexicon-feature shortcut and must learn from text alone.

### What this tells us

1. **Multi-task ≈ single-task on anxiety** (0.894 vs 0.891 F1). The shared encoder doesn't degrade the dense class. This is the standard prerequisite test for joint training — and it passes.
2. **The transformer reveals the XGBoost-linguistic lexicon-circularity.** XGBoost's apparent F1 = 1.000 on health_anxiety and F1 = 0.792 on suicidality were inflated by the model reading the same lexicons used to derive the labels. The multi-task transformer's F1 = 0.333 / 0.647 on those targets is what the underlying signal actually supports. **This is itself a methodological finding worth a paragraph in the discussion chapter** — single-feature-lexicon ML on weak labels can produce circular results that look like signal.
3. **Health-anxiety F1 = 0.333 with CI [0.000, 0.818]** is the data-efficiency ceiling for tier-1 labels: 24 total positives (3 in test) is not enough for transformer fine-tuning to learn the class reliably. The next high-impact step is self-disclosure enrichment to increase positive-label coverage, not a better model.
4. **Calibration is excellent for transformers** — ECE 0.001–0.039 with no post-hoc temperature scaling. The TF-IDF model needs Platt; the transformer doesn't.
5. **Bootstrap CIs work and matter.** The health-anxiety CI of [0.000, 0.818] honestly communicates "we don't have enough positives to claim anything." Use these CIs in the thesis instead of point estimates.

---

## Experiment 7 — User-level self-disclosure evaluation

Run via `anxiety eval-disclosure`. Evaluates models against the **self-disclosure user-level test set** — the clean evaluation signal not derived from weak labels.

- **Positives**: users with at least one verified self-disclosure post (regex + negation/hypothetical/third-party/denial filters, `src/labeling/self_disclosure.py`).
- **Controls**: subreddit-matched never-disclosed users with ≥3 posts, held out from training.
- Metrics are aggregated at the **user level** (mean / max / top-5-mean of per-post scores) and reported in two modes: disclosure posts included vs masked.

Results aggregated into `experiments/disclosure_userlevel_summary.csv` and `docs/figures/disclosure_userlevel.png` by `scripts/report_disclosure_eval.py`.

---

## Experiment 8 — r/HealthAnxiety vs r/Anxiety head-to-head

`scripts/exp_ha_vs_anxiety.py`. Binary post-level classifier (HealthAnxiety=1, Anxiety=0) with an **author-disjoint** split (Harrigian et al. protocol — no author in both train and test). Baseline to beat: Low et al. (2020) SGD-L1 weighted-F1 = 0.851.

### Results

| Model | Weighted-F1 | AUROC | vs Low 2020 |
|---|---:|---:|---:|
| TF-IDF + LogReg | — | — | — |
| MentalRoBERTa (submissions) | **0.906** | **0.955** | +0.055 |

MentalRoBERTa on submissions beats the Low 2020 baseline by +5.5 weighted-F1 points and achieves AUROC 0.955, demonstrating that health-anxiety language is separable from general-anxiety language beyond the surface features available to a linear model.

---

## Experiment 9 — Domain-adversarial training (DANN): a negative result

`scripts/exp_dann_transfer.py` (figure: `scripts/plot_dann_transfer.py`). Tests whether a **Gradient-Reversal-Layer subreddit discriminator** (Ganin et al., 2016) on top of the multi-task encoder improves cross-subreddit generalisation. Implemented in `src/models/dann.py` (`DannMultiTaskModel`): shared MentalRoBERTa encoder → 4 sigmoid target heads (weighted BCE) **+** a subreddit-discriminator head behind a GRL; loss = `Σ task-BCE + domain-CE`, with the GRL reversing the domain gradient into the encoder. λ ramps 0→0.3 on the Ganin schedule. Two domain granularities are compared (ablation): `domain: subreddit` (27 fine classes) and `domain: group` (7 coarse `configs/subreddits.yaml` groups).

**Protocol.** All three models (plain multitask, DANN-subreddit, DANN-group) train on 60k posts from the in-distribution mental-health subreddits and are evaluated at a **single operating point** (per-target threshold tuned once on the in-distribution val set, then reused everywhere) on three held sets:
- **in_dist** — held-out authors from the training subreddits.
- **cross_heldout** — anxiety-bearing subreddits *held out of training* (r/PanicAttack, r/panicdisorder, r/agoraphobia; ~11k anxiety positives in the 20k sample) → measures positive transfer to unseen subreddits.
- **neutral** — baseline subreddits (cooking, personalfinance, …; ~0 anxiety positives) → `pred_pos_rate` is the false-positive rate; lower is better.

### Results (anxiety)

| Model | in-dist AUROC | cross AUROC | in-dist AUPRC | cross F1 | neutral FP rate |
|---|---:|---:|---:|---:|---:|
| **multitask (no DANN)** | **0.986** | 0.992 | **0.891** | 0.940 | **0.012** |
| **DANN (subreddit)** | 0.961 | 0.985 | 0.726 | **0.957** | 0.014 |
| **DANN (group)** | 0.706 | 0.799 | 0.198 | 0.665 | 0.181 |

![DANN transfer](figures/dann_transfer.png)

### Conclusion — DANN does not help here, and that is informative

1. **There is no collapse to fix.** The plain multi-task MentalRoBERTa *already* transfers near-perfectly to unseen anxiety subreddits (cross AUROC **0.992**) and false-fires on only **1.2%** of neutral posts. The cross-subreddit collapse documented for **TF-IDF** in Experiment 2 is a property of bag-of-words features keying on subreddit-specific vocabulary; the transformer's representation is already essentially subreddit-invariant for this task.
2. **Subreddit-DANN matches but does not beat the baseline** (cross F1 marginally higher, cross AUROC marginally lower) and pays a real **in-distribution cost** (AUPRC 0.891 → 0.726). No net benefit.
3. **Group-DANN actively breaks** — training collapsed (val F1 → 0 by epoch 2), in-dist AUROC fell to 0.71, and it flags **18%** of neutral posts as anxious. **Mechanism:** the coarse domain label (`anxiety_primary`, `depression_primary`, …) is **collinear with the target**, so forcing invariance to "which group" is equivalent to erasing the anxiety signal. Lowering λ to 0.3 did not rescue it; the coarse-domain objective fundamentally conflicts with the task. The fine-grained 27-subreddit domain is decorrelated enough from the binary label to stay stable.

**This validates the main approach:** the fine-tuned multi-task transformer is already domain-robust, so adversarial domain alignment is unnecessary (and, when the domain is collinear with the label, harmful). A rigorous test of a strong, well-motivated hypothesis returning a clean null is a methodological result in its own right.

---

## What we used (concrete inventory)

### Models
- `sklearn.linear_model.LogisticRegression` (C=1.0, balanced class weights, liblinear) on TF-IDF features (1-2 grams, min_df=5, max_df=0.95, sublinear TF, 80k max features).
- `xgboost.XGBClassifier` (400 trees, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=auto, hist tree method) on 26 hand-crafted linguistic features.
- `mental/mental-roberta-base` (Ji et al., 2022) fine-tuned with HuggingFace `Trainer`, 4 epochs, lr=2e-5, weight_decay=0.01, warmup_ratio=0.1, max_length=256, batch_size=16 — both single-target and multi-task variants.
- Multi-task variant: pure-PyTorch shared encoder + 4 sigmoid heads, BCE-with-logits loss with per-task weights `{anxiety: 1.0, health_anxiety: 1.5, depression: 1.0, suicidality: 1.2}` and per-row confidence weights from `label_<k>_weight` (disclosure=0.85 for positives, weak=0.4).

### Linguistic features (26 total)
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
- `sklearn.metrics.roc_curve / precision_recall_curve`

---

## Headline findings (for the thesis introduction)

1. ✅ **Anxiety is detectable.** MentalRoBERTa multi-task achieves F1 = 0.894 [0.87, 0.91], AUROC = 0.987, ECE = 0.039 — new SOTA on this corpus. TF-IDF + LogReg and XGBoost-linguistic both hit F1 ≈ 0.87, so the transformer's gain is small (≈2 F1 points) but well-calibrated and free of lexicon circularity.
2. ✅ **Multi-task does not hurt the dense class.** Multi-task anxiety F1 = 0.894 vs single-task 0.891 — and the paired test (n=2458) confirms **no significant difference** (McNemar p=0.83; bootstrap ΔAUROC −0.003, 95% CI [−0.006, +0.001]). Shared encoders + per-task loss weights add three extra targets at no measurable cost to anxiety. See [significance.md](significance.md).
3. ⚠ **XGBoost-linguistic rare-class F1 is circular.** F1 = 1.000 on health_anxiety and 0.792 on suicidality come from the model reading the same lexicons used to derive the labels. The multi-task transformer's F1 = 0.333 / 0.647 are the honest scores. **This is itself a methodological finding for the discussion chapter.**
4. ✅ **Cross-subreddit transfer is partially preserved** — AUROC stays at 0.97 (in fact slightly rises) but precision-at-threshold collapses, showing the model has learned a real signal that needs per-population threshold calibration.
5. ✅ **Linguistic markers replicate clinical theory** — first-person preponderance (Pennebaker), negative-sentiment-tracks-severity, reassurance-seeking specific to health anxiety (SHAI item content).
6. ⚠️ **Health-anxiety binary classification is bottlenecked by label sparsity** (24 positives in 16k → 3 in test set, multi-task F1 CI [0.00, 0.82]). The continuous severity score correctly identifies both COVID subreddits as the most health-anxious (~2× the mean of r/Anxiety), but supervised binary modeling would benefit from self-disclosure enrichment to increase positive-label coverage.
7. ⚠️ **Calibration is a problem for the TF-IDF model** (ECE = 0.13–0.20, *under*-confident) — **fixed post-hoc by temperature scaling** (T = 0.27 sharpens its squashed probabilities; ECE 0.200 → 0.035, −82%). XGBoost-linguistic (ECE 0.03) and MentalRoBERTa (ECE 0.005–0.038) are well calibrated without adjustment; forcing temperature scaling on the already-calibrated transformer heads gives little or no gain. See [calibration.md](calibration.md).
8. 🟢 **The 9-way subreddit classifier achieves macro-F1 = 0.64** (vs ~0.10 chance), confirming substantial linguistic distinctiveness — but also confirming the risk of subreddit-style leakage that motivates rigorous cross-subreddit transfer evaluation.

---

## Caveats — what these experiments don't show

- **Experiments 1–5 are graded against tier-1 weak labels only** (subreddit prior + lexicon). The model that scores F1 = 0.87 is partly distilling its training signal back into a logistic regression. The clean evaluation is the user-level self-disclosure test set (Experiment 7) and the author-disjoint head-to-head (Experiment 8).
- **Health-anxiety F1 = 1.00 is not a real result.** Twenty-four positives, of which the XGBoost model has direct lexicon-feature access, makes that finding circular. **Do not put this number in the thesis abstract.**
- **No bootstrap CIs in `scripts/run_experiments.py`** (passes `bootstrap=False` for speed). The transformer evaluation pipeline (`src/evaluation/runner.py:evaluate_model`) does compute them — see Experiment 6 — but Experiments 1–5 don't. Add `bootstrap=True` to `full_report` calls before the methodology chapter is finalized.
- **No statistical comparison between models** is reported. Use McNemar's test on the held-out predictions before claiming Model A "beats" Model B.
- **The 22-feature space might be missing variables** that would change the marker analysis. Future work: add LIWC-2022 categories, sentence-embedding clusters, syntactic complexity.

---

## How to reproduce

```bash
# 1. Make sure the corpus is built
anxiety collect --backend scraper        # or --backend praw
anxiety preprocess
anxiety label --tier weak
anxiety label --tier disclosure          # self-disclosure regex labels
anxiety label --tier aggregate           # merges disclosure + weak

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
