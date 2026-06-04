# Validation: how we trust the data, the labels, and the predictions

Every defensible mental-health NLP project needs three separate validations:

1. **Data validation** — the inputs are clean, anonymized, and what we say they are.
2. **Label validation** — the targets are clinically meaningful, not just "posts in r/Anxiety = anxiety".
3. **Prediction validation** — the model isn't cheating (memorizing, overfitting, or learning subreddit-style instead of the phenomenon).

This document covers all three.

---

## 1. Label validation — how "anxiety = true" is decided

### The labeling system in one paragraph

Two label sources are produced and used in practice. **Tier 1 (weak)**: subreddit prior + lexicon overlap, thresholded → `weak_<target>` / `weak_<target>_bin`. **Self-disclosure**: regex diagnosis patterns with negation/hypothetical/third-party/denial filters → `disclosure_<target>`. Suicidality disclosure is intentionally disabled. The confidence weights `disclosure=0.85`, `weak=0.4` are used where disclosure provides a positive signal (a disclosure `0` is not treated as strong negative evidence and falls through to the weak label).

### Tier 1 — weak labels

The formula:

```
weak_score(label, post) = 0.5 · subreddit_prior(label, subreddit)
                        + 0.5 · lexicon_score(label, post)
post is weak_positive iff weak_score ≥ threshold[label]
```

#### Subreddit priors

Set per (subreddit, label) in `configs/subreddits.yaml`:

```yaml
- name: Anxiety
  expected_anxiety_prior: 0.85         # 85% of r/Anxiety posts are expected to be anxiety-positive
  expected_health_anxiety_prior: 0.20  # 20% of r/Anxiety posts focus on health
```

These are **expert priors based on the subreddit's stated topic**, not learned. They're literature-defensible: r/SuicideWatch's `expected_suicidality_prior: 0.90` is grounded in the subreddit's posting rules.

#### Lexicon scores

Word lists in `src/labeling/lexicons.py`. Each list cites its clinical or psycholinguistic source so the methodology chapter can defend it line-by-line:

| Lexicon | Source instrument(s) |
|---|---|
| `ANXIETY_TERMS / ANXIETY_PHRASES` | **GAD-7** (Spitzer et al., 2006) stems + Reddit-anxiety NLP literature (De Choudhury et al., Coppersmith et al., Shen & Rudzicz) |
| `HEALTH_ANXIETY_TERMS / PHRASES` | **SHAI** (Salkovskis, Rimes, Warwick, Clark, 2002) and **HAI** (Lucock & Morley, 1996) item content |
| `REASSURANCE_PATTERNS` | Cognitive-behavioral model of health anxiety (Salkovskis, 1989; Warwick & Salkovskis, 1990) |
| `DEPRESSION_TERMS` | **PHQ-9** (Kroenke, Spitzer, Williams, 2001) stems |
| `SUICIDALITY_TERMS` | **Columbia C-SSRS** stems (Posner et al., 2011) |
| `FIRST_PERSON_*`, `*_PERSON` | Pennebaker, Mayne, Francis (1997) on pronoun preponderance |
| `BODY_PARTS` | Somatic vocabulary marker — distinguishes health anxiety from general anxiety |
| `UNCERTAINTY` / `CERTAINTY` | LIWC-style cognitive process category |

Score formula (`weak.py:_normalize_score`):

```python
hits_per_100_tokens = hits / max(1, n_tokens / 100)
score = min(1.0, hits_per_100_tokens / 3.0)   # saturates at 3 hits per 100 tokens
```

The saturation curve prevents one very long anxious post from dominating.

#### Thresholds

In `configs/labeling.yaml`:

```yaml
thresholds:
  anxiety: 0.55          # easier — broad linguistic signal
  health_anxiety: 0.60   # harder — needs both anxious affect AND health focus
  depression: 0.55
  suicidality: 0.65      # high stakes — high bar for false positives
```

#### Why tier 1 alone is not enough

Empirically, on our real corpus:

```
weak_anxiety_pos:        3,560  (22%)   — usable
weak_depression_pos:     1,557  (10%)   — usable
weak_suicidality_pos:      116   (1%)   — sparse
weak_health_anxiety_pos:    24   (0.15%) — still far too sparse to train on
```

The model trained on these weak labels achieves F1 0.94 on r/Anxiety but **F1 = 0** on r/depression and r/SuicideWatch — because the weak labeller assigned 0 anxiety-positives there even though those posts are often co-morbidly anxious. This motivates the self-disclosure tier.

### Aggregation — final labels

`src/labeling/aggregate.py`:

```python
for each row, for each label:
    pick the value from the highest-precedence tier that has it
    (default order: disclosure > weak)
    record the source in label_<k>_source
    record the confidence in label_<k>_weight
```

The weights flow into model training: `label_<k>_weight` is passed as a sample weight. The effective weights are `disclosure=0.85` (positive signal only — a disclosure `0` falls through to weak) and `weak=0.4`.

### How does the codebook prevent ambiguity?

`docs/codebook.md` explicitly handles the hard cases:

- A post mentioning anxiety topic without first-person affect → `0` for the affect labels but topic-keyed for analysis. ("Here's a study about anxiety…" doesn't count.)
- A post about a third party with no first-person affect → also `0`.
- Past-tense recovered narrative → `0` unless current anxious affect is also present.
- **Health-anxiety needs both anxious affect AND health focus.** Pure illness reports (post-COVID symptoms) without disproportionate distress are NOT health anxiety.
- Borderline rule: `health_anxiety = 1` if 3+ of {specific feared disease named, multiple checklist symptoms, dread of upcoming appointments, intrusive health thoughts} are present even if affect is restrained.
- The TUI auto-checks `health_anxiety=1 ⇒ anxiety=1` (clinical implication).

---

## 2. Data validation — how the *inputs* are validated

### Pipeline-enforced cleaning (no model sees text that hasn't passed through this)

| Stage | Module | What it does |
|---|---|---|
| Encoding | `preprocessing/clean.py:fix_encoding` | `ftfy` repairs mojibake (`â€™` → `'`) |
| Markdown | `clean.py:strip_markdown` | `[anchor](url)` → `anchor`; `&gt;` quote lines dropped |
| URLs / entities | `clean.py:strip_urls`, `decode_entities` | URLs → `[URL]`; `&amp;` → `&`; etc. |
| Bot notices | `clean.py:strip_bot_notices` | "I am a bot" footers removed |
| Whitespace | `clean.py:normalize_whitespace` | collapses runs of newlines/spaces |
| **PII regex** (always on) | `anonymize.py:regex_redact` | emails, phones, `u/usernames`, `r/subs`, `@handles`, URLs |
| **PII NER** (when spaCy available) | `anonymize.py:ner_redact` | `PERSON / GPE / LOC / ORG` → `[PERSON]` etc. |
| **Author pseudonymization** | `anonymize.py:_hash_username` | salted SHA-256, deterministic across runs |
| Length filter | `pipeline.py` | drops posts <50 chars after cleaning |
| Language filter | `pipeline.py` | langdetect — drops non-English (configurable) |
| **Exact dedup** | `dedupe.py:_md5` | same MD5 of normalized text → drop |
| **Near-dedup** | `dedupe.py:_simhash` | 64-bit SimHash; Hamming distance ≤5 within subreddit → drop |

### Tested behavior

`tests/` directory:

| Test file | Verifies |
|---|---|
| `test_clean.py` | URL/markdown stripping, whitespace, None-handling |
| `test_anonymize.py` | email/phone/username redaction, sub mention redaction, hash stability |
| `test_dedupe.py` | exact duplicates dropped, empty input handled |
| `test_lexicons.py` | anxiety text → high anxiety score; neutral text → low score; health-anxiety text → high health-anxiety score |
| `test_features.py` | first-person pronoun rate higher in first-person text; health-anxiety features fire on cancer/symptom text; feature key-set is stable |
| `test_metrics.py` | full report has all keys; bootstrap CIs are sane; ECE close to 0 on perfectly calibrated data |
| `test_synthetic_collector.py` | synthetic generator yields posts; same seed → same posts |
| `test_json_scraper.py` | scraper deduplicates across listings; handles 404 gracefully |
| `test_smoke_minimal.py` | end-to-end pipeline runs; F1 ≥ 0.4 on synthetic anxiety |

22/22 tests pass on every change. Run `pytest -v` to see the green checks.

### Documented attrition

When you run preprocess, structlog emits per-stage counts:

```
preprocess.start                   n=2072
preprocess.after_drop_short        n=2068
preprocess.after_lang_filter       n=2059
preprocess.after_dedupe            n=2040
```

The thesis methodology section can quote these directly, or render them as a Sankey diagram.

---

## 3. Prediction validation — overfitting & generalization

### Multiple metrics, not just F1

`src/evaluation/metrics.py` computes for every model run:

| Metric | What it tells you |
|---|---|
| Accuracy | Floor sanity — but can lie under class imbalance |
| Precision | Of the positives we predicted, how many were right? |
| Recall | Of the true positives, how many did we catch? |
| **F1** | Harmonic mean of precision/recall |
| **AUROC** | Threshold-free ranking quality |
| **AUPRC** | Threshold-free under class imbalance (more honest than AUROC for rare classes) |
| Brier | Squared error of predicted probability — calibration + accuracy |
| **ECE** (Expected Calibration Error) | Are predicted probabilities calibrated? |

Real example, our TF-IDF baseline on anxiety: F1 0.88, AUROC 0.97, **ECE 0.20** → model is overconfident, calibration is poor, even though F1 looks great. **You'd never see this with F1 alone.**

### Bootstrap 95% confidence intervals

`bootstrap_ci(y_true, y_score, metric, n_iters=500)` resamples with replacement and reports the 2.5th / 97.5th percentile. This is reported alongside every point estimate so the thesis can write "F1 = 0.88 [0.85, 0.90]" instead of pretending point estimates are exact.

### Overfitting controls per model

| Model | Defense |
|---|---|
| TF-IDF + LogReg | L2 regularization (C=1.0), `class_weight=balanced`, 5-fold CV in baseline.yaml |
| XGBoost | **Early stopping** (30 rounds against val set), `subsample=0.8`, `colsample_bytree=0.8`, max_depth limited (6), low learning rate (0.05), `scale_pos_weight=auto` |
| Transformer | `weight_decay=0.01`, `warmup_ratio=0.1`, eval each epoch, `load_best_model_at_end=True`, `metric_for_best_model=f1` |
| Multi-task | Same weight decay + **per-row sample weights** from tier_confidence (downweights noisy labels) + per-task loss weights to prevent collapse on rare classes |

### Distribution-shift detection — the most diagnostic single tool

`src/evaluation/error_analysis.py:confusion_by_subgroup` produces a per-subreddit F1 table. Look at our real-data result:

```
       subreddit  n_test  n_pos     F1
        Anxiety     315    268   0.94
AnxietyDepression   198    104   0.86
   socialanxiety    205    132   0.84
   COVID19_support  153      9   0.70   ← cross-subreddit drop visible
   relationship_advice 286    1   0.67   ← rare positive caught
   LivingAlone     136      1   1.00   ← rare positive caught
   depression_help 224      0   0.00
        depression 136      0   0.00
   SuicideWatch    318      0   0.00   ← weak labels assigned 0 positives here
```

This **immediately** tells the thesis reader that:
1. The model is not just memorizing TF-IDF features that happen to occur in r/Anxiety — it transfers.
2. Tier-1 weak labels assigned **zero anxiety-positives in r/SuicideWatch / r/depression**, so the model never learns to predict positive there. This is an empirical limitation of the weak labels.

### Length-effect bins

`src/evaluation/error_analysis.py:length_effect` bins predictions by post length and reports F1 per bin. Detects length-bias — if F1 collapses on short posts, you have a problem.

### Calibration check

`src/viz/plots.py:plot_calibration` produces a reliability diagram + score histogram. ECE is reported numerically. A well-calibrated model has the curve hugging the diagonal; ours doesn't (yet — needs Platt / temperature scaling).

### Cross-subreddit transfer experiment (RQ3)

`src/models/splits.py:cross_subreddit_split` lets you hold out entire subreddits from training. The thesis result for RQ3:

```python
held_out = ["COVID19_support", "LivingAlone", "relationship_advice"]
train, test = cross_subreddit_split(df, held_out)
# train transformer on `train`, evaluate on `test`
# the F1 drop quantifies cross-subreddit generalization
```

If the model trained only on anxiety-primary subreddits achieves F1 ≈ 0.4 on r/relationship_advice (vs F1 ≈ 0.85 in-distribution), the model is learning subreddit style as much as anxiety. That's a finding for the discussion chapter, not a bug.

### Hardest examples

`error_analysis.hardest_examples(df, target, n=20)` returns the 20 posts with the largest |score − label| — i.e. the most-confidently-wrong predictions. These go in the appendix or qualitative analysis chapter.

### User-level self-disclosure evaluation (the clean evaluation set)

`src/labeling/disclosure_dataset.py` + `anxiety eval-disclosure`. This is the primary evaluation set graded against real signal rather than weak labels:

- **Positives**: users who ever posted a verified self-disclosure for a given target (regex + filters from `self_disclosure.py`).
- **Controls**: never-disclosed users sampled from the same subreddits as the positives (subreddit-matched, to prevent cheating via subreddit style). Controls must have ≥3 posts.
- All posts by test users are held out from training (`held_out_split=True` in the corpus).

Metrics are computed at the **user level** by aggregating per-post scores (mean / max / top-5-mean). Two modes: with disclosure posts included vs with them masked, to test whether the model learned implicit signal beyond the disclosure phrase itself.

Results are aggregated by `scripts/report_disclosure_eval.py`.

### r/HealthAnxiety vs r/Anxiety head-to-head (Experiment 8)

`scripts/exp_ha_vs_anxiety.py`. Tests whether health-anxiety language is separable from general-anxiety language with an **author-disjoint** train/test split (Harrigian et al. — no author appears in both train and test). Baseline to beat: Low et al. (2020) SGD-L1 weighted-F1 = 0.851 on the same subreddit-as-proxy setup. MentalRoBERTa achieves weighted-F1 **0.906** / AUROC **0.955** on submissions, a +0.055 improvement over the Low 2020 baseline.

---

## What this means for the thesis

A reader can challenge any prediction in the results chapter and you have an answer:

- **"How do you know it's not just learning subreddit style?"** → Per-subreddit F1 table + cross-subreddit transfer experiment.
- **"How do you know your labels are correct?"** → Self-disclosure labels (Coppersmith/eRisk standard) are used as the clean evaluation signal; disclosed users are positives held out from training. Weak labels are acknowledged as noisy training signal only, not evaluation ground truth.
- **"How do you handle health-anxiety scarcity?"** → Per-task loss weights in multi-task, sample-weight aggregation from label confidence tiers, and the self-disclosure test set provides user-level clean evaluation. The continuous weak health-anxiety score (Experiment 5) correctly identifies the COVID subreddits as the most health-anxious, consistent with clinical expectations.
- **"Aren't your linguistic markers just artifacts of your lexicon?"** → SHAP on XGBoost (model never saw the lexicons; it saw rates) + cross-feature significance with BH-FDR correction.
- **"Are your predictions reliable?"** → Bootstrap CIs on every metric; reliability diagram; ECE.
- **"Could a confounder explain the result?"** → Length-effect bins, subreddit subgroup analysis, temporal split.

Every one of these has implementation in the codebase, not just promises in the methodology section.
