# Thesis outline

A chapter-by-chapter map from the codebase to the dissertation. Each chapter cites the modules and artifacts it draws from, so writing becomes a matter of running the pipeline and reading the outputs.

## 0. Front matter

- Abstract (300 words). Centerpiece claim: *"We construct a Reddit corpus of mental-health posts, build a multi-task transformer that distinguishes health anxiety from general anxiety, and identify linguistic markers that align with clinical instruments (SHAI/HAI)."*
- Acknowledgements
- Statement of original work
- Statement of ethics review (link to `docs/ethics.md`)

## 1. Introduction

### 1.1 Motivation

Health anxiety is clinically under-recognized: sufferers seek reassurance through medical channels rather than mental-health ones, leading to repeated negative work-ups and persistent distress. Online forums host their unguarded language. NLP can help characterize this population at scale.

### 1.2 Research questions

(carry over from `README.md`):

- **RQ1.** Can transformer-based language models reliably distinguish *health anxiety* from *general anxiety* and from non-anxious mental-health discourse on Reddit?
- **RQ2.** Which linguistic markers most strongly characterize health anxiety, and do they replicate clinical findings?
- **RQ3.** How well do models trained on dedicated anxiety subreddits generalize to non-anxiety subreddits (cross-subreddit transfer)?
- **RQ4.** What did COVID-19 do to health-anxiety language on Reddit?

### 1.3 Contributions

1. A reproducible pipeline for Reddit mental-health NLP (`src/`).
2. A two-tier labeling scheme combining Tier-1 weak labels (subreddit prior + lexicon) and Tier-2 self-disclosure labels (regex diagnosis phrases with negation/hypothetical/third-party/denial filters) (`src/labeling/`).
3. A clean user-level evaluation protocol based on self-disclosed diagnoses: disclosed users as positives, subreddit-matched non-disclosed users as controls, held out from training (`src/labeling/disclosure_dataset.py`).
4. A multi-task MentalRoBERTa transformer outperforming single-task baselines on health anxiety (Exp 8: weighted-F1 0.906, AUROC 0.955 on submissions; beats Low 2020's 0.851 baseline).
5. A user-level disclosure evaluation (Exp 7) showing model performance on the held-out self-disclosure test set.
6. A linguistic analysis identifying somatic vocabulary, reassurance-seeking, and uncertainty as signature health-anxiety markers — replicating SHAI item content.
7. A cross-subreddit generalization study showing where the model fails.
8. (Optional) A COVID temporal analysis.

### 1.4 Scope and ethics statement

Public Reddit data, no diagnosis, no surveillance. Detailed in `docs/ethics.md`.

---

## 2. Background and related work

### 2.1 Clinical foundations

- DSM-5 Illness Anxiety Disorder; SHAI (Salkovskis, Rimes, Warwick, Clark 2002); HAI (Lucock & Morley 1996).
- GAD-7 (Spitzer et al. 2006); PHQ-9 (Kroenke et al. 2001); Columbia C-SSRS.
- Cognitive-behavioral model of health anxiety (Salkovskis 1989; Warwick & Salkovskis 1990).

### 2.2 NLP for mental health on social media

- De Choudhury et al. (2013) — Twitter depression detection.
- Coppersmith et al. (2014, 2015) — CLPsych.
- Yates, Cohan, Goharian (2017) — depression Reddit corpus (RSDD).
- Shen & Rudzicz (2017) — anxiety on Reddit.
- Ji et al. (2022) — MentalBERT, MentalRoBERTa.
- Harrigian et al. (2020) — model transfer & population bias.

### 2.3 Linguistic correlates of anxiety/depression

- Pennebaker, Mayne, Francis (1997) — pronoun preponderance.
- Eichstaedt et al. (2018) — Facebook depression markers.
- Coppersmith, Dredze, Harman (2014) — quantifying language.

### 2.4 Gaps this thesis addresses

- Health anxiety is rarely treated as its own class; usually subsumed under "anxiety" or absent entirely.
- Most prior work uses subreddit membership as the sole label, with no inter-annotator validation.
- Cross-subreddit transfer is rarely evaluated, masking topic-style confounds.

---

## 3. Data and ethics

### 3.1 Sources

`configs/subreddits.yaml`. Discuss why each subreddit is included and what role it plays (positive class, comorbidity, baseline, COVID enrichment).

### 3.2 Collection

`src/collection/`:
- PRAW for recent data.
- Pushshift `.zst` dumps (via `arctic_shift`) for historical depth — the only practical post-2023 option.
- Synthetic generator (`SyntheticCollector`) for reproducibility tests.

Document the *exact* collection date and parameters; release post IDs only.

### 3.3 Ethics, IRB, anonymization

Reproduce `docs/ethics.md`. Key points: no raw data redistribution; pseudonymized authors; spaCy NER + regex for PII; r/SuicideWatch handled as training-only; crisis-resource banner in any deployed artifact.

### 3.4 Preprocessing

`src/preprocessing/`: cleaning, anonymization, deduplication. Report attrition counts at every stage in a Sankey diagram — important for the reproducibility section.

### 3.5 Labeling

`src/labeling/` — the methodological centerpiece. Two tiers were executed in this experiment.

- **Tier 1 — weak** (`weak.py`): subreddit prior + lexicon overlap → `weak_<target>` (probabilistic score) and `weak_<target>_bin` (thresholded). Cite each lexicon's clinical provenance.
- **Tier 2 — self-disclosure** (`self_disclosure.py`): regex diagnosis phrases with four false-positive filters (negation, hypothetical, third-party, denial) → `disclosure_<target>` (0/1) and `disclosure_<target>_match` (matched substring for traceability). Suicidality patterns are intentionally empty. Methodology follows Coppersmith et al. (2014) and CLEF eRisk (Losada et al. 2017–).
- **Aggregation** (`aggregate.py`): `label_<target>` / `label_<target>_source` / `label_<target>_weight` with precedence disclosure > weak. `label_<target>_source` is always `'disclosure'` or `'weak'` in the corpus.

### 3.5b Disclosure test set

`src/labeling/disclosure_dataset.py`. Users with at least one verified self-disclosure are positives; subreddit-matched never-disclosed users are controls (2:1 ratio). All posts by test users are marked `held_out_split=True` and excluded from training. This is the primary clean evaluation set (Experiment 7). See `docs/disclosure_eval.md`.

### 3.6 Final corpus statistics

Generated tables: counts per subreddit × label × tier; class balance; year distribution; length distribution. Auto-produced from `data/processed/labeled.parquet`.

---

## 4. Methodology

### 4.1 Problem formulation

Multi-label binary classification: `y ∈ {0,1}^4` for {anxiety, health_anxiety, depression, suicidality}. Note the implication: `health_anxiety=1 ⇒ anxiety=1`.

### 4.2 Models

`src/models/`:

| Model | Purpose |
|---|---|
| TF-IDF + Logistic Regression | Baseline floor |
| XGBoost on linguistic features | Interpretable bridge to RQ2 |
| RoBERTa-base fine-tuned | Modern baseline |
| MentalRoBERTa fine-tuned | Domain-specific best baseline |
| Multi-task MentalRoBERTa | Novelty contribution (joint heads) |

### 4.3 Training procedure

Stratified 70/15/15 split. Per-row, per-tier confidence weights flow into the loss (`label_<k>_weight`). Optimizer / LR / batch sizes in `configs/models/*.yaml`.

### 4.4 Evaluation

`src/evaluation/`:
- Metrics: precision, recall, F1, AUROC, AUPRC, Brier, ECE.
- Bootstrap 95% CIs (`bootstrap_ci`).
- Calibration analysis with reliability diagrams.
- Subgroup F1 by subreddit.
- Hardest-example error analysis.

### 4.5 Cross-subreddit transfer (RQ3)

Train on {anxiety, social anxiety, anxiety_depression, depression, depression_help}; test on held-out {COVID19_*, LivingAlone, relationship_advice}. Report performance drop.

### 4.6 Linguistic-marker analysis (RQ2)

`src/analysis/linguistic_markers.py`. Cohen's d + Mann-Whitney U + BH-corrected p-values for every feature against `label_health_anxiety`.

### 4.7 Explainability (RQ2)

SHAP on XGBoost; gradient × input attributions on the transformer (`src/analysis/explainability.py`).

### 4.8 Temporal analysis (RQ4)

`src/analysis/temporal.py`. Pre-COVID / COVID-peak / post-peak windows.

---

## 5. Results

For each RQ:

### 5.1 RQ1: Detection performance

Headline table: model × target × {F1, AUROC, AUPRC, ECE} with 95% CIs. Discussion of where MentalRoBERTa beats RoBERTa beats baselines, and where multi-task helps health anxiety but not other targets.

**Experiment 7 (user-level disclosure eval):** Per-model performance on the self-disclosure test set (positives = users who disclosed; controls = subreddit-matched non-disclosers). Report with and without masking the disclosure posts themselves (tests implicit signal vs. explicit regex cues).

**Experiment 8 (HA-vs-Anxiety head-to-head):** MentalRoBERTa multi-task weighted-F1 0.906 / AUROC 0.955 on submissions, compared with Low et al. (2020) baseline of 0.851 weighted-F1.

### 5.2 RQ2: Linguistic markers

Top-20 markers by Cohen's d. Replication of SHAI item content (e.g., reassurance-seeking, body monitoring). SHAP top-features alignment.

### 5.3 RQ3: Cross-subreddit transfer

F1 drop in held-out subreddits. Error analysis: what kind of post does the model fail on?

### 5.4 RQ4: COVID temporal effects (if pursued)

Health-anxiety rate by quarter; differential effect across subreddits.

---

## 6. Discussion

- What the model learned about health anxiety vs. general anxiety.
- Comparison with clinical-instrument item content.
- Where the model under-performs (subgroups, length, ambiguous cases).
- Threats to validity: weak-label noise; population bias; English-only; surveillance-bias self-selection.
- Practical implications: NOT a clinical tool; possibly useful for forum-moderation triage, with informed consent.

---

## 7. Limitations and future work

- Multilingual extension.
- Clinician validation against SHAI scores from consenting users.
- Longitudinal modeling (per-user trajectories rather than per-post).
- Generative models for explanation (post → health-anxiety summary for clinicians).

---

## 8. Conclusion

Restate contributions. Restate the disclaimer. Point to released artifacts (post IDs, labels, code).

---

## Appendices

- A. Codebook (`docs/codebook.md`).
- B. Ethics statement (`docs/ethics.md`).
- C. Hyperparameters (auto-generated from configs).
- D. Full per-subreddit results.
- E. Software environment (`pyproject.toml`, lockfile).
- F. Reproducibility note: how to re-derive the corpus from post IDs + your own Reddit fetch.
