# Labeling

The labeling pipeline addresses a single fundamental problem:

> Most prior Reddit mental-health work uses **subreddit membership as the label** ("posts in r/Anxiety are anxiety-positive"). This is wrong in two directions: r/Anxiety contains plenty of non-anxious posts (questions, news, posts about a friend), and *other* subreddits (r/relationship_advice, r/COVID19_support) contain plenty of anxious ones.

Two complementary label sources are produced and used in this work: **weak labels** (cheap, noisy, corpus-wide) and **self-disclosure labels** (high-confidence, sparse). These are then aggregated into final per-row labels.

## Source 1 — Weak labels (cheap, noisy)

`src/labeling/weak.py`. For each post:

```
weak_score(label) = w_subreddit · subreddit_prior(label) + w_lex · lexicon_score(label, text)
```

- `subreddit_prior(label)` lives in `configs/subreddits.yaml` per subreddit per label. e.g. r/Anxiety has `expected_anxiety_prior: 0.85`.
- `lexicon_score(label, text)` counts hits from `src/labeling/lexicons.py` per 100 tokens, saturating at 1.0.
- Weights and thresholds live in `configs/labeling.yaml`.

The output columns are `weak_<target>` (probabilistic score in [0, 1]) and `weak_<target>_bin` (thresholded 0/1). Downstream training can use the continuous score as a soft target or as a sample weight.

### What goes in the lexicons

Each list cites a clinical instrument or a psycholinguistics paper:

| Lexicon | Source instrument(s) |
|---|---|
| `ANXIETY_TERMS` / `ANXIETY_PHRASES` | GAD-7 stems + Reddit-anxiety NLP literature (Coppersmith, Shen & Rudzicz) |
| `HEALTH_ANXIETY_TERMS` / `HEALTH_ANXIETY_PHRASES` | SHAI / HAI items + somatic-symptom literature |
| `REASSURANCE_PATTERNS` | Cognitive-behavioral health-anxiety models (Salkovskis 1989) |
| `DEPRESSION_TERMS` | PHQ-9 stems |
| `SUICIDALITY_TERMS` | Columbia C-SSRS stems |
| `FIRST_PERSON_*` / `*_PERSON` | Pennebaker et al. on pronoun preponderance |
| `UNCERTAINTY_TERMS` / `CERTAINTY_TERMS` | LIWC-style |
| `BODY_PARTS` | Somatic vocabulary marker for health anxiety |

## Source 2 — Self-disclosure labels (high-confidence, sparse)

`src/labeling/self_disclosure.py`. Implements the Coppersmith / eRisk self-disclosure protocol:

1. **Positive patterns** — regex templates anchored to first-person clinical language ("I was diagnosed with X", "I have GAD", "I suffer from generalized anxiety disorder", etc.).
2. **False-positive filters** — a candidate match is rejected if any of the following appear within ±50 characters of the match:
   - negation ("not", "never", "n't")
   - hypothetical ("if I were diagnosed", "I think I might have")
   - third-party report ("my husband was diagnosed")
   - denial / rhetorical ("turns out I don't have", "lol … depressed")

The output columns are `disclosure_<target>` (0/1) and `disclosure_<target>_match` (the matched substring, for traceability).

**Suicidality disclosure is disabled** — `SUICIDALITY_DX_PATTERNS` is empty. Suicidal ideation is not typically self-disclosed as a clinical diagnosis and carries high false-positive risk; the target falls through to the weak label only.

This is a **high-confidence proxy** (tier confidence 0.85 by default), following the established eRisk / Coppersmith methodology for clinical NLP in the absence of specialist annotation.

## Aggregation

`src/labeling/aggregate.py`. For each label and each row:

1. Take the value from the highest-precedence source that has it (disclosure before weak).
2. Record that source in `label_<target>_source`.
3. Record the corresponding confidence weight in `label_<target>_weight`.
4. Drop rows that have no label from any source (configurable).

The effective rule is:

- If `disclosure_<target> = 1` → `label_<target> = 1` (source: `disclosure`, weight: 0.85).
- Otherwise → `label_<target> = weak_<target>_bin` (source: `weak`, weight: 0.4).

Note: a `disclosure = 0` is treated asymmetrically — it means only that the regex did not fire, not that the user is definitely non-anxious. The code therefore falls through to the weak label for `disclosure = 0` rows rather than propagating a hard negative.

The `label_<target>_weight` column flows into model training as a per-sample weight.

## Evaluation set

The clean held-out evaluation set is the **self-disclosure test set** (`src/labeling/disclosure_dataset.py`):

- **Positives**: users with at least one verified self-disclosure post.
- **Controls**: subreddit-matched users with no disclosure, randomly sampled.
- All users in the test set are excluded from training via `held_out_split`.

This follows the eRisk user-level evaluation protocol.

## Walkthrough

```bash
# 1) Weak labels (corpus-wide)
anxiety label --tier weak

# 2) Self-disclosure labels (corpus-wide)
# (run automatically as part of the aggregate step, or standalone)

# 3) Aggregate into final labels
anxiety label --tier aggregate
```

After aggregation, `data/processed/labeled.parquet` has a `label_<target>` column per target, sourced from the best available tier per row, ready for model training.
