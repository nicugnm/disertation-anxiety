# Labeling

The 3-tier labeling system is the methodological centerpiece of the dissertation. It exists to address a single fundamental problem:

> Most prior Reddit mental-health work uses **subreddit membership as the label** ("posts in r/Anxiety are anxiety-positive"). This is wrong in two directions: r/Anxiety contains plenty of non-anxious posts (questions, news, posts about a friend), and *other* subreddits (r/relationship_advice, r/COVID19_support) contain plenty of anxious ones.

The 3-tier solution gives you cheap+noisy weak labels at scale, mid-cost LLM labels with reasonable quality, and a small but trustworthy human-labeled gold standard.

## Tier 1 — weak labels (cheap, noisy)

`src/labeling/weak.py`. For each post:

```
weak_score(label) = w_subreddit · subreddit_prior(label) + w_lex · lexicon_score(label, text)
```

- `subreddit_prior(label)` lives in `configs/subreddits.yaml` per subreddit per label. e.g. r/Anxiety has `expected_anxiety_prior: 0.85`.
- `lexicon_score(label, text)` counts hits from `src/labeling/lexicons.py` per 100 tokens, saturating at 1.0.
- Weights and thresholds live in `configs/labeling.yaml`.

The output is **probabilistic** (in [0, 1]) — downstream training can use it as a soft target or as a sample-weight, not just hard 0/1.

### What goes in the lexicons

Each list cites a clinical instrument or a psycholinguistics paper, so the methodology chapter can defend it line-by-line:

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

### Limits — and why we need tiers 2 and 3

The lexicon-only labeller is conservative. On the real corpus we collected:

```
weak_anxiety_pos:        3,560  (22%)
weak_depression_pos:     1,557  (10%)
weak_suicidality_pos:      116   (1%)
weak_health_anxiety_pos:    24   (0.15%)  ← still far too sparse to train on
```

The model trained on these weak labels gets F1 0.94 on r/Anxiety but **F1 = 0** on r/depression and r/SuicideWatch — because the weak labeller assigned 0 anxiety-positives there even though those posts are often co-morbidly anxious. That's the empirical motivation for tier 2.

## Tier 2 — LLM-assisted (mid-cost, reasonable quality)

`src/labeling/llm.py`. Sends posts to Claude with a prompt that operationalizes the codebook in `docs/codebook.md`. Returns:

```json
{
  "anxiety": 0|1, "anxiety_conf": 1..5,
  "health_anxiety": 0|1, "health_anxiety_conf": 1..5,
  "depression": 0|1, "depression_conf": 1..5,
  "suicidality": 0|1, "suicidality_conf": 1..5,
  "rationale": "..."
}
```

Properties:

- **Stratified sampling** across `subreddit_group` (anxiety_primary, health_anxiety_enriched, etc.) keeps cost predictable. See `tier2_llm.per_group_sample` in `configs/labeling.yaml`.
- **Cached on disk** in `.cache/llm_labels.sqlite`. Re-running the same labelling pass after a small data update is free for already-labelled rows.
- **Rate-limited** (`tier2_llm.rpm` defaults to 30 req/min). The cache hits skip the rate-limit sleep.
- **Free-text rationale** is preserved — useful when error-analysing later.

### How we know the LLM is doing a good job

Validate against tier 3: report the LLM's Cohen's κ with each human annotator. Targets in `configs/labeling.yaml` mirror the human-annotator targets:

```yaml
min_cohen_kappa:
  anxiety: 0.70
  health_anxiety: 0.60
  depression: 0.65
  suicidality: 0.75
```

If LLM-vs-human falls below these thresholds, we either tighten the prompt, switch model, or down-weight tier-2 in aggregation.

## Tier 3 — manual (gold standard)

`src/labeling/manual.py`. Minimal Rich-based TUI:

```bash
anxiety annotate --annotator-id alice
```

For each post the annotator sees:

1. A crisis-resource banner (so distress is met with help, not silence).
2. The cleaned post text.
3. A prompt for `{anxiety, health_anxiety, depression, suicidality}` 0/1.
4. A prompt for confidence 1–5.
5. An auto-check: if `health_anxiety=1` but `anxiety=0`, ask whether to auto-set `anxiety=1` (since by codebook health anxiety implies anxiety).

Annotation is **resumable** — the TUI persists every 10 rows. Two annotators can label the same posts in parallel by using different `--annotator-id` values; we then compute κ:

```bash
anxiety kappa alice bob
```

Targets:

| label | min κ |
|---|---:|
| anxiety | 0.70 |
| health_anxiety | 0.60 (harder; expected lower) |
| depression | 0.65 |
| suicidality | 0.75 (high stakes; high agreement expected) |

If we miss a threshold: refine the codebook → re-annotate. The codebook itself becomes a thesis contribution.

### Stratification

`tier3_manual.stratify_by` (default `subreddit_group`) ensures the 1000 gold posts include all corpus regions. Otherwise the test set would be 80% baseline-subs and barely able to validate the rare classes.

## Aggregation

`src/labeling/aggregate.py`. For each label and each row:

1. Take the value from the highest-precedence tier that has it (default order: `manual > llm > weak`).
2. Record that tier in `label_<k>_source`.
3. Record the corresponding confidence weight in `label_<k>_weight`.
4. Drop rows that have no label from any tier (configurable).

The weights flow into model training: `label_<k>_weight` is passed as a sample weight, so the model trusts manual labels (1.0) more than LLM labels (0.7) more than weak labels (0.4). This is critical when training on a corpus where most rows are weakly labelled and only a small subset has been hand-annotated.

## Walkthrough

```bash
# 1) Tier 1
anxiety label --tier weak

# 2) Tier 2 (needs ANTHROPIC_API_KEY)
anxiety label --tier llm

# 3) Tier 3 — first annotator
anxiety annotate --annotator-id alice

# 3') Tier 3 — second annotator (in parallel)
anxiety annotate --annotator-id bob

# 4) Inter-annotator agreement
anxiety kappa alice bob

# 5) Aggregate everything into final labels
anxiety label --tier aggregate
```

After step 5, `data/processed/labeled.parquet` has a `label_<k>` column per target, sourced from the best available tier per row, ready for model training.
