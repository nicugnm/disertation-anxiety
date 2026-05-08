# Architecture

A module-by-module deep dive. The repo is built so that any single stage can be re-run in isolation against a stable parquet schema.

## Design principles

1. **Configs drive behavior.** YAML in `configs/` controls every parameter that has a research justification. No code change should ever be needed to add a subreddit or a model.
2. **Common interfaces.** `BaseCollector` and `BaseModel` give every backend / model a uniform contract. New backends and models become drop-ins.
3. **Stable schema.** Every stage reads parquet, writes parquet, and never mutates upstream stages.
4. **Reproducibility by default.** All randomness is seeded; LLM calls are cached on disk; collection is deterministic given a config.
5. **Ethics at the layer that enforces it.** Anonymization is a module, not a checklist — `src/preprocessing/anonymize.py` is called by the pipeline so no model can ever see un-anonymized text.

## Data flow

```
                         ┌─────────────────────┐
                         │ configs/*.yaml      │
                         │ (Pydantic-validated)│
                         └──────────┬──────────┘
                                    │
        ┌─────────────────┬─────────┴────────┬────────────────┐
        ▼                 ▼                  ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ collection/  │→ │preprocessing/│→ │  labeling/   │→ │  features/   │
│              │  │              │  │              │  │              │
│ produces:    │  │ produces:    │  │ produces:    │  │ produces:    │
│ data/raw/    │  │ data/interim/│  │ data/processed│  │ feature cols │
│   *.parquet  │  │   *.parquet  │  │   labeled.pq │  │   in-memory  │
└──────────────┘  └──────────────┘  └──────────────┘  └──────┬───────┘
                                                              │
                  ┌──────────────────┬───────────────────┬────┘
                  ▼                  ▼                   ▼
          ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
          │   models/    │   │ evaluation/  │   │   analysis/  │
          │              │   │              │   │              │
          │ produces:    │   │ produces:    │   │ produces:    │
          │ runs/<name>/ │   │ eval/*.json  │   │ markers__*.csv│
          │ model + splits│  │ predictions   │  │              │
          └──────────────┘   └──────────────┘   └──────┬───────┘
                                                       │
                                                       ▼
                                                ┌──────────────┐
                                                │     viz/     │
                                                │              │
                                                │ produces:    │
                                                │ docs/figures/│
                                                └──────────────┘
```

## Module rundown

### `src/utils/`

Foundation. Touched by every other module.

- **`config.py`** — Pydantic models for every YAML config (subreddits, labeling, model). Strong typing means a typo in a config file errors at load time, not three steps into a run.
- **`io.py`** — parquet, JSONL, and `.zst`-compressed JSONL streaming. The `.zst` reader is what makes the dump backend feasible: it streams without loading the whole file into memory.
- **`logging.py`** — structlog with optional JSON output (good for batch jobs and MLflow capture).
- **`cache.py`** — SQLite-backed key-value cache. Keys are SHA-256 of (model, prompt, post) so re-running an LLM labeling pass is free.

### `src/collection/`

Four pluggable backends behind one interface.

| Backend | Auth required? | Best for |
|---|---|---|
| `JsonScraperCollector` | **No** | Default. Hits Reddit's public JSON endpoints (`old.reddit.com/r/<sub>/<listing>.json`). |
| `PrawCollector` | OAuth | Higher rate limits when you have credentials. |
| `DumpCollector` | No (just a file) | Years of historical data via Pushshift `.zst` archives. |
| `SyntheticCollector` | No | Reproducible synthetic data for tests/CI. |

Each implements `collect_subreddit(name) -> Iterator[RedditPost]`. The `RedditPost` dataclass is the canonical schema:

```python
@dataclass
class RedditPost:
    id: str
    subreddit: str
    created_utc: float
    title: str
    body: str
    author: str | None
    score: int
    num_comments: int
    permalink: str
    is_self: bool
    over_18: bool
    source: str          # 'praw' | 'json_scraper' | 'dump' | 'synthetic'
    collected_at: float
```

`runner.py` wires the chosen backend, iterates configured subreddits, and writes one parquet file per subreddit under `data/raw/`.

### `src/preprocessing/`

Transforms raw → interim. Composable functions that the `pipeline.py` runner chains.

- **`clean.py`** — encoding fixes (`ftfy`), markdown stripping, URL/HTML-entity normalization, whitespace, Reddit bot-notice removal.
- **`anonymize.py`** — two-layer:
  - regex layer (always on): emails, phones, Reddit usernames, sub mentions, URLs, @-handles
  - spaCy NER layer (when available): `PERSON`, `GPE`, `LOC`, `ORG` → placeholder tokens
  - usernames are SHA-256 + salt, deterministic across runs
- **`dedupe.py`** — exact dedup by MD5 + SimHash near-duplicate dedup within each subreddit (Hamming ≤ 5 on a 64-bit hash).
- **`pipeline.py`** — orchestrates all of the above, applies length and language filters, writes an `_all.parquet` shard for downstream stages.

### `src/labeling/`

The methodological centerpiece. Three tiers + an aggregator.

- **`lexicons.py`** — small, transparent word lists derived from clinical instruments (GAD-7, SHAI, PHQ-9, C-SSRS) and the social-media mental-health literature (Pennebaker, Coppersmith, De Choudhury). The thesis cites every list's provenance.
- **`weak.py`** — tier-1: combines per-subreddit prior with lexicon overlap. Outputs *probabilistic* weak labels, not hard 0/1, so downstream training can use them as soft targets or with confidence weighting.
- **`llm.py`** — tier-2: prompts Claude with the codebook from `docs/codebook.md`. Caches every response on disk. Stratified sampling across `subreddit_group` keeps the labelling cost predictable.
- **`manual.py`** — tier-3: minimal Rich TUI for human annotation. Resumable; supports two-annotator setups; computes Cohen's κ.
- **`aggregate.py`** — combines the tiers with precedence `manual > llm > weak`. Per-row `label_<k>_weight` is propagated to the loss for confidence-weighted training.

### `src/features/`

`linguistic.py` extracts the feature columns used by the XGBoost model and the linguistic-analysis chapter:

| Feature group | Examples |
|---|---|
| Lexical rates | anxiety / health-anxiety / depression / suicidality term & phrase counts; reassurance, body-part rates |
| Pronouns | first-singular / first-plural / second / third rates |
| Certainty | uncertainty rate, certainty rate, question-mark rate |
| Length | chars, tokens, sentences, avg-sentence-length, avg-word-length |
| Readability | Flesch reading ease, Gunning fog |
| Sentiment | VADER (compound, pos, neg, neu) |

All feature columns are prefixed `f_` so the model can fetch them with `feature_columns(df)`.

### `src/models/`

Six concrete models, all subclassing `BaseModel`:

| Class | Type | Notes |
|---|---|---|
| `TfidfLogRegModel` | sklearn | Baseline floor |
| `XgboostLinguisticModel` | XGBoost | Trains on `f_*` features only |
| `TransformerModel` | HuggingFace | Single binary head; tries MentalBERT first, falls back to RoBERTa |
| `MultiTaskTransformer` | PyTorch | Shared encoder, sigmoid head per target, BCE-with-logits, per-task loss weights |
| `LlmZeroShotModel` | Anthropic API | No training; runs prompts at predict time; cached |

`splits.py` provides stratified train/val/test and `cross_subreddit_split` for the RQ3 transfer experiment.

`registry.py` is the factory: `build_model(config) -> BaseModel`. Adding a new model = subclass + register here + new YAML.

### `src/evaluation/`

- **`metrics.py`** — accuracy / precision / recall / F1 / AUROC / AUPRC / Brier / ECE; bootstrap 95% CIs; reliability curve data; F1-optimal threshold picker.
- **`error_analysis.py`** — confusion buckets (TP/TN/FP/FN), hardest examples (largest |score − label|), per-subgroup metrics, length-effect bins.
- **`runner.py`** — wires it all together; writes JSON metrics, predictions parquet, and per-subreddit / per-length CSVs alongside each model run.

### `src/analysis/`

- **`linguistic_markers.py`** — for every `f_*` feature, compares positive vs negative posts on a target label using Mann-Whitney U with Benjamini-Hochberg FDR correction; reports Cohen's d. Drives the linguistic-analysis chapter (RQ2).
- **`explainability.py`** — SHAP for XGBoost; gradient × input attributions for transformers.
- **`temporal.py`** — pre-COVID / COVID-peak / post-peak windows for RQ4.

### `src/viz/`

- **`plots.py`** — 10 reusable matplotlib + seaborn figures: corpus overview, length distribution, temporal area chart, label-distribution heatmap, label-co-occurrence, ROC + PR, reliability diagram, confusion matrix, per-subreddit F1 bars, top-Cohen's-d markers.
- **`runner.py`** — `run_all()` regenerates every figure that has the data to support it.

### `src/cli.py`

Typer CLI. `anxiety <command> --help` for every command. See [`docs/cli_reference.md`](cli_reference.md) for full reference.

## Adding things later

| To add… | Edit |
|---|---|
| A subreddit | One entry in `configs/subreddits.yaml` |
| A new label | `LABELS` in `src/labeling/weak.py`, lexicons, codebook, multi-task config |
| A new model | Subclass `BaseModel`, register in `src/models/registry.py`, add YAML |
| A new collector | Subclass `BaseCollector`, add to `runner.make_collector` |
| A new plot | Function in `src/viz/plots.py`, call site in `runner.py` |
| A new linguistic feature | Function in `src/features/linguistic.py` returning `dict[str, float]` with `f_` prefixed keys |
