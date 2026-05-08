# Reproducibility

How to reproduce every artifact in this repo from a fresh clone.

## What's controlled

| Source of variance | How we control it |
|---|---|
| Python interpreter | `requires-python = ">=3.10"` in `pyproject.toml` |
| Dependency versions | Pinned with `>=` in `pyproject.toml`; for exact reproduction, use `pip freeze > requirements.txt` post-install |
| Random seeds | Every model config has `random_state` (default 42); splitting and bootstrap CIs use it too |
| Reddit scraping | All HTTP responses cached in `.cache/json_scraper.sqlite`; same cache file → same posts |
| LLM labeling | All API responses cached in `.cache/llm_labels.sqlite` (key = SHA-256 of model + prompt + post) |
| Synthetic data | Per-subreddit deterministic RNG seeded from `(seed, subreddit_name)` |

## Reproduction recipe

### 0. Environment

```bash
# Use the same Python that's known to work
python --version             # require >=3.10, recommended 3.11
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords vader_lexicon
```

### 1. Collect (with cache)

```bash
anxiety collect --backend scraper
```

The first run hits Reddit; subsequent runs are free (served from `.cache/json_scraper.sqlite`). To **fully reproduce a snapshot**, share the cache file.

If you want OAuth-authenticated collection instead:

```bash
cp .env.example .env  # fill REDDIT_* vars
anxiety collect --backend praw
```

If you want the historical Pushshift dump path:

```bash
# Download archives to data/external/dumps/RS_YYYY-MM.zst from arctic_shift / academic torrents
anxiety collect --backend dump
```

### 2. Preprocess

```bash
anxiety preprocess
```

Deterministic: same input parquet → same output parquet. Skip NER for a faster but slightly less aggressive PII pass:

```bash
anxiety preprocess --no-ner
```

### 3. Label

```bash
anxiety label --tier weak
anxiety label --tier llm        # only if ANTHROPIC_API_KEY is set
anxiety label --tier aggregate
```

The LLM cache (`.cache/llm_labels.sqlite`) makes tier-2 reproducible across runs and machines if you share the cache.

### 4. Annotate (manual)

```bash
anxiety annotate --annotator-id alice
anxiety annotate --annotator-id bob
anxiety kappa alice bob
```

### 5. Train

```bash
anxiety train configs/models/baseline.yaml
anxiety train configs/models/xgboost.yaml
anxiety train configs/models/transformer.yaml      # GPU/MPS recommended
anxiety train configs/models/multitask.yaml         # GPU/MPS recommended
```

Each saves to `experiments/runs/<name>/` with the model + train/val/test splits + a copy of the config YAML.

### 6. Evaluate

```bash
anxiety evaluate experiments/runs/tfidf_logreg
anxiety evaluate experiments/runs/xgboost_linguistic
# ...etc
anxiety report experiments/runs/tfidf_logreg/eval
```

### 7. Analyze + plot

```bash
anxiety analyze-markers --target anxiety
anxiety analyze-markers --target health_anxiety
anxiety plot --run-dir experiments/runs/tfidf_logreg
```

## What you can / cannot share

You **can** share:
- All code in `src/`, configs in `configs/`, docs in `docs/`.
- Your model weights from `experiments/runs/*/model/`.
- Your evaluation outputs (`*__metrics.json`, `*__predictions.parquet` after dropping `clean_text`, `*__by_subreddit.csv`).
- The `.cache/json_scraper.sqlite` cache *if and only if* the recipient is a researcher who agrees to comply with Reddit's terms — this is borderline; safer to share the post IDs only.

You **cannot** share:
- Raw Reddit text (`data/raw/`, `data/interim/`, `clean_text` column of any parquet). Reddit's Data API Terms forbid redistribution.

The acceptable artifact for public release is **post-IDs + your labels + aggregated statistics**. Recipients re-derive the corpus from a fresh Reddit fetch using the IDs.

## Snapshotting an experiment

```bash
git tag -a snapshot-YYYY-MM-DD -m "snapshot for thesis chapter X"

# Snapshot the full state (excluding raw text):
mkdir -p snapshot/
cp pyproject.toml snapshot/
pip freeze > snapshot/requirements.lock.txt
cp -r configs snapshot/
cp -r experiments/runs/*/eval snapshot/eval/
cp experiments/markers__*.csv snapshot/
# Note: do NOT copy data/raw, data/interim, or .cache/llm_labels.sqlite
# (the cache contains paraphrased post text from prompts).
```

## Reproducibility checklist for the thesis

When writing the methodology chapter, document:

- [ ] Reddit collection date (`max(collected_at)` in raw parquet).
- [ ] Backend used (PRAW / scraper / dump).
- [ ] Sub list + version of `configs/subreddits.yaml`.
- [ ] Lexicon version + provenance for each list.
- [ ] LLM model + prompt version (the prompt template is in `src/labeling/llm.py`).
- [ ] Annotator demographics (high-level only, no PII).
- [ ] Inter-annotator κ for each label.
- [ ] Pretrained model checkpoint name + revision.
- [ ] Random seeds + library versions (from `pip freeze`).
- [ ] Test set IDs (releasable).
- [ ] Compute used (CPU / GPU model + count + total wall time).
