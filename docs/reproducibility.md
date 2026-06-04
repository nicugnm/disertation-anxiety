# Reproducibility

How to reproduce every artifact in this repo from a fresh clone.

## What's controlled

| Source of variance | How we control it |
|---|---|
| Python interpreter | `requires-python = ">=3.10"` in `pyproject.toml` |
| Dependency versions | Pinned with `>=` in `pyproject.toml`; for exact reproduction, use `pip freeze > requirements.txt` post-install |
| Random seeds | Every model config has `random_state` (default 42); splitting and bootstrap CIs use it too |
| Reddit scraping | All HTTP responses cached in `.cache/json_scraper.sqlite`; same cache file → same posts |
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
anxiety label --tier disclosure
anxiety label --tier aggregate
```

### 4. Build the disclosure test set

```bash
anxiety build-disclosure-testset
```

This creates `data/processed/disclosure_testset.parquet` (posts by disclosed users + subreddit-matched controls) and marks `held_out_split=True` in `data/processed/labeled.parquet` for all test-set users. The training step automatically excludes held-out posts.

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
anxiety evaluate experiments/runs/mentalbert_anxiety
anxiety evaluate experiments/runs/multitask_anxiety_health_dep_suic
# ...etc
anxiety report experiments/runs/tfidf_logreg/eval

# User-level disclosure evaluation (Experiment 7)
anxiety eval-disclosure experiments/runs/tfidf_logreg --target anxiety
anxiety eval-disclosure experiments/runs/mentalbert_anxiety --target anxiety
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
# Note: do NOT copy data/raw or data/interim (raw text, not for redistribution).
```

## Reproducibility checklist for the thesis

When writing the methodology chapter, document:

- [ ] Reddit collection date (`max(collected_at)` in raw parquet).
- [ ] Backend used (PRAW / scraper / dump).
- [ ] Sub list + version of `configs/subreddits.yaml`.
- [ ] Lexicon version + provenance for each list.
- [ ] Self-disclosure regex patterns and false-positive filter rules (in `src/labeling/self_disclosure.py`).
- [ ] Disclosure test set construction: positives selection, subreddit-matched control sampling, `held_out_split` enforcement.
- [ ] Pretrained model checkpoint name + revision.
- [ ] Random seeds + library versions (from `pip freeze`).
- [ ] Test set IDs (releasable).
- [ ] Compute used (CPU / GPU model + count + total wall time).
