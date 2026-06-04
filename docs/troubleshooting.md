# Troubleshooting

Common errors and what to do about them.

## Collection

### "Missing Reddit credentials" when running `anxiety collect`

You're using `--backend praw` without filling out `.env`. Either fill it in (see `.env.example`) or use the no-creds default:

```bash
anxiety collect --backend scraper
```

### Scraper returns 0 posts for a subreddit

Two common causes:

1. **The subreddit is banned/quarantined.** Reddit serves 403 on the JSON endpoint for these. You'll see `scraper.unavailable` in the logs. Examples in our default config: r/COVID19positive (banned).
2. **The subreddit name is mistyped.** Reddit returns 404. Same `scraper.unavailable` log. Double-check `configs/subreddits.yaml`.

In either case the scraper caches the negative result so you don't keep hitting it. To retry, delete `.cache/json_scraper.sqlite` (or that specific row).

### `429 Too Many Requests`

You're hitting Reddit too fast. Increase `--request-interval`:

```bash
anxiety collect --backend scraper --request-interval 3.0
```

Reddit's unauthenticated tolerance is roughly 1 req/sec. The default 1.5s leaves headroom; bumping to 3s is paranoid-safe.

### Scrape stops mid-run

Re-run the same command. Cached subreddits are skipped (free); only the unfinished sub continues. Use `tail -f /tmp/collect.log` if you tee'd to a log.

## Preprocessing

### `OSError: [E050] Can't find model 'en_core_web_sm'`

Install the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

Or run preprocessing without NER:

```bash
anxiety preprocess --no-ner
```

NER-off mode still does regex-based PII redaction — a reasonable compromise for development.

### `langdetect` errors on short posts

We catch `LangDetectException` and default to `'en'` to avoid dropping posts. If you see this in logs it's purely informational.

## Labeling

### `KeyError: 'subreddit_group'` during stratified sampling

The labeled DataFrame doesn't have `subreddit_group` because no `attach_subreddit_groups` step ran. The `label_corpus` function does this automatically; if you're calling internals manually, run it first.

## Training

### `Could not load mental/mental-roberta-base or roberta-base`

Network or HuggingFace Hub issue. Try:

1. Set `HF_TOKEN` if the model is gated.
2. Manually pre-download:
   ```bash
   python -c "from transformers import AutoModel, AutoTokenizer; \
              AutoModel.from_pretrained('roberta-base'); \
              AutoTokenizer.from_pretrained('roberta-base')"
   ```
3. Switch the YAML to a model you can access.

### Transformer training runs out of memory

In `configs/models/transformer.yaml`:

- Reduce `per_device_train_batch_size` (e.g. 8 instead of 16).
- Increase `gradient_accumulation_steps` to compensate.
- Reduce `tokenizer.max_length` (e.g. 128 instead of 256). Reddit posts have a long tail but the median is short.
- On CUDA: enable `fp16: true`. (On Apple MPS, leave it false.)

### Trainer error: `evaluation_strategy` vs `eval_strategy`

HuggingFace renamed this argument in transformers 4.42. Our `transformer.py` handles both. If you see this error you've patched the file — undo your patch.

### Multi-task training F1 stuck at 0 for the rare class

Health-anxiety has very few weak-label positives. Options:

- Bump the loss weight: `loss_weights.health_anxiety: 2.0` in `configs/models/multitask.yaml`.
- Enable focal loss (not implemented; would be a small change in `multitask.py`).

## Evaluation

### Reliability diagram has long flat sections

The model is not producing scores in those bins. Either it's overconfident (only 0/1 predictions) or undertrained. Calibrate with temperature scaling — implementation in `src/evaluation/metrics.py:expected_calibration_error` for measurement; a Platt scaling fit is left to the thesis.

### `single class in labels — bootstrap CIs are NaN`

Your test set has only positives or only negatives. Stratified split should prevent this; check that the target column has both classes:

```python
df["label_anxiety"].value_counts()
```

## Plots

### "Average post length" panel labels overlap the bars

Pull latest — fixed by hiding the right-panel y-tick labels (already shown on the left).

### Plot generation hangs

Likely a font cache issue on first run. Matplotlib's font manager builds an index. Clear it and retry:

```bash
rm -rf ~/.matplotlib
```

## Imports

### `ModuleNotFoundError: No module named 'src'` when importing in Python

You forgot `pip install -e .`. The package needs to be installed (editable) for `from src.X import Y` to work outside the project directory. Inside the directory, `python -m src.cli ...` also works.

### `ModuleNotFoundError: No module named 'seaborn'` (or any optional dep)

If installed via `pip install -e .` (no `[dev]`), only the runtime deps are present. Re-install with the dev extras:

```bash
pip install -e ".[dev]"
```

## CI / tests

### Tests pass locally, fail in CI

Common cause: missing spaCy model or NLTK data. Add to your CI script:

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords vader_lexicon
```

The smoke test (`tests/test_smoke_minimal.py`) deliberately uses `use_ner=False` and `keep_only_english=False` to avoid these dependencies.

### `pytest.PytestUnknownMarkWarning: Unknown pytest.mark.timeout`

Cosmetic. Install `pytest-timeout` if you want to enforce the timeout, or remove the marker.

## Performance

### `anxiety preprocess` is slow

NER is the bottleneck. Use `--no-ner` for dev, or down-sample first:

```python
import pandas as pd
from src.utils.io import write_parquet
df = pd.read_parquet("data/raw/Anxiety.parquet").sample(500)
write_parquet(df, "data/raw/Anxiety.parquet")
```

### Training is slow on CPU

Expected. Use a GPU (CUDA) or Apple Silicon MPS:

```python
import torch
print(torch.cuda.is_available(), torch.backends.mps.is_available())
```

The transformer's `_device_select` auto-picks the best available.

## Disk usage

### `.cache/` is huge

Mostly scraper responses. Safe to delete; you'll re-pay the cost on the next run.

```bash
ls -lh .cache/
rm .cache/json_scraper.sqlite   # specific cache
```

### Parquet shards are smaller than I expected

Parquet with zstd compression typically gets 5–10× ratio on text. 14k posts compresses to ~10 MB. This is correct.
