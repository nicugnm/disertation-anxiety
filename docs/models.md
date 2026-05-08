# Models

Six concrete models, all subclassing `BaseModel`. Each implements `fit`, `predict_proba`, `save`, `load`. Add a new model by subclassing + registering in `src/models/registry.py`.

## At a glance

| Class | Config | Type | Notes |
|---|---|---|---|
| `TfidfLogRegModel` | `baseline.yaml` | sklearn | Baseline floor; **F1 0.88 on real anxiety** |
| `XgboostLinguisticModel` | `xgboost.yaml` | XGBoost | Trains on `f_*` features; SHAP-explainable |
| `TransformerModel` | `transformer.yaml` | HuggingFace | Single-target fine-tune |
| `MultiTaskTransformer` | `multitask.yaml` | PyTorch | **Dissertation novelty** — joint heads |
| `LlmZeroShotModel` | `llm_zero_shot.yaml` | Anthropic | Prompts at predict time; no fitting |

## TF-IDF + Logistic Regression

Sane sklearn pipeline: `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` → `LogisticRegression(class_weight='balanced')`.

**When to use it:** as a sanity floor and a fast iteration tool. If a deep model can't beat the TF-IDF baseline by ≥3 F1 points it isn't worth the carbon.

**Tuning hooks** in `configs/models/baseline.yaml`:

```yaml
vectorizer:
  ngram_range: [1, 2]   # try [1, 3] for a small bump on long-context features
  min_df: 5             # drop terms appearing in <5 docs
  max_df: 0.95          # drop near-stopwords
  max_features: 100000

classifier:
  C: 1.0                # smaller = more regularization
  class_weight: balanced
  solver: liblinear     # supports L1; useful for feature inspection
```

## XGBoost on linguistic features

Uses only the `f_*` columns from `src/features/linguistic.py`. The bridge to RQ2: SHAP values on this model give you "which linguistic markers drive predictions" with an interpretable, defensible feature space.

**Tuning hooks** in `configs/models/xgboost.yaml`:

```yaml
classifier:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  scale_pos_weight: auto   # auto-computes (n_neg / n_pos) for imbalance
  early_stopping_rounds: 30
```

Inspect after training:

```python
from src.models.xgboost_model import XgboostLinguisticModel
from src.utils.config import load_model_config

cfg = load_model_config("experiments/runs/xgboost_linguistic/config.yaml")
model = XgboostLinguisticModel(cfg).load("experiments/runs/xgboost_linguistic/model")
print(model.feature_importance().head(20))
```

For SHAP:

```python
from src.analysis.explainability import shap_importance_xgboost
import pandas as pd

df = pd.read_parquet("data/processed/labeled.parquet")
imp = shap_importance_xgboost(model, df, n_samples=2000)
print(imp.head(20))
```

## Transformer (single-target)

`src/models/transformer.py`. Fine-tunes a HuggingFace `AutoModelForSequenceClassification` with two output classes.

**Pretrained-model selection.** Tries `pretrained` first, falls back to `fallback_pretrained` if the first isn't available. By default this means MentalRoBERTa with a RoBERTa-base fallback.

```yaml
pretrained: mental/mental-roberta-base
fallback_pretrained: roberta-base
```

**Device.** Auto-selects CUDA → MPS (Apple Silicon) → CPU. Set `fp16: true` only on CUDA — MPS half-precision is unstable.

**Practical training tips:**

- Default `max_length: 256` is suitable for Reddit posts (median ~150 tokens). Longer seqs slow training quadratically.
- `gradient_accumulation_steps` lets you train larger effective batches when GPU RAM is tight.
- `load_best_model_at_end: true` requires `eval_strategy: epoch` and a non-empty val set.
- The `train` command persists train/val/test parquets alongside the model so `evaluate` uses the same split.

## Multi-task transformer (dissertation novelty)

`src/models/multitask.py`. Shared encoder + one sigmoid head per target. Trained with BCE-with-logits.

Why this exists: anxiety, health-anxiety, depression, and suicidality co-occur. A multi-label model with shared representations beats independent binary heads on the rare classes — particularly health anxiety, which has the lowest weak-label support.

Per-task loss weighting handles the imbalance:

```yaml
loss_weights:
  anxiety: 1.0
  health_anxiety: 1.5     # boost the focal class
  depression: 1.0
  suicidality: 1.2
```

Per-row, per-task confidence weights (from the labeling tier — manual=1.0, llm=0.7, weak=0.4) are multiplied in too, so noisy weak labels can't drown out the smaller, cleaner LLM/manual signal.

The training loop is hand-rolled in pure PyTorch (rather than using `Trainer`) because we need the per-(sample, task) sample weighting that HF Trainer doesn't natively support.

## Claude zero-/few-shot

`src/models/llm_zero_shot.py`. Sends each post to the Claude API with a prompt asking for a yes/no answer on the target label. No fine-tuning.

**Few-shot mode.** Set `n_few_shot: N` and provide `data/processed/few_shot_examples.jsonl` with N labelled examples. The prompt prepends them as in-context demonstrations.

**Cache.** Same SQLite cache pattern as tier-2 labelling. Re-evaluation against the same test set is free.

**Cost control.** `rpm` rate-limits requests; cache hits don't sleep.

**When to use it.** As a baseline for "do we even need fine-tuning?" If Claude zero-shot beats the fine-tuned MentalRoBERTa on health anxiety, that's a finding for the discussion chapter.

## Saving and loading

All models implement the same `save(path)` / `load(path)` interface, but the on-disk format differs:

- TF-IDF / XGBoost: pickle.
- Transformer: HuggingFace `save_pretrained` / `from_pretrained` on a directory.
- Multi-task: `state_dict()` + `tokenizer` + `targets.txt` in a directory.
- LLM zero-shot: just a sentinel file (model lives at the API).

`anxiety train` saves to `experiments/runs/<name>/model/`. `anxiety evaluate` expects that path layout.

## Adding a new model

1. Subclass `BaseModel` in `src/models/<name>.py`.
2. Add a branch to `src/models/registry.py:build_model`.
3. Add a YAML in `configs/models/<name>.yaml`.

The CLI, evaluation, and analysis touch only the interface, so no other code changes.
