# CLI reference

Every command exposed by `anxiety <command>`. Run `anxiety <command> --help` for the live version.

```
$ anxiety --help
Usage: anxiety [OPTIONS] COMMAND [ARGS]...

Anxiety / health-anxiety detection pipeline (dissertation).

Options:
  --log-level TEXT     Log level  [default: INFO]
  --json-logs          JSON-formatted logs

Commands:
  collect            Collect Reddit data into data/raw/<subreddit>.parquet.
  preprocess         Clean, anonymize, dedupe — raw → interim parquet.
  label              Apply tier-1 weak / tier-2 LLM / aggregate labels.
  annotate           Tier-3 manual annotation TUI.
  kappa              Compute Cohen's kappa between two annotators.
  train              Train a model from a config YAML.
  evaluate           Evaluate a saved model against its held-out test set.
  report             Aggregate evaluation runs into one comparison table.
  analyze-markers    Linguistic-feature comparison between positives/negatives.
  analyze-temporal   COVID-window analysis (RQ4).
  plot               Generate every standard figure.
  smoke-run          End-to-end pipeline on synthetic data.
```

---

## `anxiety collect`

Collect Reddit data into `data/raw/<subreddit>.parquet`.

```bash
anxiety collect [--backend BACKEND] [OPTIONS]
```

| flag | default | meaning |
|---|---|---|
| `--backend` | `scraper` | `scraper` \| `praw` \| `dump` \| `synthetic` |
| `--config` | `configs/subreddits.yaml` | Subreddit list + collection rules |
| `--out-dir` | `data/raw/` | Where to write parquet shards |
| `--n-synthetic` | 200 | Posts per subreddit (synthetic only) |
| `--dump-dir` | `data/external/dumps` | `.zst` archive directory (dump only) |
| `--request-interval` | 1.5 | Seconds between requests (scraper only) |
| `--max-pages` | 10 | Max pages per listing — 100 posts each (scraper only) |

**Backends:**

- `scraper` — **default. No credentials needed.** Hits Reddit's public JSON endpoints.
- `praw` — OAuth via `.env` (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`).
- `dump` — Reads `.zst`-compressed JSONL dumps placed under `--dump-dir`.
- `synthetic` — Reproducible synthetic data; for tests/CI.

---

## `anxiety preprocess`

```bash
anxiety preprocess [--no-ner]
```

| flag | default | meaning |
|---|---|---|
| `--raw-dir` | `data/raw` | Input |
| `--out-dir` | `data/interim` | Output |
| `--no-ner` | off | Skip spaCy NER for ~5× speedup; regex-only PII redaction |

Writes per-subreddit parquet shards plus `_all.parquet` containing all rows.

---

## `anxiety label`

```bash
anxiety label --tier {weak|llm|aggregate|all}
```

| flag | meaning |
|---|---|
| `--tier weak` | Apply tier-1 weak labels (subreddit prior + lexicon). |
| `--tier llm` | Apply tier-2 LLM labels (Claude). Needs `ANTHROPIC_API_KEY`. |
| `--tier aggregate` | Combine tiers into final `label_<k>` columns. |
| `--tier all` | Run weak → llm → aggregate sequentially. |
| `--interim-dir` | Default `data/interim`. |
| `--out-path` | Default `data/processed/labeled.parquet`. |
| `--config` | Subreddits config path. |
| `--labeling-config` | Labeling config path. |

---

## `anxiety annotate`

Tier-3 manual annotation. Resumable, supports two-annotator κ workflows.

```bash
anxiety annotate --annotator-id alice [--input-path PATH] [--output-path PATH]
```

| flag | default | meaning |
|---|---|---|
| `--annotator-id` | (required) | Free-form identifier for this annotator |
| `--input-path` | `data/processed/labeled.parquet` | Source of posts to label |
| `--output-path` | `tier3_manual.output_path` | Where to write annotations |
| `--labeling-config` | `configs/labeling.yaml` | Tier-3 settings |

The TUI shows a crisis-resource banner; you can `s` to skip a distressing post. Progress is flushed to disk every 10 rows.

---

## `anxiety kappa`

```bash
anxiety kappa <annotator_a> <annotator_b>
```

Computes Cohen's κ for each label across the two annotators on the rows they both labelled. Less than 5 shared rows for a label returns `n/a`.

---

## `anxiety train`

```bash
anxiety train CONFIG [OPTIONS]
```

| arg | default | meaning |
|---|---|---|
| `CONFIG` | (required) | Path to a model YAML |
| `--data-path` | `data/processed/labeled.parquet` | Labeled corpus |
| `--output-dir` | `experiments/runs/<name>` | Where to save the trained model + splits |

Saves the trained model + `train.parquet` / `val.parquet` / `test.parquet` (so `evaluate` reuses the exact split) + a copy of the config YAML.

---

## `anxiety evaluate`

```bash
anxiety evaluate RUN_DIR [--out-dir DIR]
```

`RUN_DIR` should be the directory produced by `anxiety train`. The evaluator:
- loads the model from `<RUN_DIR>/model`,
- loads the test split from `<RUN_DIR>/test.parquet`,
- writes per-target metrics, predictions, by-subreddit CSV, by-length CSV into `<RUN_DIR>/eval/` (or `--out-dir`).

For multi-target models (the multitask transformer), evaluates every configured target.

---

## `anxiety report`

```bash
anxiety report EVAL_DIR
```

Walks `EVAL_DIR` for `*__metrics.json` files and prints a unified table sorted by (target, F1 desc).

---

## `anxiety analyze-markers`

```bash
anxiety analyze-markers --target {anxiety|health_anxiety|depression|suicidality}
```

For every linguistic feature (`f_*`), runs a Mann-Whitney U test between positive and negative posts on the chosen target, computes Cohen's d, applies Benjamini-Hochberg FDR, prints the top 20 by `|d|`, writes the full table to `experiments/markers__<target>.csv`.

---

## `anxiety analyze-temporal`

```bash
anxiety analyze-temporal [--data-path PATH] [--out-path CSV]
```

Bins posts into pre-COVID / COVID-peak / post-peak windows and reports per-window per-subreddit positive rates for each label. Output is a CSV.

---

## `anxiety plot`

```bash
anxiety plot [--run-dir RUN] [--figures-dir DIR] [--labeled-path PATH]
```

Regenerates every figure that has the data to support it:

- corpus_overview / length_distribution / temporal / label_distribution / label_cooccurrence (need `data/processed/labeled.parquet`)
- pr_roc / calibration / confusion / subreddit_f1 (need `--run-dir`)
- markers (need `experiments/markers__*.csv`)

---

## `anxiety smoke-run`

```bash
anxiety smoke-run
```

End-to-end pipeline on synthetic data: collect → preprocess → label → train baseline → evaluate → marker analysis. Takes ~30 seconds. No credentials required. The CI smoke-test relies on this.
