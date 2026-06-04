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
  collect                  Collect Reddit data into data/raw/.
  collect-authors          Fetch full Reddit histories for the disclosure-test cohort.
  preprocess               Clean, anonymize, dedupe — raw → interim parquet.
  label                    Apply tier-1 weak / self-disclosure / aggregate labels.
  train                    Train a model from a config YAML.
  evaluate                 Evaluate a saved model against its held-out test set.
  report                   Aggregate evaluation runs into one comparison table.
  analyze-markers          Linguistic-feature comparison between positives/negatives.
  analyze-temporal         COVID-window analysis (RQ4).
  plot                     Generate every standard figure.
  audit                    Comprehensive corpus + labeling audit.
  build-disclosure-testset Build user-level disclosure test set with matched controls.
  eval-disclosure          Evaluate a trained model at user-level on the disclosure test set.
  erisk-load               Load an eRisk 2025 file into a parquet.
  erisk-eval               Compute eRisk-style per-user metrics from per-post predictions.
  smoke-run                End-to-end pipeline on synthetic data.
```

---

## `anxiety collect`

Collect Reddit data into `data/raw/<subreddit>.parquet`.

```bash
anxiety collect [--backend BACKEND] [OPTIONS]
```

| flag | default | meaning |
|---|---|---|
| `--backend` | `scraper` | `scraper` \| `search` \| `praw` \| `dump` \| `synthetic` |
| `--config` | `configs/subreddits.yaml` | Subreddit list + collection rules |
| `--out-dir` | `data/raw/` | Where to write parquet shards |
| `--n-synthetic` | 200 | Posts per subreddit (synthetic only) |
| `--dump-dir` | `data/external/dumps` | `.zst` archive directory (dump only) |
| `--request-interval` | 1.5 | Seconds between requests (scraper / search) |
| `--max-pages` | 10 | Max pages per listing (scraper) or per query (search) |
| `--include-comments` | off | Also fetch comments per submission (scraper only) |
| `--min-submission-comments` | 5 | Skip submissions with fewer comments (scraper + comments) |
| `--max-comments-per-post` | 40 | Max comments to keep per submission |
| `--min-comment-score` | 1 | Drop comments below this score |

**Backends:**

- `scraper` — **default. No credentials needed.** Hits Reddit's public JSON endpoints (`old.reddit.com/r/<sub>/<listing>.json`).
- `search` — Reddit-wide search for self-disclosure phrases (e.g. "I was diagnosed with depression"). Writes one parquet per query.
- `praw` — OAuth via `.env` (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`).
- `dump` — Reads `.zst`-compressed JSONL dumps placed under `--dump-dir`.
- `synthetic` — Reproducible synthetic data; for tests/CI.

---

## `anxiety collect-authors`

Fetch full Reddit post histories for the disclosure-test cohort (both classes).

```bash
anxiety collect-authors [OPTIONS]
```

| flag | default | meaning |
|---|---|---|
| `--users-csv` | `data/processed/disclosure_testset__users.csv` | Cohort CSV produced by `build-disclosure-testset` |
| `--raw-dir` | `data/raw` | Where to recover usernames from |
| `--out-dir` | `data/raw/authors` | Per-user history shards (one `<hash>.parquet` each) |
| `--request-interval` | 1.5 | Seconds between requests |
| `--max-pages` | 10 | Pages per user listing (~1 000-item cap) |
| `--config` | `configs/subreddits.yaml` | Subreddit config |

Resumable: re-running skips users whose `<hash>.parquet` already exists. Exits non-zero (code 1) if the cohort CSV is missing — run `build-disclosure-testset` first.

---

## `anxiety preprocess`

```bash
anxiety preprocess [OPTIONS]
```

| flag | default | meaning |
|---|---|---|
| `--raw-dir` | `data/raw` | Input |
| `--out-dir` | `data/interim` | Output |
| `--no-ner` | off | Skip spaCy NER (regex-only PII redaction; ~10× faster) |
| `--n-process-ner` | 1 | spaCy worker processes (>1 gives linear speedup on multi-core) |
| `--near-dup-threshold` | 5 | SimHash Hamming radius; -1 disables near-dedup (exact-only) |

Writes per-subreddit parquet shards plus `_all.parquet` containing all rows.

---

## `anxiety label`

```bash
anxiety label --tier {weak|disclosure|aggregate|all}
```

| flag | meaning |
|---|---|
| `--tier weak` | Apply tier-1 weak labels (subreddit prior + lexicon overlap). |
| `--tier disclosure` | Apply self-disclosure labels (regex diagnosis patterns + negation/hypothetical/third-party/denial filters). Suicidality disclosure is disabled. Runs an automatic audit afterwards. |
| `--tier aggregate` | Combine tiers into final `label_<k>` columns (precedence: disclosure > weak). |
| `--tier all` | Run weak → disclosure → aggregate sequentially. |
| `--interim-dir` | Default `data/interim`. |
| `--out-path` | Default `data/processed/labeled.parquet`. |
| `--config` | Subreddits config path. |
| `--labeling-config` | Labeling config path. |

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
| `--include-held-out` | off | Include held-out disclosure-test-set users in training. Off by default to preserve train/test separation. |

Saves the trained model + `train.parquet` / `val.parquet` / `test.parquet` (so `evaluate` reuses the exact split) + a copy of the config YAML. Posts marked `held_out_split=True` (members of the user-level disclosure test set) are excluded by default.

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

| flag | default | meaning |
|---|---|---|
| `--target` | `health_anxiety` | Which label column to compare |
| `--data-path` | `data/processed/labeled.parquet` | Labeled corpus |
| `--out-path` | `experiments/markers__<target>.csv` | Output CSV |

For every linguistic feature (`f_*`), runs a Mann-Whitney U test between positive and negative posts on the chosen target, computes Cohen's d, applies Benjamini-Hochberg FDR, prints the top 20 by `|d|`, writes the full table to `experiments/markers__<target>.csv`.

---

## `anxiety analyze-temporal`

```bash
anxiety analyze-temporal [--data-path PATH] [--out-path CSV]
```

| flag | default | meaning |
|---|---|---|
| `--data-path` | `data/processed/labeled.parquet` | Labeled corpus |
| `--out-path` | `experiments/temporal.csv` | Output CSV |

Bins posts into pre-COVID / COVID-peak / post-peak windows and reports per-window per-subreddit positive rates for each label. Drives RQ4 analysis.

---

## `anxiety plot`

```bash
anxiety plot [--run-dir RUN] [--figures-dir DIR] [--labeled-path PATH]
```

| flag | default | meaning |
|---|---|---|
| `--figures-dir` | `docs/figures` | Where to write PNGs |
| `--labeled-path` | `data/processed/labeled.parquet` | Labeled corpus |
| `--run-dir` | (optional) | A trained model run dir (for performance plots) |

Regenerates every figure that has the data to support it:

- corpus_overview / length_distribution / temporal / label_distribution / label_cooccurrence (need `data/processed/labeled.parquet`)
- pr_roc / calibration / confusion / subreddit_f1 (need `--run-dir`)
- markers (need `experiments/markers__*.csv`)

---

## `anxiety audit`

```bash
anxiety audit [--data-path PATH] [--out-path PATH] [--n-examples N] [--top-n-subreddits N]
```

| flag | default | meaning |
|---|---|---|
| `--data-path` | `data/processed/labeled.parquet` | Labeled corpus |
| `--out-path` | `docs/audit_report.md` | Markdown report path; pass empty to skip writing |
| `--n-examples` | 5 | Example disclosure matches per target |
| `--top-n-subreddits` | 25 | Subreddits shown in the per-subreddit × target matrix |

Reports corpus stats, per-tier label counts, per-subreddit × target matrix, user-level disclosure stats, and sample disclosure matches for sanity-checking the regex. Writes a self-contained Markdown report for the thesis appendix. Also runs automatically after `anxiety label --tier disclosure`.

---

## `anxiety build-disclosure-testset`

```bash
anxiety build-disclosure-testset [OPTIONS]
```

| flag | default | meaning |
|---|---|---|
| `--in-path` | `data/processed/labeled.parquet` | Labeled corpus parquet |
| `--test-path` | `data/processed/disclosure_testset.parquet` | Output test-set parquet |
| `--controls-per-positive` | 2 | Matched controls per positive user |
| `--min-posts-per-user` | 3 | Skip users with fewer posts than this |
| `--seed` | 42 | Random seed |
| `--cohort-users-csv` | (optional) | Regroup only these users (author_hash column) within their enriched histories instead of building a fresh corpus-wide test set |

Builds a user-level self-disclosure test set. Disclosed users become positives; non-disclosed users from the same subreddits become subreddit-matched controls. All their posts are marked `held_out_split=True` in the corpus so `anxiety train` excludes them automatically, keeping the noisy-train / clean-test split genuinely user-disjoint and label-source-disjoint.

Outputs:
- `disclosure_testset.parquet` — test posts
- `disclosure_testset__users.csv` — user-level summary (input to `collect-authors`)
- Updates `labeled.parquet` with `held_out_split` flag

---

## `anxiety eval-disclosure`

```bash
anxiety eval-disclosure RUN_DIR [OPTIONS]
```

| arg/flag | default | meaning |
|---|---|---|
| `RUN_DIR` | (required) | `experiments/runs/<name>` directory |
| `--target` | `anxiety` | Target label to evaluate |
| `--test-path` | `data/processed/disclosure_testset.parquet` | Disclosure test parquet |
| `--aggregation` | `mean` | `mean` \| `max` \| `topk_mean` — how to aggregate per-post scores to user level |
| `--mask-disclosure-posts` | off | Drop the explicit disclosure utterances before aggregating per user |

Evaluates a trained model at **user level** on the disclosure test set. Aggregates per-post scores to a single user score, then computes precision / recall / F1 / AUROC / AUPRC. Always emits both "with disclosure posts" and "disclosure masked" variants in the terminal table and writes both to `<RUN_DIR>/eval/<name>__<target>__disclosure_userlevel.json`. This is the clean evaluation used in this work (not a gold/manually-annotated set).

---

## `anxiety erisk-load`

```bash
anxiety erisk-load PATH [--task {auto|task1|task2}] [--out-path PATH]
```

| arg/flag | default | meaning |
|---|---|---|
| `PATH` | (required) | Path to an eRisk file or directory |
| `--task` | `auto` | Format: `auto` (detect from extension/content) \| `task1` (TREC) \| `task2` (JSON) |
| `--out-path` | `data/external/erisk_<task>.parquet` | Output parquet |

Loads an eRisk 2025 dataset (Task 1 TREC or Task 2 JSON) into a parquet for use with the rest of the pipeline.

---

## `anxiety erisk-eval`

```bash
anxiety erisk-eval PREDICTIONS [OPTIONS]
```

| arg/flag | default | meaning |
|---|---|---|
| `PREDICTIONS` | (required) | Per-post predictions parquet |
| `--user-col` | `author_hash` | Column identifying the user |
| `--label-col` | `label_anxiety` | True label column |
| `--score-col` | `score_anxiety` | Model score column |
| `--threshold` | 0.5 | Score threshold for positive decision |
| `--require-consecutive` | 1 | Posts above threshold required to commit to a positive decision |

Computes eRisk-style per-user metrics (ERDE_5, ERDE_50, F_latency, P@k) from per-post predictions. Prints results as JSON.

---

## `anxiety smoke-run`

```bash
anxiety smoke-run
```

End-to-end pipeline on synthetic data: collect → preprocess → label (weak only) → train baseline → evaluate → marker analysis. Takes ~30 seconds. No credentials required. The CI smoke-test relies on this.
