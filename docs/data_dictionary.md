# Data dictionary

Every column at every pipeline stage. All times are Unix timestamps; all `bool` columns are 0/1 in parquet (pandas-native bool).

## Stage 1 — `data/raw/<subreddit>.parquet`

Direct output of any collector. Schema is identical regardless of backend.

| column | type | description |
|---|---|---|
| `id` | string | Reddit base-36 post ID. Unique within Reddit. |
| `subreddit` | string | Source subreddit name (without `r/`). |
| `created_utc` | float64 | Post creation time, seconds since epoch UTC. |
| `title` | string | Post title (raw). |
| `body` | string | Post selftext (raw). May be empty for link posts; we filter `is_self=True` by default so this is usually populated. |
| `author` | string\|null | Original Reddit username. **Stripped before downstream stages**; kept here for traceability only. Null if account deleted. |
| `score` | int64 | Net upvotes - downvotes at collection time. |
| `num_comments` | int64 | Comment count at collection time. |
| `permalink` | string | Path under reddit.com (no host). |
| `is_self` | bool | True if a text post (selftext-bearing). |
| `over_18` | bool | NSFW flag. |
| `source` | string | `'praw'` \| `'json_scraper'` \| `'dump'` \| `'synthetic'`. |
| `collected_at` | float64 | Unix timestamp of the moment this row was scraped. Useful for the methodology section. |

## Stage 2 — `data/interim/<subreddit>.parquet`

After `clean → anonymize → dedupe`. Same schema as raw, **with these changes**:

| column | type | change |
|---|---|---|
| `clean_text` | string | **NEW.** Title + body, encoding-fixed, markdown-stripped, URL-redacted, NER-redacted (PERSON/LOC/ORG), entity-decoded, whitespace-normalized. |
| `lang` | string | **NEW** (when language filter is on). Predicted ISO-639-1 code. |
| `author` | — | **REMOVED.** |
| `author_hash` | string | **NEW.** SHA-256 of `salt + author`, prefixed `u_`. Same author always maps to same hash. |

`_all.parquet` is a concat of all subreddit shards in this stage.

## Stage 3 — `data/processed/labeled.parquet`

All Stage-2 columns plus the labeling columns produced by `apply_weak_labels`, `label_corpus` (LLM), `annotate` (manual), and `aggregate_labels`.

### Tier-1 (weak) columns

| column | type | description |
|---|---|---|
| `weak_anxiety` | float64 | Probabilistic weak score in [0, 1]: subreddit_prior_weight × subreddit_prior + lexicon_weight × lex_score. |
| `weak_health_anxiety` | float64 | Same, for health-anxiety target. |
| `weak_depression` | float64 | Same. |
| `weak_suicidality` | float64 | Same. |
| `weak_<k>_bin` | int64 | 1 if `weak_<k> ≥ tier1_weak.thresholds[<k>]`, else 0. |

### Tier-2 (LLM) columns — present only on rows that were sent to the LLM

| column | type | description |
|---|---|---|
| `llm_anxiety` | int64 | 0 or 1. |
| `llm_anxiety_conf` | int64 | LLM-reported confidence 1–5. |
| `llm_<k>` / `llm_<k>_conf` | int64 | Same shape for the other 3 targets. |
| `llm_rationale` | string | ≤30-word free-text rationale from the LLM. Useful for error analysis. |

### Tier-3 (manual) columns — present only after `anxiety annotate`

| column | type | description |
|---|---|---|
| `annotator_id` | string | Free-form annotator identifier (used to compute κ). |
| `manual_anxiety` | int64 | 0 or 1. |
| `manual_<k>` | int64 | Same for the other 3 targets. |
| `manual_confidence` | int64 | 1–5 self-reported. |

### Aggregated columns (final labels, used for training)

| column | type | description |
|---|---|---|
| `label_<k>` | float64 | Final label for target `<k>`, picked from manual > llm > weak based on `aggregate.precedence`. |
| `label_<k>_source` | string | `'manual'` \| `'llm'` \| `'weak'` \| null. |
| `label_<k>_weight` | float64 | Confidence weight from `aggregate.tier_confidence`. Flows into the loss for confidence-weighted training. |

### Helper columns added at runtime

| column | type | description |
|---|---|---|
| `subreddit_group` | string | Group from `subreddits.yaml` (e.g. `anxiety_primary`, `health_anxiety_enriched`). Used for stratified sampling. |

## Stage 4 — `experiments/runs/<name>/{train,val,test}.parquet`

Same schema as Stage 3 — these are the splits used by `anxiety train`. Persisted alongside the model so `anxiety evaluate` can reproduce the same split.

## Stage 5 — `experiments/runs/<name>/eval/<model>__<target>__predictions.parquet`

Output of the evaluator. One row per test example.

| column | type | description |
|---|---|---|
| `id` | string | Reddit post ID. |
| `subreddit` | string |  |
| `label_<target>` | float64 | Ground-truth (final aggregated label). |
| `score_<target>` | float64 | Model probability output. |
| `pred_<target>` | int64 | Hard prediction at the chosen threshold (default = F1-optimal). |
| `bucket_<target>` | string | `'TP'` \| `'TN'` \| `'FP'` \| `'FN'`. |

## Other artifacts

| Path | Contents |
|---|---|
| `experiments/runs/<name>/eval/<model>__<target>__metrics.json` | Headline metrics + bootstrap CIs + ECE + chosen threshold. |
| `experiments/runs/<name>/eval/<model>__<target>__by_subreddit.csv` | Per-subreddit F1, P, R, n, n_pos. |
| `experiments/runs/<name>/eval/<model>__<target>__by_length.csv` | Equal-frequency length-bin F1. |
| `experiments/markers__<target>.csv` | Linguistic-marker comparison: Cohen's d, Mann-Whitney U, BH-corrected p-values. |
| `.cache/llm_labels.sqlite` | Tier-2 response cache. |
| `.cache/llm_zero_shot.sqlite` | Zero-shot model response cache. |
| `.cache/json_scraper.sqlite` | HTTP response cache for the no-creds scraper. |
