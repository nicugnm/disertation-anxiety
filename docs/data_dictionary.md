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

All Stage-2 columns plus the labeling columns produced by `apply_weak_labels` (Tier-1) and `apply_disclosure_labels` (Tier-2 self-disclosure), followed by `aggregate_labels`.

### Tier-1 (weak) columns — always present

| column | type | description |
|---|---|---|
| `weak_anxiety` | float64 | Probabilistic weak score in [0, 1]: `subreddit_prior_weight × subreddit_prior + lexicon_weight × lex_score`. |
| `weak_health_anxiety` | float64 | Same, for health-anxiety target. |
| `weak_depression` | float64 | Same. |
| `weak_suicidality` | float64 | Same. |
| `weak_<k>_bin` | int64 | 1 if `weak_<k> ≥ tier1_weak.thresholds[<k>]`, else 0. |

### Tier-2 (self-disclosure) columns — always present

Produced by `src/labeling/self_disclosure.py`: regex diagnosis phrases filtered by negation, hypothetical, third-party, and denial checks. Suicidality patterns are intentionally empty (no `disclosure_suicidality` positives).

| column | type | description |
|---|---|---|
| `disclosure_anxiety` | int64 | 1 if a verified first-person anxiety diagnosis disclosure was detected; 0 otherwise. |
| `disclosure_health_anxiety` | int64 | Same, for health-anxiety. |
| `disclosure_depression` | int64 | Same, for depression. |
| `disclosure_suicidality` | int64 | Always 0 — suicidality self-disclosure is disabled by design. |
| `disclosure_<k>_match` | string\|null | The matched substring that triggered the disclosure flag (for traceability / audit). Null when `disclosure_<k>=0`. |

### Aggregated columns (final labels, used for training)

| column | type | description |
|---|---|---|
| `label_<k>` | float64 | Final label for target `<k>`, resolved with precedence disclosure > weak (see `aggregate.py`): `disclosure_<k>` when the disclosure flag is 1, otherwise `weak_<k>_bin`. |
| `label_<k>_source` | string | `'disclosure'` or `'weak'`. Null only if neither tier produced a label. |
| `label_<k>_weight` | float64 | Confidence weight from `aggregate.tier_confidence` (disclosure = 0.85, weak = 0.5 by default). Flows into the loss for confidence-weighted training. |

### Held-out split column — present after `build-disclosure-testset`

| column | type | description |
|---|---|---|
| `held_out_split` | bool | True for every post authored by a user in the disclosure test set (positives + matched controls). These rows are excluded from training by default. |

### Helper columns added at runtime

| column | type | description |
|---|---|---|
| `subreddit_group` | string | Group from `subreddits.yaml` (e.g. `anxiety_primary`, `health_anxiety_enriched`). Used for stratified sampling. |

## Stage 3b — `data/processed/disclosure_testset.parquet`

Posts by test users (disclosed positives + subreddit-matched controls), produced by `anxiety build-disclosure-testset`. Contains all Stage-3 columns plus:

| column | type | description |
|---|---|---|
| `user_anxiety` | int64 | 1 if the post's author ever disclosed anxiety (user-level label). |
| `user_health_anxiety` | int64 | Same, for health-anxiety. |
| `user_depression` | int64 | Same, for depression. |
| `user_group` | string | `'disclosed_<target>[+<target>...]'` for positives or `'matched_control'` for controls. |
| `is_disclosure_post` | int64 | 1 if this specific post is one of the disclosure utterances. Lets evaluators mask them and report both "full history" and "implicit-signal-only" metrics. |

`data/processed/disclosure_testset__users.csv` — one row per test user (`author_hash`, `user_<target>`, `user_group`, `n_posts`, `subreddits`).

## Stage 3c — `experiments/runs/<name>/eval/<model>__<target>__disclosure_userlevel.json`

Output of `anxiety eval-disclosure`. User-level metrics (precision, recall, F1, AUROC) on the disclosure test set, including results with `mask_disclosure_posts=True` (implicit-signal-only evaluation).

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
| `.cache/json_scraper.sqlite` | HTTP response cache for the no-creds scraper. |
