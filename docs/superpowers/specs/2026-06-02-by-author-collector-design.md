# By-author full-history collector — design

_Date: 2026-06-02 · Status: approved (brainstorm) · Branch: `feature/by-author-collector`_

## 1. Context & motivation

The corpus (`data/processed/labeled.parquet`, 743,881 posts / 38 subreddits) was collected
**by subreddit**, so for any given user we only have the posts they happened to make in the
scraped subreddits. Consequences, measured from the live data:

- **93.2% of the corpus is comments**, 6.6% submissions; median **~4 posts per test user**.
- Self-disclosure positives are small and imbalanced — user-level: anxiety 348, depression 807,
  **health_anxiety 186** (107 with ≥3 posts). Health anxiety, the focal class, is the bottleneck.
- The user-level disclosure test set (`disclosure_testset__users.csv`) holds **3,943 users**
  (1,323 disclosed positives + 2,620 matched controls), all `held_out_split=True`.

The richest lever for both *signal* (more posts per user → better user-level aggregation) and the
class imbalance is to fetch each test user's **full Reddit history**. Crucially, although
preprocessing dropped the `author` column, **`data/raw/*.parquet` still contains `author`**, and by
recomputing the pipeline's salted hash we recover **1,322 / 1,323** disclosed usernames (1 lost to a
deleted account); controls recover the same way.

## 2. Goal & non-goals

**Goal:** Enrich the full post + comment history of the **3,943 disclosure-test users (both
classes)**, re-detect disclosures on the richer histories, re-group the cohort (promoting controls
who turn out to have disclosed), retrain a fresh TF-IDF baseline, and re-run the user-level
evaluation to see how the honest metrics move.

**Non-goals (explicitly out of scope for this work):**
- The user-level *training-label* redesign (using disclosed users as training positives / splitting
  them across train/test). Disclosed users remain fully held out here; enrichment changes the
  **evaluation** set, not the training set.
- Retraining the transformer / multi-task / XGBoost models (GPU, long). Baseline TF-IDF only, to see
  directional values quickly.
- Fetching histories for users outside the 3,943-user cohort.
- Reworking whether `data/raw/` belongs in git (only `data/raw/authors/` is addressed).

## 3. Decisions locked in brainstorming

| # | Decision | Choice |
|---|---|---|
| Scope | Whose histories | **Both classes** — all 3,943 test users |
| Depth | Content + cap | **Submissions + comments, uncapped** (to Reddit's ~1,000/listing cap) |
| Reclassify | Control reveals a disclosure | **Promote to positive** (re-detect + re-group) |
| Execution | Pilot vs full | **Full send** (no pilot) → resumability is a hard requirement |
| Approach | Collection mechanism | **A — no-auth public JSON, standalone CLI command** |

Power-user skew (≥1 user has 2,722 posts) is handled at **aggregation time** (top-k / sampling),
not by capping collection.

## 4. Architecture & components

All new code is small and isolated.

### 4.1 `src/collection/author_history.py`
- `recover_author_usernames(users_df, raw_dir="data/raw") -> dict[author_hash, username]`
  Rebuilds the salted-hash → username map from `data/raw/` (recomputing
  `preprocessing.anonymize._hash_username` on raw `author`), then looks up the cohort's hashes.
  Skips deleted/`[removed]`/null authors.
- `AuthorHistoryCollector(JsonScraperCollector)`
  Inherits the polite User-Agent, `_get_json` with 429/5xx backoff, and `SqliteCache`.
  Adds `collect_user(username) -> Iterator[RedditPost]`, paging
  `{BASE}/user/<name>/submitted.json` (kind `t3`) and `{BASE}/user/<name>/comments.json`
  (kind `t1`) via the `after` cursor up to `max_pages_per_listing` (~1,000 items/listing).
  Each record maps to `RedditPost` with `source="author_history"`, correct `kind`/`parent_id`,
  `subreddit` = whatever the post's subreddit is (arbitrary, Reddit-wide). Applies the same
  `passes_filters`. 403/404 (suspended/deleted/private) → skip + cache-negative.

### 4.2 `anxiety collect-authors` (new CLI command in `src/cli.py`)
- Reads the cohort from `data/processed/disclosure_testset__users.csv` (default; `--users-csv`
  override), recovers usernames, scrapes each user.
- Writes **one parquet per user named by `author_hash`** (never the username) under
  `data/raw/authors/` (`--out-dir` override).
- **Resumable two ways:** skip users whose `<hash>.parquet` already exists; page-level HTTP cache
  (`.cache/author_history.sqlite`) skips already-fetched pages.
- Rich progress bar over users; logs recovered/total, skipped (unrecoverable / 404), per-user yield.
- Flags mirror the scraper: `--request-interval`, `--max-pages`, `--out-dir`, `--users-csv`,
  `--raw-dir`.

### 4.3 `rebuild_groups_within_cohort(df, cohort_hashes, targets)` in `src/labeling/disclosure_dataset.py`
The methodology piece. Instead of letting `build_disclosure_test_users` recruit *new* controls from
the thin (un-enriched) corpus, re-group **only within the enriched cohort**:
- positives = cohort users now disclosed for any target (re-detected on full histories),
- controls  = cohort users still never disclosed even with full history.
This keeps both classes depth-comparable and honors the "both classes balanced" choice. Produces the
same user-level schema (`user_<target>`, `user_group`, `n_posts`, `subreddits`) so
`materialize_test_posts` / `mark_held_out` / `evaluate_user_level` are reused unchanged.

## 5. Data flow ("re-run all")

```
collect-authors  →  data/raw/authors/<hash>.parquet      (full histories, with `author`)
        ↓  preprocess (clean → anonymize → dedupe; --raw-dir data/raw/authors --out-dir data/interim/authors)
   data/interim/authors/*  →  merge with existing data/interim/* + global exact-MD5 dedup
        ↓  label --tier weak ; label --tier disclosure     (re-detect disclosures on enriched text)
   labeled.parquet (enriched)
        ↓  rebuild_groups_within_cohort  →  enriched disclosure_testset.parquet + __users.csv
        ↓  train tfidf (excludes held_out_split)  →  eval-disclosure
   fresh user-level values (with / without disclosure-masking)
```

Re-fetching a user pulls posts we already had → removed by the existing **global exact-MD5 dedup**
when the merged corpus is deduplicated. Everything after `collect-authors` is orchestration of
existing commands plus the merge/dedup step and the cohort re-group; the only genuinely new logic is
the collector, username recovery, and `rebuild_groups_within_cohort`.

## 6. Ethics & anonymization

- Enriched histories pass through the **existing** preprocessing unchanged: always-on regex
  redaction + spaCy NER, `author` → salted hash, `author` dropped. The processed corpus stays
  anonymized exactly as today.
- **Required:** add `data/raw/authors/` and `.cache/author_history.sqlite` to `.gitignore`. The
  per-user raw shards and the HTTP cache contain **real usernames + full mental-health histories**;
  they must not be committed. (`data/raw/*.parquet` is currently git-tracked — that broader posture
  is noted but out of scope.)
- Fetching full per-user histories for mental-health profiling is more invasive than subreddit
  scraping. `docs/ethics.md` exists; IRB/ethics-board alignment and Reddit ToS compliance remain the
  author's responsibility. Flagged, not blocked.

## 7. Error handling

- 403/404 (suspended / deleted / private account) → skip user, cache-negative, log, continue.
- Network / 5xx → bounded retries (inherited); 429 → uncapped polite backoff honoring `Retry-After`
  (inherited).
- Resume: existing `<hash>.parquet` + page-level HTTP cache → a crashed multi-hour run continues.
- Unrecoverable username (1 deleted positive) or a fully unfetchable user → keep their original thin
  history (graceful degradation), logged so the count is visible.
- Empty history → write nothing for that user, log.

## 8. Testing (no live network)

`tests/test_author_history.py`, mirroring `tests/test_json_scraper.py`:
- Mock `_session.get` with fake `submitted.json` + `comments.json` pages → assert both kinds
  yielded, `source="author_history"`, `parent_id` set, pagination via `after`, cross-listing dedup.
- 404/403 user → yields nothing, no crash.
- `recover_author_usernames` → correct hash→username mapping on synthetic raw; deleted author
  skipped.
- `rebuild_groups_within_cohort` → a former control that discloses in enriched history is promoted to
  positive; no external controls recruited; user-level schema/counts correct.

## 9. Success criteria

- `anxiety collect-authors` produces resumable per-user shards for the recoverable cohort users.
- After re-run: enriched test set with **median posts/user ≫ 4**; positive count **grown vs 1,323**;
  fresh user-level disclosure values (with / without masking) emitted.
- No usernames committed to git; full test suite passes offline.
