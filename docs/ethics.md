# Ethics statement

This project conducts NLP research on Reddit posts discussing anxiety, depression, and suicidality. Although the data is publicly accessible, mental-health content from identifiable users requires heightened care. This document codifies the ethical commitments enforced by the pipeline.

## 1. Legal and platform basis

- **Reddit Data API Terms** (effective 2023, updated since) prohibit redistribution of raw user content. We comply by:
  - Releasing **post IDs + derived labels + aggregated statistics only**, never raw text dumps.
  - Documenting collection method and date in every released artifact so others can re-derive the corpus from a current Reddit fetch.
- All collection respects the API's rate limits and the platform's `robots.txt` for any scraping fallback.
- **No use of suspended/deleted/removed content** beyond what the API exposes; we drop posts marked `[deleted]` or `[removed]` during preprocessing.

## 2. Anonymization (enforced by pipeline)

`src/preprocessing/anonymize.py` applies these transforms before any model sees the data:

1. Drop `author` field after retention checks; replace with hashed pseudonym.
2. Strip URLs, email addresses, phone numbers, and `u/username` mentions.
3. Replace named entities tagged `PERSON` and `GPE` (city/country) with placeholder tokens (`[PERSON]`, `[LOC]`) when present in user-provided personal details.
4. Discard posts under a minimum length threshold that are too short to anonymize meaningfully.
5. Subreddit-of-origin is retained as a categorical feature but never linked back to a user across posts in published outputs.

## 3. Sensitive-content handling

- **r/SuicideWatch** posts are included for *training* the comorbidity head only. They are excluded from any released sample, demo, or quoted excerpt in the thesis. Any quoted phrase in writing is paraphrased or strictly anonymized.
- **No verbatim quotations** of user content longer than 8 tokens appear in the thesis without explicit paraphrase.
- The annotation interface displays a crisis-resource banner and supports skipping distressing posts.

## 4. IRB / Ethics committee

- Public-data social-media research often qualifies as exempt under most university IRB frameworks, but **the student MUST verify with their institution before publishing**.
- This file is a stand-in for an IRB protocol summary; replace with the actual approval reference when obtained.

## 5. Crisis resources

Any deployed artifact (web demo, notebook, README) using this model must surface, as the first thing the reader sees:

> If you are in crisis or thinking about harming yourself:
> - **US:** Call or text **988** (Suicide & Crisis Lifeline).
> - **UK & ROI:** Samaritans **116 123**.
> - **EU:** [list of national lines](https://www.befrienders.org/).
> - **International directory:** https://findahelpline.com/

## 6. What this model is *not*

- **Not a diagnostic instrument.** It is a research artifact for studying language. Any clinical claim requires controlled, IRB-approved validation against gold-standard instruments (SHAI, GAD-7, HAI) administered by clinicians. The thesis must state this prominently in the abstract, introduction, and conclusion.
- **Not a screening tool for surveillance.** The model must not be deployed to flag individual users without their explicit, informed consent.

## 7. Bias considerations

- **Linguistic bias:** English-only; underrepresents non-Western expressions of distress.
- **Population bias:** Reddit users skew young, male, technologically literate, Western. Findings do not generalize to general populations.
- **Self-selection bias:** Posts in r/Anxiety come from people willing to discuss anxiety online. This is not equivalent to clinical anxiety.
- These limitations are reported transparently in every results table and discussed in the thesis Limitations chapter.

## 8. Data retention

- Raw collected data lives in `data/raw/` on the researcher's machine only.
- After thesis defense, raw data is destroyed; processed labels (without text) are retained as a reproducibility artifact.
- No raw data is committed to git. `.gitignore` enforces this; reviewers are expected to verify before any push.
