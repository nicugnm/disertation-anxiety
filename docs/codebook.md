# Annotation codebook

Version 0.1 — to be refined after pilot annotation. This codebook is the basis of the gold-standard test set and a contribution of the dissertation.

## Labels

Each post receives **four binary labels** plus a confidence score 1–5.

| Label | Definition |
|---|---|
| `anxiety` | Author expresses present-day anxious affect, anxious cognition, or anxious physiological experience about *any* topic. |
| `health_anxiety` | Anxiety is specifically about one's own (or a loved one's) physical health, illness, symptoms, medical procedures, or fear of disease. Implies the `anxiety` label. |
| `depression` | Author expresses depressive symptoms (anhedonia, hopelessness, persistent low mood, worthlessness). |
| `suicidality` | Author expresses suicidal ideation, intent, plan, or recent attempt. |

Posts may have any combination. A post entirely about a third party (asking for advice on someone else) is labeled `0` for the affect labels but may still be labeled by topic for analysis.

## Decision rules

### Anxiety (general)

Positive markers (any one is sufficient when explicit):
- First-person statement of feeling anxious / nervous / on edge / panicking ("I can't stop worrying", "my chest is tight", "I had a panic attack").
- Description of present worry-loops, catastrophizing, or intrusive thoughts.
- Avoidance behaviors framed as driven by fear ("I haven't left the house in...").
- Physiological symptoms attributed by the author to anxiety (racing heart, dissociation).

Negative markers (label `0` even if anxiety topic is mentioned):
- Strictly informational ("Here's a study about anxiety...").
- Asking on behalf of another with no first-person affect.
- Past-tense recovered narrative ("I used to have anxiety, here's what worked") — unless the post also contains current anxious affect.

### Health anxiety (subtype — the hard case)

Positive markers — health anxiety requires **(a) anxious affect** AND **(b) health/illness focus**:

(a) anxious affect — same as general anxiety
(b) health focus — at least one of:
- Catastrophic interpretation of bodily sensation ("I felt a twinge in my chest, am I having a heart attack?").
- Reassurance-seeking ("Please tell me this isn't cancer").
- Repeated medical checking — describing multiple doctor visits, googling symptoms, body-scanning.
- Persistent fear of having or developing a serious illness despite medical reassurance.
- Symptom monitoring with disproportionate distress.

Borderline cases — annotate `health_anxiety=1` if **3 or more** of the following are present even if affect is restrained:
- Mention of specific feared disease(s) by name.
- Multiple symptoms listed in checklist style.
- Explicit anticipation of medical appointments with dread.
- Description of intrusive health-related thoughts.

Distinguish from:
- **Acute illness reports** (post-COVID symptoms, recovery questions) → not health anxiety unless distress is disproportionate / persistent.
- **Caregiving worry** about a loved one's diagnosis → annotate as `health_anxiety=1` only if author expresses persistent disproportionate fear, not normative concern.

### Depression

Positive: anhedonia, persistent low mood ≥2 weeks (or unspecified duration with severity), hopelessness, worthlessness, suicidal ideation (also triggers `suicidality`), sleep/appetite/energy disturbance attributed to mood.

Negative: situational sadness without symptom cluster, grief without symptoms beyond expected.

### Suicidality

Positive: any expression of wanting to die, planning self-harm, recent attempt, or "I won't be here much longer" type statements.

**Annotator safety:** if a post triggers strong distress, skip with `flag=skip`. The annotation interface tracks skipped posts separately.

## Confidence scale

- **5** — unambiguous, multiple explicit markers
- **4** — clear, single strong marker
- **3** — present but understated
- **2** — borderline; depends on charitable reading
- **1** — guess

In analysis we report results both on the full set and on confidence ≥ 4 only.

## Inter-annotator agreement target

- Cohen's κ ≥ 0.70 on `anxiety`
- Cohen's κ ≥ 0.60 on `health_anxiety` (harder, expected lower)
- Cohen's κ ≥ 0.65 on `depression`
- Cohen's κ ≥ 0.75 on `suicidality` (high stakes, expected high agreement)

Below these thresholds: refine the codebook and re-annotate.

## Procedure

1. Annotators independently label a calibration set of 50 posts.
2. Compute κ; review disagreements; refine codebook.
3. Annotate the gold test set (target: 1000 posts, stratified by subreddit).
4. Adjudicate disagreements via discussion; for irresolvable cases, drop from test set.
5. Final test set published with majority label + per-annotator labels for transparency.
