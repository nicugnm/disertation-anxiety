# Label definitions (codebook)

Version 0.1. This codebook defines the conceptual constructs that underpin both the weak lexicon labels and the self-disclosure regex patterns used in this work. The decision rules below are the **operationalization** of each label: they inform which terms appear in the lexicons (`src/labeling/lexicons.py`) and which first-person patterns count as a valid self-disclosure (`src/labeling/self_disclosure.py`).

## Labels

Each post is assigned **four binary labels**.

| Label | Definition |
|---|---|
| `anxiety` | Author expresses present-day anxious affect, anxious cognition, or anxious physiological experience about *any* topic. |
| `health_anxiety` | Anxiety is specifically about one's own (or a loved one's) physical health, illness, symptoms, medical procedures, or fear of disease. Implies the `anxiety` label. |
| `depression` | Author expresses depressive symptoms (anhedonia, hopelessness, persistent low mood, worthlessness). |
| `suicidality` | Author expresses suicidal ideation, intent, plan, or recent attempt. |

Posts may have any combination. A post entirely about a third party (asking for advice on someone else) is labeled `0` for the affect labels but may still be labeled by topic for analysis.

## Decision rules

These rules define what the labels mean. They are used to:
- select and justify the terms in the lexicons (weak labeling), and
- design the self-disclosure regex patterns and filters.

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

Borderline cases — label `health_anxiety=1` if **3 or more** of the following are present even if affect is restrained:
- Mention of specific feared disease(s) by name.
- Multiple symptoms listed in checklist style.
- Explicit anticipation of medical appointments with dread.
- Description of intrusive health-related thoughts.

Distinguish from:
- **Acute illness reports** (post-COVID symptoms, recovery questions) → not health anxiety unless distress is disproportionate / persistent.
- **Caregiving worry** about a loved one's diagnosis → label `health_anxiety=1` only if author expresses persistent disproportionate fear, not normative concern.

### Depression

Positive: anhedonia, persistent low mood ≥2 weeks (or unspecified duration with severity), hopelessness, worthlessness, suicidal ideation (also triggers `suicidality`), sleep/appetite/energy disturbance attributed to mood.

Negative: situational sadness without symptom cluster, grief without symptoms beyond expected.

### Suicidality

Positive: any expression of wanting to die, planning self-harm, recent attempt, or "I won't be here much longer" type statements.

**Note:** self-disclosure detection for `suicidality` is **disabled** in `src/labeling/self_disclosure.py` — suicidal ideation is not typically self-disclosed as a clinical diagnosis and carries a high false-positive risk. Only weak labels are produced for this target.
