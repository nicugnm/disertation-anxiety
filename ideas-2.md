# Ideas-2: Whisper-paradigm brainstorm (pre-research synthesis)

This is the initial brainstorm produced from domain knowledge alone — before the deep web-research sweep that became `ideas-1.md`. It maps the Whisper paradigm onto the dissertation's tier-1/2/3 labeling architecture and proposes a full landscape of upgrades.

---

This is a great direction — your friend's Whisper reference is more apt than they might realize. Whisper's main thesis was *"scale + diversity of weak supervision beats careful clean-label training"* — which maps directly onto your tier-1/tier-2/tier-3 architecture, just dialed way up. Let me lay out a full landscape and then collapse it to what I'd actually do if I were you.

---

## 1. The Whisper paradigm, translated to your problem

Whisper trained on 680,000 hours of paired audio+transcripts scraped from the web with no manual labeling — 650× more data than previous SOTA. Three key ideas they relied on:

| Whisper idea | What they did | Direct analog for your project |
|---|---|---|
| **Scale weak labels** | Web-scraped audio + closed captions | Scrape 100k–1M Reddit posts, LLM-label all of them with Claude (your tier-2). The thesis already has the scaffold — just blow up `tier2_llm.max_posts` from 8k → 100k+ |
| **Filter, don't perfect** | Drop low-quality (audio, transcript) pairs aggressively, never edit labels | After LLM labels, train a small "label quality" classifier on (weak vs LLM) agreement. Drop or down-weight the noisy fraction. Don't try to fix labels — drop them. |
| **Multitask single forward pass** | One model does ASR + VAD + language ID + translation via task tokens | Your multitask transformer already does this for 4 binary labels. Whisper-style next step: add *auxiliary* tasks (sentiment, subreddit, post length bucket, severity regression) to the same shared encoder. Auxiliary tasks regularize the rare-class heads. |
| **Diversity over precision** | 96 languages, many domains, no benchmark-tuning | Expand from 10 to 30+ subreddits including r/HealthAnxiety (already in your `configs/subreddits.yaml` commented out), r/HypochondriacsUnited, r/ContaminationOCD, r/ROCD, r/PanicAttack, r/PTSD, r/Mentalillness, r/Mentalhealth, r/SocialAnxiety, etc. Add cross-platform: HuggingFace's `Mental-Health-FineTuning-Data` mirrors. |
| **Zero-shot generalization** | No fine-tuning on benchmark test sets | You're already partway there with the cross-subreddit transfer experiment. Add a cross-corpus eval against CLPsych/eRisk (next section). |

Concretely, a Whisper-style training pipeline for you would be:

```
Stage 0 (DOMAIN MLM): continue masked-language-model pretraining on
        your raw 16k posts before any fine-tuning. ~1 hour on the 4090.
        Free F1 bump of 1-3 points typically.

Stage 1 (NOISY): collect 100k+ posts → LLM-label all → train multi-task
        transformer with soft labels (LLM confidence as target) and per-
        row weights.

Stage 2 (TEACHER-FILTERED): use Stage 1 model to score the LLM labels.
        Drop or downweight posts where the model strongly disagrees with
        the LLM. Retrain. (Whisper's "data filtering" step.)

Stage 3 (GOLD FINE-TUNE): fine-tune Stage 2 on your tier-3 gold standard
        (the 1000 human-annotated posts) with low LR. Tiny dataset,
        large effect.
```

This is the single biggest possible methodological upgrade — and the framework is already in your repo, just running on 8k posts instead of 100k.

---

## 2. New datasets — clinically labeled + public

Eight datasets that are downloadable and clinician/expert-labeled for mental-health NLP. I'd seriously consider integrating 1–2 of them.

| Dataset | Labels | Size | Why it's useful for you |
|---|---|---|---|
| **eRisk** (CLEF 2017–2023, Losada et al.) | Depression, anorexia, self-harm, gambling addiction — clinician-validated | ~2k users, longitudinal | **Best fit.** Reddit; anorexia ≈ health-anxiety-on-body which dovetails with your SHAI angle. Public, downloadable. Strong cross-corpus eval. |
| **CLPsych Shared Task 2015/2019/2022** (Coppersmith, Shing et al.) | Depression, PTSD, suicidality (4-level risk grading) | ~1.5k users | Gold-standard clinician annotations. The 2019 UMD suicidality subset has 4-level severity, which would let you frame suicidality as ordinal not binary. |
| **RSDD** (Yates, Cohan, Goharian 2017) | Self-reported depression diagnosis (regex-extracted from posts) | 9,210 dep + 107k control users | Massive matched controls. Used by 100+ papers. Good for pre-training. |
| **SMHD** (Cohan et al. 2018) | 9 conditions (anxiety, depression, bipolar, PTSD, OCD, autism, schizophrenia, eating disorder, ADHD) | 350k users | Multi-condition. Useful if you ever want to add comorbidity analysis. |
| **DAIC-WOZ** (Gratch et al. 2014, USC) | PHQ-8 by clinicians + transcripts + audio | 189 interviews | Adds a clinical-interview modality. Could be a chapter on "Reddit ≠ clinical setting" with concrete evidence. |
| **Reddit Mental Health Dataset** (Low et al. 2020) | Subreddit-membership across 28 mental-health subs, pre/post COVID windows | 800k posts | Public, already split by time periods. Drop-in replacement / supplement to your collection — and has the COVID windows pre-cut for your RQ4. |
| **Mental Health FAQ + 7Cups peer counseling** (Sharma et al. 2020, Empathic Replies) | Peer counselor responses, labeled for empathy/effectiveness | ~10k conversations | Lets you extend the thesis toward intervention/response generation, not just detection. |
| **GoEmotions** (Demszky et al. 2020) | 27 fine-grained emotions, Reddit comments | 58k comments | Excellent auxiliary task. Co-train your transformer on GoEmotions + your 4 targets — emotion taxonomy regularizes the affect detection. |

**Health-anxiety specifically — what you can do today without external data:**
- Uncomment `r/HealthAnxiety` in `configs/subreddits.yaml` and re-collect. That single subreddit will likely contribute 1–5k posts and dramatically grow your `health_anxiety_enriched` group.
- Add `r/HypochondriacsUnited`, `r/Hypochondria`, `r/ContaminationOCD`, `r/ROCD`. These are smaller but high-purity.

**Clinician-labeled bespoke dataset (highest effort, highest payoff):**
- A formal collaboration with 2 clinical psychologists annotating 500–1000 posts against SHAI item content. ~30 hours of clinician time. Would let you publish the dataset itself as a thesis contribution (cite-able, reusable).
- Cost equivalent: ~$5/post on Prolific to clinically-trained MTurk workers. ~$5k for 1000 posts.

---

## 3. Labeling-strategy upgrades

Beyond just "more tier-2 labels", several ideas to make the tier-2 step itself smarter:

1. **Multi-LLM ensemble labeling.** Run Claude AND GPT-4 AND a small open model (Llama-3) on the same prompts, take majority vote. Disagreements become tier-3 candidates. Reduces single-model bias and gives an inter-LLM kappa to report.

2. **Chain-of-thought labels with structured rationale.** Currently tier-2 returns binary + 1-line rationale. Instead, ask the LLM to return **per-SHAI-item** scores (catastrophic interpretation, reassurance-seeking, body scanning, fear of disease, dread of appointments). You then train a multi-task head per SHAI item — gives both richer training signal AND interpretability.

3. **Active learning for tier-3.** Right now `manual.py` stratifies by `subreddit_group`. Better: use the trained multi-task transformer's *uncertainty* (entropy of sigmoid output) to pick the 1000 most-uncertain posts. The thesis literature consistently shows 2–3× value per annotated example from active learning vs random sampling.

4. **Snorkel-style labeling functions.** Beyond the lexicon, add many noisy weak labelers (regex rules, sentence-level sentiment, subreddit prior, BERT embedding similarity to SHAI items) and combine them via a generative model that learns each labeler's accuracy. This is the standard upgrade from your current "subreddit prior + lexicon" combiner. Public library: `snorkel`.

5. **Confidence-weighted soft labels.** Currently aggregate.py picks `manual > llm > weak` and uses a tier-level confidence weight. Better: use the LLM's *per-example* confidence (1-5) as a soft label in [0, 1], not a hard binary. Train with BCE-with-logits where the target is the soft label. Loses the threshold, keeps the gradient.

6. **Counter-example mining.** For each positive, find the *minimal text edit* that flips the label (LLM-generated). Negate phrases, add disclaimers, change tense. These hard negatives prevent the model from over-relying on keywords.

7. **Synthetic minority-class generation.** Use Claude to *generate* health-anxiety posts conditioned on each SHAI item. 500 synthetic + 100 real ≫ 100 real alone for training (with the obvious caveat about train-test contamination — easy to control by generating from DSM-5 items not from your test set).

---

## 4. Model / architecture ideas

| Idea | Effort | Expected impact |
|---|---|---|
| **Domain MLM pretraining**: continue MLM on raw Reddit corpus 1 epoch before fine-tuning | Low | +1–3 F1 typically |
| **DeBERTa-v3-base** instead of MentalRoBERTa-base | Low | +1–2 F1, faster |
| **Larger model**: MentalRoBERTa-large / Llama-3-8B with LoRA | Medium | +2–4 F1 on rare classes typically |
| **MentalLLaMA** (Yang et al. 2023): instruction-tuned for mental health | Medium | Direct fit, strong zero-shot |
| **Severity regression head**: predict GAD-7/SHAI score from text in addition to binary | Medium | More clinically actionable thesis |
| **Hierarchical attention**: sentence → post pooling for long posts | Medium | Helps very long posts; minor average effect |
| **Adapter / LoRA fine-tuning**: per-target adapters sharing one frozen base | Medium | Parameter-efficient, lets you fit 10+ targets |
| **Per-subreddit threshold calibration**: pick threshold per group to hit target precision | Low | Directly addresses your RQ3 finding (cross-sub F1 collapse) |
| **Domain-adversarial training**: gradient reversal on subreddit predictor → subreddit-invariant features | Medium | Forces the model to learn semantic anxiety, not subreddit style. *Very* on-thesis for RQ3. |
| **Retrieval-augmented classification**: at inference, retrieve k nearest labeled examples and concatenate to input | High | Big help for rare classes |
| **Mixture-of-experts**: per-condition LoRA experts gated by a router | High | PhD-scale; probably overkill |

The two single highest-value architecture moves for your specific thesis:
- **Domain-adversarial training** — it makes the cross-sub transfer finding bulletproof methodologically (you'd show that the standard model leaks subreddit style, and your debiased model doesn't, while AUROC is preserved).
- **Severity regression head** — turns the health-anxiety story from "we classify it" to "we predict its severity on the SHAI scale", which is far more clinically defensible.

---

## 5. Evaluation strengthening

Several high-leverage methodology hardenings:

1. **McNemar's test or paired bootstrap** for "Model A vs Model B" claims. Currently the README says "XGBoost beats TF-IDF" without statistical evidence. McNemar is two lines of scipy.
2. **Per-subreddit threshold calibration** as a *deployment* recommendation in the discussion chapter — and report calibrated F1 alongside the global-threshold F1.
3. **Counterfactual robustness**: take 100 high-confidence predictions, perturb the text (swap synonyms, add disclaimers, negate the affect verb), check how many predictions flip. A robust model flips <10%; a keyword-dependent model flips 50%+.
4. **Adversarial validation**: train a classifier to distinguish r/Anxiety from r/relationship_advice using only your linguistic features. If that classifier is highly accurate AND its top features overlap with your anxiety classifier's top features, you've identified confounding features (subreddit-style markers that look like anxiety markers).
5. **Fairness audit**: infer author gender/age from text (https://github.com/SaffronMarchant/age-gender-from-text) and report F1 per inferred demographic bin. Highlights bias.
6. **External-corpus zero-shot eval**: trained on your Reddit corpus, run it on eRisk or CLPsych without any fine-tuning. The F1 there is the *real* generalization number.
7. **Calibration improvements**: temperature scaling (Platt) on top of the TF-IDF logits to fix the ECE 0.13. ~10 lines of code.
8. **Bootstrap CIs in `run_experiments.py`** (already on your radar).

---

## 6. Beyond classification — new thesis angles

If you want to *expand* the thesis rather than just polish RQ1–RQ4, here are higher-ambition directions:

1. **Severity prediction**: predict numeric GAD-7 / SHAI / PHQ-9 scores from post text. Trained on (post, self-reported-score) pairs from r/Anxiety users who post their scores in their posts (mine regex). More clinically meaningful than binary.
2. **Longitudinal user trajectories**: per-user sequence model over their post history. Detect onset, relapse, recovery. Adds an entire dimension to RQ4 (temporal).
3. **Symptom-level decomposition**: instead of one binary health-anxiety label, predict each of the 14 SHAI items separately. Each is its own classification head. The thesis becomes "we decompose health anxiety into its clinical constituents."
4. **Counter-speech / helpful-response generation**: given an anxious post, generate a clinically-appropriate peer response. Train on 7Cups data. Becomes a full system, not just a classifier.
5. **Span-level explanation**: train the model to highlight the specific phrases driving its prediction. Useful for clinical handoff. (Doable cheap with attention rollout or LIME.)
6. **Pre/post intervention impact**: for users who received a supportive reply, did their subsequent posts decrease in anxiety markers? Causal inference angle, very novel.
7. **Cross-cultural / multilingual extension**: extend to Romanian, Spanish, or any language with a mental-health forum. Whisper-style multilingual is a *huge* thesis pivot but very tractable for one extra language using XLM-R as the encoder.

---

## 7. If I were you, top 5 priorities (ranked)

Given you're one person, finite time, want this to defend well, and have an RTX 4090 + Anthropic API access:

| # | Move | Effort | Expected thesis impact |
|---|---|---|---|
| **1** | **Tier-2 LLM labeling at Whisper scale (50k–100k posts)** + filter-by-disagreement → retrain multi-task transformer. Single biggest fix for the health-anxiety F1 = 0.33 ceiling and the cleanest Whisper-style story. | 1 day setup + ~$50–200 API + 1 GPU-hour | ★★★★★ |
| **2** | **Add eRisk or CLPsych as external zero-shot test set.** Demonstrates your model generalizes off-Reddit / cross-corpus. Single strongest RQ3 result. | 2 days (download + reformat + eval) | ★★★★ |
| **3** | **Domain MLM pretraining + severity regression head.** Pre-train MentalRoBERTa on your raw 16k posts 1 epoch, then add a SHAI-score regression head trained on r/HealthAnxiety self-report posts. Turns "binary classifier" into "clinical instrument approximator." | 3 days | ★★★★ |
| **4** | **Domain-adversarial training (gradient reversal on subreddit ID).** Makes the cross-sub transfer story bulletproof. Cleanest novel methodological contribution for RQ3. | 2 days (write a custom Trainer subclass) | ★★★★ |
| **5** | **Active-learning gold standard** — pick the 1000 manual-annotation posts using model uncertainty instead of stratified random. 2–3× value per annotation. | 1 day code + the same annotation time | ★★★ |

What I'd *skip* for a master's thesis (PhD-scale or marginal):
- Mixture of experts / large LLM fine-tuning beyond DeBERTa-large
- Multi-LLM ensemble (Claude + GPT-4 + Llama) — premise good, but cost/effort not justified at thesis stage
- Cross-platform Twitter extension (Twitter API access has gotten horrible)
- Pre/post intervention causal inference — beautiful but PhD-thesis-scale
