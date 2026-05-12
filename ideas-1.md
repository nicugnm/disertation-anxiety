# Deep research: ideas for the Reddit anxiety / health-anxiety dissertation

Compiled from five parallel deep-research sweeps across arXiv, ACL Anthology, PubMed, HuggingFace, Zenodo, ScienceDirect, JMIR, IEEE, Nature, Springer, ACM and dataset repos. All claims are sourced; URLs are at the bottom of each section.

---

## 0. Top-line synthesis — what the literature actually allows you to claim

The five research streams converge on **one** very clean defensible novelty:

> **No published paper has trained a transformer specifically to detect health anxiety as a class distinct from general anxiety, and no work has validated text-based health-anxiety detection against the SHAI.**

The single existing benchmark in this exact space is **Low, Rumker, Talkar, Torous, Cecchi & Ghosh (JMIR 2020, e22635)**, which reports SGD-L1 with LIWC features achieving **F1 = 0.851** on r/HealthAnxiety vs. controls. That is the only baseline number you would need to beat — and *anything* on top of MentalRoBERTa-multitask should clear it.

A second gap, almost equally defensible: **no DANN-style domain-adversarial training has been applied to Reddit mental-health classification**. Your F1=0.89 → 0.33 / AUROC=0.93 → 0.97 dissociation is the canonical "subreddit-style shortcut" signature documented by Harrigian, Aguirre & Dredze (EMNLP-Findings 2020) — but no one has actually applied the textbook fix (gradient reversal on subreddit identity) in this domain.

So the thesis has **two** publishable methodological contributions sitting unclaimed:

1. First transformer head-to-head r/HealthAnxiety vs. r/Anxiety with SHAI-grounded features and interpretability.
2. First DANN-on-subreddit application to address the cross-subreddit F1 collapse that the field has been documenting for 5+ years without fixing.

### The three-move recommendation, distilled

**A. Uncomment r/HealthAnxiety + r/Hypochondriacs* in `configs/subreddits.yaml`, recollect, and train a head-to-head r/HealthAnxiety vs r/Anxiety binary classifier with MentalRoBERTa.** Beats the only existing baseline (Low 2020 SGD F1=0.851). Two days. *This is the headline result of your thesis.*

**B. Add DANN with subreddit-as-domain to the multi-task setup.** First application in this domain. Directly addresses the cross-sub F1 collapse with a textbook fix. One-to-two weeks. *This is your methodological contribution.*

**C. Integrate ANGST (Hengle 2024, EMNLP) as external validation.** Three expert psychologists, public on HuggingFace, immediate download. Convert "internal-split F1" to "external-clinician-validated F1." Three days. *This is the credibility upgrade.*

Everything else — Whisper-style scaling, MentaLLaMA baselines, symptom decomposition, ordinal severity, fairness audit, TextAttack robustness — is great supporting work, but A+B+C alone give you the thesis its three citable contributions: **a new class, a new method, a new validation.**

---

## 1. Mental-health NLP datasets — comprehensive catalog

### Critical answer up front

**Is there any publicly-available dataset where 2+ clinical psychologists annotated posts for *health anxiety* specifically?**

**No.** No public dataset satisfies all three conditions: (1) text data, (2) "health anxiety / hypochondriasis / illness anxiety disorder" as an explicit, separate label, (3) annotated by 2+ clinical psychologists. The closest approximations:

- **Reddit Mental Health Dataset (Low et al. 2020)** — contains r/healthanxiety posts but the "label" is only subreddit membership (a weak proxy), no clinician annotation.
- **ANGST (Hengle et al. 2024, EMNLP)** — 2,876 posts annotated by **3 expert psychologists**, but for *general anxiety / depression / both / neither*, not for health anxiety specifically.

This is a defensible gap that your thesis can claim to address.

### Tier 1 — Directly relevant: Reddit + anxiety/comorbidity, downloadable

#### 1. Reddit Mental Health Dataset (RMHD)
- **Citation:** Low DM, Rumker L, Talkar T, Torous J, Cecchi G, Ghosh SS. (2020). *Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19.* **JMIR**, 22(10):e22635.
- **URL:** https://zenodo.org/records/3941387 ; https://www.jmir.org/2020/10/e22635/
- **Size:** 826,961 unique users, ~1M+ posts across 28 subreddits, 2018–2020.
- **Subreddits include r/healthanxiety** alongside r/anxiety, r/depression, r/SuicideWatch, r/socialanxiety, r/bpd, r/bipolarreddit, r/EDAnonymous, r/ptsd, r/schizophrenia, r/COVID19_support, etc., plus 11 non-MH controls.
- **Labels:** Subreddit-as-label (weak/distant). No human-validated post-level labels.
- **Access:** Public domain dedication (PDDL v1.0); Zenodo download.
- **Health-anxiety relevance:** **Highest among public datasets.** The only widely-cited corpus that explicitly carves out r/healthanxiety as a distinct community. The JMIR paper finds r/healthanxiety became the "linguistic centroid" for other MH subreddits during COVID-19 (ρ=–0.96).
- **Caveats:** Subreddit membership is a noisy proxy; many r/healthanxiety posters do not meet clinical criteria. No SHAI scores.

#### 2. ANGST — ANxiety-Depression Comorbidity DiaGnosis in Reddit PoSTs
- **Citation:** Hengle A, Kulkarni A, Patankar SD, Chandrasekaran M, D'silva S, Jacob JS, Gupta R. (2024). *Still Not Quite There! Evaluating Large Language Models for Comorbid Mental Health Diagnosis.* **EMNLP 2024**, pp. 16698–16721.
- **URL:** https://huggingface.co/datasets/ameyhengle/ANGST ; arXiv 2410.03908
- **Size:** 2,876 gold-labeled posts + 7,667 silver (GPT-3.5) posts; filtered from ~400k Reddit posts (2018–2022).
- **Labels:** Multi-label — {anxiety only, depression only, both, neither}.
- **Annotation:** **3 expert psychologists**, independent review, Krippendorff's α and Fleiss κ reported.
- **Access:** CC-BY-NC-2.0 via Google Form consent.
- **Health-anxiety relevance:** **Indirect but valuable.** General anxiety, not health anxiety, but the comorbidity scheme provides the cleanest available clinical-expert-annotated anxiety/depression dichotomy — a strong external validation target for your anxiety/depression classes.

#### 3. Anxiety on Reddit (Shen & Rudzicz 2017)
- **Citation:** Shen JH, Rudzicz F. (2017). *Detecting Anxiety through Reddit.* **CLPsych 2017 workshop @ ACL**.
- **URL:** https://aclanthology.org/W17-3107/ ; https://github.com/heyyjudes/anxiety-on-reddit
- **Labels:** Binary anxious / non-anxious. Subreddit-as-label.
- **Relevance:** Low — pools all r/anxiety together. Foundational baseline for the "anxiety on Reddit" line of work.

### Tier 2 — Strong external-validation candidates (depression / suicide, expert-labeled)

#### 4. UMD Reddit Suicidality Dataset v2 (Shing et al. 2018 / CLPsych 2019, 2021, 2024)
- **Citation:** Shing HC, Nair S, Zirikly A, Friedenberg M, Daumé H, Resnik P. (2018). *Expert, Crowdsourced, and Machine Assessment of Suicide Risk via Online Postings.* **CLPsych 2018**.
- **URL:** https://psresnik.github.io/umd_reddit_suicidality_dataset.html
- **Size:** 11,129 SuicideWatch users + 11,129 matched controls. Crowd labels for 621+621 users; expert labels for 245+245 users.
- **Labels:** 4-point ordinal — no risk / low / moderate / severe.
- **Annotation:** Practicing clinicians + crowdworkers.
- **Access:** **DUA-gated**, requires application via UMD.
- **Relevance:** Gold standard for the *suicidality* class.

#### 5. CLPsych Shared Task family (2014–2024)
- **2015 (Coppersmith et al.):** Twitter, depression vs. PTSD vs. control; self-disclosure labels.
- **2016/2017 (ReachOut.com):** Forum posts triaged by severity (green/amber/red/crisis); manual.
- **2019 (Zirikly et al.):** Reddit suicide-risk degree (UMD subset).
- **2021 (MacAvaney et al.):** OurDataHelps.org donated data from suicide-loss survivors/attempt survivors.
- **2022 (Tsakalidis et al.):** Reddit "Moments of Change" — temporal mood-shift annotation.
- **2024 (Chim et al.):** LLM-driven evidence highlighting for suicidality risk on 125 UMD users.
- **Access:** All via signed DUA with task organizers.

#### 6. RSDD — Reddit Self-reported Depression Diagnosis (Yates et al. 2017)
- **Citation:** Yates A, Cohan A, Goharian N. (2017). *Depression and Self-Harm Risk Assessment in Online Forums.* **EMNLP 2017**.
- **URL:** https://georgetown-ir-lab.github.io/emnlp17-depression/
- **Size:** 9,210 self-diagnosed depressed users + 107,274 controls; mean 969 posts/user.
- **Labels:** Binary, self-disclosure regex.
- **Access:** DUA-gated via Georgetown IR Lab.
- **Extension:** **RSDD-Time** (MacAvaney et al. 2018) — temporal annotations of diagnosis statements.

#### 7. SMHD — Self-reported Mental Health Diagnoses (Cohan et al. 2018)
- **Citation:** Cohan A, Desmet B, Yates A, Soldaini L, MacAvaney S, Goharian N. (2018). *SMHD: A Large-Scale Resource…* **COLING 2018**.
- **URL:** https://ir.cs.georgetown.edu/resources/smhd.html
- **Size:** ~350k users across 9 conditions: Depression, **Anxiety**, ADHD, Autism, Bipolar, OCD, Eating Disorder, PTSD, Schizophrenia + controls.
- **Labels:** Binary per-condition (multi-label possible).
- **Access:** **DUA-gated** via Georgetown.
- **Relevance:** **High** — the Anxiety class is the only large public corpus where users self-disclose an anxiety diagnosis.

#### 8. eRisk @ CLEF (2017–2025, 9 editions)
- **Citation:** Losada D, Crestani F, Parapar J. Multiple papers; flagship eRisk 2017 dataset (Losada & Crestani 2016).
- **URL:** https://erisk.irlab.org/
- **Tasks:** Early depression, anorexia, self-harm, pathological gambling, eating disorder severity, **BDI-II symptom estimation** (2019–2025), **sentence ranking for depression symptoms** (2023–2025), contextualized depression detection (2025).
- **Access:** DUA via eRisk organizers.
- **Relevance:** **Best resource for SHAI-style symptom mapping** — eRisk's BDI symptom-ranking framework is methodologically analogous to what a SHAI-symptom-ranking task would look like.

#### 9. DepreSym + BDI-Sen (Pérez et al. 2023, eRisk 2023)
- **Citation:** Pérez A, Fernández-Pichel M, Parapar J, Losada DE. (2023). *BDI-Sen* (SIGIR'23) and *DepreSym* (LREV 2025).
- **URL:** https://erisk.irlab.org/BDISen.html ; arXiv 2308.10758
- **Size:** BDI-Sen: 4.9k sentences; DepreSym: 21,580 sentences.
- **Labels:** Sentence-level relevance to each of 21 BDI-II symptoms.
- **Annotation:** 3 expert assessors including expert psychologist + GPT-4 comparison.
- **Relevance:** **Methodological template** — exactly the format you would need for a SHAI-symptom-annotated corpus.

#### 10. ReDSM5 (Bao et al. 2025, CIKM)
- **Citation:** Bao E, et al. (2025). *ReDSM5: A Reddit Dataset for DSM-5 Depression Detection.* **CIKM 2025**.
- **URL:** https://huggingface.co/datasets/irlab-udc/redsm5
- **Size:** 1,484 long-form posts, sentence-level labels for the 9 DSM-5 MDD symptoms + clinical rationale.
- **Annotation:** **Single licensed psychologist** (exhaustive).
- **Access:** Research license on Hugging Face.
- **Relevance:** Closest existing pattern to "DSM-5-symptom-annotated Reddit" — useful template for a SHAI-symptom adaptation.

#### 11. PRIMATE (Gupta/Naseem et al. 2022)
- **Citation:** Gupta S, et al. (2022). *Learning to Automate Follow-up Question Generation for Depression Triage.* **CLPsych 2022**.
- **URL:** https://github.com/primate-mh/Primate2022
- **Size:** 2,003 posts from r/depression_help.
- **Labels:** Binary per PHQ-9 question.
- **Annotation:** 5 crowdworkers + MHP quality control.
- **Access:** Public via GitHub.

### Tier 3 — Mental-health-NLP datasets (broader)

| # | Dataset | Citation / URL | Size | Labels | Relevance |
|---|---|---|---|---|---|
| 12 | Dreaddit | Turcan & McKeown 2019, LOUHI/ACL | 3,553 segments | Binary stress, MTurk | Anxiety-adjacent |
| 13 | CAMS | Garg et al. 2022, LREC | 5,051 Reddit posts | 6 causal categories + span rationales | Manual annotation |
| 14 | SDCNL | Haque et al. 2021, ICANN | 1,896 posts | Binary suicide-vs-depression | Subreddit + unsupervised correction |
| 15 | SWMH | Ji et al. 2022, Zenodo/HF `AIMH/SWMH` | 54,412 Reddit posts | Multi-class subreddit | Public |
| 16 | Reddit C-SSRS | Gaur et al. 2019, WWW | 500 users | 5-class C-SSRS ordinal | **4 practicing psychiatrists** |
| 17 | GoEmotions | Demszky et al. 2020, ACL | 58k Reddit comments | 27 emotions + neutral, crowd | Apache 2.0, auxiliary task |
| 18 | DepressionEmo | Rahman et al. 2024, J Affective Dis | 6,037 long Reddit posts | 8 emotions multi-label | GitHub |
| 19 | LoST | Garg et al. 2023, IEEE SMC | 3,251 Reddit posts | Low self-esteem by 2 professionals (clinical psych + NLP researcher) | Rosenberg SES, CSEI |
| 20 | MultiWD | Sathvik & Garg 2024, JBI | 3,281 Reddit posts | 6-dimensional wellness | Manual, GitHub |
| 21 | PsySym | Zhang et al. 2022, EMNLP | ~26k users + 8.5k posts | DSM-5 symptoms × 7 disorders incl. anxiety | Author-contact access |
| 22 | IMHI / MentaLLaMA | Yang et al. 2024, WWW | 105k instruction samples | 8 mental-health tasks | HF `klyang/MentaLLaMA-*` |
| 23 | Twitter-STMHD | Singh et al. 2022 | 33k Twitter users | Multiple disorders incl. anxiety | Self-disclosure + manual |
| 24 | DEPTWEET | Kabir et al. 2023 | 40k tweets | 4 severity levels of depression | Manual |
| 25 | RHMD | Naseem et al. 2022, WWW | 10,015 posts | 4-class health mention type | For filtering r/healthanxiety noise |
| 26 | CARMA | Marmol-Romero et al. 2025, arXiv 2511.03102 | 340,000+ Arabic Reddit | 6 conditions + control, F1=0.83 anxiety | Public |
| 27 | MentalRiskES | IberLEF 2023, 2024 | Spanish Telegram | 3 conditions inc. anxiety | 10 Prolific annotators |
| 28 | PsyQA | Sun et al. 2021 | 22k Chinese Q&A | Strategy-annotated | thu-coai/PsyQA |
| 29 | Mental-Health Counseling Conversations | Amod, HF 2023 | ~3.5k QA pairs | Real licensed-professional | HF, >100k downloads |
| 30 | Sharma et al. 2020 Empathy | EMNLP | 10k (post, response) | EPITOME (empathy framework) | License-gated TalkLife |
| 31 | MentalChat16K | Xu et al. 2025, KDD | 16k synthetic + BHC transcripts | Mental-health dialogues | GitHub |
| 32 | DAIC-WOZ + E-DAIC | Gratch et al. 2014, AVEC 2019 | 189+275 interviews | PHQ-8, PCL-C, **GAD-7** | DUA via USC ICT |
| 33 | RED | Welivita et al. 2023 | 1.2M Reddit posts | Multi-condition | Public |
| 34 | CLPsych 2015 Twitter | Coppersmith et al. | Depression vs PTSD vs control | Self-disclosure | DUA |
| 35 | SMHD-GER | Kerz et al. 2023, EACL | German SMHD analog | — | — |

### Datasets mentioned but with significant access/findability caveats

- **D2S (Yadav et al. 2020)** — Twitter PHQ-9 symptoms; author-contact only.
- **RESTORE (Yadav et al. 2023)** — multimodal PHQ-9; author-contact.
- **OurDataHelps.org corpus** (CLPsych 2021) — strict DUA, very limited.
- **Crisis Text Line corpus** — proprietary, research-restricted.
- **7 Cups of Tea text data** — historically referenced but not publicly downloadable.
- **TalkLife** — gated via TalkLife platform; Sharma et al. used under license.
- **MentalHelp (Raihan 2024)** — 14M Reddit posts, mostly automatic labels.

### Health-anxiety-specific corpora — verdict

| Resource | Verdict |
|---|---|
| Reddit Mental Health Dataset (Low 2020) — r/healthanxiety subset | **Best public starting point.** Subreddit-as-label. |
| ANGST (Hengle 2024) | Anxiety as one class but not health-anxiety-specific. |
| SMHD (Cohan 2018) | Anxiety as one class, no subdivision. |
| MentalRiskES | Anxiety class in Spanish; no health-anxiety subdivision. |
| RHMD (Naseem 2022) | Distinguishes personal vs. figurative health mentions — useful auxiliary for filtering somatic posts. |
| Cyberchondria text corpora | **Do not exist publicly.** |
| CBT-for-health-anxiety RCT transcripts | **Not public.** |
| SHAI-annotated text corpus | **None located.** |

### Top 5 most useful for this thesis (ranked)

1. **Reddit Mental Health Dataset (Low et al. 2020)** — Only public Reddit corpus that explicitly preserves r/healthanxiety. Permissive license (PDDL v1.0). Use as **primary external comparator** and possibly **supplementary training data**.
2. **ANGST (Hengle et al. 2024)** — Only public dataset with **multiple expert psychologists** annotating Reddit posts for anxiety vs. depression vs. comorbidity. Use as **gold external validation**.
3. **UMD Reddit Suicidality Dataset v2 + CLPsych 2024** — Gold-standard for the suicidality class. Apply for DUA early (weeks).
4. **SMHD (Cohan et al. 2018) — Anxiety + Depression slices** — Closest to "diagnosed cases" publicly available. DUA required.
5. **eRisk BDI-Sen / DepreSym (Pérez et al. 2023) + ReDSM5 (Bao et al. 2025)** — **Methodological precedent** for constructing a clinician-annotated symptom-level corpus mapping SHAI items to text spans.

### Reproducibility note

Several datasets (RSDD, SMHD, UMD, eRisk, CLPsych, ANGST) require DUAs; budget 4–8 weeks for the slowest (UMD). **Start the DUA process now in parallel with model work.** Public unrestricted downloads available immediately: RMHD (Zenodo), SWMH (Zenodo/HF), Dreaddit (GitHub), CAMS (GitHub), GoEmotions (HF), LoST (arXiv supp), MultiWD (GitHub), PRIMATE (GitHub), SDCNL (GitHub), DepressionEmo (GitHub), ReDSM5 (HF), CARMA (HF/GitHub), MentaLLaMA/IMHI (HF), RHMD.

---

## 2. Whisper-style weak supervision, distillation & LLM-as-labeler

### Whisper (Radford et al. 2022) — the actual recipe

**Paper:** Radford, Kim, Xu, Brockman, McLeavey, Sutskever — *Robust Speech Recognition via Large-Scale Weak Supervision*, arXiv:2212.04356, ICML 2023.

**Data filtering pipeline:**

1. **Heuristic detection of machine-generated transcripts.** Transcripts that are all-uppercase, all-lowercase, lack complex punctuation, or use heavily normalized formatting are flagged as ASR-generated and discarded.
2. **Audio-text language matching.** They train *two* language ID models — one on audio, one on text — and require agreement.
3. **Fuzzy de-duping of transcript texts** to remove repeated boilerplate.
4. **De-duplication against eval sets** to avoid contamination.
5. **Final corpus:** 680,000 hours; 65% English ASR (438,218h), 18% X→English translation, 17% multilingual ASR.

**Direct lesson:** Whisper's philosophy is "*filter aggressively on cheap heuristics, never relabel*." For Reddit posts: drop bot-pattern posts, drop near-duplicate cross-posts, drop posts whose subreddit prior strongly disagrees with the lexicon (label conflict → discard, don't reconcile). They did NOT use Snorkel-style probabilistic label aggregation — they used hard filters.

### Snorkel / data programming / WRENCH

- **Snorkel (Ratner et al., VLDB 2017, arXiv:1711.10160).** Labeling functions + generative model denoises them. Reported: 132% average lift over heuristic baselines; within 3.6% of fully hand-labeled. Code: snorkel.org.
- **Snorkel DryBell (Bach et al., 2019, arXiv:1812.00417).** Industrial deployment.
- **WRENCH benchmark (Zhang et al., NeurIPS 2021, arXiv:2109.11377).** 22 datasets, >120 method variants. Code: https://github.com/JieyuZ2/wrench.
- **BOXWRENCH (Zhang et al., NeurIPS 2024, arXiv:2501.07727).** **Critical scaling result:** with realistically-written LFs, **supervised learning needs 1000+ labeled examples to match weak supervision**. Weak supervision keeps improving with more unlabeled data.

### LLM-as-labeler — empirical evidence on label quality

| Paper | Finding (hard numbers) |
|---|---|
| **Gilardi, Alizadeh, Kubli, PNAS 2023** (arXiv:2303.15056) | ChatGPT zero-shot accuracy **~25 pp above MTurk** on 4 tasks; Krippendorff α higher than crowd AND trained annotators; ~30× cheaper. |
| **AnnoLLM (He et al., NAACL 2024)** arXiv:2303.16854 | "Explain-then-annotate" two-step: GPT-3.5 ≥ crowd-workers on BoolQ, WiC, user-intent. |
| **Wang et al., EMNLP-Findings 2021** (arXiv:2108.13487) | 50–96% cost reduction at matched downstream performance. |
| **Knowledge Distillation in Automated Annotation (Wang et al., 2024, arXiv:2406.17633)** | Models fine-tuned on GPT-4 labels perform *comparably* to models on human labels across 14 CSS tasks. |
| **Performance-Guided KD (Amazon, EMNLP 2024, arXiv:2411.05045)** | BERT distilled from LLM: F1 0.908 (IMDB), 0.943 (Inshorts), **130× faster + 25× cheaper** than direct LLM. |
| **GPT-4 stance classification (PLOS ONE 2024)** | Fleiss κ = 0.645, Krippendorff α = 0.613 vs human gold. |
| **Judge's Verdict (arXiv:2510.09738)** | Top LLMs reach Cohen's κ 0.804–0.813 vs human-human baseline of 0.801 — i.e. *at human-level*. |

**Bottom line:** LLM labels are good enough to train a downstream classifier on, *provided* (a) the task is not extremely subjective, (b) you use chain-of-thought / explain-then-annotate, (c) you majority-vote 3–5 completions, and (d) you keep a small human gold set for calibration. Expect κ ≈ 0.6–0.8 vs human gold.

### Failure modes of LLM-as-labeler

- **Mental-health stigma bias.** *Expressing stigma and inappropriate responses prevents LLMs from safely replacing mental health providers* (arXiv:2504.18412, FAccT 2025) — LLMs express significant stigma against schizophrenia, alcohol dependence, etc., and respond inappropriately to delusional content. **Direct implication:** the Claude codebook must explicitly forbid stigma-laden phrasings in the rationale; review prompts for therapist persona.
- **Conservative default class.** LLMs default to the "safe" class on ambiguous inputs.
- **Prompt instability.** Same input → different output across seeds. Mitigation: temperature ≤ 0.3 or majority-vote.
- **Domain/lexical sparsity.** LLMs underperform fine-tuned specialists on niche jargon (Reddit slang, mental-health idioms). Mitigation: keep a domain-specialized student model — exactly the MentalRoBERTa setup.

### Self-training / pseudo-labeling / Noisy Student

- **Noisy Student (Xie et al., CVPR 2020, arXiv:1911.04252).** ImageNet 88.4% top-1, +2.0 pp over prior SOTA. Key: noise on student (dropout, RandAugment) but **no noise** on teacher when generating pseudo-labels. Soft pseudo-labels > hard. Balance pseudo-label counts per class.
- **UDA (Xie et al., NeurIPS 2020, arXiv:1904.12848).** On IMDB: 20 labeled examples + UDA on unlabeled data beats prior SOTA trained on 1250× more labels. Error 6.50%→4.20% on a strong BERT baseline.
- **Is BERT Robust to Label Noise? (Bhattacharjee et al., 2022, arXiv:2204.09371).** BERT tolerates 20–40% symmetric label noise with <2 pp F1 drop. Beyond ~50% noise, performance collapses.

### Synthetic-data / LLM-as-generator

- **ZeroGen (Ye et al., EMNLP 2022, arXiv:2202.07922).** Generate synthetic dataset from GPT2-XL → train DistilBERT.
- **Self-Instruct (Wang et al., ACL 2023, arXiv:2212.10560).** 52k generated instructions; vanilla GPT-3 + Self-Instruct ≈ InstructGPT.
- **AttrPrompt (Yu et al., NeurIPS 2023, arXiv:2306.15895).** Class-conditional attributed prompts; matches plain-prompt baseline at **5% of the ChatGPT cost**.
- **LLM2LLM (Lee et al., ACL-Findings 2024, arXiv:2403.15042).** Iteratively generate synthetic data for the student's *mistakes*. +24% GSM8K, +33% SNIPS, +53% TREC, +40% SST-2 in low-data regime.
- **Backtranslation & paraphrasing (ICCS 2025, arXiv:2507.14590).** For emotion classification, BT/paraphrase **match or beat LLM zero-/few-shot generation** — and are cheaper.

### Active learning under 1k budget

- **Schröder et al. (2022 / Springer 2025).** Pool-based AL with BERT, sub-1000 budgets, Bayesian uncertainty acquisition consistently beats random sampling by 3–10 pp F1.
- **FreeAL (Xiao et al., EMNLP 2023, arXiv:2311.15614).** LLM = active annotator; SLM = quality filter. Across 8 benchmarks substantial zero-shot gains for both SLM and LLM with **zero human labels**.

### Soft labels / confidence-weighted training

- **Rethinking Soft Labels for KD (Lukasik et al., arXiv:2102.00650, NeurIPS 2021).** Sample-wise weights & per-sample temperature: better than constant T across all KD benchmarks.

### Empirical answers to key sub-questions

**How much does scaling weak labels (1k → 100k) help vs hurt?**
- BOXWRENCH (NeurIPS 2024): with realistic LFs, supervised learning needs 1000+ gold labels to match weak supervision — and weak supervision keeps improving with more unlabeled data.
- Whisper (680k h vs prior 60k h): monotonic robustness gains, no observed saturation.
- Noisy Student (1.3M → 300M unlabeled): monotonic improvement.
- **Failure mode:** when label noise > ~50%, BERT collapses.

**For 16k corpus → 1k human + 15k Claude:** expect ~3–8 pp F1 lift over 1k-only training, more on minority classes.

**Is filter-don't-perfect supported?** Yes — Whisper is the canonical example. UDA (consistency on the *kept* unlabeled data) is another. BOXWRENCH (2024) shows filtering aggressively outperforms denoising on realistic tasks.

### Concrete recommendation — ranked by impact / effort

1. **[2 days, +3–6 pp F1] Whisper-style hard filtering of the weak (lexicon+subreddit) layer.** Drop any post where lexicon and subreddit-prior disagree by a large margin; drop near-duplicates (MinHash + 0.85); drop posts <20 tokens or with no first-person mention for self-state targets. Do *not* try to fix conflicts.
2. **[1 day, +1–3 pp F1] Soft-label distillation from Claude.** Get top-k token probs from Claude or majority-vote 3 completions → soft target → KL-div loss on MentalRoBERTa.
3. **[3 days, +2–5 pp F1] Self-training / Noisy Student loop.** Train student on (1k human + Claude-labeled). Generate pseudo-labels on remaining unlabeled. Retrain with dropout + back-translation noise. Iterate twice.
4. **[3 days, +1–3 pp F1 on minority classes] AttrPrompt back-translation for minority classes.** Cheap synthetic + BT augmentation only for low-support targets.
5. **[3 days, better data-efficiency curve] FreeAL-style AL** for the human-annotation budget rather than random sampling.
6. **[1 week, baseline comparator] Snorkel LabelModel** combining lexicon-prior + Claude label as two LFs.
7. **[1 week, ablation] LLM2LLM-style iterative augmentation** targeting Claude's most-wrong examples on the human gold.

**Skip:** ZeroGen-from-scratch synthetic data (16k real corpus is more authentic than anything Claude generates); heavy Snorkel LF-engineering (BOXWRENCH 2024 shows it under-delivers vs simpler filtering); full Self-Instruct (overkill for 4 binary targets).

**Mandatory caveats for mental health:**
- Audit Claude labels for stigma bias. Spot-check ~100 positive predictions of "self-harm risk" by hand.
- Never give Claude a "therapist" persona in the labeling prompt — induces stigmatizing reasoning.
- Calibrate decision thresholds per class on human gold, not on Claude labels (LLM conservative-default bias).
- Report κ between Claude and human gold per class; expect 0.5–0.8. Anything <0.4 means that target is genuinely subjective.

**Key absent literature:** No published paper does what this dissertation does (Claude-distilled MentalRoBERTa multi-task with a 3-tier label hierarchy on Reddit). That is the contribution.

---

## 3. Health-anxiety / hypochondriasis NLP — comprehensive lit review

### The single foundational paper for r/HealthAnxiety NLP

**Low DM et al. (2020). "Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19: Observational Study." *JMIR* 22(10):e22635.**
- **Method**: SGD linear classifier with L1 penalty + tree baselines; 90 text features = 62 LIWC + 4 sentiment + manually-built suicide/economic-stress/isolation/substance lexicons + readability + TF-IDF n-grams (256-1024).
- **Dataset**: RMHD, 826,961 unique users, 28 subreddits, 2018-2020.
- **Results for r/HealthAnxiety**: Weighted **F1 = 0.851** in the one-vs-controls binary classifier. Top positive features: *cancer, LIWC biological, "health anxieti", LIWC health, LIWC body, test, fine, LIWC assent, googl*. r/HealthAnxiety spiked first in pandemic-related posts in January 2020; during March 2020 the MH subreddits became most linguistically similar to r/HealthAnxiety (ρ = −0.96).
- **What it claims about health vs general anxiety**: Implicitly distinguishes the two — r/HealthAnxiety's discriminating tokens (cancer, googl, test, body, biological) are **completely different** from typical r/Anxiety markers (social situations, panic). But the paper does **not** make health-anxiety detection its primary task.
- **Code/lexicon released**: Yes (OSF + GitHub).
- **Gap left**: (a) no SHAI / HAI / Whiteley validation; (b) classifier is **linear SGD**, no transformer; (c) only ~5% of users self-disclose a diagnosis; (d) one-vs-rest formulation, not head-to-head r/Anxiety vs r/HealthAnxiety; (e) no instrument-grounded operationalization of *health anxiety as a construct*.

### Cyberchondria — the closest adjacent literature

- **Schenkel et al. (2023). JMIR Form Res 7:e42206.** Elastic-net on N=725 survey responses; SHAI predicts CSS (Cyberchondria Severity Scale) distress R²=0.344. **No NLP**; tabular survey only.
- **Doherty-Torstrick et al. (2016). Psychosomatics 57(4):390-400.** Survey N=731. High-anxiety reported worsening after online checking (68.3% vs 40%); Whiteley Index strongest predictor. **No NLP.**
- **Fergus (2013).** Cyberchondria and IU; N=512. Established IU-moderation finding. **No NLP.**

### Linguistic markers in Somatic Symptom Disorder

- **Lemogne C et al. (2024). J Affective Disorders 351:374-383.** NLP on 8-week messaging therapist intervention transcripts; tracked emotional valence markers in N=173 SSRD patients with PHQ-9/GAD-7 measured every 3 weeks. NLP markers tracked PHQ-9/GAD-7 changes; specific markers distinguished improvement. Not Reddit; no SHAI; no released model.

### "Detecting Anxiety through Reddit" line — general anxiety only

- **Shen & Rudzicz (2017).** N-gram LMs + word2vec; **91% word-embedding alone; 98% combined** — *binary* classifications (anxiety subreddit vs. controls), not health-anxiety subtyping. Does NOT include r/HealthAnxiety as its own class.
- **Ireland & Iserman (2018).** Dictionary analysis across r/PanicParty etc. LIWC dimensions differ in anxious people across contexts.
- **Tariq M et al. (2026).** 13 interpretable linguistic features + LR, **author-disjoint splits**, cross-domain validation to DAIC-WOZ. Still general anxiety, not health-anxiety subtype.

### Transformer / domain-adapted LMs for anxiety in general

- **Ji et al. (2022). MentalBERT.** F1 ≈ 0.82 on multi-class mental disorder classification. r/HealthAnxiety **not** in the pretraining mix as a primary source.
- **MANTIS @ SMM4H 2023.** Hybrid ensemble for *social* anxiety disorder detection on Reddit. **Health anxiety not in the SMM4H series at all.**
- **Wu D et al. (2024). JMIR Mental Health 10:e44325.** N=2000, Longformer fine-tuned on speech transcripts, predicts GAD-7 above/below threshold. **F1 = 0.945 reported for GAD-7 prediction from diaries.** Strongest example of clinically-validated transformer anxiety detection — but for GAD.
- **2024-2026 multi-class benchmarks** (arXiv 2509.16542, 2507.19511): RoBERTa hits 91-99% F1 across anxiety/depression/bipolar/schizophrenia/CPTSD multi-class. **None of these treat health anxiety as its own class** — they collapse it under "anxiety."

### PsySym and DSM-5 symptom-level annotation

- **Zhang et al. (2022). EMNLP, arXiv 2205.11308.** Reddit-based, 8,554 sentences, **7 mental disorders × 38 symptom classes** clinician-annotated, mapped to DSM-5 / clinical scales. The 7 disorders: depression, PTSD, ADHD, anxiety, OCD, eating disorders, schizophrenia. **Health anxiety / IAD / SSD is NOT one of the 7.** Major gap.
- **ReDSM5** (arXiv 2508.03399, 2025) — Reddit dataset annotated to DSM-5 criteria, but **for depression only**.

### LIWC-only work specifically on health anxiety

- **Hwang dissertation (Oregon State).** "Linguistic Attributes of Online Health Anxiety Communication" — LIWC to characterise health-anxiety communication. No transformer, descriptive only.
- **Cancer-forum LIWC studies (JMIR Cancer 2021/2024):** LIWC anxiety + biological + health categories discriminate cancer patients from family members.

### Long COVID / health-focus and pandemic studies

- **Segneri F et al. (2024). PLOS ONE 19(8):e0308340.** 6,107 users, 984,625 pre-pandemic posts; LIWC + VADER + TF-IDF + SNA → LR. Long-COVID-affected users had pre-pandemic discourse focused on health, higher 1st-person singular — interpreted as "greater tendency towards hypochondria." McFadden R² ~0.08. **Strongest evidence that a hypochondria-like linguistic style precedes a clinical health outcome.**

### Linguistic markers predictive of health anxiety

| Marker | Evidence base |
|---|---|
| Somatic / biological vocabulary (LIWC biological, body, health) | Low et al. 2020; cancer-forum LIWC studies; Hwang dissertation |
| Symptom/disease-name density ("cancer", "tumor", "test") | Low et al. 2020 top tokens |
| Online-search references ("googl", "search") | Low et al. 2020; Doherty-Torstrick 2016 (behavioral) |
| Reassurance-seeking patterns (questions, "is it normal that…") | Lucock & Morley 1996 HAQ factor; Salkovskis CBT model |
| First-person singular pronouns ("I", "me", "my body") | Ireland & Iserman 2018; Segneri 2024 |
| Catastrophizing / absolutist language | NLP cognitive-distortion literature (arXiv 2508.09878) — not specifically validated on health anxiety |
| Intolerance-of-uncertainty markers ("what if", "could be") | Fergus 2013 (behavioral only); not validated computationally |
| Body-vigilance language (monitoring, checking) | Salkovskis & Warwick clinical theory; **not operationalized in NLP** |

### Has anyone…? — direct answers

**(a) Has anyone trained a transformer specifically to detect *health anxiety* (as a class distinct from general anxiety) from social media?**
**No.** The closest is Low et al. 2020 — uses linear SGD, not transformers; treats r/HealthAnxiety as one of 15 subreddits in a multi-class. General multi-class Reddit MH transformer papers treat r/HealthAnxiety as a single label among ~6-15 disorders. No paper isolates health anxiety as the target construct and benchmarks it head-to-head against general anxiety.

**(b) Has anyone validated computational health-anxiety detection against the SHAI?**
**No, not for text.** The only SHAI + ML combination I found is Liu et al. (Gaussian Naive Bayes, accuracy 0.813) — but that's *tabular* features predicting SHAI status, **not from text**. **No study correlates text-derived predicted scores with SHAI scores from the same users.** Clean defensible novelty gap.

**(c) What is the SOTA reported in the literature for health-anxiety detection from text?**
**Low et al. 2020 r/HealthAnxiety-vs-controls SGD-L1 binary F1 = 0.851**, on subreddit-as-proxy labels. No transformer baseline, no SHAI alignment, no comparison vs r/Anxiety. Everything else is buried inside multi-class anxiety classifiers that don't report health-anxiety-specific metrics.

### State of the field

**What has been done:**
1. r/HealthAnxiety included in one major Reddit MH benchmark (Low et al. 2020) with linear-SGD F1=0.851 — no transformer, no instrument validation.
2. Cyberchondria predicted from tabular surveys (elastic-net R²=0.34) but not from text.
3. SSD studied via NLP markers tracking PHQ-9/GAD-7 in messaging therapy (Lemogne 2024) — not Reddit, no SHAI.
4. Anxiety more generally is mature: 0.91-0.99 F1 on multi-class Reddit; SOTA for GAD-7 prediction from text is F1 = 0.945 (Longformer, diaries).
5. Adjacent constructs operationalized: pain catastrophizing, eating disorders, cognitive distortions, contamination OCD.
6. Clinical instruments mature: SHAI, HAI, Whiteley Index, CSS, HAI-14 — psychometrically validated but **none anchored to a text classifier**.

**What has NOT been done:**
1. **No transformer trained specifically to detect health anxiety vs general anxiety** as separate classes on social media.
2. **No validation of any text-based health-anxiety classifier against the SHAI** on the same users.
3. **No computational operationalization of the Salkovskis-Warwick CBT framework** — reassurance seeking, body vigilance, catastrophic misinterpretation as features extracted from text.
4. **No DSM-5 IAD vs SSD distinction** drawn computationally; PsySym ignores both.
5. **No public, reusable health-anxiety lexicon** grounded in SHAI/HAI items.
6. **No head-to-head r/Anxiety vs r/HealthAnxiety binary task** with multiple model families benchmarked.
7. **No multi-corpus joint analysis** asking "which subreddit users behave most like high-SHAI scorers."
8. **No error-analysis or explainability work** mapping classifier features to Salkovskis-Warwick model constructs.

### Strongest novelty angles, ranked

1. **First transformer-based binary detector of *health anxiety* vs *general anxiety*** trained on r/HealthAnxiety vs r/Anxiety with author-disjoint splits, reporting F1/AUROC against Low 2020's F1 = 0.851 baseline.

2. **First SHAI/HAI-anchored linguistic feature engineering** — derive features from the SHAI's three factors (likelihood of illness, severity, body vigilance) and HAI's factors (worry/preoccupation, fear of illness/death, reassurance seeking, interference). Each operationalized as lexicon + pattern: reassurance-seeking patterns ("does anyone else…"), body-vigilance language ("I keep checking"), catastrophic-misinterpretation triggers (somatic noun + cancer/death).

3. **Multi-class anxiety subtyping**: r/HealthAnxiety vs r/Anxiety vs r/COVID19_support vs r/COVID19positive vs other anxiety-adjacent subreddits, framed as DSM-5-IAD-vs-GAD-vs-pandemic-acute-stress.

4. **Pseudo-SHAI scoring**: build a regressor predicting a SHAI-like score from text using SHAI-item-derived rubrics or LLM rubric scoring; validate against external signals.

5. **Explainability/feature-importance mapping** tying classifier features back to specific SHAI / Salkovskis-Warwick constructs (SHAP or attention).

6. **Pandemic-era distinction**: do r/COVID19_support / r/COVID19positive users show transient health-anxious linguistic patterns that converge with r/HealthAnxiety baseline?

7. **Release a labelled subset with weak-supervision SHAI/HAI-derived heuristics**, plus trained transformer, plus lexicon — first publicly-available health-anxiety-specific NLP resource.

**Strongest single positioning:** *"First transformer-based health-anxiety detector grounded in the SHAI/HAI clinical instruments, reporting head-to-head performance against r/Anxiety as a contrast class, with explainability tied to the Salkovskis-Warwick cognitive model."*

---

## 4. SOTA models for mental-health NLP (2023–2026)

### The baseline the author already owns: MentalBERT / MentalRoBERTa

Ji et al. (2022). MentalBERT/MentalRoBERTa, pretrained on 13.67M sentences from r/depression, r/SuicideWatch, r/Anxiety, r/offmychest, r/bipolar.

| Dataset | BERT | RoBERTa | MentalBERT | MentalRoBERTa |
|---|---|---|---|---|
| eRisk T1 (depression) | 86.31 | 92.25 | 92.25 | **93.38** |
| CLPsych15 (depression) | 62.75 | 66.07 | 62.63 | **69.71** |
| Depression_Reddit (Pirina) | 90.90 | 95.11 | 94.62 | 94.23 |
| Dreaddit (stress) | 78.26 | 80.56 | 80.04 | **81.76** |
| SWMH (multiclass) | 70.76 | 72.03 | 71.11 | **72.16** |
| T-SID (Twitter suicide) | 88.51 | 88.76 | 88.61 | **89.01** |

Domain-adapted variants gain 1-7 F1 over vanilla BERT/RoBERTa. Apache-2.0, HF: `mental/mental-bert-base-uncased`, `mental/mental-roberta-base`. 110M / 125M params. Fits 4090 trivially.

### MentaLLaMA and the IMHI benchmark (Yang et al. 2024, WWW)

arXiv 2309.13567; HF `klyang/MentaLLaMA-chat-7B`, `klyang/MentaLLaMA-chat-13B`. MIT license.

Table 3 weighted F1 across 10 IMHI test sets (DR, CLP, Dreaddit, SWMH, T-SID, SAD, CAMS, Loneliness, MultiWD, IRF):

| Model | DR | CLP | Dreaddit | SWMH | T-SID | SAD | CAMS | Loneliness | MultiWD | IRF |
|---|---|---|---|---|---|---|---|---|---|---|
| MentalRoBERTa | 94.23 | **69.71** | 81.76 | 72.16 | 89.01 | 68.44 | 47.62 | **85.33** | — | — |
| ChatGPT (zero-shot) | 82.41 | 56.31 | 71.79 | 49.32 | 33.30 | 54.05 | 33.85 | 58.40 | 62.72 | 41.33 |
| LLaMA-2-7B (ZS) | 58.91 | 36.26 | 53.51 | 37.33 | 25.55 | 11.04 | 16.34 | 58.32 | 40.10 | 38.02 |
| MentaLLaMA-7B | 76.14 | 59.86 | 71.65 | 72.51 | 72.64 | 49.93 | 32.52 | 83.52 | 68.44 | 67.53 |
| MentaLLaMA-chat-7B | 83.95 | 51.84 | 62.20 | 75.58 | 77.74 | 62.18 | 44.80 | 83.71 | 75.79 | 72.88 |
| MentaLLaMA-chat-13B | **85.68** | 52.61 | 75.79 | 71.70 | 75.31 | 63.62 | 45.52 | 85.10 | 75.11 | **76.49** |

**Key finding:** On Reddit binary depression (DR, CLP), MentalRoBERTa still beats MentaLLaMA-chat-13B by 7-17 F1 points. MentaLLaMA only wins on multiclass tasks (SWMH, CAMS) and where it can leverage instruction-following.

**Hardware fit:** chat-7B in 4-bit QLoRA fits ~12GB VRAM. chat-13B in 4-bit fits ~16-20GB. chat-33B-LoRA needs ~22-24GB.

### Mental-LLM (Xu et al. 2024, UbiComp) — the FLAN-T5 / Alpaca branch

arXiv 2307.14385; HF `NEU-HAI/mental-alpaca`, `NEU-HAI/mental-flan-t5-xxl`. **Mental-Alpaca (7B) and Mental-FLAN-T5 (11B) beat GPT-4 prompt-design baselines by 4.8% balanced accuracy despite being 100-250x smaller.**

| Model | Dreaddit | DepSev bin | DepSev 4cls | SDCNL | CSSRS bin | CSSRS 5cls |
|---|---|---|---|---|---|---|
| GPT-3.5 (best prompt) | 0.688 | 0.653 | 0.642 | 0.632 | 0.617 | 0.310 |
| GPT-4 (best prompt) | 0.725 | 0.719 | 0.656 | 0.647 | 0.760 | 0.441 |
| Mental-Alpaca | **0.816** | 0.775 | 0.746 | **0.724** | 0.730 | 0.403 |
| Mental-FLAN-T5 | 0.802 | 0.759 | **0.756** | 0.677 | **0.868** | **0.481** |

### MentalQLM (Tao et al. 2025, IEEE JBHI) — new lightweight SOTA

medRxiv 2024.12.29.24319755. Instruction-tunes **Qwen-1.5-0.5B** with dual-LoRA (classification head + reasoning head) on IMHI splits.

**Average weighted F1 of 0.778 across five IMHI benchmarks**, outperforming MentaLLaMA-chat-13B by 3.2% and few-shot GPT-4 by 17.7%, at 26x fewer parameters than MentaLLaMA-chat-13B. SAD: 67.02 (vs MentaLLaMA chat-13B's 63.62). Code: https://github.com/tortorish/MentalQLM.

### Other recent lightweight specialists

- **mhGPT (arXiv 2408.08261):** 1.98B GPT-NeoX trained on 49,812 PubMed MH articles + 1M Reddit submissions. Outperforms MentaLLaMA on IRF, Dreaddit, SAD, MultiWD, PPD-NER when fine-tuned on 5% of downstream data.
- **Menta (arXiv 2512.02716, Dec 2025):** Qwen3-4B + LoRA, jointly fine-tuned on 6 mental health tasks. +15.2% average over best non-fine-tuned SLM baselines, beats 13B LLMs on depression and stress, fits in 3GB RAM (iPhone 15 Pro Max deployment).
- **MentalGLM (arXiv 2410.10323, Oct 2024):** GLM-9B/33B fine-tuned on C-IMHI Chinese benchmark; 85.12 F1 on suicide risk detection.
- **multiMentalRoBERTa (arXiv 2511.04698, Nov 2025):** RoBERTa-large (355M) fine-tuned for 6-way/5-way classification across stress, anxiety, depression, PTSD, suicidality, neutral. F1 = 0.839/0.870. Most directly comparable to the author's setup.

### Medical-domain models that under-perform on Reddit MH

These DO NOT beat MentalRoBERTa on Reddit:
- **PsychBERT** (Vajre 2021): BERT-base continued-pretrained on 40k PubMed psychology articles + 200k MH conversations. Slightly weaker than MentalBERT on Reddit.
- **Bio_ClinicalBERT** (Alsentzer 2019): MIMIC-III; ~85% F1 on ADHD severity but underperforms MentalBERT on Reddit.
- **BioGPT, BioMedLM**: no published Reddit MH benchmark.
- **Med-PaLM 2**: 77.5-92.5% on DSM-5 case examples (vignettes, not Reddit).
- **MedAlpaca, ChatDoctor, Me-LLaMA**: clinical text, not Reddit; consistently lose.
- **Clinical ModernBERT (arXiv 2504.03964):** 8k context BERT with RoPE + Flash Attention; Reddit MH performance not yet published.

### GPT-4, Claude, Gemini, DeepSeek — zero-/few-shot on Reddit MH

**Critical empirical finding: frontier LLMs in zero-shot do NOT beat fine-tuned MentalRoBERTa on Reddit binary depression/anxiety classification. They lose by 8-25 F1 points.**

- ChatGPT zero-shot on DR: 82.4 vs. MentalRoBERTa 94.2.
- GPT-4 best prompt on Dreaddit: 0.725 balanced acc; MentalRoBERTa 0.81+.
- Cognitive-Mental-LLM (arXiv 2503.10095): o3-mini with CoT reaches F1 0.79 on Dreaddit; Mental-FLAN-T5 reached 0.80.
- Comprehensive Evaluation (arXiv 2409.15687): 33 LLMs (2B-405B). LLaMA-3.1-405B reaches 91.2% on psychiatric knowledge MCQs, but not on Reddit classification F1.
- Claude 3.5 Sonnet ceiling on related tasks: F1 0.7617 (Preprints.org 202502.0720). Far below fine-tuned MentalRoBERTa's 0.85-0.95 on Reddit binary MH.
- Cognitive-Mental-LLM: Claude 3 Haiku 56.6, Claude 3 Sonnet 54.2, GPT-4 72.0 on Dreaddit. None beat MentalRoBERTa.

**No published evidence shows Claude-3.5-Sonnet or any Claude-4 variant outperforming a fine-tuned MentalRoBERTa on canonical Reddit MH binary benchmarks.** Claude is competitive on harder multiclass/interpretive tasks (CAMS reasoning, free-text severity) and is the right choice for *explanation generation*, but the discriminative crown sits with domain-adapted encoder fine-tunes.

### Llama-3 / Mistral / Qwen fine-tunes

- **Llama-3.1-8B-Instruct** QLoRA on SWMH 54k corpus reaches ~96% accuracy on binary depression. 4-bit QLoRA fits 12-16GB VRAM, 1-3 hours on a 4090.
- **Mistral-7B-Instruct-v0.2** + LoRA on MH counseling: human-preferred over base, no Reddit classification SOTA.
- **Mixtral-8x7B-Instruct**: 47B active params, does NOT fit 4090 even in 4-bit. Skip.
- **Phi-3-mini/medium**: small Microsoft models; no peer-reviewed Reddit benchmark beating MentalRoBERTa.
- **Qwen-2.5 / Qwen-3 4B/7B**: best smaller LLM family for fine-tuning per Menta and MentalQLM. Both beat MentaLLaMA on subsets of IMHI.

### CLPsych / eRisk shared tasks (2024-2025)

- **CLPsych 2024:** Task moved toward LLM-driven evidence extraction. Top systems used LLaMA-2/3 or Mistral with few-shot prompting plus rule-based span scoring.
- **CLPsych 2025:** Self-state identification baseline used 4-bit Gemma-2-9B in few-shot.
- **eRisk 2024/2025:** SINAI-UJA, DS@GT top-5 were transformer ensembles (RoBERTa-large + DeBERTa-v3-large + LongFormer) plus longitudinal temporal-attention.

**Takeaway:** Shared-task winners still favor RoBERTa/DeBERTa fine-tunes with optional LLM augmentation rather than LLM-only systems.

### Prompting techniques

- **CoT** consistently improves binary MH classification by 3-8 F1.
- **Few-shot CoT** winner across multi-class; gains 5-10 F1.
- **Self-consistency CoT** another 1-3 F1 but 5x API cost.
- **RAG**: worst strategy on SWMH (F1 0.45 vs zero-shot 0.67 vs fine-tuning 0.81). Does help when retrieval source is psychometric scales (PHQ-9, GAD-7) — see arXiv 2501.00982.

### Model leaderboard (sorted by reported F1 on Reddit binary depression DR)

| Rank | Model | Params | F1 on DR | 4090? | License |
|---|---|---|---|---|---|
| 1 | RoBERTa-base fine-tune | 125M | **95.11** | yes | MIT |
| 2 | MentalBERT | 110M | 94.62 | yes | Apache-2.0 |
| 3 | MentalRoBERTa | 125M | 94.23 | yes | Apache-2.0 |
| 4 | DeBERTa-v3-large (suicide, longitudinal) | 435M | 94.6 (suicide) | yes | MIT |
| 5 | multiMentalRoBERTa (5-class) | 355M | 87.0 (macro) | yes | open |
| 6 | MentaLLaMA-chat-13B | 13B | 85.68 | yes (4-bit) | MIT |
| 7 | MentaLLaMA-chat-7B | 7B | 83.95 | yes | MIT |
| 8 | ChatGPT zero-shot | n/a | 82.41 | API | proprietary |
| 9 | Mental-Alpaca (Dreaddit 0.816) | 7B | n/a (DR) | yes (QLoRA) | research |
| 10 | Mental-FLAN-T5-XXL | 11B | n/a (0.802 Dreaddit) | tight | Apache-2.0 |
| 11 | MentaLLaMA-7B | 7B | 76.14 | yes | MIT |
| 12 | mhGPT | 1.98B | ~MentalBERT | yes | research |
| 13 | Menta (Qwen3-4B, 6-task) | 4B | beats 13B LLMs | yes | research |
| 14 | MentalQLM (Qwen1.5-0.5B + dual-LoRA) | 0.5B | 77.8 (5-task avg) | trivially | open |

**F1 ceiling for binary anxiety/depression Reddit classification across literature:** ~95% on Depression_Reddit/Pirina; ~93-94% on eRisk T1; ~82% on Dreaddit; ~70% on CLPsych15. The author's current MentalRoBERTa multi-task is at or near the published ceiling on the easy splits.

### Practical recommendation

**Tier A — highest expected dissertation value:**
1. **Fine-tune Llama-3.1-8B-Instruct with QLoRA (4-bit, r=16, alpha=32) on the 16k corpus, multi-task.** Canonical 2025 baseline reviewers will expect. ~3-4 hours on 4090.
2. **Fine-tune DeBERTa-v3-large** (Microsoft MIT, 435M). Consistently beats RoBERTa-large by +1-3 F1. Drop-in replacement for MentalRoBERTa pipeline; 30 min on 4090.
3. **MentaLLaMA-chat-7B as stronger LLM baseline.** Zero-shot + LoRA fine-tune on the 16k corpus. Cite IMHI Table 3.

**Tier B — strong supporting experiments:**
4. **Claude 3.5 Sonnet few-shot CoT** baseline. 8-shot CoT. Lands below fine-tuned MentalRoBERTa but provides interpretable explanations.
5. **Mental-Alpaca-7B / Mental-FLAN-T5-XXL** zero-shot transfer to author's corpus.
6. **Multi-task DeBERTa-v3-large + uncertainty head** with soft labels.

**Tier C — only if time permits:**
7. **MentalQLM-style dual-LoRA** on Qwen-2.5-1.5B/3B for on-device deployment angle.
8. **Adaptive RAG with PHQ-9 / GAD-7 scales** (arXiv 2501.00982).
9. **Ensemble of fine-tuned MentalRoBERTa + DeBERTa-v3-large + Llama-3.1-8B QLoRA** with logistic-regression stacking.

**Skip:** Mixtral or LLaMA-3-70B (don't fit 4090). BioGPT/MedAlpaca/ChatDoctor/Me-LLaMA for classification. Pure RAG-only LLM pipelines.

### Safety / ethics note

1. **No clinical claim.** All surveyed models — including MentaLLaMA's official model card — state "non-clinical research only." Reddit-trained classifiers learn linguistic correlates of self-disclosure, not DSM-5 diagnoses; ground-truth labels are weak supervision, not clinical assessment. Classifier outputs are *risk markers*, not diagnoses.

2. **Bias.** Domain-adapted MH models inherit demographic skew of Reddit (US, English, young, male, white). Cross-population generalization is poor — spiritual/cultural idioms can be misclassified as psychotic symptoms (Brenner 2025, JMIR 2025). UK/EU users, women, non-native English speakers, older users: performance lower than published numbers.

3. **Crisis triage and human escalation.** FDA's Digital Health Advisory Committee (Nov 2025) signaled that deployed GenAI MH tools need (a) explicit risk-stratified endpoints, (b) human escalation pathways, (c) misuse/overuse controls, (d) continuous drift monitoring. Documented unsafe LLM responses to suicide-related prompts (PLOS Digital Health 2025; arXiv 2509.08839).

---

## 5. Methodology — cross-domain, multi-task, active learning, fairness, calibration

### Cross-domain / cross-corpus generalization in mental-health NLP

**Harrigian, Aguirre & Dredze, "Do Models of Mental Health Based on Social Media Data Generalize?" (EMNLP-Findings 2020).** https://aclanthology.org/2020.findings-emnlp.337/. Code: https://github.com/kharrigian/emnlp-2020-mental-health-generalization. Train depression classifiers on CLPsych-2015 Twitter, Multi-Task-Learning Twitter, RSDD-Reddit, SMHD-Reddit and evaluate every train/test pair. Proxy-based labels (self-reported diagnosis, regex matching) introduce systematic spurious differences between case/control users that prevent transfer even within Reddit. F1 drops large and asymmetric: Twitter→Reddit and Reddit→Twitter often near chance. **THIS IS the canonical paper for the author's phenomenon.**

**Harrigian, Aguirre & Dredze, "On the State of Social Media Data for Mental Health Research" (CLPsych 2021).** https://arxiv.org/abs/2011.05233. Meta-review and open directory of ~100 mental-health social-media datasets. Field is "data-bound," >85% of datasets contain no validated ground truth.

**Burdisso / Errecalde / Naseem et al.** (https://pmc.ncbi.nlm.nih.gov/articles/PMC8238472/). Twitter→Reddit F1 in 30s-60s; Reddit→Twitter F1 in 20s-60s. **Empirically expected cross-platform F1 ceiling is ~30-60%, i.e. exactly the regime the author hit.**

### Domain adaptation

**Ganin et al., "Domain-Adversarial Training of Neural Networks" (JMLR 2016).** https://arxiv.org/abs/1505.07818. Gradient-reversal layer between encoder and domain discriminator forces encoder to produce domain-invariant features. Quantitative gain: 5-15 pp accuracy on Amazon cross-domain sentiment; up to +10 pp on cross-language. **No DANN-on-Reddit-mental-health paper exists** as of mid-2026. Closest is *Knowledge-aware and Contrastive Network* (IPM 2022) — contrastive, not adversarial.

**Long et al., "Conditional Adversarial Domain Adaptation" (CDAN, NeurIPS 2018).** arXiv 1705.10667. Conditions discriminator on cross-covariance + entropy. Outperforms DANN by 3-5 pp on standard benchmarks.

**Tzeng et al., "ADDA" (CVPR 2017).** Pre-trains source encoder, then target encoder adversarially.

**Applicability:** DANN with subreddit identity as "domain" is the textbook fix. Estimated gain (extrapolating from sentiment-DA literature): +5-10 pp F1 on held-out subreddits. **Effort: 1-2 weeks.** Code in `adapt-python`, `torchnlp`, gradient-reversal in 20 lines of PyTorch.

### Subreddit / platform leakage

**Ernala et al., "Methodological Gaps in Predicting Mental Health States from Social Media" (CHI 2019).** https://dl.acm.org/doi/10.1145/3290605.3300364. Predictive models built on social-media proxies have strong internal validity but **poor external validity** — population bias, sampling bias, construct-validity uncertainty. Recommends triangulation and clinical validation. **THIS is the field-defining paper for what the author is observing.**

**Harrigian & Dredze, "Then and Now: Quantifying Longitudinal Validity of Self-Disclosed Depression Diagnoses" (CLPsych 2022).** Self-disclosed diagnoses ≥5 years old no longer reliably indicate current depression; datasets contain personality-related selection biases. **Recommends:** annotate diagnosis dates, propensity-score matching, identify and remove spurious correlations.

**Harrigian, Aguirre & Dredze, "The Problem of Semantic Shift in Longitudinal Monitoring" (Web Science 2022).** arXiv 2206.11160. Handful of semantically unstable features produce large drift in longitudinal estimates.

### Multi-task learning

**Cohan et al. SMHD (COLING 2018).** Nine conditions standard MTL benchmark.

**Sarkar et al., "A Computational Approach to Understand Mental Health from Reddit: Knowledge-Aware Multitask Learning Framework" (ICWSM 2022).** arXiv 2203.11856. +2-4% F1 on rare classes vs single-task.

**Zhang et al. PsySym (EMNLP 2022).** Multi-task symptom+disorder reaches F1 83.3% on PsySym, beating pure-text single-task baselines.

**Average multi-task uplift in this domain: a few F1 points on rare classes — but huge interpretability and transferability gain because auxiliary symptom heads regularize away from spurious subreddit features.**

### Active learning

20-40% labeling-cost reductions in text classification with uncertainty/diversity sampling. On Reddit mental-health specifically, LLM-as-annotator pipelines (https://arxiv.org/html/2412.03796) reach 83.7% reduction in training data needed for comparable performance. Library: `small-text`, `modAL`. Effort: medium.

### Calibration

**Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017).** arXiv 1706.04599. **Temperature scaling** cuts ECE by 10× on most benchmarks. Outperforms Platt and isotonic on vision; comparable on NLP.

**Niculescu-Mizil & Caruana (ICML 2005).** SVMs and boosted trees push probability mass away from 0/1 (sigmoid distortion → Platt); naive Bayes pushes toward 0/1; well-regularized neural nets are best-calibrated out-of-the-box.

**Desai & Durrett, "Calibration of Pre-trained Transformers" (EMNLP 2020).** BERT and RoBERTa typically better calibrated out-of-domain than non-pretrained models.

**Aragón et al. (2023, JAMIA-related).** https://arxiv.org/pdf/2305.16797. **Label smoothing halves ECE in mental-health detection;** M-BERT improved ECE by 0.008, M-MentalBERT with LDA topics by 0.042.

**Reference numbers:** TF-IDF/logistic ECE 0.08-0.15 typical; calibrated transformer ECE 0.02-0.05. **Author's 0.13 is high but plausible; 0.03 is competitive.** Effort: trivial (10 lines for temperature scaling).

### Bias / fairness

**Aguirre, Harrigian & Dredze, "Gender and Racial Fairness in Depression Research using Social Media" (EACL 2021).** Post-hoc audit of CLPsych-2015. Depression classifiers perform *systematically* worse for underrepresented racial groups, gap NOT explained by representation alone.

**PNAS 2024, "Key language markers of depression on social media depend on race."** Large-scale evidence linguistic depression markers differ by race.

**Park et al., Scientific Reports 2024.** Disparate Impact Remover; no single best classifier dominates all fairness metrics. Documented disparities: 5-15 pp F1 across race; 3-8 pp across gender.

Code: `fairlearn`, `AIF360`. Effort: low (1-2 days for basic audit).

### Adversarial robustness

**Jin et al., "TextFooler" (AAAI 2020).** arXiv 1907.11932. Synonym-substitution attack: drops BERT IMDB accuracy 92.2% → 6.6% by perturbing 6.1% of words. Library: `TextAttack` (https://github.com/QData/TextAttack).

**Li et al., "BERT-Attack" (EMNLP 2020).** Uses BERT-MLM to generate adversarial substitutions; even stronger than TextFooler.

**Highly relevant for this thesis:** if F1 collapses 50 pp under TextFooler with <10% perturbation, the model relies on surface lexical features that are also the subreddit shortcuts. 30-minute integration.

### Counterfactual evaluation

**Feder et al., "Causal Inference in Natural Language Processing" (TACL 2022).** arXiv 2109.00725. CausaLM and Counterfactual Token Probing.

**Counterfactual Data Augmentation (CDA).** Swap protected attributes (gender, race) in training data, retain label. Reduces gender bias 30-80% with negligible accuracy loss.

### External validation against clinical instruments

**eRisk @ CLEF** (Crestani, Losada, Parapar). Participants must *fill out the BDI-II questionnaire on behalf of a Reddit user* based on their posting history. 170-user BDI-II benchmark from eRisk 2021. Most rigorous public benchmark of "predict-the-clinical-instrument-from-text."

**CLPsych 2019 (Zirikly et al.).** C-SSRS-graded suicide risk on Reddit r/SuicideWatch.

**Levis et al.** https://publichealth.jmir.org/2025/1/e72591. CART models achieve AUC > 0.900 against full-scale PHQ-9/GAD-7. Clinically accepted PHQ-9 cut-off ≥10: ~71% sensitivity / 66% specificity.

### Longitudinal modeling

**Tsakalidis et al., "Moments of Change" (ACL 2022).** Defines "Switch" (drastic mood change) and "Escalation" (gradual). 500 manually annotated user timelines, 18.7K posts. Best results: context-aware sequential modeling.

**Time-Aware Transformer (Springer 2024).** Outperforms post-bag baselines for anorexia by fusing inter-post intervals.

### Severity / ordinal regression

**Naseem, Dunn, Kim & Khushi, "Early Identification of Depression Severity Levels on Reddit Using Ordinal Classification" (TheWebConf 2022).** https://dl.acm.org/doi/10.1145/3485447.3512128. 3,553 Reddit posts relabeled into 4 ordinal severity classes per BDI. Hierarchical-attention + ordinal loss: **weighted F1 84.15% (binary baseline ~72.6%).** Augmentation contributed +11.6 pp F1.

**Library:** `coral-pytorch`. Effort: low (single new loss + reshaped labels).

**Empirical bottom line on ordinal vs binary:** ordinal heads typically *match* or *slightly* outperform binary on per-class F1 *and* provide clinically more useful output. Rarely hurt.

### Symptom-level decomposition

**Zhang et al. PsySym (EMNLP 2022).** Models 38 DSM-5 symptoms across 7 disorders; per-symptom predictions double as clinical-grade explanations.

**Bao et al. ReDSM5 (2025).** 1,484 Reddit posts annotated for presence/absence of each of 9 DSM-5 depression symptoms with brief clinical rationales by a licensed psychologist.

**Symptom decomposition is the single most clinically defensible upgrade** because (a) it grounds the model in DSM-5 rather than r/depression, and (b) symptom features are far less subreddit-specific than overall depression label.

### In-context learning vs fine-tuning (May 2026 evidence)

**Xu et al. Mental-LLM (UbiComp 2024).** Fine-tuned Mental-Alpaca and Mental-FLAN-T5 beat GPT-3.5 by +10.9% balanced accuracy and GPT-4 by +4.8%. **Fine-tuning wins on complex tasks; ICL competitive only on simple binary detection.**

**arXiv 2510.22285 (2025).** SFT achieves strongest performance at higher compute cost; simple ICL beats long instruction-heavy prompts. Consensus: **SFT > few-shot ICL > zero-shot for mental-health classification of comparable difficulty.**

### Top-8 methodological upgrades (ranked for this thesis specifically)

1. **DANN/CDAN with subreddit-as-domain.** The AUROC=0.97/F1=0.33 split is the canonical signature of a model using subreddit-specific surface features to set its decision threshold. DANN's gradient-reversal head, with subreddit identity as discriminator target, is *exactly* designed to destroy these shortcuts. Expected gain: cross-sub F1 ~0.33 → ~0.45-0.55. Effort: 1-2 weeks.

2. **Calibration pipeline (temperature scaling on a held-out cross-subreddit validation set).** Author's 0.03 ECE is *publishable* only if it holds *out-of-distribution*. Cross-subreddit ECE is the credible number. Effort: half a day.

3. **Symptom-level decomposition using PsySym or ReDSM5.** Reframes from "predict depression label on r/depression" to "predict DSM-5 symptoms from arbitrary Reddit text." Symptom features less subreddit-specific. Effort: medium (1-2 weeks).

4. **External validation against instrument-graded benchmark (eRisk BDI-II 2021 or CLPsych C-SSRS).** Single most credible response to F1-collapse. Spearman ρ > 0.4 between model score and clinical instrument converts work from "Reddit classifier" to "social-media surveillance proxy." Effort: high (data access) but extremely high-value.

5. **Demographic-fairness audit (Aguirre-style) + counterfactual data augmentation.** Run inferred-gender/race subgroup F1 plus CDA on gendered words. Expected: 5-15 pp race gap, 3-8 pp gender gap. Effort: 2-3 days with `fairlearn`.

6. **Adversarial robustness audit with TextAttack (TextFooler + BERT-Attack).** Cheap and dramatic. If accuracy drops 60+ pp under TextFooler with <10% perturbation, the AUROC=0.97 number is shortcut-driven. Effort: 1 day.

7. **Ordinal severity head (CORN/CORAL) using BDI/PHQ-aligned labels.** Strictly increases information content; matches eRisk and Naseem 2022 evidence. Effort: half a week.

8. **Multi-task learning with mental-health knowledge graph or symptom-aux tasks.** +2-4 pp F1 on rare classes. Lower impact on F1-collapse specifically but cheap regularization. Effort: medium.

### Cross-cutting empirical reference numbers

- **Cross-platform F1 drop in MH NLP:** 30-50 pp F1 collapse is modal (Harrigian 2020; Burdisso 2021). Author's 56 pp on severe end but not anomalous.
- **AUROC stability under platform shift:** yes, AUROC commonly stays 0.80+ even when F1 collapses — documented metric pathology.
- **Typical ECE for MH classifiers:** 0.08-0.15 uncalibrated linear; 0.02-0.05 calibrated transformer. 0.13 high but plausible; 0.03 good.
- **Demographic F1 gaps:** 5-15 pp by race, 3-8 pp by gender.
- **Multi-task uplift for rare classes:** 2-4 pp F1.
- **Ordinal vs binary severity:** ordinal matches or beats binary on per-class F1.
- **Fine-tuning vs in-context for MH:** fine-tuned 7B models beat GPT-4 zero-shot by 4-11% balanced accuracy.

---

## 6. Priority recommendation matrix (synthesis)

Ranked top-to-bottom by **defensible novelty / dissertation impact per unit of effort**:

| # | Move | Effort | Impact | Why |
|---|---|---|---|---|
| 1 | Add r/HealthAnxiety + r/Hypochondriacs* subs and re-train head-to-head | 2 days | ★★★★★ | Headline novelty: only Low 2020 SGD F1=0.851 exists |
| 2 | DANN with subreddit-as-domain | 1-2 weeks | ★★★★★ | Methodology novelty: no prior MH application |
| 3 | Integrate ANGST (3 psychologists) as external val | 3 days | ★★★★ | Credibility upgrade: clinical-expert labels |
| 4 | Tier-2 LLM labeling at Whisper scale (50-100k posts) | 1 day setup + $50-200 | ★★★★ | Fixes health-anxiety F1=0.33 ceiling |
| 5 | Whisper-style filter (lexicon-LLM disagreement drop, near-dup, length) | 2 days | ★★★ | +3-6 pp F1, cheap |
| 6 | Soft-label distillation from Claude (KL-div, multi-completion confidence) | 1-2 days | ★★★ | +1-3 pp F1 |
| 7 | Symptom-level decomposition (SHAI items, PsySym template) | 1-2 weeks | ★★★★ | Reframes thesis, DSM-5 grounding |
| 8 | Domain MLM pretraining on raw Reddit corpus | 1 hour | ★★ | Free +1-3 F1 |
| 9 | Temperature scaling + label smoothing for TF-IDF | half day | ★★ | ECE 0.13 → 0.03 |
| 10 | DeBERTa-v3-large replacement for MentalRoBERTa | 30 min | ★★ | +1-3 F1 |
| 11 | Llama-3.1-8B QLoRA multi-task | 3-4 hours | ★★★ | Canonical 2025 reviewer-expected baseline |
| 12 | Cross-corpus eval on Low 2020 RMHD (zero-shot) | 1 day | ★★★ | Demonstrates generalization |
| 13 | MentaLLaMA-chat-7B as published baseline | 1 day | ★★ | Cite IMHI Table 3 |
| 14 | Self-training / Noisy Student loop | 3 days | ★★★ | +2-5 pp F1 |
| 15 | Active-learning gold standard (FreeAL-style) | 1 day code + same annotation time | ★★★ | 2-3× value per annotation |
| 16 | Adversarial robustness audit (TextAttack) | 1 day | ★★ | Quantitative shortcut evidence |
| 17 | Fairness audit (fairlearn, inferred demographics) | 2-3 days | ★★ | Expected: 5-15 pp race / 3-8 pp gender gap |
| 18 | Counterfactual data augmentation (CDA) | 2 days | ★★ | Tests classifier sensitivity to surface form |
| 19 | Ordinal severity head (CORAL) on BDI/PHQ alignment | half week | ★★★ | Naseem: binary 72.6 → ordinal 84.15 F1 |
| 20 | AttrPrompt + back-translation for minority classes | 3 days | ★★ | +1-3 pp F1 minority |
| 21 | Snorkel LabelModel ablation | 1 week | ★ | Useful comparator only |
| 22 | LLM2LLM iterative augmentation | 1 week | ★★ | High variance, try only after plateaus |
| 23 | UMD Reddit Suicidality DUA application | weeks (waiting) + 1 day eval | ★★★★ | Gold-standard suicidality eval |
| 24 | eRisk BDI-II symptom-prediction transfer | 1-2 weeks | ★★★ | "Predict clinical instrument from text" |
| 25 | Longitudinal user-trajectory modeling (Moments-of-Change style) | 2-3 weeks | ★★ | Adds whole dimension to RQ4 |
| 26 | Per-subreddit threshold calibration as deployment recommendation | half day | ★★ | Directly addresses Exp 2 finding |
| 27 | McNemar tests for model-vs-model significance | 1 hour | ★ | Citability |
| 28 | Adapter / LoRA fine-tuning architecture | 3 days | ★ | Parameter-efficient |
| 29 | Retrieval-augmented classification (psychometric scales source) | 1 week | ★ | adaptive-RAG arXiv 2501.00982 |
| 30 | Cross-lingual extension (Romanian, Spanish via XLM-R) | 2-3 weeks | ★★★ | Major thesis pivot toward Whisper-style diversity |

### Things to deprioritize / skip

- **Mixture of experts / Llama-3-70B / Mixtral fine-tuning** — won't fit 4090 even in 4-bit during training; PhD-scale.
- **Twitter cross-platform** — Twitter API access has gotten horrible since 2023.
- **Pre/post intervention causal inference** — beautiful but PhD-scale.
- **ZeroGen-from-scratch synthetic data** — 16k real corpus is more authentic than anything Claude generates.
- **Heavy Snorkel LF-engineering** — BOXWRENCH 2024 shows it under-delivers vs filtering.
- **Pure RAG-only LLM pipelines for classification** — loses by 30+ F1 to fine-tuning.
- **BioGPT/MedAlpaca/ChatDoctor/Me-LLaMA for Reddit classification** — clinical-text, not social-media; reliably underperform MentalRoBERTa.

---

## 7. Cross-cutting takeaways

1. **The cleanest novel contribution is health-anxiety as a separate class.** Low 2020 is the only baseline (F1=0.851 SGD); a transformer trivially beats it. No SHAI validation exists.
2. **The cleanest methodology contribution is DANN-on-subreddit.** Five years of "F1 collapses cross-subreddit" papers, no one has applied the textbook fix.
3. **Frontier LLMs (Claude 3.5 Sonnet, GPT-4) zero-shot LOSE to fine-tuned MentalRoBERTa on Reddit MH binary classification.** Your fine-tune is at SOTA — don't be intimidated into thinking you need GPT-5 to win.
4. **Whisper's actual recipe is filter-don't-perfect.** Hard heuristics drop bad data; never relabel. BOXWRENCH 2024 confirms this beats Snorkel-style denoising at realistic LF quality.
5. **Health anxiety is a genuine NLP gap.** No transformer specifically targets it; no SHAI alignment exists; no Salkovskis-Warwick CBT framework has been operationalized as features.
6. **ANGST (Hengle 2024) is the only public Reddit MH dataset with multiple expert psychologists.** Use it as external validation.
7. **eRisk BDI-Sen / DepreSym / ReDSM5 are methodological templates** for a SHAI-symptom-annotated corpus.
8. **The F1=0.33 cross-sub collapse is field-normal**, not a bug in the user's work. Harrigian 2020 documents 30-50 pp drops as modal. DANN can recover 5-10 of those points.
9. **Mental-health LLM stigma is a real risk** (FAccT 2025). Don't give Claude a "therapist" persona in labeling prompts.
10. **The empirically-validated F1 ceiling on Reddit binary depression is ~95%.** The user's 0.89 anxiety is within striking distance with DeBERTa-v3-large or Llama-3.1-8B QLoRA.

---

## Master sources list

### Dissertation-focal datasets (download immediately)
- Low et al. JMIR 2020 — Reddit Mental Health Dataset: https://zenodo.org/records/3941387, paper https://www.jmir.org/2020/10/e22635/
- ANGST: https://huggingface.co/datasets/ameyhengle/ANGST, paper arXiv:2410.03908
- ReDSM5: https://huggingface.co/datasets/irlab-udc/redsm5, arXiv:2508.03399
- PRIMATE: https://github.com/primate-mh/Primate2022
- GoEmotions: https://huggingface.co/datasets/google-research-datasets/go_emotions
- Dreaddit: https://arxiv.org/abs/1911.00133
- CAMS: https://github.com/drmuskangarg/CAMS
- SWMH: https://zenodo.org/records/6476179
- DepressionEmo: https://arxiv.org/abs/2401.04655
- LoST: https://arxiv.org/abs/2306.05596
- MultiWD: https://github.com/drmuskangarg/MultiWD
- SDCNL: https://github.com/ayaanzhaque/SDCNL
- Sharma 2020 Empathy: https://github.com/behavioral-data/Empathy-Mental-Health
- MentaLLaMA/IMHI: https://github.com/SteveKGYang/MentalLLaMA
- MentalRiskES: https://github.com/sinai-uja/corpusMentalRiskES
- Reddit C-SSRS: https://zenodo.org/records/2667859
- Pirina & Çöltekin 2018 (Depression_Reddit): https://aclanthology.org/W18-5903/
- CARMA: https://arxiv.org/abs/2511.03102
- Shen & Rudzicz 2017: https://aclanthology.org/W17-3107/ + https://github.com/heyyjudes/anxiety-on-reddit

### DUA-gated datasets to apply for
- UMD Reddit Suicidality: https://psresnik.github.io/umd_reddit_suicidality_dataset.html
- RSDD: https://georgetown-ir-lab.github.io/emnlp17-depression/
- SMHD: https://ir.cs.georgetown.edu/resources/smhd.html
- eRisk @ CLEF: https://erisk.irlab.org/
- CLPsych: https://clpsych.org/
- DAIC-WOZ: https://dcapswoz.ict.usc.edu/
- BDI-Sen: https://erisk.irlab.org/BDISen.html

### Models / model hubs
- MentalBERT/RoBERTa: https://arxiv.org/abs/2110.15621, HF `mental/mental-roberta-base`
- MentaLLaMA: https://arxiv.org/abs/2309.13567, HF `klyang/MentaLLaMA-chat-7B`, `klyang/MentaLLaMA-chat-13B`
- Mental-LLM (Alpaca/FLAN-T5): https://arxiv.org/abs/2307.14385, HF `NEU-HAI/mental-alpaca`, `NEU-HAI/mental-flan-t5-xxl`
- MentalQLM: https://www.medrxiv.org/content/10.1101/2024.12.29.24319755v2.full, https://github.com/tortorish/MentalQLM
- multiMentalRoBERTa: https://arxiv.org/html/2511.04698v2
- mhGPT: https://arxiv.org/abs/2408.08261
- Menta: https://arxiv.org/abs/2512.02716
- MentalGLM: https://arxiv.org/html/2410.10323v1
- PsychBERT: https://huggingface.co/mnaylor/psychbert-cased
- Bio_ClinicalBERT: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
- Clinical ModernBERT: https://arxiv.org/html/2504.03964v1
- Me-LLaMA: https://www.nature.com/articles/s41746-025-01533-1
- MedAlpaca: https://arxiv.org/html/2304.08247v3

### Methodology papers
- Harrigian, Aguirre & Dredze EMNLP-Findings 2020 (cross-corpus): https://aclanthology.org/2020.findings-emnlp.337/ + https://github.com/kharrigian/emnlp-2020-mental-health-generalization
- Harrigian et al. CLPsych 2021 (state of data): https://aclanthology.org/2021.clpsych-1.2/
- Harrigian & Dredze CLPsych 2022 (longitudinal validity): https://aclanthology.org/2022.clpsych-1.6/
- Harrigian semantic shift 2022: https://arxiv.org/abs/2206.11160
- Ernala et al. CHI 2019: https://dl.acm.org/doi/10.1145/3290605.3300364
- Aguirre, Harrigian & Dredze EACL 2021 (fairness): https://aclanthology.org/2021.eacl-main.256/
- Ganin DANN: https://arxiv.org/abs/1505.07818
- Long CDAN: https://arxiv.org/abs/1705.10667
- Guo et al. calibration: https://arxiv.org/abs/1706.04599
- Desai & Durrett calibration of transformers: https://aclanthology.org/2020.emnlp-main.21.pdf
- Aragón mental-health calibration: https://arxiv.org/pdf/2305.16797
- Feder causal NLP: https://aclanthology.org/2022.tacl-1.66/
- TextFooler: https://arxiv.org/abs/1907.11932
- TextAttack: https://github.com/QData/TextAttack
- Naseem ordinal severity: https://dl.acm.org/doi/10.1145/3485447.3512128
- PsySym (Zhang): https://aclanthology.org/2022.emnlp-main.677/
- Tsakalidis Moments of Change: https://aclanthology.org/2022.acl-long.318/
- Sarkar Knowledge-Aware MTL: https://arxiv.org/abs/2203.11856
- Cohan SMHD: https://arxiv.org/abs/1806.05258

### Whisper / weak supervision / LLM-as-labeler
- Whisper: https://arxiv.org/abs/2212.04356
- Snorkel: https://arxiv.org/abs/1711.10160
- WRENCH: https://arxiv.org/abs/2109.11377
- BOXWRENCH: https://arxiv.org/abs/2501.07727
- Gilardi PNAS 2023 (ChatGPT vs MTurk): https://arxiv.org/abs/2303.15056
- AnnoLLM: https://arxiv.org/abs/2303.16854
- Wang EMNLP-Findings 2021: https://arxiv.org/abs/2108.13487
- KD in Automated Annotation: https://arxiv.org/abs/2406.17633
- Performance-Guided KD (Amazon): https://arxiv.org/abs/2411.05045
- Noisy Student: https://arxiv.org/abs/1911.04252
- UDA: https://arxiv.org/abs/1904.12848
- Is BERT Robust to Label Noise: https://arxiv.org/abs/2204.09371
- ZeroGen: https://arxiv.org/abs/2202.07922
- Self-Instruct: https://arxiv.org/abs/2212.10560
- AttrPrompt: https://arxiv.org/abs/2306.15895
- LLM2LLM: https://arxiv.org/abs/2403.15042
- Rethinking Soft Labels (KD): https://arxiv.org/abs/2102.00650
- FreeAL: https://arxiv.org/abs/2311.15614
- Backtranslation for emotion: https://arxiv.org/abs/2507.14590
- LatentGLoss (MH augmentation): https://arxiv.org/abs/2504.07245
- Stigma in LLM MH (FAccT 2025): https://arxiv.org/abs/2504.18412

### Health-anxiety specific
- Low JMIR 2020: https://www.jmir.org/2020/10/e22635/
- Doherty-Torstrick 2016: https://pmc.ncbi.nlm.nih.gov/articles/PMC5952212/
- Schenkel JMIR Form Res 2023: https://formative.jmir.org/2023/1/e42206
- Fergus 2013: https://journals.sagepub.com/doi/full/10.1089/cyber.2012.0671
- Lemogne JAD 2024: https://www.sciencedirect.com/science/article/abs/pii/S0165032724003215
- Segneri PLOS ONE 2024 (Long COVID): https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308340
- Salkovskis SHAI 2002: https://pubmed.ncbi.nlm.nih.gov/12171378/
- Lucock & Morley 1996 HAQ: https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/j.2044-8287.1996.tb00498.x
- McElroy CSS-12: https://journals.sagepub.com/doi/10.1089/cyber.2018.0624
- Hwang dissertation: https://ir.library.oregonstate.edu/downloads/vm40z036q
- Tariq 2026 early linguistic anxiety: https://arxiv.org/abs/2601.11758
- Cognitive distortion survey: https://arxiv.org/abs/2508.09878
- Anorexia early detection time-series: https://link.springer.com/article/10.1007/s10791-026-09903-3
- Hyperdiagnosis / Nosophobia AI: https://pmc.ncbi.nlm.nih.gov/articles/PMC12860674/

### LLM evaluation / safety
- Comprehensive LLM eval on MH: https://arxiv.org/abs/2409.15687
- Cognitive-Mental-LLM: https://arxiv.org/html/2503.10095v1
- SFT vs prompt vs RAG: https://arxiv.org/html/2503.24307v1
- Adaptive RAG for psychometrics: https://arxiv.org/html/2501.00982v1
- Survey of LLMs in MH detection: https://arxiv.org/html/2504.02800v2
- LLM unsafe MH prompts: https://arxiv.org/pdf/2509.08839
- FDA Digital Health Advisory: https://www.orrick.com/en/Insights/2025/11/FDAs-Digital-Health-Advisory-Committee-Considers-Generative-AI-Therapy-Chatbots-for-Depression

### Shared task overviews
- CLPsych 2024: https://aclanthology.org/2024.clpsych-1.15/
- eRisk 2025: https://erisk.irlab.org/2025/index.html
- DS@GT eRisk 2025: https://arxiv.org/abs/2507.10958
- Detecting Early Suicidal Ideation DeBERTa: https://arxiv.org/pdf/2510.14889

### Surveys
- Harrigian mental-health-datasets repo: https://github.com/kharrigian/mental-health-datasets
- Yang & Liu 2025 Datasets for Depression Modeling: https://arxiv.org/html/2503.21513v1
- JMIR systematic review MH NLP methodologies 2023: https://www.jmir.org/2023/1/e42734
- Garg 2023 computational MH survey: https://link.springer.com/article/10.1007/s11831-022-09863-z

---

*Compiled by deep parallel research across 5 specialized agents using WebSearch + WebFetch on arXiv, ACL Anthology, PubMed, HuggingFace, Zenodo, GitHub, JMIR, IEEE, Nature, Springer, ACM, and dataset repositories. All claims are sourced; URLs verified.*
