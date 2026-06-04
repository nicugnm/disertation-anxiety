# Whisper-style weak-label filtering

Confident-learning cleanup of the anxiety weak labels: out-of-fold scores flag examples where the model confidently disagrees with the weak label (score < 0.25 for weak-positives; > 0.75 for weak-negatives). Remove them, retrain, and evaluate on the held-out self-disclosure test set (disclosure users excluded from training). `src/labeling/filtering.py`, `scripts/weak_label_filtering.py`.

- Flagged: **881** (127 likely false-positives, 754 likely false-negatives) of 80,000.

_Regenerate: `python scripts/weak_label_filtering.py`_

| setting | n_train | flagged_removed | disclosure_user_auroc |
|---|---|---|---|
| original (all weak labels) | 80000 | 0 | 0.7447 |
| cleaned (confident issues removed) | 79119 | 881 | 0.7445 |

![weak-label filtering](figures/weak_label_filtering.png)

## Example flagged weak-POSITIVES the model rejects (likely off-topic in an anxiety sub)
- [agoraphobia, oof=0.090] I'm an overthinker so I do think about everything. But I don't make assumptions. From trying to save up money, to taking care of parents, to spending more time 
- [agoraphobia, oof=0.092] My therapist said this to me and I want you to say it to yourself every day: there is no age at which losing a parent to suicide will be okay for child. You nee
- [agoraphobia, oof=0.108] People can enjoy controlled adrenaline when the rest of their lives are safe. Nobody enjoys uncontrolled adrenaline. Someone who frequently bungee jumps would l
- [HealthAnxiety, oof=0.118] Hyperventilating . I could not control myself , hasn't happened in over 5 years .
- [agoraphobia, oof=0.119] It's season 16 I think where the kid needs the blood and this agoraphobic in England had special blood that he needed

## Example flagged weak-NEGATIVES the model flags (likely anxiety in a neutral sub)
- [ibs, oof=0.997] It's the opposite for me. Feels like my anxiety and stress cause my gut issues.
- [CPTSD, oof=0.997] Saying you have anxiety is not the same thing as having Generalized Anxiety Disorder.
- [SuicideWatch, oof=0.997] Maybe no sativa because of anxiety nothernlights is my recommendation
- [mentalhealth, oof=0.996] it's a beautiful movie but it freaked me out tbh. I'm so scared of going crazy already and it skyrocketed that anxiety
- [CasualConversation, oof=0.995] It's the anxiety for me. I need to stop. Glad you were able to.