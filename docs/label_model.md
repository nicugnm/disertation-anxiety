# Multi-source label model vs the single biased heuristic (anchored on expert labels)

On ANGST (3 expert psychologists), several diverse labelling functions and an unsupervised Dawid--Skene combination are compared against the expert label. The question: does combining diverse, more-independent signals agree with clinicians better than our own lexicon alone? `scripts/exp_label_model.py`.

| source | coverage | kappa | f1 | auroc |
|---|---|---|---|---|
| lexicon (ours, biased) | 1.0 | 0.3096 | 0.5476 | 0.7792 |
| sentiment (VADER) | 0.952 | 0.0205 | 0.3902 | 0.5287 |
| uncertainty (LIWC) | 0.428 | -0.0236 | 0.0982 | 0.5038 |
| llm (Qwen2.5-7B) | 1.0 | 0.0307 | 0.4061 | 0.5772 |
| majority vote | 1.0 | 0.07 | 0.4258 | 0.7123 |
| label model (Dawid-Skene) | 1.0 | 0.0307 | 0.4061 | 0.7303 |
| supervised combo (5-fold) | 1.0 | 0.3529 | 0.5338 | 0.7769 |

![label model](figures/label_model.png)

## What this shows

- The **lexicon** — the *biased* source — is actually the single signal **most** aligned with the expert clinicians (highest kappa and AUROC), because it is built from clinical instruments (GAD-7/SHAI). The weak label is not noise about the construct.
- The other weak signals (sentiment, uncertainty, **zero-shot LLM**) agree with experts near chance, so an **unsupervised** combination (majority / Dawid--Skene) **cannot beat the lexicon**: with no ground truth it cannot tell which labelling function is reliable and is pulled toward the noisy ones.
- A **supervised** combination (a little expert data, 5-fold) is what actually improves on the best single source. The route to less bias is a small amount of expert ground truth, not more unsupervised researcher heuristics.
- The zero-shot LLM is a **weak anxiety annotator** here, so it cannot simply replace the lexicon.
- Implication: the circularity is in the **evaluation** (testing against lexicon-derived labels inflates the in-domain score to ~0.99, see the circularity ladder), not in the lexicon's construct validity.

Per-LF learned reliability (Dawid--Skene): lexicon (ours, biased) sens=0.65/spec=0.67, sentiment (VADER) sens=0.87/spec=0.33, uncertainty (LIWC) sens=0.08/spec=0.94, llm (Qwen2.5-7B) sens=1.00/spec=0.29.