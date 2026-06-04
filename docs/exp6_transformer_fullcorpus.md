# Experiment 6 (full corpus) — multitask transformer per-target

Full-corpus multitask MentalRoBERTa (experiments/runs/multitask_fullcorpus/model, trained on a 205k sample) evaluated on a held-out test set of 30,000 posts disjoint from training. Best-threshold F1 / AUROC / AUPRC / ECE. `scripts/exp6_transformer_fullcorpus.py`.

_Regenerate: `python scripts/exp6_transformer_fullcorpus.py`_

| target | n | n_pos | f1 | auroc | auprc | ece | threshold |
|---|---|---|---|---|---|---|---|
| anxiety | 30000 | 2827 | 0.8617 | 0.9931 | 0.9366 | 0.0096 | 0.372 |
| health_anxiety | 30000 | 96 | 0.5729 | 0.9839 | 0.4937 | 0.0014 | 0.199 |
| depression | 30000 | 442 | 0.6855 | 0.9923 | 0.7558 | 0.0018 | 0.406 |
| suicidality | 30000 | 19 | 0.5 | 0.9954 | 0.4386 | 0.0003 | 0.088 |

![exp6 transformer](figures/exp6_transformer_fullcorpus.png)