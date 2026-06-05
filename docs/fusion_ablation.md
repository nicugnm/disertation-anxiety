# Experiment 10 — fusion architecture-surgery ablation

Ablation of `FusionMultiTaskModel` (`src/models/fusion.py`): clinical-feature fusion (26 linguistic + 7 SHAI), attention pooling, focal loss, on an author-disjoint split of the full corpus. In-domain held-out test + zero-shot RMHD/ANGST transfer (anxiety). `scripts/exp_fusion_ablation.py`. baseline == plain multitask.

_Regenerate: `python scripts/exp_fusion_ablation.py`_

| variant | anxiety_f1 | health_anxiety_f1 | depression_f1 | suicidality_f1 | anxiety_auroc | rmhd_auroc | angst_auroc |
|---|---|---|---|---|---|---|---|
| baseline | 0.8451 | 0.5079 | 0.6734 | 0.4444 | 0.9907 | 0.8945 | 0.7782 |
| fusion | 0.8447 | 0.5397 | 0.6537 | 0.56 | 0.9914 | 0.9052 | 0.7717 |
| attn | 0.8543 | 0.4727 | 0.6398 | 0.381 | 0.991 | 0.9006 | 0.7914 |
| focal | 0.8462 | 0.4889 | 0.6243 | 0.4545 | 0.992 | 0.9001 | 0.8307 |
| fusion_focal | 0.8519 | 0.5593 | 0.6397 | 0.5217 | 0.992 | 0.9309 | 0.8108 |
| all | 0.8547 | 0.4812 | 0.6305 | 0.5 | 0.9923 | 0.9103 | 0.8078 |

![fusion ablation](figures/fusion_ablation.png)