# Statistical significance of paired model comparisons

McNemar's test on each model's saved decisions + paired-bootstrap ΔAUROC (95% CI, two-sided p). Models aligned on shared post `id`s. **`significant_95 = True` means the AUROC difference is real at the 0.05 level.**

_Regenerate: `python scripts/significance.py`_

| target | model_a | model_b | n_common | n_pos | mcnemar_b | mcnemar_c | mcnemar_p | auroc_a | auroc_b | delta_auroc | ci_lo | ci_hi | boot_p | significant_95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anxiety | mentalbert_anxiety | multitask_anxiety_health_dep_suic | 2458 | 534 | 40 | 43 | 0.82624 | 0.9846 | 0.9871 | -0.0026 | -0.006 | 0.0009 | 0.154 | False |
| anxiety | mentalbert_anxiety | tfidf_logreg | 225 | 57 | 5 | 8 | 0.58105 | 0.9765 | 0.9692 | 0.0073 | -0.0191 | 0.0393 | 0.656 | False |
| anxiety | multitask_anxiety_health_dep_suic | tfidf_logreg | 225 | 57 | 5 | 13 | 0.09625 | 0.9751 | 0.9692 | 0.006 | -0.0218 | 0.042 | 0.768 | False |

![forest plot](figures/significance.png)