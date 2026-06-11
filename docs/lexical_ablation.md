# Lexical-ablation probe (keyword reliance)

Models trained on r/HealthAnxiety vs r/Anxiety, then evaluated on normal test text and on test text with every clinical-lexicon word/phrase removed (the same lexicons that build the weak labels). The drop measures how much of the score is keyword spotting. `scripts/exp_lexical_ablation.py`.

| model | clean_f1 | masked_f1 | f1_drop | clean_auroc | masked_auroc | auroc_drop |
|---|---|---|---|---|---|---|
| TF-IDF (n=5 seeds) | 0.8871 | 0.8376 | 0.0495 | 0.9378 | 0.8993 | 0.0384 |
| MentalRoBERTa (1 seed) | 0.9091 | 0.8546 | 0.0545 | 0.9581 | 0.9263 | 0.0318 |

![lexical ablation](figures/lexical_ablation.png)