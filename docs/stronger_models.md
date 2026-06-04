# Stronger encoders — r/HealthAnxiety vs r/Anxiety (submissions, author-disjoint)

Experiment-8 pipeline with larger/newer encoders. Baseline: Low 2020 SGD-L1 weighted-F1 = 0.851. `scripts/exp_stronger_models.py`.

_Regenerate: `python scripts/exp_stronger_models.py`_

| model | pretrained | weighted_f1 | auroc | f1 | note |
|---|---|---|---|---|---|
| Low 2020 (SGD-L1) | - | 0.851 | None |  | published baseline |
| TF-IDF + LogReg | - | 0.886 | 0.944 |  | Exp 8 |
| MentalRoBERTa | mental/mental-roberta-base | 0.906 | 0.955 |  | Exp 8 |
| roberta-large | roberta-large | 0.9156 | 0.962 | 0.8889 | this run |
| deberta-v3-base | microsoft/deberta-v3-base | None | None | None | FAILED: Input contains NaN. |

![stronger models](figures/stronger_models.png)

## Heavier variants (recipes — not run here)

- **Domain-adaptive MLM**: continue masked-LM pretraining of RoBERTa on the 744k-post corpus (`run_mlm.py`, ~1–2 GPU-hours), then fine-tune. Expected small gain on the noisy disclosure task; larger on in-domain classification.
- **Llama-3.1-8B QLoRA**: 4-bit `bitsandbytes` + `peft` LoRA adapters, instruction-framed binary classification. Needs `pip install peft bitsandbytes` and ~2–4 GPU-hours; fits in 24 GB at 4-bit. Decoder-only LLMs rarely beat a fine-tuned encoder on short-text binary classification, so this is a completeness check, not an expected SOTA.