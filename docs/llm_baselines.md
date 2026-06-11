# Phase 2 — generative-LLM baselines (r/HealthAnxiety vs r/Anxiety)

Same submissions-only, author-disjoint split as Experiment 8 / Idea 12. Decoder-only LLMs scored with a yes/no verbalizer over next-token logits (`src/models/llm_causal.py`); LoRA fine-tune uses 4-bit NF4 (QLoRA). LLMs evaluated on an author-disjoint test subsample (n=636); encoder rows are cited from the full-test prior runs, and a TF-IDF anchor is re-scored on the exact subsample for an apples-to-apples comparison. `scripts/exp_llm_baselines.py`.

_Regenerate: `python scripts/exp_llm_baselines.py`_

| model | kind | weighted_f1 | auroc | f1 | note |
|---|---|---|---|---|---|
| Low 2020 (SGD-L1) | linear | 0.851 | None |  | published baseline (full test) |
| TF-IDF + LogReg | linear | 0.886 | 0.944 |  | Exp 8 (full test) |
| MentalRoBERTa (125M, fine-tuned) | encoder | 0.906 | 0.955 |  | Exp 8 (full test) |
| RoBERTa-large (355M, fine-tuned) | encoder | 0.916 | 0.958 |  | Idea 12 (full test) |
| TF-IDF (this eval subsample, n=636) | linear | 0.8855 | 0.9444 | 0.8588 | apples-to-apples anchor |
| mentallama-zeroshot | llm | 0.228 | 0.4718 | 0.5708 | eval subsample n=636 |
| qwen-zeroshot | llm | 0.7817 | 0.8156 | 0.7426 | eval subsample n=636 |
| qwen-qlora | llm | 0.9169 | 0.963 | 0.8894 | eval subsample n=636 |
| llama31-zeroshot | llm | 0.642 | 0.6687 | 0.6021 | eval subsample n=636 |
| llama31-qlora | llm | 0.9114 | 0.962 | 0.8862 | eval subsample n=636 |

![llm baselines](figures/llm_baselines.png)

## Interpretation

- **Zero-shot generative LLMs lose.** Prompted zero-shot, the 7B models trail the fine-tuned encoders and even TF-IDF — consistent with the literature that fine-tuned encoders beat prompted LLMs on short-text mental-health classification.
- **QLoRA reaches parity, not dominance.** One epoch of 4-bit LoRA lifts the 7B model up to the best encoder's level (differences at n≈636 are within noise), but at 20-55x the parameters of MentalRoBERTa (125M) / RoBERTa-large (355M) — the small fine-tuned encoder remains the efficient choice. Fine-tuning, not prompting, closes the gap.
- **Verbalizer caveat.** A zero-shot row with AUROC approximately 0.5 and a degenerate weighted-F1 reflects a prompt-format mismatch, not a capability measure: a model tuned for long-form answers and lacking a chat template (e.g. MentaLLaMA-chat-7B, a LLaMA-2-chat model fine-tuned on IMHI) is not elicited well by a yes/no next-token probe — it needs its native `[INST]...[/INST]` format or generate-and-parse decoding for a fair number.

- **Llama-3.1-8B confirms the pattern (access now granted).** Zero-shot it scores 0.642 / 0.669 — it loses, even below Qwen2.5-7B zero-shot (verbalizer fit varies by model). After one epoch of QLoRA it reaches **0.911 / 0.962**, tying RoBERTa-large (0.916) and Qwen-QLoRA (0.917). An 8B model fine-tuned only *matches* a 125M encoder, at ~23x the parameters — the efficient-encoder conclusion holds across two independent 7-8B model families.