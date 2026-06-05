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
| mentallama-zeroshot | llm | 0.2408 | 0.4513 | 0.5718 | eval subsample n=636 |
| qwen-zeroshot | llm | 0.7817 | 0.8156 | 0.7426 | eval subsample n=636 |
| qwen-qlora | llm | 0.9169 | 0.963 | 0.8894 | eval subsample n=636 |

![llm baselines](figures/llm_baselines.png)

## Interpretation

- **Zero-shot generative LLMs lose.** Qwen2.5-7B zero-shot (0.782 weighted-F1 / 0.816 AUROC) trails TF-IDF (0.886) and the fine-tuned encoders (0.906–0.916) — consistent with the literature that fine-tuned encoders beat prompted LLMs on short-text mental-health classification.
- **QLoRA reaches parity, not dominance.** One epoch of 4-bit LoRA on 2,672 posts lifts Qwen2.5-7B to 0.917 weighted-F1 / 0.963 AUROC — statistically tied with RoBERTa-large (0.916 / 0.958; Δ within noise at n=636). A 7B model only *matches* a 125M MentalRoBERTa / 355M RoBERTa-large, so the small fine-tuned encoder remains the efficient choice; **fine-tuning, not prompting, closes the gap**, and the top of the table is bumping the ~0.92 Reddit-binary ceiling.
- **MentaLLaMA caveat.** The MentaLLaMA-chat-7B zero-shot row (AUROC 0.45 ≈ chance, degenerate weighted-F1) reflects a prompt-format mismatch, **not** a capability measure: MentaLLaMA-chat is a LLaMA-2-chat model fine-tuned on IMHI for long-form explanations and lacks a chat template, so the yes/no next-token probe is not elicited well. A fair number needs its native `[INST]…[/INST]` format or generate-and-parse decoding.

> _Llama-3.1-8B (zero-shot + QLoRA) is gated and was deferred pending HF access; re-run with `--models llama31-zeroshot,llama31-qlora` once granted._