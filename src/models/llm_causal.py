"""Local open-weight causal-LM baseline (zero-shot + QLoRA fine-tune).

Phase 2 of the architecture work: a generative-LLM baseline for the headline
r/HealthAnxiety-vs-r/Anxiety task, to answer "do decoder-only LLMs beat a
fine-tuned 125M encoder on short-text mental-health classification?".

One model class covers both regimes (a single ``model_type: llm_causal``):

* **zero-shot** (``lora.enabled: false``): no training. We frame each post as a
  yes/no instruction, format it with the model's chat template, and read the
  next-token logits — the score is ``P("yes") / (P("yes") + P("no"))`` summed
  over surface forms. Deterministic, gives a continuous score for AUROC.
* **QLoRA fine-tune** (``lora.enabled: true``): load the base model in 4-bit
  (bitsandbytes NF4), attach LoRA adapters (peft), and train the verbalizer
  answer token with the prompt masked out of the loss. Fits an 8B model in 24 GB.

``predict_proba`` returns shape ``(n,)`` (single-target), matching ``BaseModel``
so the existing evaluation / ``full_report`` path is unchanged. Heavy deps
(``torch``/``transformers`` always; ``peft``/``bitsandbytes`` only for 4-bit /
LoRA) are imported lazily so the test suite imports this module on CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.logging import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are an expert NLP annotator for a mental-health research project. "
    "For each post, decide whether the labeled phenomenon is present. "
    'Answer with a single word: "yes" or "no". Do not explain.'
)

PROMPT_TEMPLATE = (
    "Label: {target_description}\n\n"
    'Post:\n"""\n{post}\n"""\n\n'
    'Is the label present? Answer "yes" or "no" only.'
)

# Reuses the descriptions from the Claude zero-shot baseline, plus the
# head-to-head "ha" target used by exp_ha_vs_anxiety / exp_stronger_models.
TARGET_DESCRIPTIONS = {
    "anxiety": "the author expresses present-day anxious affect, anxious cognition, or anxious physiological experience",
    "health_anxiety": "the author expresses anxiety specifically about their own (or a loved one's) physical health, illness, or fear of disease",
    "depression": "the author expresses depressive symptoms (anhedonia, hopelessness, persistent low mood, worthlessness)",
    "suicidality": "the author expresses suicidal ideation, intent, plan, or recent attempt",
    "ha": "the post is about HEALTH anxiety (fear or worry about having a serious physical illness) as opposed to GENERAL anxiety",
}

_YES_FORMS = ["yes", " yes", "Yes", " Yes", "YES", " YES"]
_NO_FORMS = ["no", " no", "No", " No", "NO", " NO"]


class HfCausalLmModel(BaseModel):
    """Decoder-only LLM as a single-target binary classifier (zero-shot or QLoRA)."""

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        e = config.extra
        self._pretrained = e.get("pretrained", "Qwen/Qwen2.5-7B-Instruct")
        self._char_cap = int(e.get("char_cap", 1600))
        self._max_length = int(e.get("tokenizer", {}).get("max_length", 1024))
        self._batch_size = int(e.get("batch_size", 8))
        # "auto" = chat_template if present else plain; "llama2" = [INST]<<SYS>> format
        # (for LLaMA-2-chat models like MentaLLaMA that ship without a chat_template);
        # "plain" = system + prompt + "Answer:".
        self._prompt_style = e.get("prompt_style", "auto")
        # 4-bit only makes sense on CUDA; disabled automatically on CPU (tests).
        self._want_4bit = bool(e.get("load_in_4bit", True))
        lora = e.get("lora", {}) or {}
        self._lora_enabled = bool(lora.get("enabled", False))
        self._lora_cfg = {
            "r": int(lora.get("r", 16)),
            "alpha": int(lora.get("alpha", 32)),
            "dropout": float(lora.get("dropout", 0.05)),
            "target_modules": lora.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        }
        tr = e.get("train", {}) or {}
        self._epochs = float(tr.get("num_train_epochs", 1))
        self._lr = float(tr.get("learning_rate", 1e-4))
        self._train_bs = int(tr.get("per_device_train_batch_size", 8))
        self._grad_accum = int(tr.get("gradient_accumulation_steps", 2))
        self._max_train = int(tr.get("max_train", 12000))

        self._model = None
        self._tok = None
        self._yes_ids: list[int] = []
        self._no_ids: list[int] = []
        self._adapter_dir: str | None = None

    # ------------------------------------------------------------------ #
    # Lazy backbone construction
    # ------------------------------------------------------------------ #
    def _use_4bit(self) -> bool:
        import torch

        return self._want_4bit and torch.cuda.is_available()

    def _device(self):
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_tokenizer(self):
        if self._tok is not None:
            return self._tok
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self._pretrained)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        # left padding => last position is the real final prompt token for every row
        tok.padding_side = "left"
        self._tok = tok
        self._yes_ids = self._first_token_ids(tok, _YES_FORMS)
        self._no_ids = self._first_token_ids(tok, _NO_FORMS)
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError("Could not resolve yes/no verbalizer token ids for this tokenizer.")
        return tok

    @staticmethod
    def _first_token_ids(tok, forms: list[str]) -> list[int]:
        ids: set[int] = set()
        for w in forms:
            enc = tok.encode(w, add_special_tokens=False)
            if enc:
                ids.add(int(enc[0]))
        return sorted(ids)

    def _load_model(self, for_training: bool):
        import torch
        from transformers import AutoModelForCausalLM

        kwargs: dict = {}
        if self._use_4bit():
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = {"": 0}
            kwargs["dtype"] = torch.bfloat16
        elif torch.cuda.is_available():
            kwargs["dtype"] = torch.bfloat16
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["dtype"] = torch.float32  # CPU test path

        model = AutoModelForCausalLM.from_pretrained(self._pretrained, **kwargs)
        if not self._use_4bit() and not torch.cuda.is_available():
            model = model.to("cpu")
        return model

    # ------------------------------------------------------------------ #
    # Prompt formatting
    # ------------------------------------------------------------------ #
    def _format(self, text: str) -> str:
        tok = self._tok
        desc = TARGET_DESCRIPTIONS.get(self.target, self.target)
        user = PROMPT_TEMPLATE.format(target_description=desc, post=str(text)[: self._char_cap])
        if self._prompt_style == "llama2":
            # LLaMA-2-chat instruction format (MentaLLaMA ships without a chat_template)
            return f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{user} [/INST]"
        if self._prompt_style != "plain" and getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return f"{SYSTEM_PROMPT}\n\n{user}\nAnswer:"

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        import torch

        self._load_tokenizer()
        if self._model is None:
            self._model = self._load_model(for_training=False)
        self._model.eval()
        from tqdm.auto import tqdm

        tok = self._tok
        texts = df[self.config.text_field].astype(str).fillna("").tolist()
        scores: list[float] = []
        tag = self._pretrained.split("/")[-1]
        for i in tqdm(range(0, len(texts), self._batch_size),
                      desc=f"score[{self.target}] {tag}", unit="batch"):
            chunk = [self._format(t) for t in texts[i : i + self._batch_size]]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=self._max_length)
            enc = {k: v.to(self._model.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits[:, -1, :].float().cpu()
            scores.extend(self._scores_from_logits(logits, self._yes_ids, self._no_ids).tolist())
        return np.asarray(scores, dtype=float)

    @staticmethod
    def _scores_from_logits(logits, yes_ids: list[int], no_ids: list[int]) -> np.ndarray:
        """P(yes) / (P(yes)+P(no)) summed over surface forms. (B, V) -> (B,)."""
        import torch

        probs = torch.softmax(logits.float(), dim=-1)
        p_yes = probs.index_select(1, torch.tensor(yes_ids)).sum(dim=1)
        p_no = probs.index_select(1, torch.tensor(no_ids)).sum(dim=1)
        return (p_yes / (p_yes + p_no + 1e-9)).cpu().numpy()

    # ------------------------------------------------------------------ #
    # QLoRA fine-tuning
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, val=None, sample_weight=None) -> "HfCausalLmModel":  # noqa: ANN001
        if not self._lora_enabled:
            self._fitted = True  # zero-shot: nothing to train
            return self

        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        tok = self._load_tokenizer()
        model = self._load_model(for_training=True)
        if self._use_4bit():
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(
            r=self._lora_cfg["r"], lora_alpha=self._lora_cfg["alpha"],
            lora_dropout=self._lora_cfg["dropout"], bias="none", task_type="CAUSAL_LM",
            target_modules=self._lora_cfg["target_modules"],
        ))
        model.config.use_cache = False
        model.train()
        self._model = model

        # build supervised examples: prompt (masked) + single answer token
        df = train
        if self._max_train and len(df) > self._max_train:
            df = df.sample(n=self._max_train, random_state=42).reset_index(drop=True)
        y = self.y_from_df(df)
        texts = df[self.config.text_field].astype(str).fillna("").tolist()
        yes_id, no_id = self._yes_ids[0], self._no_ids[0]
        examples = []
        from tqdm.auto import tqdm

        for text, label in tqdm(list(zip(texts, y)), desc=f"tokenize[{self.target}]", unit="ex"):
            prompt_ids = tok(self._format(text), add_special_tokens=False,
                             truncation=True, max_length=self._max_length).input_ids
            ans_id = yes_id if int(label) == 1 else no_id
            input_ids = prompt_ids + [ans_id]
            labels = [-100] * len(prompt_ids) + [ans_id]
            examples.append((input_ids, labels))

        from tqdm.auto import tqdm

        optim = self._make_optimizer(model)
        device = model.device
        bs, accum = self._train_bs, self._grad_accum
        epochs = int(np.ceil(self._epochs))
        n_batches = int(np.ceil(len(examples) / bs))
        log.info("llm_causal.qlora_fit", n=len(examples), steps=n_batches * epochs, lr=self._lr,
                 fourbit=self._use_4bit())
        rng = np.random.default_rng(42)
        bar = tqdm(total=n_batches * epochs, desc=f"QLoRA[{self.target}] {self._pretrained.split('/')[-1]}",
                   unit="batch")
        for epoch in range(epochs):
            order = rng.permutation(len(examples))
            optim.zero_grad()
            for bi in range(0, len(order), bs):
                batch = [examples[j] for j in order[bi : bi + bs]]
                input_ids, attn, labels = self._collate(batch, tok.pad_token_id)
                out = model(input_ids=input_ids.to(device),
                            attention_mask=attn.to(device),
                            labels=labels.to(device))
                (out.loss / accum).backward()
                if ((bi // bs) + 1) % accum == 0:
                    optim.step()
                    optim.zero_grad()
                bar.update(1)
                bar.set_postfix(loss=round(float(out.loss), 4), epoch=epoch + 1)
        bar.close()
        model.config.use_cache = True
        self._fitted = True
        return self

    @staticmethod
    def _collate(batch, pad_id: int):
        import torch

        maxlen = max(len(x[0]) for x in batch)
        ids, attn, labs = [], [], []
        for input_ids, labels in batch:
            pad = maxlen - len(input_ids)
            # right-pad for training (loss masks pads via labels=-100)
            ids.append(input_ids + [pad_id] * pad)
            attn.append([1] * len(input_ids) + [0] * pad)
            labs.append(labels + [-100] * pad)
        return (torch.tensor(ids), torch.tensor(attn), torch.tensor(labs))

    def _make_optimizer(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        if self._use_4bit():
            try:
                import bitsandbytes as bnb

                return bnb.optim.PagedAdamW8bit(params, lr=self._lr)
            except Exception:  # noqa: BLE001
                pass
        import torch

        return torch.optim.AdamW(params, lr=self._lr)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "pretrained": self._pretrained,
            "target": self.config.target,
            "lora_enabled": self._lora_enabled,
            "load_in_4bit": self._want_4bit,
            "char_cap": self._char_cap,
            "max_length": self._max_length,
        }
        if self._lora_enabled and self._model is not None and hasattr(self._model, "save_pretrained"):
            adapter = p / "adapter"
            self._model.save_pretrained(adapter)
            meta["adapter_dir"] = "adapter"
        (p / "llm_causal_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> "HfCausalLmModel":
        p = Path(path)
        meta = json.loads((p / "llm_causal_meta.json").read_text(encoding="utf-8"))
        self._pretrained = meta["pretrained"]
        self._lora_enabled = meta.get("lora_enabled", False)
        self._want_4bit = meta.get("load_in_4bit", True)
        self._char_cap = meta.get("char_cap", self._char_cap)
        self._max_length = meta.get("max_length", self._max_length)
        self._load_tokenizer()
        base = self._load_model(for_training=False)
        adapter = meta.get("adapter_dir")
        if adapter and (p / adapter).exists():
            from peft import PeftModel

            base = PeftModel.from_pretrained(base, str(p / adapter))
        self._model = base
        self._fitted = True
        return self
