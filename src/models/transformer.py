"""Single-target transformer fine-tuning (RoBERTa / MentalBERT)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.logging import get_logger

log = get_logger(__name__)


class TransformerModel(BaseModel):
    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self._device = None

    def _load_pretrained(self) -> tuple[object, object]:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        e = self.config.extra
        primary = e.get("pretrained", "roberta-base")
        fallback = e.get("fallback_pretrained", "roberta-base")
        for name in (primary, fallback):
            try:
                tok = AutoTokenizer.from_pretrained(name)
                mdl = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
                log.info("transformer.load", model=name)
                return tok, mdl
            except Exception as ex:  # noqa: BLE001
                log.warning("transformer.load_failed", model=name, error=str(ex))
        raise RuntimeError(f"Could not load {primary} or {fallback}")

    def _device_select(self):
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _make_dataset(self, df: pd.DataFrame, has_labels: bool):
        from datasets import Dataset

        records = {self.config.text_field: self.x_from_df(df)}
        if has_labels:
            records["labels"] = self.y_from_df(df).tolist()
        return Dataset.from_dict(records)

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,  # not used directly; kept for interface parity
    ) -> TransformerModel:
        from transformers import (
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        e = self.config.extra
        tok_cfg = e.get("tokenizer", {})
        train_cfg = e.get("train", {})

        self.tokenizer, self.model = self._load_pretrained()
        self._device = self._device_select()
        log.info("transformer.device", device=str(self._device))

        max_len = tok_cfg.get("max_length", 256)
        text_field = self.config.text_field

        def tokenize_fn(batch):
            return self.tokenizer(
                batch[text_field],
                truncation=tok_cfg.get("truncation", True),
                max_length=max_len,
            )

        ds_train = self._make_dataset(train, has_labels=True).map(tokenize_fn, batched=True)
        ds_val = (
            self._make_dataset(val, has_labels=True).map(tokenize_fn, batched=True)
            if val is not None and not val.empty
            else None
        )
        ds_train = ds_train.remove_columns([text_field])
        if ds_val is not None:
            ds_val = ds_val.remove_columns([text_field])

        # `evaluation_strategy` was renamed to `eval_strategy` in transformers 4.42+;
        # build kwargs defensively.
        ta_kwargs = dict(
            output_dir="checkpoints/transformer",
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 32),
            num_train_epochs=train_cfg.get("num_train_epochs", 4),
            learning_rate=train_cfg.get("learning_rate", 2.0e-5),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
            fp16=train_cfg.get("fp16", False),
            save_strategy=train_cfg.get("save_strategy", "epoch"),
            load_best_model_at_end=train_cfg.get("load_best_model_at_end", True) and ds_val is not None,
            metric_for_best_model=train_cfg.get("metric_for_best_model", "f1"),
            greater_is_better=train_cfg.get("greater_is_better", True),
            seed=train_cfg.get("random_state", 42),
            report_to=[],
            logging_steps=50,
        )
        if ds_val is not None:
            ta_kwargs["eval_strategy"] = train_cfg.get("evaluation_strategy", "epoch")

        try:
            args = TrainingArguments(**ta_kwargs)
        except TypeError:
            # Older transformers expect "evaluation_strategy"
            ta_kwargs["evaluation_strategy"] = ta_kwargs.pop("eval_strategy", "no")
            args = TrainingArguments(**ta_kwargs)

        def compute_metrics(eval_pred):
            from sklearn.metrics import f1_score, precision_recall_fscore_support

            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
            return {"precision": p, "recall": r, "f1": f1}

        # transformers 5.x renamed `tokenizer` → `processing_class`. Try the new
        # name first, fall back for older versions.
        trainer_kwargs = dict(
            model=self.model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics,
        )
        try:
            trainer = Trainer(processing_class=self.tokenizer, **trainer_kwargs)
        except TypeError:
            trainer = Trainer(tokenizer=self.tokenizer, **trainer_kwargs)
        trainer.train()
        self._trainer = trainer
        self._fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader

        from transformers import DataCollatorWithPadding

        self.model.eval()  # type: ignore[union-attr]
        self.model.to(self._device)  # type: ignore[union-attr]
        max_len = self.config.extra.get("tokenizer", {}).get("max_length", 256)
        texts = self.x_from_df(df)
        enc = self.tokenizer(  # type: ignore[union-attr]
            texts, truncation=True, max_length=max_len, padding=False
        )
        # Build a tiny torch dataset
        from datasets import Dataset

        ds = Dataset.from_dict(enc)
        collator = DataCollatorWithPadding(self.tokenizer)  # type: ignore[arg-type]
        loader = DataLoader(ds, batch_size=64, collate_fn=collator)
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                logits = self.model(**batch).logits  # type: ignore[union-attr]
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.append(p)
        return np.concatenate(probs)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(p)  # type: ignore[union-attr]
        self.tokenizer.save_pretrained(p)  # type: ignore[union-attr]

    def load(self, path: str | Path) -> TransformerModel:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self._device = self._device_select()
        self._fitted = True
        return self
