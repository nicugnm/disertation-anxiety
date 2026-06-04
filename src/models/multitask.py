"""Multi-task transformer: shared encoder + one binary head per label.

This is the dissertation's modeling-novelty contribution: jointly model
{anxiety, health_anxiety, depression, suicidality} with a per-task loss
weight, so health-anxiety predictions benefit from related-label signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.logging import get_logger

log = get_logger(__name__)


class MultiTaskTransformer(BaseModel):
    """Shared encoder, sigmoid head per target. Trained with BCE-with-logits."""

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self._device = None

    def _build_model(self):
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        e = self.config.extra
        primary = e.get("pretrained", "roberta-base")
        fallback = e.get("fallback_pretrained", "roberta-base")
        for name in (primary, fallback):
            try:
                tok = AutoTokenizer.from_pretrained(name)
                cfg = AutoConfig.from_pretrained(name)
                encoder = AutoModel.from_pretrained(name)
                log.info("multitask.load", model=name)
                break
            except Exception as ex:  # noqa: BLE001
                log.warning("multitask.load_failed", model=name, error=str(ex))
        else:
            raise RuntimeError(f"Could not load {primary} or {fallback}")

        n_targets = len(self.targets)
        hidden = cfg.hidden_size

        class MultiHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.dropout = nn.Dropout(0.1)
                self.head = nn.Linear(hidden, n_targets)

            def forward(self, input_ids, attention_mask=None, **_kw):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                # Mean-pool with attention mask (more stable than CLS for many encoders)
                last = out.last_hidden_state
                mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None else None
                if mask is not None:
                    summed = (last * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1)
                    pooled = summed / counts
                else:
                    pooled = last.mean(dim=1)
                return self.head(self.dropout(pooled))

        return tok, MultiHead()

    def _device_select(self):
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _y_multi(self, df: pd.DataFrame) -> np.ndarray:
        cols = [f"label_{t}" for t in self.targets]
        y = df[cols].astype(float).fillna(0.0).values
        return (y >= 0.5).astype(np.float32)

    def _w_multi(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row, per-task confidence weights (1.0 if missing weight column)."""
        cols = [f"label_{t}_weight" for t in self.targets]
        w = np.ones((len(df), len(self.targets)), dtype=np.float32)
        for i, c in enumerate(cols):
            if c in df.columns:
                col = df[c].astype(float).fillna(1.0).values
                w[:, i] = col
        return w

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> MultiTaskTransformer:
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader, Dataset

        e = self.config.extra
        tok_cfg = e.get("tokenizer", {})
        train_cfg = e.get("train", {})
        loss_w = e.get("loss_weights", {})

        self.tokenizer, self.model = self._build_model()
        self._device = self._device_select()
        self.model.to(self._device)
        log.info("multitask.device", device=str(self._device))

        max_len = tok_cfg.get("max_length", 256)
        epochs = int(train_cfg.get("num_train_epochs", 5))
        bs = int(train_cfg.get("per_device_train_batch_size", 16))
        lr = float(train_cfg.get("learning_rate", 2e-5))
        wd = float(train_cfg.get("weight_decay", 0.01))

        # Per-task loss weights as tensor
        w_task = torch.tensor(
            [loss_w.get(t, 1.0) for t in self.targets], dtype=torch.float32, device=self._device
        )

        class _DS(Dataset):
            def __init__(self, df, tokenizer, text_field, max_len, y, w_row):
                self.texts = df[text_field].astype(str).fillna("").tolist()
                self.tokenizer = tokenizer
                self.max_len = max_len
                self.y = y
                self.w = w_row

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, i):
                enc = self.tokenizer(
                    self.texts[i],
                    truncation=True,
                    max_length=self.max_len,
                    padding=False,
                    return_tensors=None,
                )
                return {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": self.y[i],
                    "weights": self.w[i],
                }

        def collate(batch):
            from torch.nn.utils.rnn import pad_sequence

            input_ids = pad_sequence(
                [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id or 0,
            )
            attention_mask = pad_sequence(
                [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch],
                batch_first=True,
                padding_value=0,
            )
            labels = torch.tensor(np.stack([b["labels"] for b in batch]), dtype=torch.float32)
            weights = torch.tensor(np.stack([b["weights"] for b in batch]), dtype=torch.float32)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "weights": weights}

        y_train = self._y_multi(train)
        w_train = self._w_multi(train)
        ds_train = _DS(train, self.tokenizer, self.config.text_field, max_len, y_train, w_train)
        dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, collate_fn=collate)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        from tqdm.auto import tqdm

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            bar = tqdm(dl_train, desc=f"multitask epoch {epoch + 1}/{epochs}", leave=False)
            for batch in bar:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                optimizer.zero_grad()
                logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                # Loss per (sample, task) -> weighted sum
                raw_loss = loss_fn(logits, batch["labels"])  # (B, T)
                weighted = raw_loss * batch["weights"] * w_task
                loss = weighted.mean()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
                bar.set_postfix(loss=f"{loss.item():.3f}")
            log.info("multitask.epoch", epoch=epoch + 1, loss=total_loss / max(1, n_batches))

            if val is not None and not val.empty:
                self._eval_log(val)

        self._fitted = True
        return self

    def _eval_log(self, val: pd.DataFrame) -> None:
        from sklearn.metrics import f1_score

        probs = self.predict_proba(val)
        y = self._y_multi(val)
        preds = (probs >= 0.5).astype(int)
        f1s = {t: float(f1_score(y[:, i], preds[:, i], zero_division=0)) for i, t in enumerate(self.targets)}
        log.info("multitask.val_f1", **f1s)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader, Dataset

        max_len = self.config.extra.get("tokenizer", {}).get("max_length", 256)
        texts = df[self.config.text_field].astype(str).fillna("").tolist()
        tok = self.tokenizer

        class _DSPred(Dataset):
            def __len__(self): return len(texts)
            def __getitem__(self, i):
                enc = tok(texts[i], truncation=True, max_length=max_len, padding=False)
                return enc["input_ids"], enc["attention_mask"]

        def collate(batch):
            from torch.nn.utils.rnn import pad_sequence
            ids = pad_sequence([torch.tensor(b[0], dtype=torch.long) for b in batch], batch_first=True, padding_value=tok.pad_token_id or 0)
            mask = pad_sequence([torch.tensor(b[1], dtype=torch.long) for b in batch], batch_first=True, padding_value=0)
            return ids, mask

        from tqdm.auto import tqdm

        loader = DataLoader(_DSPred(), batch_size=64, collate_fn=collate)
        self.model.eval()
        all_p: list[np.ndarray] = []
        with torch.no_grad():
            for ids, mask in tqdm(loader, desc="predict", leave=False):
                ids = ids.to(self._device)
                mask = mask.to(self._device)
                logits = self.model(input_ids=ids, attention_mask=mask)
                p = torch.sigmoid(logits).cpu().numpy()
                all_p.append(p)
        return np.concatenate(all_p, axis=0)

    def save(self, path: str | Path) -> None:
        import torch

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p / "model.pt")
        self.tokenizer.save_pretrained(p)
        # Persist target order so load() can rebuild correctly.
        (p / "targets.txt").write_text("\n".join(self.targets))

    def load(self, path: str | Path) -> MultiTaskTransformer:
        import torch
        from transformers import AutoTokenizer

        p = Path(path)
        self.tokenizer, self.model = self._build_model()
        self.tokenizer = AutoTokenizer.from_pretrained(p)
        state = torch.load(p / "model.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self._device = self._device_select()
        self.model.to(self._device)
        self._fitted = True
        return self
