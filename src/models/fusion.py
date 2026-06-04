"""FusionMultiTaskModel — architecture surgery on the multi-task transformer.

Extends MultiTaskTransformer with four independently-ablatable modifications:
  1. fusion       — concat the pooled encoder embedding with 26 hand-crafted
                    linguistic features (src/features/linguistic.py) + 7 SHAI
                    dimensions (src/features/shai.py), z-normalised, through a
                    FusionMLP before the per-target heads. Hypothesis: stable
                    clinical/lexical features improve cross-corpus transfer and
                    rare-class F1.
  2. attn_pool    — learned additive attention pooling over tokens (vs mean-pool).
  3. focal        — focal loss for the imbalanced rare classes (vs plain BCE).
  4. activation   — gelu | relu | silu in the heads/fusion MLP.

predict_proba still returns (n_samples, n_targets) sigmoid, so the existing eval
(runner.py, external.py, eval-disclosure) works unchanged. The 33 features are
computed from text inside fit/predict_proba (the eval contract only guarantees a
`clean_text` column), z-normalised with train-fit stats persisted in
fusion_meta.json alongside the resolved architecture flags.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.multitask import MultiTaskTransformer
from src.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Reusable nn pieces (module-level so they are unit-testable without an encoder)
# --------------------------------------------------------------------------- #
def make_activation(name: str):
    from torch import nn

    return {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}.get(name, nn.GELU)()


def build_attention_pool(hidden: int, attn_dim: int):
    import torch
    from torch import nn

    class AttentionPool(nn.Module):
        """Additive (Bahdanau) attention pooling over the token dimension."""

        def __init__(self):
            super().__init__()
            self.w = nn.Linear(hidden, attn_dim)
            self.v = nn.Linear(attn_dim, 1, bias=False)

        def forward(self, last_hidden, attention_mask=None):
            scores = self.v(torch.tanh(self.w(last_hidden))).squeeze(-1)  # (B, L)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)          # (B, L, 1)
            return (last_hidden * weights).sum(dim=1)                      # (B, H)

    return AttentionPool()


def build_fusion_mlp(hidden: int, n_feats: int, activation: str, dropout: float, use_layernorm: bool):
    import torch
    from torch import nn

    class FusionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(hidden + n_feats, hidden)
            self.act = make_activation(activation)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden) if use_layernorm else nn.Identity()

        def forward(self, pooled, feats):
            x = torch.cat([pooled, feats], dim=1)         # (B, H + n_feats)
            return self.norm(self.dropout(self.act(self.proj(x))))

    return FusionMLP()


def focal_bce(logits, targets, gamma: float):
    """Focal binary cross-entropy, reduction='none' -> (B, T). gamma=0 == plain BCE."""
    import torch
    import torch.nn.functional as F

    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma <= 0:
        return ce
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    return ((1.0 - p_t) ** gamma) * ce


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class FusionMultiTaskModel(MultiTaskTransformer):
    """MentalRoBERTa multi-task with optional clinical-feature fusion, attention
    pooling, focal loss, and configurable activation."""

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        e = config.extra
        fz = e.get("fusion", {}) if isinstance(e.get("fusion"), dict) else {"enabled": bool(e.get("fusion"))}
        ap = e.get("attn_pool", {}) if isinstance(e.get("attn_pool"), dict) else {"enabled": bool(e.get("attn_pool"))}
        fc = e.get("focal", {}) if isinstance(e.get("focal"), dict) else {"enabled": bool(e.get("focal"))}
        self._fusion = bool(fz.get("enabled", False))
        self._fusion_dropout = float(fz.get("dropout", 0.1))
        self._use_layernorm = bool(fz.get("use_layernorm", True))
        self._attn_pool = bool(ap.get("enabled", False))
        self._attn_dim = int(ap.get("attn_dim", 128))
        self._focal = bool(fc.get("enabled", False))
        self._gamma = float(fc.get("gamma", 2.0))
        self._class_balanced = bool(fc.get("class_balanced", False))
        self._beta = float(fc.get("beta", 0.999))
        self._activation = str(e.get("activation", "gelu"))
        # feature-normalisation state (persisted)
        self._feature_cols: list[str] | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None

    # ---- clinical features --------------------------------------------- #
    def _raw_feats(self, df: pd.DataFrame) -> pd.DataFrame:
        """26 linguistic + 7 SHAI features as a DataFrame with a pinned column order."""
        from src.features.linguistic import extract_dataframe, feature_columns
        from src.features.shai import score_shai, shai_dimensions

        feat_df = extract_dataframe(df, text_col=self.config.text_field)
        ling = feat_df[feature_columns(feat_df)].reset_index(drop=True)
        dims = shai_dimensions()
        shai = pd.DataFrame(
            [score_shai(t) for t in df[self.config.text_field].astype(str).fillna("")],
            columns=dims,
        ).rename(columns={d: f"shai_{d}" for d in dims}).reset_index(drop=True)
        out = pd.concat([ling, shai], axis=1)
        if self._feature_cols is None:
            self._feature_cols = list(out.columns)
        return out.reindex(columns=self._feature_cols, fill_value=0.0)

    def _fit_norm(self, feats: pd.DataFrame) -> None:
        self._feat_mean = feats.mean(axis=0).to_numpy(dtype=np.float32)
        std = feats.std(axis=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)
        self._feat_std = std

    def _apply_norm(self, feats: pd.DataFrame) -> np.ndarray:
        arr = feats.to_numpy(dtype=np.float32)
        return ((arr - self._feat_mean) / self._feat_std).astype(np.float32)

    # ---- model --------------------------------------------------------- #
    def _build_model(self):
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        e = self.config.extra
        primary = e.get("pretrained", "roberta-base")
        fallback = e.get("fallback_pretrained", "roberta-base")
        tok = encoder = cfg = None
        for name in (primary, fallback):
            try:
                tok = AutoTokenizer.from_pretrained(name)
                cfg = AutoConfig.from_pretrained(name)
                encoder = AutoModel.from_pretrained(name)
                log.info("fusion.load", model=name)
                break
            except Exception as ex:  # noqa: BLE001
                log.warning("fusion.load_failed", model=name, error=str(ex))
        else:
            raise RuntimeError(f"Could not load {primary} or {fallback}")

        n_targets = len(self.targets)
        hidden = cfg.hidden_size
        n_feats = len(self._feature_cols) if (self._fusion and self._feature_cols) else 33
        attn = build_attention_pool(hidden, self._attn_dim) if self._attn_pool else None
        fusion = build_fusion_mlp(hidden, n_feats, self._activation, self._fusion_dropout, self._use_layernorm) if self._fusion else None

        class FusionMultiHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.attn = attn
                self.fusion = fusion
                self.dropout = nn.Dropout(0.1)
                self.head = nn.Linear(hidden, n_targets)

            @staticmethod
            def _mean_pool(last, attention_mask):
                if attention_mask is None:
                    return last.mean(dim=1)
                mask = attention_mask.unsqueeze(-1).float()
                return (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            def forward(self, input_ids, attention_mask=None, feats=None, **_kw):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                last = out.last_hidden_state
                pooled = self.attn(last, attention_mask) if self.attn is not None else self._mean_pool(last, attention_mask)
                if self.fusion is not None and feats is not None:
                    pooled = self.fusion(pooled, feats)
                return self.head(self.dropout(pooled))

        return tok, FusionMultiHead()

    # ---- training ------------------------------------------------------ #
    def fit(self, train, val=None, sample_weight=None) -> "FusionMultiTaskModel":  # noqa: ANN001
        import torch
        from torch import optim
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        e = self.config.extra
        tok_cfg = e.get("tokenizer", {})
        train_cfg = e.get("train", {})
        loss_w = e.get("loss_weights", {})

        # features (only when fusion is on)
        feats_train = None
        if self._fusion:
            raw = self._raw_feats(train)
            self._fit_norm(raw)
            feats_train = self._apply_norm(raw)

        self.tokenizer, self.model = self._build_model()
        self._device = self._device_select()
        self.model.to(self._device)
        log.info("fusion.fit.start", fusion=self._fusion, attn_pool=self._attn_pool,
                 focal=self._focal, activation=self._activation, device=str(self._device))

        max_len = tok_cfg.get("max_length", 256)
        epochs = int(train_cfg.get("num_train_epochs", 5))
        bs = int(train_cfg.get("per_device_train_batch_size", 16))
        lr = float(train_cfg.get("learning_rate", 2e-5))
        wd = float(train_cfg.get("weight_decay", 0.01))

        y_train = self._y_multi(train)
        w_train = self._w_multi(train)
        w_task = torch.tensor([loss_w.get(t, 1.0) for t in self.targets], dtype=torch.float32, device=self._device)
        if self._class_balanced:  # Cui et al. 2019 effective-number weighting, folded into w_task
            pos = y_train.sum(axis=0).clip(min=1)
            cb = (1.0 - self._beta) / (1.0 - np.power(self._beta, pos))
            cb = cb / cb.mean()
            w_task = w_task * torch.tensor(cb, dtype=torch.float32, device=self._device)

        texts = train[self.config.text_field].astype(str).fillna("").tolist()
        tok = self.tokenizer

        class _DS(Dataset):
            def __len__(self):
                return len(texts)

            def __getitem__(self, i):
                enc = tok(texts[i], truncation=True, max_length=max_len, padding=False)
                item = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                        "labels": y_train[i], "weights": w_train[i]}
                if feats_train is not None:
                    item["feats"] = feats_train[i]
                return item

        def collate(batch):
            from torch.nn.utils.rnn import pad_sequence

            ids = pad_sequence([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch],
                               batch_first=True, padding_value=tok.pad_token_id or 0)
            mask = pad_sequence([torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch],
                                batch_first=True, padding_value=0)
            labels = torch.tensor(np.stack([b["labels"] for b in batch]), dtype=torch.float32)
            weights = torch.tensor(np.stack([b["weights"] for b in batch]), dtype=torch.float32)
            out = {"input_ids": ids, "attention_mask": mask, "labels": labels, "weights": weights}
            if "feats" in batch[0]:
                out["feats"] = torch.tensor(np.stack([b["feats"] for b in batch]), dtype=torch.float32)
            return out

        dl = DataLoader(_DS(), batch_size=bs, shuffle=True, collate_fn=collate)
        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(epochs):
            self.model.train()
            tot, nb = 0.0, 0
            bar = tqdm(dl, desc=f"fusion epoch {epoch + 1}/{epochs}", leave=False)
            for batch in bar:
                feats = batch.get("feats")
                ids = batch["input_ids"].to(self._device)
                mask = batch["attention_mask"].to(self._device)
                labels = batch["labels"].to(self._device)
                weights = batch["weights"].to(self._device)
                if feats is not None:
                    feats = feats.to(self._device)
                opt.zero_grad()
                logits = self.model(input_ids=ids, attention_mask=mask, feats=feats)
                raw = focal_bce(logits, labels, self._gamma) if self._focal else \
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                loss = (raw * weights * w_task).mean()
                loss.backward()
                opt.step()
                tot += float(loss.item()); nb += 1
                bar.set_postfix(loss=f"{loss.item():.3f}")
            log.info("fusion.epoch", epoch=epoch + 1, loss=tot / max(1, nb))
            if val is not None and not val.empty:
                self._eval_log(val)

        self._fitted = True
        return self

    # ---- inference ----------------------------------------------------- #
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        max_len = self.config.extra.get("tokenizer", {}).get("max_length", 256)
        texts = df[self.config.text_field].astype(str).fillna("").tolist()
        tok = self.tokenizer
        feats_arr = self._apply_norm(self._raw_feats(df)) if self._fusion else None

        class _DSPred(Dataset):
            def __len__(self):
                return len(texts)

            def __getitem__(self, i):
                enc = tok(texts[i], truncation=True, max_length=max_len, padding=False)
                if feats_arr is not None:
                    return enc["input_ids"], enc["attention_mask"], feats_arr[i]
                return enc["input_ids"], enc["attention_mask"], 0

        def collate(batch):
            from torch.nn.utils.rnn import pad_sequence

            ids = pad_sequence([torch.tensor(b[0], dtype=torch.long) for b in batch], batch_first=True, padding_value=tok.pad_token_id or 0)
            mask = pad_sequence([torch.tensor(b[1], dtype=torch.long) for b in batch], batch_first=True, padding_value=0)
            feats = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32) if feats_arr is not None else None
            return ids, mask, feats

        loader = DataLoader(_DSPred(), batch_size=64, collate_fn=collate)
        self.model.eval()
        all_p: list[np.ndarray] = []
        with torch.no_grad():
            for ids, mask, feats in tqdm(loader, desc="predict", leave=False):
                ids = ids.to(self._device); mask = mask.to(self._device)
                if feats is not None:
                    feats = feats.to(self._device)
                logits = self.model(input_ids=ids, attention_mask=mask, feats=feats)
                all_p.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(all_p, axis=0)

    # ---- persistence --------------------------------------------------- #
    def _meta(self) -> dict:
        return {
            "fusion": self._fusion, "fusion_dropout": self._fusion_dropout, "use_layernorm": self._use_layernorm,
            "attn_pool": self._attn_pool, "attn_dim": self._attn_dim,
            "focal": self._focal, "gamma": self._gamma, "class_balanced": self._class_balanced, "beta": self._beta,
            "activation": self._activation,
            "feature_cols": self._feature_cols,
            "feat_mean": self._feat_mean.tolist() if self._feat_mean is not None else None,
            "feat_std": self._feat_std.tolist() if self._feat_std is not None else None,
        }

    def save(self, path) -> None:  # noqa: ANN001
        import torch

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p / "model.pt")
        self.tokenizer.save_pretrained(p)
        (p / "targets.txt").write_text("\n".join(self.targets))
        (p / "fusion_meta.json").write_text(json.dumps(self._meta(), indent=2))

    def load(self, path) -> "FusionMultiTaskModel":  # noqa: ANN001
        import torch
        from transformers import AutoTokenizer

        p = Path(path)
        meta = json.loads((p / "fusion_meta.json").read_text())
        self._fusion = meta["fusion"]; self._fusion_dropout = meta["fusion_dropout"]; self._use_layernorm = meta["use_layernorm"]
        self._attn_pool = meta["attn_pool"]; self._attn_dim = meta["attn_dim"]
        self._focal = meta["focal"]; self._gamma = meta["gamma"]; self._class_balanced = meta["class_balanced"]; self._beta = meta["beta"]
        self._activation = meta["activation"]
        self._feature_cols = meta["feature_cols"]
        self._feat_mean = np.asarray(meta["feat_mean"], dtype=np.float32) if meta["feat_mean"] is not None else None
        self._feat_std = np.asarray(meta["feat_std"], dtype=np.float32) if meta["feat_std"] is not None else None
        self.tokenizer, self.model = self._build_model()
        self.tokenizer = AutoTokenizer.from_pretrained(p)
        self.model.load_state_dict(torch.load(p / "model.pt", map_location="cpu"))
        self._device = self._device_select()
        self.model.to(self._device)
        self._fitted = True
        return self
