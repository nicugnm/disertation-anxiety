"""HierUserModel — hierarchical user-level classifier.

Frozen post-encoder (MentalRoBERTa) → per-post embedding (mean-pool) → learned
aggregation over a user's chronological posts (attention | mean) → user-level head.
The encoder is encoded ONCE per fit/predict (frozen), so only the small aggregator
trains — cheap and fits 24GB easily.

`predict_proba(df)` groups by author, scores each user, and broadcasts the user
score back to that user's post rows (original order) → the existing user-level
disclosure eval (`evaluate_user_level(aggregation="mean")`) yields identical user
metrics with no eval-code changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.models.hier_dataset import build_user_sequences
from src.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Aggregators over the post dimension (module-level, unit-testable)
# --------------------------------------------------------------------------- #
def build_aggregator(kind: str, embed_dim: int, hidden: int, dropout: float):
    import torch
    from torch import nn

    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(embed_dim, hidden)
            self.dropout = nn.Dropout(dropout)

    class AttnAgg(_Base):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(hidden, 1)

        def forward(self, post_emb, post_mask):           # (B,N,E), (B,N) bool
            h = torch.tanh(self.proj(post_emb))            # (B,N,hidden)
            scores = self.attn(h).squeeze(-1)              # (B,N)
            scores = scores.masked_fill(~post_mask, float("-inf"))
            w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,N,1)
            self.last_weights = w.detach()                  # for interpretability
            return self.dropout((h * w).sum(dim=1))         # (B,hidden)

    class MeanAgg(_Base):
        def forward(self, post_emb, post_mask):
            h = torch.tanh(self.proj(post_emb))
            m = post_mask.unsqueeze(-1).float()
            return self.dropout((h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1))

    return {"attention": AttnAgg, "mean": MeanAgg}.get(kind, AttnAgg)()


class HierUserModel(BaseModel):
    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        e = config.extra
        pe = e.get("post_encoder", {})
        um = e.get("user_model", {})
        self._pretrained = pe.get("pretrained", "roberta-base")
        self._fallback = pe.get("fallback_pretrained", "roberta-base")
        self._max_length = int(pe.get("max_length", 256))
        self._aggregator = um.get("aggregator", "attention")
        self._hidden = int(um.get("hidden", 256))
        self._dropout = float(um.get("dropout", 0.2))
        self._max_posts = int(um.get("max_posts", 64))
        self._order = um.get("order", "recent")
        self.tokenizer = None
        self.encoder = None
        self.user_net = None
        self._embed_dim = None
        self._device = None

    # ---- encoder ------------------------------------------------------- #
    def _device_select(self):
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_encoder(self):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        for name in (self._pretrained, self._fallback):
            try:
                tok = AutoTokenizer.from_pretrained(name)
                cfg = AutoConfig.from_pretrained(name)
                enc = AutoModel.from_pretrained(name)
                log.info("hier.load", model=name)
                break
            except Exception as ex:  # noqa: BLE001
                log.warning("hier.load_failed", model=name, error=str(ex))
        else:
            raise RuntimeError(f"Could not load {self._pretrained} or {self._fallback}")
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()
        self._embed_dim = cfg.hidden_size
        return tok, enc

    def _encode_posts(self, df: pd.DataFrame) -> np.ndarray:
        """Frozen mean-pooled per-post embeddings, (n_posts, embed_dim)."""
        import torch
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        texts = df[self.config.text_field].astype(str).fillna("").tolist()
        tok = self.tokenizer
        max_len = self._max_length

        class _DS(Dataset):
            def __len__(self):
                return len(texts)

            def __getitem__(self, i):
                enc = tok(texts[i], truncation=True, max_length=max_len, padding=False)
                return enc["input_ids"], enc["attention_mask"]

        def collate(batch):
            from torch.nn.utils.rnn import pad_sequence

            ids = pad_sequence([torch.tensor(b[0], dtype=torch.long) for b in batch], batch_first=True, padding_value=tok.pad_token_id or 0)
            mask = pad_sequence([torch.tensor(b[1], dtype=torch.long) for b in batch], batch_first=True, padding_value=0)
            return ids, mask

        loader = DataLoader(_DS(), batch_size=64, collate_fn=collate)
        out = []
        self.encoder.eval()
        with torch.no_grad():
            for ids, mask in tqdm(loader, desc="encode posts", leave=False):
                ids = ids.to(self._device); mask = mask.to(self._device)
                last = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.unsqueeze(-1).float()
                pooled = (last * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                out.append(pooled.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def _build_user_net(self):
        from torch import nn

        agg = build_aggregator(self._aggregator, self._embed_dim, self._hidden, self._dropout)
        hidden = self._hidden
        n_targets = len(self.targets)

        class UserNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.agg = agg
                self.head = nn.Linear(hidden, n_targets)

            def forward(self, post_emb, post_mask):
                return self.head(self.agg(post_emb, post_mask))

        return UserNet()

    # ---- training ------------------------------------------------------ #
    def fit(self, train, val=None, sample_weight=None) -> "HierUserModel":  # noqa: ANN001
        import torch
        from torch import nn, optim
        from tqdm.auto import tqdm

        train_cfg = self.config.extra.get("train", {})
        loss_w = self.config.extra.get("loss_weights", {})
        epochs = int(train_cfg.get("num_train_epochs", 8))
        bs = int(train_cfg.get("per_device_train_batch_size", 16))
        lr = float(train_cfg.get("learning_rate", 1e-3))
        wd = float(train_cfg.get("weight_decay", 0.01))

        self.tokenizer, self.encoder = self._build_encoder()
        self._device = self._device_select()
        self.encoder.to(self._device)
        self.user_net = self._build_user_net().to(self._device)
        w_task = torch.tensor([loss_w.get(t, 1.0) for t in self.targets], dtype=torch.float32, device=self._device)

        emb = self._encode_posts(train)
        users, post_idx, post_mask, y = build_user_sequences(
            train, self.targets, self._max_posts, self._order)
        log.info("hier.fit.start", n_users=len(users), aggregator=self._aggregator, device=str(self._device))
        emb_t = torch.tensor(emb, device=self._device)

        opt = optim.AdamW(self.user_net.parameters(), lr=lr, weight_decay=wd)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        n = len(users)
        order = np.arange(n)
        for epoch in range(epochs):
            np.random.RandomState(epoch).shuffle(order)
            self.user_net.train()
            tot, nb = 0.0, 0
            bar = tqdm(range(0, n, bs), desc=f"hier epoch {epoch + 1}/{epochs}", leave=False)
            for s in bar:
                b = order[s:s + bs]
                pe = emb_t[torch.tensor(post_idx[b], device=self._device)]       # (B,N,E)
                pm = torch.tensor(post_mask[b], device=self._device)             # (B,N)
                yb = torch.tensor(y[b], dtype=torch.float32, device=self._device)
                opt.zero_grad()
                logits = self.user_net(pe, pm)
                loss = (bce(logits, yb) * w_task).mean()
                loss.backward(); opt.step()
                tot += float(loss.item()); nb += 1
                bar.set_postfix(loss=f"{loss.item():.3f}")
            log.info("hier.epoch", epoch=epoch + 1, loss=tot / max(1, nb))
        self._fitted = True
        return self

    # ---- inference (broadcast user score to post rows) ----------------- #
    def _user_scores(self, df: pd.DataFrame):
        import torch

        emb = self._encode_posts(df)
        users, post_idx, post_mask, _ = build_user_sequences(
            df, self.targets, self._max_posts, self._order, min_posts=1)
        emb_t = torch.tensor(emb, device=self._device)
        self.user_net.eval()
        scores = {}
        with torch.no_grad():
            for s in range(0, len(users), 256):
                idx = post_idx[s:s + 256]
                pm = torch.tensor(post_mask[s:s + 256], device=self._device)
                pe = emb_t[torch.tensor(idx, device=self._device)]
                p = torch.sigmoid(self.user_net(pe, pm)).cpu().numpy()
                for j, u in enumerate(users[s:s + 256]):
                    scores[u] = p[j]
        return scores

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Score each user, then broadcast the user score to all of that user's rows."""
        scores = self._user_scores(df)
        default = np.zeros(len(self.targets), dtype=np.float32)
        return np.stack([scores.get(a, default) for a in df["author_hash"].to_numpy()])

    # ---- persistence --------------------------------------------------- #
    def save(self, path) -> None:  # noqa: ANN001
        import torch

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.user_net.state_dict(), p / "user_net.pt")
        self.tokenizer.save_pretrained(p)
        (p / "hier_meta.json").write_text(json.dumps({
            "pretrained": self._pretrained, "fallback": self._fallback, "max_length": self._max_length,
            "aggregator": self._aggregator, "hidden": self._hidden, "dropout": self._dropout,
            "max_posts": self._max_posts, "order": self._order, "embed_dim": self._embed_dim,
            "targets": self.targets,
        }, indent=2))

    def load(self, path) -> "HierUserModel":  # noqa: ANN001
        import torch
        from transformers import AutoTokenizer

        p = Path(path)
        meta = json.loads((p / "hier_meta.json").read_text())
        self._pretrained = meta["pretrained"]; self._fallback = meta["fallback"]; self._max_length = meta["max_length"]
        self._aggregator = meta["aggregator"]; self._hidden = meta["hidden"]; self._dropout = meta["dropout"]
        self._max_posts = meta["max_posts"]; self._order = meta["order"]; self._embed_dim = meta["embed_dim"]
        self.tokenizer, self.encoder = self._build_encoder()
        self.tokenizer = AutoTokenizer.from_pretrained(p)
        self._device = self._device_select()
        self.encoder.to(self._device)
        self.user_net = self._build_user_net().to(self._device)
        self.user_net.load_state_dict(torch.load(p / "user_net.pt", map_location="cpu"))
        self.user_net.to(self._device)
        self._fitted = True
        return self
