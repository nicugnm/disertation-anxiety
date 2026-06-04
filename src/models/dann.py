"""DANN — domain-adversarial multi-task transformer (Ganin et al. 2016).

Extends the multi-task model with a **subreddit-discriminator head behind a
gradient-reversal layer (GRL)**. The shared encoder is trained both to predict
the targets and to *fool* the subreddit classifier, yielding subreddit-invariant
features — the textbook fix for the cross-subreddit F1 collapse in Experiment 2.

Loss = Σ task-BCE  +  domain-CE, with the GRL reversing the domain gradient into
the encoder (so the encoder minimises task loss while *maximising* domain
confusion). The reversal strength α ramps 0→λ_max on the Ganin schedule.

`config.extra['domain']`:
  'subreddit' — discriminator predicts the exact subreddit (fine-grained)
  'group'     — discriminator predicts the configs/subreddits.yaml group (coarse)

Reuses MultiTaskTransformer for device selection, label/weight extraction,
`predict_proba`, and eval logging; only the head, training loop, and (de)ser
differ.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.multitask import MultiTaskTransformer
from src.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Gradient Reversal Layer
# --------------------------------------------------------------------------- #


class GradientReversal(torch.autograd.Function):
    """Identity on the forward pass; negates and scales the gradient on backward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha: float = 1.0):
    return GradientReversal.apply(x, alpha)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class DannMultiTaskModel(MultiTaskTransformer):
    """Shared encoder + per-target sigmoid heads + GRL subreddit discriminator."""

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        self._domain_values: list[str] | None = None  # class index -> domain label

    # ---- domain labels -------------------------------------------------- #

    def _domain_series(self, df) -> "pd.Series":  # noqa: F821
        """Map each row's subreddit to its domain label ('subreddit' or 'group')."""
        mode = self.config.extra.get("domain", "subreddit")
        if mode == "group":
            from src.utils.config import load_subreddits

            cfg = load_subreddits(self.config.extra.get("subreddits_config", "configs/subreddits.yaml"))
            name_to_group = {s.name.lower(): s.group for s in cfg.subreddits}
            return df["subreddit"].astype(str).str.lower().map(name_to_group).fillna("baseline")
        return df["subreddit"].astype(str)

    # ---- model -------------------------------------------------------- #

    def _build_dann_model(self, n_domains: int):
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
                log.info("dann.load", model=name)
                break
            except Exception as ex:  # noqa: BLE001
                log.warning("dann.load_failed", model=name, error=str(ex))
        else:
            raise RuntimeError(f"Could not load {primary} or {fallback}")

        n_targets = len(self.targets)
        hidden = cfg.hidden_size

        class DannHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.dropout = nn.Dropout(0.1)
                self.target_head = nn.Linear(hidden, n_targets)
                self.domain_head = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(hidden, n_domains),
                )

            @staticmethod
            def _pool(last, attention_mask):
                if attention_mask is None:
                    return last.mean(dim=1)
                mask = attention_mask.unsqueeze(-1).float()
                summed = (last * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                return summed / counts

            def forward(self, input_ids, attention_mask=None, alpha=None, **_kw):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                pooled = self._pool(out.last_hidden_state, attention_mask)
                target_logits = self.target_head(self.dropout(pooled))
                if alpha is None:  # inference path — reused by MultiTaskTransformer.predict_proba
                    return target_logits
                domain_logits = self.domain_head(grad_reverse(pooled, alpha))
                return target_logits, domain_logits

        return tok, DannHead()

    # ---- training ----------------------------------------------------- #

    def fit(self, train, val=None, sample_weight=None) -> "DannMultiTaskModel":  # noqa: ANN001
        from torch import optim
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        e = self.config.extra
        tok_cfg = e.get("tokenizer", {})
        train_cfg = e.get("train", {})
        loss_w = e.get("loss_weights", {})
        gamma = float(e.get("grl_gamma", 10.0))
        lambda_max = float(e.get("lambda_max", 1.0))

        dom_series = self._domain_series(train)
        self._domain_values = sorted(dom_series.dropna().astype(str).unique().tolist())
        dom_to_idx = {d: i for i, d in enumerate(self._domain_values)}
        n_domains = len(self._domain_values)

        self.tokenizer, self.model = self._build_dann_model(n_domains)
        self._device = self._device_select()
        self.model.to(self._device)
        log.info("dann.fit.start", domain=e.get("domain", "subreddit"),
                 n_domains=n_domains, device=str(self._device))

        max_len = tok_cfg.get("max_length", 256)
        epochs = int(train_cfg.get("num_train_epochs", 5))
        bs = int(train_cfg.get("per_device_train_batch_size", 16))
        lr = float(train_cfg.get("learning_rate", 2e-5))
        wd = float(train_cfg.get("weight_decay", 0.01))
        w_task = torch.tensor([loss_w.get(t, 1.0) for t in self.targets],
                              dtype=torch.float32, device=self._device)

        y = self._y_multi(train)
        w = self._w_multi(train)
        dom_idx = dom_series.astype(str).map(dom_to_idx).fillna(0).astype(int).values
        texts = train[self.config.text_field].astype(str).fillna("").tolist()
        tok = self.tokenizer

        class _DS(Dataset):
            def __len__(self):
                return len(texts)

            def __getitem__(self, i):
                enc = tok(texts[i], truncation=True, max_length=max_len, padding=False)
                return enc["input_ids"], enc["attention_mask"], y[i], w[i], int(dom_idx[i])

        def collate(batch):
            from torch.nn.utils.rnn import pad_sequence

            ids = pad_sequence([torch.tensor(b[0], dtype=torch.long) for b in batch],
                               batch_first=True, padding_value=tok.pad_token_id or 0)
            mask = pad_sequence([torch.tensor(b[1], dtype=torch.long) for b in batch],
                                batch_first=True, padding_value=0)
            labels = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
            weights = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
            domains = torch.tensor([b[4] for b in batch], dtype=torch.long)
            return ids, mask, labels, weights, domains

        dl = DataLoader(_DS(), batch_size=bs, shuffle=True, collate_fn=collate)
        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        ce = nn.CrossEntropyLoss()
        total_steps = max(1, epochs * len(dl))
        step = 0

        for epoch in range(epochs):
            self.model.train()
            tot_task = tot_dom = 0.0
            nb = 0
            alpha = 0.0
            bar = tqdm(dl, desc=f"DANN epoch {epoch + 1}/{epochs}", leave=False)
            for ids, mask, labels, weights, domains in bar:
                ids = ids.to(self._device); mask = mask.to(self._device)
                labels = labels.to(self._device); weights = weights.to(self._device)
                domains = domains.to(self._device)
                p = step / total_steps
                alpha = lambda_max * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)
                opt.zero_grad()
                tgt_logits, dom_logits = self.model(
                    input_ids=ids, attention_mask=mask, alpha=alpha
                )
                task_loss = (bce(tgt_logits, labels) * weights * w_task).mean()
                dom_loss = ce(dom_logits, domains)
                (task_loss + dom_loss).backward()
                opt.step()
                tot_task += float(task_loss.item()); tot_dom += float(dom_loss.item()); nb += 1
                step += 1
                bar.set_postfix(task=f"{task_loss.item():.3f}", dom=f"{dom_loss.item():.3f}",
                                alpha=f"{alpha:.2f}")
            log.info("dann.epoch", epoch=epoch + 1, task_loss=tot_task / max(1, nb),
                     domain_loss=tot_dom / max(1, nb), alpha=round(alpha, 3))
            if val is not None and not val.empty:
                self._eval_log(val)

        self._fitted = True
        return self

    # ---- persistence -------------------------------------------------- #

    def save(self, path) -> None:  # noqa: ANN001
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p / "model.pt")
        self.tokenizer.save_pretrained(p)
        (p / "targets.txt").write_text("\n".join(self.targets))
        (p / "domains.txt").write_text("\n".join(self._domain_values or []))

    def load(self, path) -> "DannMultiTaskModel":  # noqa: ANN001
        from transformers import AutoTokenizer

        p = Path(path)
        self._domain_values = (
            [ln for ln in (p / "domains.txt").read_text().splitlines() if ln]
            if (p / "domains.txt").exists() else []
        )
        self.tokenizer, self.model = self._build_dann_model(len(self._domain_values))
        self.tokenizer = AutoTokenizer.from_pretrained(p)
        self.model.load_state_dict(torch.load(p / "model.pt", map_location="cpu"))
        self._device = self._device_select()
        self.model.to(self._device)
        self._fitted = True
        return self
