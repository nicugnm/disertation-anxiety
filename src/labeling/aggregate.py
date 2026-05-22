"""Combine the three tiers into final labels + per-row confidence weights.

Precedence: manual > llm > weak. The output also keeps the raw tier columns
so the thesis can do ablation analysis (e.g. train on weak only vs weak+llm).
"""
from __future__ import annotations

import pandas as pd

from src.labeling.weak import LABELS
from src.utils.config import LabelingConfig
from src.utils.logging import get_logger

log = get_logger(__name__)


def aggregate_labels(df: pd.DataFrame, cfg: LabelingConfig) -> pd.DataFrame:
    """Add `label_<k>` and `label_<k>_source` and `label_<k>_weight` columns."""
    out = df.copy()
    precedence = cfg.aggregate.precedence
    conf_map = cfg.aggregate.tier_confidence

    def pick(row: pd.Series, k: str) -> tuple[float | None, str | None, float | None]:
        for tier in precedence:
            col_map = {
                "manual": f"manual_{k}",
                "llm": f"llm_{k}",
                "disclosure": f"disclosure_{k}",
                "weak": f"weak_{k}_bin",
            }
            col = col_map.get(tier)
            if col and col in row.index and pd.notna(row[col]):
                val = float(row[col])
                # Disclosure is asymmetric: a positive (=1) is a high-confidence
                # clinical claim, but a 0 only means "regex didn't fire" — NOT
                # evidence the user is non-anxious. Fall through to weak for
                # the negative signal. Same logic for LLM if the prompt is
                # asymmetric (here it isn't — LLM returns 0/1 both meaningfully).
                if tier == "disclosure" and val == 0:
                    continue
                return val, tier, conf_map.get(tier, 0.5)
        return None, None, None

    for k in LABELS:
        labels, sources, weights = [], [], []
        for _, row in out.iterrows():
            v, src, w = pick(row, k)
            labels.append(v)
            sources.append(src)
            weights.append(w)
        out[f"label_{k}"] = labels
        out[f"label_{k}_source"] = sources
        out[f"label_{k}_weight"] = weights

    if cfg.aggregate.require_at_least_one_tier:
        any_label = out[[f"label_{k}" for k in LABELS]].notna().any(axis=1)
        dropped = (~any_label).sum()
        if dropped:
            log.warning("aggregate.dropped_no_labels", n=int(dropped))
        out = out[any_label].reset_index(drop=True)

    log.info(
        "aggregate.done",
        n=len(out),
        sources={k: out[f"label_{k}_source"].value_counts(dropna=False).to_dict() for k in LABELS},
    )
    return out
