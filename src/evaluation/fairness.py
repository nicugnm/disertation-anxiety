"""Subgroup fairness audit utilities.

Protected demographics are unavailable (the corpus is anonymized) and inferring
them with a classifier would be unreliable and inject bias — so we use only
**self-reported** gender/age (regex on the text, partial coverage, exploratory)
plus observable strata such as post length. Reports per-group TPR/FPR/F1/selection
rate and standard fairness gaps (equal-opportunity, equalized-odds, demographic
parity).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_TAG = re.compile(r"\b(\d{2})\s?([mf])\b|\b([mf])\s?(\d{2})\b", re.I)
_GENDER_PHRASE = re.compile(
    r"\b(?:i am|i'?m|as)\s+a\s+(?:\d{1,2}\s+year[- ]old\s+)?(man|woman|male|female|guy|girl|dude|lady)\b", re.I
)
_AGE_PHRASE = re.compile(r"\b(\d{1,2})\s?(?:years?[- ]old|yo|y/o)\b|\b(?:i am|i'?m)\s+(\d{1,2})\b", re.I)
_MALE = {"man", "male", "guy", "dude"}
_FEMALE = {"woman", "female", "girl", "lady"}


def extract_gender(text: str) -> str | None:
    """Return 'M'/'F' if the author self-reports gender, else None (None if conflicting)."""
    t = str(text)
    found = set()
    for m in _TAG.finditer(t):
        g = (m.group(2) or m.group(3)).upper()
        found.add(g)
    for m in _GENDER_PHRASE.finditer(t):
        w = m.group(1).lower()
        found.add("M" if w in _MALE else "F")
    return found.pop() if len(found) == 1 else None


def extract_age(text: str) -> int | None:
    """Return a self-reported age in [13, 99], else None."""
    t = str(text)
    cands = []
    for m in _TAG.finditer(t):
        cands.append(int(m.group(1) or m.group(4)))
    for m in _AGE_PHRASE.finditer(t):
        cands.append(int(m.group(1) or m.group(2)))
    cands = [a for a in cands if 13 <= a <= 99]
    return cands[0] if cands else None


def subgroup_metrics(y_true, y_pred, groups, min_n: int = 30, min_pos: int = 5) -> pd.DataFrame:
    """Per-group TPR / FPR / F1 / selection rate, for groups with enough data."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    groups = np.asarray(groups, dtype=object)
    rows = []
    for g in pd.unique(groups):
        if g is None or (isinstance(g, float) and np.isnan(g)):
            continue
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        npos = int(yt.sum())
        if m.sum() < min_n or npos < min_pos:
            continue
        tp = int(((yt == 1) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum()); tn = int(((yt == 0) & (yp == 0)).sum())
        rows.append({
            "group": g, "n": int(m.sum()), "n_pos": npos,
            "tpr": tp / (tp + fn) if tp + fn else np.nan,
            "fpr": fp / (fp + tn) if fp + tn else np.nan,
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "selection_rate": float(yp.mean()),
        })
    return pd.DataFrame(rows)


def fairness_gaps(sub: pd.DataFrame) -> dict:
    """Max-min gaps across subgroups: equal-opportunity (TPR), FPR, demographic
    parity (selection rate), and equalized odds (max of TPR & FPR gaps)."""
    if len(sub) < 2:
        return {"tpr_gap": np.nan, "fpr_gap": np.nan, "selection_rate_gap": np.nan, "equalized_odds_diff": np.nan}
    tpr_gap = float(np.nanmax(sub["tpr"]) - np.nanmin(sub["tpr"]))
    fpr_gap = float(np.nanmax(sub["fpr"]) - np.nanmin(sub["fpr"]))
    sel_gap = float(np.nanmax(sub["selection_rate"]) - np.nanmin(sub["selection_rate"]))
    return {"tpr_gap": tpr_gap, "fpr_gap": fpr_gap, "selection_rate_gap": sel_gap,
            "equalized_odds_diff": max(tpr_gap, fpr_gap)}
