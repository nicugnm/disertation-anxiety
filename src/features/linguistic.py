"""Hand-crafted linguistic features for the XGBoost baseline + linguistic analysis.

Each feature is documented with its psycholinguistic motivation. These are
the features the *thesis* discusses qualitatively (which markers correlate
with health anxiety?), so transparency matters.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.labeling.lexicons import (
    ANXIETY_PHRASES,
    ANXIETY_TERMS,
    BODY_PARTS,
    CERTAINTY_TERMS,
    DEPRESSION_TERMS,
    FIRST_PERSON_PLURAL,
    FIRST_PERSON_SINGULAR,
    HEALTH_ANXIETY_PHRASES,
    HEALTH_ANXIETY_TERMS,
    REASSURANCE_PATTERNS,
    SECOND_PERSON,
    SUICIDALITY_TERMS,
    THIRD_PERSON,
    UNCERTAINTY_TERMS,
)

RE_TOKEN = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
RE_SENT = re.compile(r"[.!?]+\s+|\n+")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _tokens(text: str) -> list[str]:
    return RE_TOKEN.findall((text or "").lower())


def _sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s for s in RE_SENT.split(text) if s.strip()]


def _ratio(num: int, denom: int) -> float:
    return num / denom if denom else 0.0


def _term_count(tokens: list[str], lex: set[str]) -> int:
    return sum(1 for t in tokens if t in lex)


def _phrase_count(text: str, lex: set[str]) -> int:
    t = (text or "").lower()
    return sum(1 for p in lex if p in t)


# --------------------------------------------------------------------------- #
# Individual features
# --------------------------------------------------------------------------- #


def lexical_features(text: str) -> dict[str, float]:
    tokens = _tokens(text)
    n = max(1, len(tokens))
    return {
        "f_anx_term_rate": _ratio(_term_count(tokens, ANXIETY_TERMS), n),
        "f_anx_phrase_count": _phrase_count(text, ANXIETY_PHRASES),
        "f_health_anx_term_rate": _ratio(_term_count(tokens, HEALTH_ANXIETY_TERMS), n),
        "f_health_anx_phrase_count": _phrase_count(text, HEALTH_ANXIETY_PHRASES),
        "f_reassurance_count": _phrase_count(text, REASSURANCE_PATTERNS),
        "f_dep_term_rate": _ratio(_term_count(tokens, DEPRESSION_TERMS), n),
        "f_suic_term_rate": _ratio(_term_count(tokens, SUICIDALITY_TERMS), n),
        "f_body_part_rate": _ratio(_term_count(tokens, BODY_PARTS), n),
    }


def pronoun_features(text: str) -> dict[str, float]:
    """First-person preponderance is robustly associated with depression and anxiety."""
    tokens = _tokens(text)
    n = max(1, len(tokens))
    return {
        "f_first_sing_rate": _ratio(_term_count(tokens, FIRST_PERSON_SINGULAR), n),
        "f_first_plur_rate": _ratio(_term_count(tokens, FIRST_PERSON_PLURAL), n),
        "f_second_rate": _ratio(_term_count(tokens, SECOND_PERSON), n),
        "f_third_rate": _ratio(_term_count(tokens, THIRD_PERSON), n),
    }


def certainty_features(text: str) -> dict[str, float]:
    tokens = _tokens(text)
    n = max(1, len(tokens))
    unc = _term_count(tokens, UNCERTAINTY_TERMS) + _phrase_count(text, UNCERTAINTY_TERMS)
    cer = _term_count(tokens, CERTAINTY_TERMS) + _phrase_count(text, CERTAINTY_TERMS)
    return {
        "f_uncertainty_rate": _ratio(unc, n),
        "f_certainty_rate": _ratio(cer, n),
        "f_question_rate": _ratio((text or "").count("?"), n),
    }


def length_features(text: str) -> dict[str, float]:
    tokens = _tokens(text)
    sents = _sentences(text)
    n_chars = len(text or "")
    return {
        "f_n_chars": float(n_chars),
        "f_n_tokens": float(len(tokens)),
        "f_n_sents": float(len(sents)),
        "f_avg_sent_len": _ratio(len(tokens), len(sents)) if sents else 0.0,
        "f_avg_word_len": (np.mean([len(t) for t in tokens]) if tokens else 0.0),
    }


def readability_features(text: str) -> dict[str, float]:
    """Wraps `textstat`; handles import failure gracefully."""
    try:
        import textstat
    except ImportError:
        return {"f_flesch": 0.0, "f_gunning_fog": 0.0}
    text = text or ""
    if len(text) < 20:
        return {"f_flesch": 0.0, "f_gunning_fog": 0.0}
    try:
        return {
            "f_flesch": float(textstat.flesch_reading_ease(text)),
            "f_gunning_fog": float(textstat.gunning_fog(text)),
        }
    except Exception:  # noqa: BLE001
        return {"f_flesch": 0.0, "f_gunning_fog": 0.0}


def sentiment_features(text: str) -> dict[str, float]:
    """VADER sentiment — fast, no model download required."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        # Module-level singleton via function attribute
        if not hasattr(sentiment_features, "_an"):
            sentiment_features._an = SentimentIntensityAnalyzer()
        s = sentiment_features._an.polarity_scores(text or "")
        return {
            "f_sent_compound": float(s["compound"]),
            "f_sent_pos": float(s["pos"]),
            "f_sent_neg": float(s["neg"]),
            "f_sent_neu": float(s["neu"]),
        }
    except ImportError:
        return {"f_sent_compound": 0.0, "f_sent_pos": 0.0, "f_sent_neg": 0.0, "f_sent_neu": 0.0}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


FEATURE_GROUPS = ("lexical", "pronoun", "certainty", "length", "readability", "sentiment")


def extract_one(text: str, groups: Iterable[str] = FEATURE_GROUPS) -> dict[str, float]:
    out: dict[str, float] = {}
    if "lexical" in groups:
        out.update(lexical_features(text))
    if "pronoun" in groups:
        out.update(pronoun_features(text))
    if "certainty" in groups:
        out.update(certainty_features(text))
    if "length" in groups:
        out.update(length_features(text))
    if "readability" in groups:
        out.update(readability_features(text))
    if "sentiment" in groups:
        out.update(sentiment_features(text))
    return out


def extract_dataframe(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    groups: Iterable[str] = FEATURE_GROUPS,
) -> pd.DataFrame:
    rows = [extract_one(t, groups) for t in df[text_col].fillna("")]
    feats = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]
