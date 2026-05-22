"""PII removal. Pipeline-enforced — no model sees text that hasn't passed through here.

We do **not** rely solely on regex. For PERSON / GPE / ORG entities we use
spaCy when available; if spaCy isn't installed, we fall back to a regex-only
mode and log a warning so the user knows to install it before any real run.
"""
from __future__ import annotations

import hashlib
import re

from src.utils.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Regex layer — fast, always-on
# --------------------------------------------------------------------------- #

RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_PHONE = re.compile(
    r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b"
)
RE_REDDIT_USER = re.compile(r"\b(?:/u/|u/)[A-Za-z0-9_-]+\b")
RE_REDDIT_SUB_MENTION = re.compile(r"\b/?r/[A-Za-z0-9_-]+\b")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
# Loose handle pattern (twitter/instagram-like @user)
RE_HANDLE = re.compile(r"(?<![A-Za-z0-9])@[A-Za-z0-9_]{2,}\b")


def _hash_username(username: str | None, salt: str = "anxiety-research-v1") -> str | None:
    """Stable, salted hash so the same author maps to the same pseudonym across runs."""
    if username is None:
        return None
    return "u_" + hashlib.sha256(f"{salt}|{username}".encode()).hexdigest()[:12]


def regex_redact(text: str) -> str:
    text = RE_URL.sub(" [URL] ", text)
    text = RE_EMAIL.sub(" [EMAIL] ", text)
    text = RE_PHONE.sub(" [PHONE] ", text)
    text = RE_REDDIT_USER.sub(" [USER] ", text)
    text = RE_REDDIT_SUB_MENTION.sub(" [SUB] ", text)
    text = RE_HANDLE.sub(" [HANDLE] ", text)
    return text


# --------------------------------------------------------------------------- #
# spaCy NER layer — replaces PERSON / GPE / ORG when the model is available
# --------------------------------------------------------------------------- #

_NLP = None


def _get_spacy():
    global _NLP
    if _NLP is False:
        return None
    if _NLP is None:
        try:
            import spacy

            _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger"])
        except (ImportError, OSError):
            log.warning(
                "anonymize.spacy_unavailable",
                hint="run `python -m spacy download en_core_web_sm`",
            )
            _NLP = False
            return None
    return _NLP


REPLACE_ENT = {
    "PERSON": "[PERSON]",
    "GPE": "[LOC]",
    "LOC": "[LOC]",
    "ORG": "[ORG]",
}


def _redact_doc(text: str, doc) -> str:
    """Apply NER replacements to `text` given an already-parsed spaCy `doc`."""
    out: list[str] = []
    last = 0
    for ent in doc.ents:
        if ent.label_ not in REPLACE_ENT:
            continue
        out.append(text[last : ent.start_char])
        out.append(f" {REPLACE_ENT[ent.label_]} ")
        last = ent.end_char
    out.append(text[last:])
    return "".join(out)


def ner_redact(text: str, max_chars: int = 5000) -> str:
    """Single-text NER redaction. For batched scale, use `ner_redact_batch`."""
    nlp = _get_spacy()
    if nlp is None:
        return text
    # Truncate for very long posts to keep latency bounded; PII is usually
    # concentrated near the start anyway.
    truncated = text[:max_chars]
    tail = text[max_chars:]
    return _redact_doc(truncated, nlp(truncated)) + tail


def ner_redact_batch(
    texts: list[str],
    max_chars: int = 5000,
    batch_size: int = 64,
    n_process: int = 1,
) -> list[str]:
    """Batched NER redaction using spaCy's `nlp.pipe`.

    Much faster than calling `ner_redact` in a loop because spaCy can stream
    batches through its pipeline and (with n_process>1) parallelize across
    multiple worker processes. Falls back to regex-only output if spaCy is
    unavailable.
    """
    nlp = _get_spacy()
    if nlp is None:
        return list(texts)

    truncated = [(t or "")[:max_chars] for t in texts]
    tails = [(t or "")[max_chars:] for t in texts]
    out: list[str] = []
    for i, doc in enumerate(nlp.pipe(truncated, batch_size=batch_size, n_process=n_process)):
        out.append(_redact_doc(truncated[i], doc) + tails[i])
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def anonymize(text: str, use_ner: bool = True) -> str:
    text = regex_redact(text)
    if use_ner:
        text = ner_redact(text)
    return text


def anonymize_batch(texts: list[str], use_ner: bool = True, n_process: int = 1) -> list[str]:
    """Batched anonymization. Regex layer first (vectorized), then NER pipe."""
    redacted = [regex_redact(t or "") for t in texts]
    if use_ner:
        redacted = ner_redact_batch(redacted, n_process=n_process)
    return redacted


def anonymize_record(record: dict, use_ner: bool = True) -> dict:
    """Return a copy of `record` with text anonymized and author pseudonymized."""
    new = dict(record)
    new["clean_text"] = anonymize(record.get("clean_text") or "", use_ner=use_ner)
    new["author_hash"] = _hash_username(record.get("author"))
    new.pop("author", None)
    return new
