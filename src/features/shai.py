"""SHAI-item symptom decomposition (Short Health Anxiety Inventory; Salkovskis 2002).

Decomposes health-anxiety language into the SHAI's clinical construct dimensions so
that a model's health-anxiety signal can be read against the instrument's structure
(rather than as one opaque score). Each dimension is a curated lexicon of terms /
phrases drawn from the SHAI items + the project's existing health-anxiety lexicons.

`score_shai(text)` returns a per-dimension rate (hits per token), comparable across
posts and subreddits.
"""
from __future__ import annotations

import re

# Dimensions mapped to SHAI construct groups (Salkovskis et al. 2002).
SHAI_DIMENSIONS: dict[str, set[str]] = {
    # SHAI: worry about health / preoccupation
    "illness_worry": {
        "health anxiety", "illness anxiety", "hypochondria", "hypochondriac",
        "worried about my health", "preoccupied", "obsessed with my health",
        "anxious about my health", "fixated on",
    },
    # SHAI: awareness of bodily sensations & changes
    "bodily_vigilance": {
        "twinge", "tingling", "numbness", "lightheaded", "dizzy", "dizziness",
        "palpitations", "tremor", "twitch", "twitching", "shortness of breath",
        "chest pain", "every sensation", "aware of my body", "notice every",
        "checking pulse", "took my pulse", "felt my pulse", "checking heart rate",
    },
    # SHAI: fear of having a serious illness
    "serious_illness_fear": {
        "cancer", "tumor", "tumour", "stroke", "heart attack", "aneurysm",
        "blood clot", "embolism", "als", "multiple sclerosis", "lupus", "leukemia",
        "lymphoma", "sepsis", "meningitis", "convinced i have", "convinced i'm dying",
        "convinced im dying", "am i dying", "is this cancer", "terrified i have",
        "scared it's", "serious illness", "dying",
    },
    # SHAI/behavioral: symptom checking & online searching
    "symptom_checking": {
        "googled my symptoms", "googling symptoms", "googled symptoms", "google symptoms",
        "webmd", "dr google", "spent hours googling", "self-diagnose", "self-diagnosing",
        "self diagnosed", "monitoring my symptoms", "checking my symptoms", "researching",
    },
    # SHAI: reassurance seeking
    "reassurance_seeking": {
        "please tell me", "please reassure me", "tell me i'm okay", "tell me im okay",
        "is this normal", "anyone else", "has anyone had", "did anyone",
        "should i be worried", "am i overreacting", "reassure me", "reassurance",
    },
    # SHAI: not reassured by normal results / persistent doubt
    "difficulty_reassured": {
        "tests came back normal", "labs were normal", "ekg was normal", "mri was clear",
        "blood work was", "doctor said i'm fine", "doctors say i'm fine",
        "can't accept the test results", "won't accept", "don't believe the doctor",
        "second opinion", "third opinion", "but i still",
    },
    # SHAI/behavioral: medical help seeking
    "medical_help_seeking": {
        "should i go to the er", "should i go to the hospital", "doctors keep telling me",
        "ct scan", "blood test", "went to the er", "urgent care", "emergency room",
        "made an appointment", "saw a specialist", "doctors appointment",
    },
}

_WORD_RE = re.compile(r"[a-z']+")


def score_shai(text: str) -> dict[str, float]:
    """Per-SHAI-dimension rate = matched terms/phrases per token (case-insensitive)."""
    t = str(text).lower()
    tokens = _WORD_RE.findall(t)
    n = max(len(tokens), 1)
    token_set: dict[str, int] = {}
    for tok in tokens:
        token_set[tok] = token_set.get(tok, 0) + 1
    out: dict[str, float] = {}
    for dim, terms in SHAI_DIMENSIONS.items():
        hits = 0
        for term in terms:
            if " " in term or "-" in term:        # phrase: substring match
                hits += t.count(term)
            else:                                   # single token: exact word match
                hits += token_set.get(term, 0)
        out[dim] = hits / n
    return out


def shai_dimensions() -> list[str]:
    return list(SHAI_DIMENSIONS)
