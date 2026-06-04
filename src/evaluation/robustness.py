"""Lightweight, dependency-free robustness perturbations (TextBugger-style).

Each perturbation maps text -> minimally edited text that a human still reads the
same way, applied to a fraction `p` of eligible words. Seeded via an np.random
Generator for reproducibility. Used to measure how often a model's decision flips
under meaning-preserving noise — TF-IDF (exact tokens) is expected to be far more
fragile than a subword transformer.
"""
from __future__ import annotations

import numpy as np

_KB = {
    "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj",
    "u": "yihjk", "i": "uojkl", "o": "ipkl", "p": "ol", "a": "qwsz", "s": "qweadzx",
    "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn", "j": "yuihknm",
    "k": "uiojlm", "l": "iopk", "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb",
    "b": "ghvn", "n": "hjbm", "m": "jkn",
}
_VOWELS = "aeiou"


def _edit_words(text, rng, p, fn, min_len=4):
    """Apply char-edit `fn` to each word (len>=min_len) with probability p."""
    words = text.split()
    out = []
    for w in words:
        if len(w) >= min_len and rng.random() < p:
            out.append(fn(w, rng))
        else:
            out.append(w)
    return " ".join(out)


def char_swap(text, rng, p=0.15):
    def f(w, rng):
        i = int(rng.integers(0, len(w) - 1))
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return _edit_words(text, rng, p, f)


def char_delete(text, rng, p=0.15):
    def f(w, rng):
        i = int(rng.integers(0, len(w)))
        return w[:i] + w[i + 1:]
    return _edit_words(text, rng, p, f)


def keyboard_typo(text, rng, p=0.15):
    def f(w, rng):
        i = int(rng.integers(0, len(w)))
        c = w[i].lower()
        if c in _KB:
            sub = _KB[c][int(rng.integers(0, len(_KB[c])))]
            sub = sub.upper() if w[i].isupper() else sub
            return w[:i] + sub + w[i + 1:]
        return w
    return _edit_words(text, rng, p, f)


def case_flip(text, rng, p=0.15):
    def f(w, rng):
        return w.swapcase()
    return _edit_words(text, rng, p, f, min_len=1)


def punct_strip(text, rng, p=1.0):
    return "".join(ch for ch in text if ch.isalnum() or ch.isspace())


def social_elongate(text, rng, p=0.15):
    def f(w, rng):
        idxs = [i for i, ch in enumerate(w) if ch.lower() in _VOWELS]
        if not idxs:
            return w
        i = idxs[int(rng.integers(0, len(idxs)))]
        return w[:i] + w[i] * int(rng.integers(2, 4)) + w[i + 1:]
    return _edit_words(text, rng, p, f, min_len=2)


PERTURBATIONS = {
    "char_swap": char_swap,
    "char_delete": char_delete,
    "keyboard_typo": keyboard_typo,
    "case_flip": case_flip,
    "punct_strip": punct_strip,
    "social_elongate": social_elongate,
}


def flip_rate(clean_pred, pert_pred) -> float:
    """Fraction of examples whose binary decision changed under perturbation."""
    return float(np.mean(np.asarray(clean_pred) != np.asarray(pert_pred)))


def mean_abs_score_drift(clean_score, pert_score) -> float:
    return float(np.mean(np.abs(np.asarray(clean_score, float) - np.asarray(pert_score, float))))
