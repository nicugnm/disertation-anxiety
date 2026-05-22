"""Self-disclosure labeling — the field-standard proxy for clinical ground truth.

Mental-health NLP research from Coppersmith et al. (2014, 2015) onwards uses
**self-disclosed diagnosis** as a proxy for true clinical labels in cases where
specialist annotation is infeasible. The CLEF eRisk workshops (Crestani, Losada,
Parapar, 2017–present) codified this into a standard protocol:

  1. Find candidate posts via regex templates ("I was diagnosed with X").
  2. Filter false positives: negations, hypotheticals, third-party reports,
     denials, jokes/rhetorical.
  3. Human-verify the *users* who pass the filter (eRisk does this offline).
  4. Treat verified users as the positive class; randomly sample negatives
     as a control group.

This module implements steps 1–2. Step 3 (human verification) is described in
`docs/codebook.md`. Step 4 happens at split time.

References:
  - Coppersmith, Dredze, Harman (2014). "Quantifying Mental Health Signals in
    Twitter." ACL Workshop on Computational Linguistics and Clinical Psychology.
  - Losada, Crestani, Parapar (2017–). CLEF eRisk Overview papers.
  - De Choudhury et al. (2013) on Twitter depression detection.

Outputs per row:
  disclosure_<target>          int   0/1 — verified disclosure detected
  disclosure_<target>_match    str   the matched substring (for traceability)

This is a high-confidence label source (tier_confidence = 0.85 by default),
ranked above LLM (0.7) but below human annotation (1.0).
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Positive patterns per target
# --------------------------------------------------------------------------- #

# Each pattern captures the matched span via `re.search`. Patterns are
# case-insensitive (compiled with re.IGNORECASE). Surrounding context is
# checked against NEGATION / HYPOTHETICAL / THIRD_PARTY filters below.

ANXIETY_DX_PATTERNS: list[str] = [
    # Diagnosis-verb patterns — strong signal
    r"\b(i|i've|i have|i am|i was|i got|i'm) (?:been )?diagnosed (?:with|as) "
    r"(?:having |an? )?(?:generali[sz]ed |severe |chronic |clinical )?anxiety(?: disorder)?\b",
    r"\bmy (?:doctor|therapist|psychiatrist|psychologist|gp|md|shrink) "
    r"(?:diagnosed me with|said i have|told me i (?:had|have)) "
    r"(?:an? )?(?:generali[sz]ed )?anxiety(?: disorder)?\b",
    # Specific subtype/code mentions — strong signal
    r"\bi (?:have|was diagnosed with|got diagnosed with) GAD\b",
    r"\bi (?:have|suffer from) generali[sz]ed anxiety disorder\b",
    r"\bi have panic disorder\b",
    # Possession patterns — moderately strong; require qualifier
    r"\bi (?:have|suffer from|struggle with|am dealing with) (?:an? )?anxiety disorder\b",
    r"\bi'm on (?:medication|meds) for (?:my )?anxiety\b",
]

HEALTH_ANXIETY_DX_PATTERNS: list[str] = [
    # Direct condition naming
    r"\b(i|i've|i have|i am|i was|i got|i'm) (?:been )?diagnosed (?:with|as) "
    r"(?:having )?(?:health anxiety|illness anxiety|hypochondria(?:sis)?)\b",
    r"\bi (?:have|suffer from|struggle with|am dealing with) health anxiety\b",
    r"\bi (?:have|suffer from|struggle with) illness anxiety(?: disorder)?\b",
    r"\bi have IAD\b",  # Illness Anxiety Disorder DSM-5 code
    r"\bi(?:'m| am) (?:a )?hypochondriac\b",
    r"\bi have hypochondria(?:sis)?\b",
    r"\bmy (?:doctor|therapist|psychiatrist|psychologist) "
    r"(?:diagnosed me with|told me i have) "
    r"(?:health anxiety|illness anxiety|hypochondria(?:sis)?)\b",
]

DEPRESSION_DX_PATTERNS: list[str] = [
    # Diagnosis-verb patterns
    r"\b(i|i've|i have|i am|i was|i got|i'm) (?:been )?diagnosed (?:with|as) "
    r"(?:having |an? )?(?:clinical |major |severe |chronic )?depress(?:ion|ive disorder)\b",
    r"\bmy (?:doctor|therapist|psychiatrist|psychologist|gp|md|shrink) "
    r"(?:diagnosed me with|said i have|told me i (?:had|have)) "
    r"(?:clinical |major |severe )?depress(?:ion|ive disorder)\b",
    # Subtype/code mentions
    r"\bi (?:have|was diagnosed with|got diagnosed with) MDD\b",
    r"\bi (?:have|suffer from) major depressive disorder\b",
    # Possession patterns
    r"\bi (?:have|suffer from|struggle with|am dealing with) "
    r"(?:clinical |major |severe )?depression\b",
    r"\bi(?:'m| am) (?:clinically |severely |majorly )?depressed\b",
    r"\bi'm on (?:antidepressants|medication|meds) for (?:my )?depression\b",
]

# Suicidality is not typically self-disclosed as a "diagnosis". Past suicide
# attempts may be reported but require very careful handling. We disable
# self-disclosure for this target to avoid false positives that could
# misclassify highly sensitive content.
SUICIDALITY_DX_PATTERNS: list[str] = []

PATTERNS_BY_TARGET: dict[str, list[str]] = {
    "anxiety": ANXIETY_DX_PATTERNS,
    "health_anxiety": HEALTH_ANXIETY_DX_PATTERNS,
    "depression": DEPRESSION_DX_PATTERNS,
    "suicidality": SUICIDALITY_DX_PATTERNS,
}

# --------------------------------------------------------------------------- #
# False-positive filters
# --------------------------------------------------------------------------- #

# A match within ±NEG_WINDOW characters of any of these patterns is rejected.

NEG_WINDOW_BEFORE = 50  # look back this many chars
NEG_WINDOW_AFTER = 0    # negations/hypotheticals are almost always before

NEGATION_PATTERNS: list[str] = [
    r"\b(?:not|never|no|n't|nor)\b",
    r"\bundiagnosed\b",
    r"\bcan't be\b",
    r"\bnever been\b",
    r"\bdidn't get\b",
    r"\bwon't be\b",
    r"\bwithout (?:being )?diagnosed\b",
]

HYPOTHETICAL_PATTERNS: list[str] = [
    r"\bif (?:i|you|she|he|they) (?:were|was|am|get|got|could|might|had|have)\b",
    r"\bmight (?:have|be|need)\b",
    r"\bcould (?:have|be|get)\b",
    r"\bmaybe (?:i|you)\b",
    r"\bperhaps (?:i|you)\b",
    r"\bi think i (?:have|might|am|may)\b",
    r"\bi suspect (?:i|you)\b",
    r"\bi (?:fear|worry) (?:that )?i (?:have|might|may)\b",
    r"\bi wonder if (?:i|you)\b",
    r"\bwhat if (?:i|you|she|he)\b",
    r"\bdo i have\b",
    r"\bam i (?:depressed|anxious|going)\b",
    r"\bwould (?:probably )?be\b",
    r"\bseems like i (?:have|might)\b",
]

THIRD_PARTY_PATTERNS: list[str] = [
    r"\bmy (?:husband|wife|partner|boyfriend|girlfriend|spouse|fiancee?|"
    r"son|daughter|kid|kids|child|children|mom|mum|mother|dad|father|"
    r"sister|brother|sibling|friend|colleague|coworker|parent|parents|"
    r"family member|aunt|uncle|cousin|grandma|grandpa|grandmother|grandfather|"
    r"roommate|neighbor|ex)\b",
    r"\b(?:she|he|they) (?:was|has been|is|got) diagnosed\b",
    r"\b(?:someone i know|a friend|a relative) (?:has|with|suffers)\b",
]

DENIAL_PATTERNS: list[str] = [
    r"\bdoctors? said i don't have\b",
    r"\bturns out i don't have\b",
    r"\bdoesn't actually have\b",
    r"\bnot a real diagnosis\b",
    r"\bjust kidding\b",
    r"\blol\b.{0,30}\b(?:depressed|anxious)\b",  # rhetorical
]

ALL_FILTERS: dict[str, list[str]] = {
    "negation": NEGATION_PATTERNS,
    "hypothetical": HYPOTHETICAL_PATTERNS,
    "third_party": THIRD_PARTY_PATTERNS,
    "denial": DENIAL_PATTERNS,
}

# Pre-compile for speed
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    target: [re.compile(p, re.IGNORECASE) for p in pats]
    for target, pats in PATTERNS_BY_TARGET.items()
}
_COMPILED_FILTERS: dict[str, list[re.Pattern]] = {
    name: [re.compile(p, re.IGNORECASE) for p in pats]
    for name, pats in ALL_FILTERS.items()
}


# --------------------------------------------------------------------------- #
# Core detection
# --------------------------------------------------------------------------- #


@dataclass
class DisclosureResult:
    """One disclosure decision for one (text, target)."""

    is_disclosure: bool
    matched_span: str | None
    matched_pattern: str | None
    filter_triggered: str | None  # 'negation' | 'hypothetical' | 'third_party' | 'denial' | None


def _has_filter_nearby(text: str, match_start: int, match_end: int) -> str | None:
    """Return the name of the first filter triggered in the window, or None."""
    window_start = max(0, match_start - NEG_WINDOW_BEFORE)
    window_end = min(len(text), match_end + NEG_WINDOW_AFTER)
    window = text[window_start:window_end]
    for name, patterns in _COMPILED_FILTERS.items():
        for p in patterns:
            if p.search(window):
                return name
    return None


def detect_disclosure(text: str, target: str) -> DisclosureResult:
    """Detect a verified self-disclosure of `target` in `text`.

    Algorithm:
      1. Find candidate matches from positive patterns for the target.
      2. For each candidate, check ±window for negation/hypothetical/etc.
      3. Return the first candidate that survives all filters.
      4. If none survive, return is_disclosure=False with the last filter triggered.
    """
    if not text or target not in _COMPILED_PATTERNS:
        return DisclosureResult(False, None, None, None)
    patterns = _COMPILED_PATTERNS[target]
    if not patterns:
        return DisclosureResult(False, None, None, None)

    last_filter: str | None = None
    for p in patterns:
        for m in p.finditer(text):
            triggered = _has_filter_nearby(text, m.start(), m.end())
            if triggered is None:
                return DisclosureResult(
                    is_disclosure=True,
                    matched_span=m.group(0),
                    matched_pattern=p.pattern,
                    filter_triggered=None,
                )
            last_filter = triggered
    return DisclosureResult(False, None, None, last_filter)


def apply_disclosure_labels(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    targets: Iterable[str] = ("anxiety", "health_anxiety", "depression", "suicidality"),
    show_progress: bool = True,
) -> pd.DataFrame:
    """Add `disclosure_<target>` and `disclosure_<target>_match` columns.

    Returns a copy of `df`. Idempotent: re-running overwrites the columns.
    """
    targets = list(targets)
    out = df.copy()
    texts = out[text_col].fillna("").astype(str).tolist()

    if show_progress:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )
        with progress:
            task = progress.add_task("Self-disclosure scan", total=len(texts) * len(targets))
            results: dict[str, list[DisclosureResult]] = {t: [] for t in targets}
            for text in texts:
                for target in targets:
                    results[target].append(detect_disclosure(text, target))
                    progress.advance(task)
    else:
        results = {target: [detect_disclosure(t, target) for t in texts] for target in targets}

    for target in targets:
        out[f"disclosure_{target}"] = [int(r.is_disclosure) for r in results[target]]
        out[f"disclosure_{target}_match"] = [r.matched_span for r in results[target]]

    log.info(
        "self_disclosure.done",
        n=len(out),
        positives={t: int(out[f"disclosure_{t}"].sum()) for t in targets},
    )
    return out


def disclosure_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize disclosure counts per (target × subreddit) for inspection."""
    targets = [c.replace("disclosure_", "") for c in df.columns
               if c.startswith("disclosure_") and not c.endswith("_match")]
    if not targets:
        return pd.DataFrame()
    rows = []
    for target in targets:
        for sub, grp in df.groupby("subreddit"):
            rows.append({
                "subreddit": sub,
                "target": target,
                "n_disclosures": int(grp[f"disclosure_{target}"].sum()),
                "rate": float(grp[f"disclosure_{target}"].mean()),
                "n_posts": len(grp),
            })
    return pd.DataFrame(rows).sort_values(["target", "n_disclosures"], ascending=[True, False])
