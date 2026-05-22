"""Unit tests for the self-disclosure labeler.

Covers: positive matches per target, negation, hypothetical, third-party,
denial, and edge cases (empty, non-English-ish, ambiguous).
"""
from __future__ import annotations

from src.labeling.self_disclosure import detect_disclosure


# --------------------------------------------------------------------------- #
# Positive cases — should be flagged as disclosure
# --------------------------------------------------------------------------- #


def test_anxiety_diagnosis_explicit():
    r = detect_disclosure("I was diagnosed with generalized anxiety disorder last year.", "anxiety")
    assert r.is_disclosure
    assert "diagnosed" in (r.matched_span or "").lower()


def test_anxiety_gad_shorthand():
    r = detect_disclosure("I have GAD and it's been rough.", "anxiety")
    assert r.is_disclosure


def test_anxiety_clinician_phrasing():
    r = detect_disclosure(
        "My psychiatrist diagnosed me with anxiety disorder when I was 22.", "anxiety"
    )
    assert r.is_disclosure


def test_health_anxiety_explicit():
    r = detect_disclosure("I have health anxiety and it controls my life.", "health_anxiety")
    assert r.is_disclosure


def test_health_anxiety_hypochondriac():
    r = detect_disclosure("I'm a hypochondriac and I know it.", "health_anxiety")
    assert r.is_disclosure


def test_health_anxiety_iad_code():
    r = detect_disclosure("I was diagnosed with illness anxiety disorder (IAD).", "health_anxiety")
    assert r.is_disclosure


def test_depression_explicit():
    r = detect_disclosure("I have been diagnosed with major depressive disorder.", "depression")
    assert r.is_disclosure


def test_depression_mdd_shorthand():
    r = detect_disclosure("I have MDD and bipolar, both diagnosed.", "depression")
    assert r.is_disclosure


def test_depression_meds_for():
    r = detect_disclosure(
        "I'm on antidepressants for my depression but they're not working great.", "depression"
    )
    assert r.is_disclosure


# --------------------------------------------------------------------------- #
# Negation — should NOT be flagged
# --------------------------------------------------------------------------- #


def test_negation_not_diagnosed():
    # "I have not been diagnosed" — the first-person verb pattern can't match
    # because "not" breaks the verb phrase. Either the pattern rejects outright
    # (no candidate found) or the negation filter catches it. Both are safe.
    r = detect_disclosure("I have not been diagnosed with depression but I worry.", "depression")
    assert not r.is_disclosure


def test_negation_never():
    r = detect_disclosure("I've never been diagnosed with anxiety formally.", "anxiety")
    assert not r.is_disclosure


def test_negation_dont_have():
    r = detect_disclosure("I don't have health anxiety, just normal worry.", "health_anxiety")
    assert not r.is_disclosure


# --------------------------------------------------------------------------- #
# Hypothetical — should NOT be flagged
# --------------------------------------------------------------------------- #


def test_hypothetical_i_think():
    r = detect_disclosure("I think I have depression but I haven't seen anyone.", "depression")
    assert not r.is_disclosure
    assert r.filter_triggered == "hypothetical"


def test_hypothetical_if_i_were():
    r = detect_disclosure(
        "If I were diagnosed with anxiety I'd be relieved to know.", "anxiety"
    )
    assert not r.is_disclosure


def test_hypothetical_what_if():
    r = detect_disclosure("What if I have health anxiety and don't know it?", "health_anxiety")
    assert not r.is_disclosure


def test_hypothetical_might_have():
    r = detect_disclosure("I might have depression based on what I read.", "depression")
    assert not r.is_disclosure


# --------------------------------------------------------------------------- #
# Third-party — should NOT be flagged
# --------------------------------------------------------------------------- #


def test_third_party_partner():
    # First-person patterns don't match third-person subjects at all; this is the
    # safer outcome (no candidate, no filter needed).
    r = detect_disclosure(
        "My partner was diagnosed with depression and I'm trying to support them.", "depression"
    )
    assert not r.is_disclosure


def test_third_party_partner_mixed_with_self_claim_is_conservative():
    # Documented behavior: when a self claim appears within ±50 chars of a
    # third-party reference, the conservative filter rejects it. This is by
    # design — false negatives are cheaper than false positives for a label
    # source intended to give clinical-grade ground truth.
    r = detect_disclosure(
        "My partner has depression. I have depression too apparently.", "depression"
    )
    assert not r.is_disclosure
    assert r.filter_triggered == "third_party"


def test_third_party_friend():
    r = detect_disclosure("My friend has anxiety and panics often.", "anxiety")
    # "I have anxiety" not present, "my friend has" is the only signal. But the
    # patterns require first-person framing; this won't match anyway, but verify.
    assert not r.is_disclosure


def test_third_party_mom():
    r = detect_disclosure(
        "My mom has health anxiety so I know what it looks like; my dad is a hypochondriac too.",
        "health_anxiety",
    )
    # The "my mom has" and "my dad is" pattern keep the surrounding hypochondriac match third-party.
    assert not r.is_disclosure


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_empty_text():
    assert not detect_disclosure("", "anxiety").is_disclosure
    assert not detect_disclosure(None, "anxiety").is_disclosure  # type: ignore[arg-type]


def test_no_signal_text():
    r = detect_disclosure("Just moved to a new apartment, looking for furniture recs.", "anxiety")
    assert not r.is_disclosure


def test_unknown_target():
    r = detect_disclosure("I have GAD", "made_up_target")
    assert not r.is_disclosure


def test_suicidality_disabled():
    r = detect_disclosure("I was diagnosed with major depression and attempted last year.",
                          "suicidality")
    # We disable disclosure for suicidality (no patterns).
    assert not r.is_disclosure


# --------------------------------------------------------------------------- #
# Apply across DataFrame
# --------------------------------------------------------------------------- #


def test_apply_disclosure_labels_smoke():
    import pandas as pd

    from src.labeling.self_disclosure import apply_disclosure_labels

    df = pd.DataFrame({
        "clean_text": [
            "I was diagnosed with generalized anxiety disorder.",
            "My mom has depression.",
            "Just buying groceries today.",
        ],
        "subreddit": ["Anxiety", "depression", "LivingAlone"],
    })
    out = apply_disclosure_labels(df, show_progress=False)
    assert int(out["disclosure_anxiety"].iloc[0]) == 1
    assert int(out["disclosure_depression"].iloc[1]) == 0  # third-party
    assert int(out["disclosure_anxiety"].iloc[2]) == 0
    assert "disclosure_anxiety_match" in out.columns
