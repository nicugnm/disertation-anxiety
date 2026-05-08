from src.labeling.weak import lexicon_scores


def test_anxiety_signal_present():
    s = lexicon_scores("I had a panic attack and my heart was racing.")
    assert s["anxiety"] > 0


def test_health_anxiety_signal():
    s = lexicon_scores(
        "I keep googling my symptoms and I'm convinced this lump is cancer. "
        "Three doctors said I'm fine but I can't accept it."
    )
    assert s["health_anxiety"] > 0
    # Should be at least competitive with general anxiety signal
    assert s["health_anxiety"] >= s["anxiety"] - 0.05


def test_neutral_text_low_signal():
    s = lexicon_scores("I just moved to a new apartment and need furniture recommendations.")
    assert s["anxiety"] < 0.1
    assert s["health_anxiety"] < 0.1
    assert s["depression"] < 0.1
