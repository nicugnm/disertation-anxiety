from src.features.linguistic import extract_one


def test_pronoun_features_first_person_dominant():
    feats = extract_one("I am worried about myself. I keep thinking about my health.")
    assert feats["f_first_sing_rate"] > feats["f_third_rate"]


def test_health_anxiety_features_present():
    feats = extract_one("I felt a twinge in my chest. Is this a heart attack? Should I go to the ER?")
    assert feats["f_health_anx_term_rate"] > 0
    assert feats["f_health_anx_phrase_count"] >= 1


def test_features_stable_keys():
    a = set(extract_one("hello").keys())
    b = set(extract_one("worried about my heart").keys())
    assert a == b
