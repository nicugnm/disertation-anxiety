"""Lexicons used by tier-1 weak labeling and as features for the XGBoost model.

These are deliberately small, transparent word lists derived from clinical
instruments (GAD-7, SHAI/HAI, PHQ-9, Columbia C-SSRS) and the social-media
mental-health literature (e.g. De Choudhury et al., Coppersmith et al.,
Yates et al.). They are NOT a clinical instrument — they are a weak signal.
For the dissertation, cite each list's provenance in the methodology chapter.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# General anxiety markers (drawn from GAD-7 stems + Reddit-anxiety NLP work)
# --------------------------------------------------------------------------- #

ANXIETY_TERMS: set[str] = {
    "anxiety",
    "anxious",
    "anxiously",
    "panic",
    "panicking",
    "panicked",
    "worry",
    "worrying",
    "worried",
    "worries",
    "nervous",
    "nervousness",
    "scared",
    "afraid",
    "fear",
    "fearful",
    "stress",
    "stressed",
    "stressful",
    "overwhelmed",
    "overwhelming",
    "edge",  # 'on edge'
    "restless",
    "racing thoughts",
    "racing heart",
    "intrusive thoughts",
    "tense",
    "tension",
    "dread",
    "dreading",
    "agoraphobia",
    "agoraphobic",
    "hyperventilate",
    "hyperventilating",
    "shaking",
    "trembling",
    "trembled",
    "freaking out",
    "freak out",
    "spiral",
    "spiraling",
    "catastrophize",
    "catastrophizing",
    "ruminating",
    "rumination",
}

ANXIETY_PHRASES: set[str] = {
    "panic attack",
    "anxiety attack",
    "on edge",
    "racing thoughts",
    "can't breathe",
    "cant breathe",
    "heart racing",
    "chest tight",
    "tight chest",
    "what if",
    "i'm worried",
    "im worried",
    "i can't stop",
    "cant stop worrying",
}

# --------------------------------------------------------------------------- #
# Health anxiety markers (drawn from SHAI/HAI items + somatic-symptom literature)
# --------------------------------------------------------------------------- #

HEALTH_ANXIETY_TERMS: set[str] = {
    "hypochondria",
    "hypochondriac",
    "health anxiety",
    "illness anxiety",
    "googled my symptoms",
    "googling symptoms",
    "googled symptoms",
    "google symptoms",
    "webmd",
    "dr google",
    "checking pulse",
    "took my pulse",
    "felt my pulse",
    "checking heart rate",
    "self-diagnose",
    "self-diagnosing",
    "self diagnosed",
    "convinced i have",
    "convinced i'm dying",
    "convinced im dying",
    "doctor said i'm fine",
    "doctors say i'm fine",
    "tests came back normal",
    "labs were normal",
    "ekg was normal",
    "mri was clear",
    "ct scan",
    "ct scan was",
    "blood work was",
    "blood test was",
    "lump",
    "tumor",
    "tumour",
    "cancer",
    "stroke",
    "heart attack",
    "aneurysm",
    "blood clot",
    "embolism",
    "pulmonary embolism",
    "als",
    "ms",
    "multiple sclerosis",
    "lupus",
    "leukemia",
    "lymphoma",
    "hiv",
    "aids",
    "rabies",
    "meningitis",
    "sepsis",
    "appendicitis",
    "twinge",
    "tingling",
    "numbness",
    "lightheaded",
    "lightheadedness",
    "dizzy",
    "dizziness",
    "shortness of breath",
    "chest pain",
    "palpitations",
    "tremor",
    "twitch",
    "twitching",
    "muscle twitch",
    "rash",
    "lesion",
    "mole",
    "headache",
    "migraine",
}

HEALTH_ANXIETY_PHRASES: set[str] = {
    "am i dying",
    "is this cancer",
    "is this a tumor",
    "is this a heart attack",
    "do i have",
    "could this be",
    "is this serious",
    "should i go to the er",
    "should i go to the hospital",
    "monitoring my symptoms",
    "checking my symptoms",
    "scared it's",
    "terrified i have",
    "spent hours googling",
    "won't accept",
    "can't accept the test results",
    "doctors keep telling me",
    "second opinion",
    "third opinion",
}

# --------------------------------------------------------------------------- #
# Reassurance-seeking patterns (specific to health anxiety)
# --------------------------------------------------------------------------- #

REASSURANCE_PATTERNS: set[str] = {
    "please tell me",
    "please reassure me",
    "tell me i'm okay",
    "tell me im okay",
    "is this normal",
    "anyone else",
    "has anyone had",
    "did anyone",
    "should i be worried",
    "am i overreacting",
}

# --------------------------------------------------------------------------- #
# Depression markers (PHQ-9 stems)
# --------------------------------------------------------------------------- #

DEPRESSION_TERMS: set[str] = {
    "depressed",
    "depression",
    "depressing",
    "hopeless",
    "hopelessness",
    "worthless",
    "worthlessness",
    "empty",
    "numb",
    "numbness",
    "anhedonia",
    "no pleasure",
    "lost interest",
    "exhausted",
    "fatigue",
    "tired all the time",
    "no energy",
    "can't get out of bed",
    "cant get out of bed",
    "slept all day",
    "crying",
    "crying spells",
    "no point",
    "pointless",
    "meaningless",
    "useless",
    "lonely",
    "loneliness",
    "isolated",
    "isolation",
}

# --------------------------------------------------------------------------- #
# Suicidality markers (Columbia C-SSRS stems — handle with extreme care)
# --------------------------------------------------------------------------- #

SUICIDALITY_TERMS: set[str] = {
    "suicide",
    "suicidal",
    "kill myself",
    "killing myself",
    "end my life",
    "ending my life",
    "end it all",
    "want to die",
    "wish i was dead",
    "wish i were dead",
    "better off dead",
    "no reason to live",
    "no point living",
    "self harm",
    "self-harm",
    "selfharm",
    "cutting myself",
    "overdose",
    "od'd",
    "took pills",
}

# --------------------------------------------------------------------------- #
# Pronouns (first-person preponderance is a robust depression/anxiety marker
# across the literature — Pennebaker et al., Eichstaedt et al.)
# --------------------------------------------------------------------------- #

FIRST_PERSON_SINGULAR: set[str] = {"i", "me", "my", "mine", "myself", "i'm", "im", "i've", "ive"}
FIRST_PERSON_PLURAL: set[str] = {"we", "us", "our", "ours", "ourselves"}
SECOND_PERSON: set[str] = {"you", "your", "yours", "yourself", "yourselves"}
THIRD_PERSON: set[str] = {
    "he",
    "she",
    "they",
    "them",
    "their",
    "theirs",
    "him",
    "her",
    "his",
    "hers",
}

# --------------------------------------------------------------------------- #
# Certainty / uncertainty markers (Pennebaker LIWC-style)
# --------------------------------------------------------------------------- #

UNCERTAINTY_TERMS: set[str] = {
    "maybe",
    "perhaps",
    "might",
    "could",
    "possibly",
    "probably",
    "i think",
    "i guess",
    "i suppose",
    "not sure",
    "unsure",
    "what if",
    "wonder",
    "wondering",
}

CERTAINTY_TERMS: set[str] = {
    "definitely",
    "certainly",
    "absolutely",
    "always",
    "never",
    "completely",
    "totally",
    "exactly",
    "i'm sure",
    "im sure",
    "i know",
    "obviously",
}

# --------------------------------------------------------------------------- #
# Somatic / body parts (signal for health anxiety vs. general anxiety)
# --------------------------------------------------------------------------- #

BODY_PARTS: set[str] = {
    "heart",
    "chest",
    "lung",
    "lungs",
    "stomach",
    "abdomen",
    "head",
    "brain",
    "neck",
    "throat",
    "skin",
    "arm",
    "arms",
    "leg",
    "legs",
    "back",
    "spine",
    "kidney",
    "liver",
    "thyroid",
    "lymph",
    "node",
    "nodes",
    "breast",
    "testicle",
    "ovary",
    "uterus",
    "blood",
    "vein",
    "artery",
    "muscle",
    "muscles",
    "joint",
    "joints",
    "nerve",
    "nerves",
}


def all_lexicons() -> dict[str, set[str]]:
    return {
        "anxiety_terms": ANXIETY_TERMS,
        "anxiety_phrases": ANXIETY_PHRASES,
        "health_anxiety_terms": HEALTH_ANXIETY_TERMS,
        "health_anxiety_phrases": HEALTH_ANXIETY_PHRASES,
        "reassurance": REASSURANCE_PATTERNS,
        "depression_terms": DEPRESSION_TERMS,
        "suicidality_terms": SUICIDALITY_TERMS,
        "first_person_singular": FIRST_PERSON_SINGULAR,
        "first_person_plural": FIRST_PERSON_PLURAL,
        "second_person": SECOND_PERSON,
        "third_person": THIRD_PERSON,
        "uncertainty": UNCERTAINTY_TERMS,
        "certainty": CERTAINTY_TERMS,
        "body_parts": BODY_PARTS,
    }
