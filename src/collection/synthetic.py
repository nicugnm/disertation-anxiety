"""Synthetic post generator for end-to-end testing without Reddit credentials.

Generates posts whose linguistic profile *roughly* matches each subreddit's
expected priors. Output is realistic enough to exercise the full pipeline
(preprocessing, labeling, training, evaluation) but is **not** suitable for
any actual research conclusion — it's a smoke-test fixture only.
"""
from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Iterator

from src.collection.base import BaseCollector, RedditPost

# --------------------------------------------------------------------------- #
# Phrase banks. Real research uses real data; these exist purely so `make smoke`
# can exercise every code path without needing API keys.
# --------------------------------------------------------------------------- #

ANXIETY_PHRASES = [
    "I can't stop worrying about everything.",
    "My heart is racing and I feel like I can't breathe.",
    "I had another panic attack last night.",
    "The intrusive thoughts won't go away.",
    "I'm constantly on edge, feeling like something bad will happen.",
    "I keep replaying every conversation in my head.",
    "Tomorrow's meeting is going to destroy me.",
    "Why does my chest feel tight all the time?",
]

HEALTH_ANXIETY_PHRASES = [
    "I felt a weird twinge in my chest, am I having a heart attack?",
    "I've been googling my symptoms for hours and now I'm convinced it's cancer.",
    "Please tell me this lump is nothing serious.",
    "Three doctors said I'm fine but I can't accept it.",
    "Every new sensation in my body sends me spiraling.",
    "I can't stop checking my pulse.",
    "I'm terrified that this headache means a brain tumor.",
    "I keep monitoring symptoms even though my labs were normal.",
]

DEPRESSION_PHRASES = [
    "Nothing brings me joy anymore.",
    "I've been in bed for three days.",
    "I feel hollow and empty.",
    "I'm so tired all the time, even after sleeping.",
    "Everything feels pointless.",
    "I don't see the point of getting up tomorrow.",
]

SUICIDALITY_PHRASES = [
    "I don't want to be here anymore.",
    "I've been thinking about ending things.",
    "It would be easier for everyone if I were gone.",
]

NEUTRAL_PHRASES = [
    "Just moved to a new city last week.",
    "Has anyone tried this new recipe?",
    "My partner and I are figuring out our finances.",
    "I started a new hobby and it's actually fun.",
    "Looking for advice on a tough situation with my friend.",
    "Today was a regular day, nothing special to report.",
    "Trying to organize my apartment finally.",
]

COVID_PHRASES = [
    "Day 5 of COVID, lingering cough but improving.",
    "Just tested positive, feeling exhausted.",
    "Long COVID symptoms are wearing me down.",
    "Worried about my parents catching this.",
]


def _seeded_choice(rng: random.Random, items: list[str]) -> str:
    return rng.choice(items)


def _generate_body(rng: random.Random, profile: dict[str, float]) -> str:
    """Mix phrase-bank sentences according to the subreddit's expected profile."""
    sentences: list[str] = []
    p_anx = profile.get("anxiety", 0.0)
    p_ha = profile.get("health_anxiety", 0.0)
    p_dep = profile.get("depression", 0.0)
    p_suic = profile.get("suicidality", 0.0)
    p_covid = profile.get("covid", 0.0)

    n_sents = rng.randint(3, 8)
    for _ in range(n_sents):
        r = rng.random()
        if r < p_ha:
            sentences.append(_seeded_choice(rng, HEALTH_ANXIETY_PHRASES))
        elif r < p_ha + p_anx:
            sentences.append(_seeded_choice(rng, ANXIETY_PHRASES))
        elif r < p_ha + p_anx + p_dep:
            sentences.append(_seeded_choice(rng, DEPRESSION_PHRASES))
        elif r < p_ha + p_anx + p_dep + p_suic:
            sentences.append(_seeded_choice(rng, SUICIDALITY_PHRASES))
        elif r < p_ha + p_anx + p_dep + p_suic + p_covid:
            sentences.append(_seeded_choice(rng, COVID_PHRASES))
        else:
            sentences.append(_seeded_choice(rng, NEUTRAL_PHRASES))

    return " ".join(sentences)


def _profile_for(subreddit_entry) -> dict[str, float]:  # noqa: ANN001
    """Build the per-sentence probability profile from a SubredditEntry."""
    base = {
        "anxiety": subreddit_entry.expected_anxiety_prior * 0.6,
        "health_anxiety": subreddit_entry.expected_health_anxiety_prior * 0.6,
        "depression": subreddit_entry.expected_depression_prior * 0.6,
        "suicidality": subreddit_entry.expected_suicidality_prior * 0.5,
    }
    if "covid" in subreddit_entry.name.lower():
        base["covid"] = 0.4
    return base


class SyntheticCollector(BaseCollector):
    """Reproducible synthetic data — same seed yields same posts."""

    def __init__(self, config, n_per_subreddit: int = 200, seed: int = 42) -> None:  # noqa: ANN001
        super().__init__(config)
        self.n_per_subreddit = n_per_subreddit
        self.seed = seed

    def collect_subreddit(self, name: str) -> Iterator[RedditPost]:
        entry = self.config.by_name(name)
        if entry is None:
            return
        # Per-subreddit deterministic RNG so adding a new subreddit later
        # doesn't change posts in others.
        sub_seed = int(hashlib.sha256(f"{self.seed}-{name}".encode()).hexdigest()[:8], 16)
        rng = random.Random(sub_seed)
        profile = _profile_for(entry)

        for i in range(self.n_per_subreddit):
            body = _generate_body(rng, profile)
            title = body.split(".")[0][:80]
            post = RedditPost(
                id=f"syn_{name}_{i:05d}",
                subreddit=name,
                created_utc=time.time() - rng.randint(0, 60 * 60 * 24 * 365 * 3),
                title=title,
                body=body,
                author=f"user_{rng.randint(1000, 9999)}",
                score=rng.randint(0, 1000),
                num_comments=rng.randint(0, 200),
                permalink=f"/r/{name}/comments/syn_{i:05d}/",
                is_self=True,
                over_18=False,
                source="synthetic",
                collected_at=time.time(),
            )
            if self.passes_filters(post):
                yield post
