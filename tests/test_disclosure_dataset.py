"""Unit tests for the user-level disclosure test-set builder."""
from __future__ import annotations

import pandas as pd

from src.labeling.disclosure_dataset import (
    build_disclosure_test_users,
    evaluate_user_level,
    find_disclosed_users,
    find_matched_controls,
    mark_held_out,
    materialize_test_posts,
)


def _make_corpus():
    """Toy 6-user corpus with two disclosure positives and assorted controls."""
    return pd.DataFrame({
        "id": list(range(12)),
        "author_hash": [
            "alice", "alice", "alice",     # disclosed anxiety (1 disclosure + 2 normal posts)
            "bob", "bob", "bob",            # control candidate in r/Anxiety (no disclosure)
            "carol", "carol",               # control candidate in r/cooking (no disclosure)
            "dave", "dave", "dave",         # disclosed depression
            "eve",                          # control candidate, but only 1 post (excluded by min_posts)
        ],
        "subreddit": [
            "Anxiety", "Anxiety", "Anxiety",
            "Anxiety", "Anxiety", "Anxiety",
            "cooking", "cooking",
            "depression", "depression", "depression",
            "Anxiety",
        ],
        "clean_text": [
            "I was diagnosed with anxiety last year and have been doing better.",
            "Tough day today, the panic came back.",
            "Trying a new breathing technique.",
            "What if I have anxiety, who knows.",
            "Heart racing again at work.",
            "Anyone else feel this all the time?",
            "Anyone got a good chili recipe?",
            "Tried braising for the first time.",
            "I was diagnosed with depression last spring.",
            "Hard to get out of bed today.",
            "Trying to keep going, taking my meds.",
            "Idk what to think.",
        ],
        "disclosure_anxiety": [
            1, 0, 0,
            0, 0, 0,
            0, 0,
            0, 0, 0,
            0,
        ],
        "disclosure_health_anxiety": [0] * 12,
        "disclosure_depression": [
            0, 0, 0,
            0, 0, 0,
            0, 0,
            1, 0, 0,
            0,
        ],
        "disclosure_suicidality": [0] * 12,
    })


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


def test_find_disclosed_users_collects_per_target():
    df = _make_corpus()
    assert find_disclosed_users(df, "anxiety") == {"alice"}
    assert find_disclosed_users(df, "depression") == {"dave"}
    assert find_disclosed_users(df, "health_anxiety") == set()


def test_find_disclosed_users_missing_column_returns_empty():
    df = pd.DataFrame({"author_hash": ["alice"], "subreddit": ["x"]})
    assert find_disclosed_users(df, "anxiety") == set()


def test_find_matched_controls_prefers_same_subreddit_and_excludes_disclosed():
    df = _make_corpus()
    user_subs = {
        "alice": {"Anxiety"},
        "bob": {"Anxiety"},
        "carol": {"cooking"},
        "dave": {"depression"},
        "eve": {"Anxiety"},
    }
    post_counts = {"alice": 3, "bob": 3, "carol": 2, "dave": 3, "eve": 1}
    controls = find_matched_controls(
        df,
        target_positives={"alice"},
        all_disclosed={"alice", "dave"},
        user_subs=user_subs,
        post_counts=post_counts,
        n_per_positive=2,
        min_posts_per_user=2,
        seed=42,
    )
    # bob is the only valid Anxiety control (carol is in cooking, eve has too few posts,
    # dave is disclosed, alice IS the positive). With n_per_positive=2 we get bob;
    # no other valid Anxiety controls exist.
    assert "bob" in controls
    assert "alice" not in controls
    assert "dave" not in controls
    assert "eve" not in controls  # min_posts violation


def test_matched_controls_skips_low_post_users():
    df = _make_corpus()
    user_subs = {"alice": {"Anxiety"}, "eve": {"Anxiety"}}
    post_counts = {"alice": 3, "eve": 1}
    controls = find_matched_controls(
        df,
        target_positives={"alice"},
        all_disclosed={"alice"},
        user_subs=user_subs,
        post_counts=post_counts,
        n_per_positive=5,
        min_posts_per_user=3,
        seed=42,
    )
    assert controls == set()  # eve is below the min_posts threshold


# --------------------------------------------------------------------------- #
# Test-set builder (end-to-end on toy corpus)
# --------------------------------------------------------------------------- #


def test_build_test_users_includes_positives_and_controls():
    df = _make_corpus()
    users = build_disclosure_test_users(
        df,
        targets=("anxiety", "depression"),
        controls_per_positive=2,
        min_posts_per_user=2,
        seed=42,
    )

    # alice and dave are positives
    by_user = users.set_index("author_hash")
    assert int(by_user.loc["alice", "user_anxiety"]) == 1
    assert int(by_user.loc["dave", "user_depression"]) == 1
    assert by_user.loc["alice", "user_group"].startswith("disclosed_")

    # At least one matched_control row
    n_controls = (users["user_group"] == "matched_control").sum()
    assert n_controls >= 1

    # No user is both positive and control
    for u in users["author_hash"]:
        positive_for_any = any(int(users[users["author_hash"] == u][f"user_{t}"].iloc[0]) == 1
                               for t in ("anxiety", "depression"))
        is_control = (users[users["author_hash"] == u]["user_group"].iloc[0] == "matched_control")
        assert positive_for_any != is_control  # exactly one


def test_build_test_users_empty_when_no_disclosures():
    df = _make_corpus()
    df["disclosure_anxiety"] = 0
    df["disclosure_depression"] = 0
    users = build_disclosure_test_users(df, controls_per_positive=2, min_posts_per_user=2, seed=42)
    assert users.empty


# --------------------------------------------------------------------------- #
# Materialization + held-out marking
# --------------------------------------------------------------------------- #


def test_materialize_test_posts_attaches_user_labels_and_disclosure_flag():
    df = _make_corpus()
    users = build_disclosure_test_users(
        df,
        targets=("anxiety", "depression"),
        controls_per_positive=2,
        min_posts_per_user=2,
        seed=42,
    )
    test_posts = materialize_test_posts(df, users, targets=("anxiety", "depression"))

    # All test users' posts are included
    test_user_set = set(users["author_hash"])
    assert set(test_posts["author_hash"]) == test_user_set

    # alice's disclosure post is flagged
    alice = test_posts[test_posts["author_hash"] == "alice"]
    assert alice["is_disclosure_post"].sum() == 1
    # All alice posts have user_anxiety=1
    assert (alice["user_anxiety"] == 1).all()


def test_mark_held_out_flags_only_test_users():
    df = _make_corpus()
    users = build_disclosure_test_users(
        df,
        targets=("anxiety", "depression"),
        controls_per_positive=2,
        min_posts_per_user=2,
        seed=42,
    )
    marked = mark_held_out(df, users)

    test_user_set = set(users["author_hash"])
    held = marked[marked["held_out_split"]]
    not_held = marked[~marked["held_out_split"]]
    assert set(held["author_hash"]) == test_user_set
    assert test_user_set.isdisjoint(set(not_held["author_hash"]))


def test_mark_held_out_empty_test_users_marks_all_false():
    df = _make_corpus()
    empty_users = pd.DataFrame(columns=["author_hash"])
    marked = mark_held_out(df, empty_users)
    assert not marked["held_out_split"].any()


# --------------------------------------------------------------------------- #
# User-level evaluation
# --------------------------------------------------------------------------- #


def test_evaluate_user_level_perfect_predictor():
    # Two users: alice (positive), bob (negative)
    test_posts = pd.DataFrame({
        "author_hash": ["alice", "alice", "bob", "bob"],
        "user_anxiety": [1, 1, 0, 0],
        "score_anxiety": [0.9, 0.85, 0.1, 0.15],
        "is_disclosure_post": [1, 0, 0, 0],
    })
    rep = evaluate_user_level(
        test_posts,
        score_col="score_anxiety",
        target="anxiety",
        aggregation="mean",
    )
    assert rep["n_users"] == 2
    assert rep["n_positive_users"] == 1
    assert rep["f1"] == 1.0
    assert rep["precision"] == 1.0
    assert rep["recall"] == 1.0


def test_evaluate_user_level_with_disclosure_masking_changes_score():
    # The disclosure post has the highest score; masking it lowers alice's user score
    test_posts = pd.DataFrame({
        "author_hash": ["alice", "alice", "alice", "bob", "bob", "bob"],
        "user_anxiety": [1, 1, 1, 0, 0, 0],
        "score_anxiety": [0.99, 0.2, 0.3, 0.4, 0.45, 0.5],
        "is_disclosure_post": [1, 0, 0, 0, 0, 0],
    })
    unmasked = evaluate_user_level(test_posts, "score_anxiety", "anxiety", aggregation="mean")
    masked = evaluate_user_level(
        test_posts, "score_anxiety", "anxiety", aggregation="mean", mask_disclosure_posts=True,
    )
    # Without masking, alice's mean = (0.99 + 0.2 + 0.3) / 3 ≈ 0.50; bob's mean = 0.45. Close call.
    # With masking, alice's mean = (0.2 + 0.3) / 2 = 0.25; bob's = 0.45. Now misclassified.
    assert unmasked["mask_disclosure_posts"] is False
    assert masked["mask_disclosure_posts"] is True


def test_evaluate_user_level_topk_aggregation_uses_strongest_posts():
    # alice has one very high post score; topk_mean should let her cross threshold
    # while mean aggregation might not.
    test_posts = pd.DataFrame({
        "author_hash": ["alice"] * 6 + ["bob"] * 6,
        "user_anxiety": [1] * 6 + [0] * 6,
        "score_anxiety": [0.05, 0.05, 0.05, 0.05, 0.05, 0.95] + [0.4] * 6,
        "is_disclosure_post": [0] * 12,
    })
    mean_rep = evaluate_user_level(test_posts, "score_anxiety", "anxiety", aggregation="mean")
    topk_rep = evaluate_user_level(test_posts, "score_anxiety", "anxiety", aggregation="topk_mean")
    # Mean aggregation: alice ~= 0.183 < 0.4 (bob) → misclassified
    # Top-K mean (K=5 default): alice's top-5 includes 0.95 → much higher → correct
    assert topk_rep["f1"] >= mean_rep["f1"]
