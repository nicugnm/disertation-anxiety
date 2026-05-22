"""Build a user-level disclosure test set with subreddit-matched controls.

Methodology (eRisk / Coppersmith / CLPsych standard):

  1. A user is a *positive* for target T if they have ever posted a verified
     self-disclosure for T (output of `src/labeling/self_disclosure.py`).
  2. Negatives ("controls") are NEVER-disclosed users sampled from the SAME
     subreddits as the positives. This stops the classifier from cheating via
     subreddit style.
  3. All posts by test users (positives + controls) are HELD OUT — they never
     appear in training. This breaks circular evaluation.

References:
  - Coppersmith, Dredze, Harman (2014) "Quantifying Mental Health Signals in Twitter"
  - Losada, Crestani (2016) "A Test Collection for Research on Depression and Language Use"
  - Parapar et al. (2025) eRisk 2025

Output schema:
  data/processed/disclosure_testset.parquet — one row per post by a test user.
    All original corpus columns +
      user_<target>           int   user-level label (1 if author ever disclosed)
      user_group              str   'disclosed_<target>+...' or 'matched_control'
      is_disclosure_post      int   1 if this specific post is one of the
                                    disclosure utterances (so an evaluator
                                    can mask them and report both numbers).

  data/processed/labeled.parquet — same file, with a new column:
      held_out_split          bool  True for posts by any test user.

  `anxiety train` automatically filters held_out_split=True from its training
  data, so the noisy-train / clean-test split is enforced at the pipeline level.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_TARGETS = ("anxiety", "health_anxiety", "depression")


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


def find_disclosed_users(df: pd.DataFrame, target: str) -> set[str]:
    """Return the set of `author_hash` values with at least one disclosure for `target`."""
    col = f"disclosure_{target}"
    if col not in df.columns:
        return set()
    mask = df[col].fillna(0).astype(int) == 1
    if "author_hash" not in df.columns:
        return set()
    return set(df.loc[mask, "author_hash"].dropna().astype(str).tolist()) - {""}


def _user_subreddit_index(df: pd.DataFrame) -> dict[str, set[str]]:
    """Map each `author_hash` to the set of subreddits they've posted in."""
    if "author_hash" not in df.columns:
        return {}
    grouped = df.groupby("author_hash")["subreddit"].agg(
        lambda s: set(str(x) for x in s if pd.notna(x))
    )
    return grouped.to_dict()


def find_matched_controls(
    df: pd.DataFrame,
    target_positives: set[str],
    all_disclosed: set[str],
    user_subs: dict[str, set[str]],
    post_counts: dict[str, int],
    n_per_positive: int = 2,
    min_posts_per_user: int = 3,
    seed: int = 42,
) -> set[str]:
    """Sample non-disclosed users from the same subreddits as each positive.

    For each positive `u`, look at the subreddits `u` posts in and sample
    `n_per_positive` users from those subreddits who have *never* disclosed
    any condition (across all targets) and have at least `min_posts_per_user`
    posts (so the model has enough signal per user).

    Sampling is **without replacement across the whole call** — each control
    user is used once even if it could match several positives.
    """
    rng = random.Random(seed)

    # Pool of usable controls: not disclosed, sufficient post count.
    valid = {
        u for u in user_subs
        if u
        and u not in all_disclosed
        and post_counts.get(u, 0) >= min_posts_per_user
    }

    # Reverse index: subreddit -> list of valid control candidates.
    sub_index: dict[str, list[str]] = {}
    for u in valid:
        for sub in user_subs[u]:
            sub_index.setdefault(sub, []).append(u)
    # Shuffle each list once so order isn't biased.
    for k in sub_index:
        rng.shuffle(sub_index[k])

    chosen: set[str] = set()
    # Iterate positives in a stable order so reruns with same seed agree.
    for pos_user in sorted(target_positives):
        subs = list(user_subs.get(pos_user, set()))
        if not subs:
            continue
        rng.shuffle(subs)
        n_needed = n_per_positive
        for sub in subs:
            if n_needed <= 0:
                break
            for cand in sub_index.get(sub, []):
                if n_needed <= 0:
                    break
                if cand in chosen:
                    continue
                chosen.add(cand)
                n_needed -= 1
    return chosen


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def build_disclosure_test_users(
    df: pd.DataFrame,
    targets: Iterable[str] = DEFAULT_TARGETS,
    controls_per_positive: int = 2,
    min_posts_per_user: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the user-level test table.

    One row per test user with:
      author_hash, user_<target>... user_group, n_posts, subreddits.
    """
    targets = list(targets)
    log.info("disclosure_testset.start", n_posts=len(df), targets=targets)

    if "author_hash" not in df.columns:
        raise ValueError("Corpus is missing `author_hash` — run preprocess first.")

    user_subs = _user_subreddit_index(df)
    post_counts = df.groupby("author_hash").size().to_dict()

    disclosed_per_target: dict[str, set[str]] = {}
    for t in targets:
        disclosed_per_target[t] = find_disclosed_users(df, t)
        log.info("disclosure_testset.positives", target=t, n=len(disclosed_per_target[t]))

    all_disclosed: set[str] = set().union(*disclosed_per_target.values()) if disclosed_per_target else set()

    matched_controls: set[str] = set()
    for t in targets:
        controls = find_matched_controls(
            df,
            target_positives=disclosed_per_target[t],
            all_disclosed=all_disclosed,
            user_subs=user_subs,
            post_counts=post_counts,
            n_per_positive=controls_per_positive,
            min_posts_per_user=min_posts_per_user,
            seed=seed + sum(ord(c) for c in t) % 1000,
        )
        log.info("disclosure_testset.controls", target=t, n=len(controls))
        matched_controls.update(controls)

    test_users = sorted(all_disclosed | matched_controls)

    rows: list[dict] = []
    for u in test_users:
        if not u:
            continue
        row: dict = {"author_hash": u}
        for t in targets:
            row[f"user_{t}"] = int(u in disclosed_per_target[t])

        if u in all_disclosed:
            disc_targets = [t for t in targets if u in disclosed_per_target[t]]
            row["user_group"] = "disclosed_" + "+".join(disc_targets)
        else:
            row["user_group"] = "matched_control"

        row["n_posts"] = int(post_counts.get(u, 0))
        row["subreddits"] = ",".join(sorted(user_subs.get(u, set())))
        rows.append(row)

    # Always materialize the full expected schema — this prevents downstream
    # `KeyError: user_group` when the corpus happens to have zero positives
    # or every positive gets filtered out (e.g. all under min_posts_per_user).
    expected_cols = (
        ["author_hash"]
        + [f"user_{t}" for t in targets]
        + ["user_group", "n_posts", "subreddits"]
    )
    if not rows:
        out = pd.DataFrame(columns=expected_cols)
    else:
        out = pd.DataFrame(rows)
        # If for some reason a column didn't appear (defensive — shouldn't
        # happen with the loop above), backfill it.
        for c in expected_cols:
            if c not in out.columns:
                out[c] = 0 if c.startswith("user_") and c != "user_group" else None

    if out.empty:
        log.info("disclosure_testset.users_built", n_users=0, n_positives={}, n_controls=0)
    else:
        log.info(
            "disclosure_testset.users_built",
            n_users=len(out),
            n_positives={t: int(out[f"user_{t}"].sum()) for t in targets},
            n_controls=int((out["user_group"] == "matched_control").sum()),
        )
    return out


def materialize_test_posts(
    df: pd.DataFrame,
    test_users: pd.DataFrame,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> pd.DataFrame:
    """Return all posts by test users, with user-level labels + a disclosure flag.

    Each post carries the user-level label (so a post by a disclosed user is
    `user_<target>=1` even when the post itself isn't a disclosure). The
    `is_disclosure_post` column lets the evaluator separately report "F1 with
    disclosure posts included" vs "F1 with disclosure posts masked out".
    """
    targets = list(targets)
    if test_users.empty:
        return df.head(0).assign(
            **{f"user_{t}": 0 for t in targets},
            user_group=pd.Series(dtype=object),
            is_disclosure_post=pd.Series(dtype=int),
        )
    test_user_set = set(test_users["author_hash"].astype(str))
    posts = df[df["author_hash"].astype(str).isin(test_user_set)].copy()

    user_cols = ["author_hash"] + [f"user_{t}" for t in targets] + ["user_group"]
    posts = posts.merge(test_users[user_cols], on="author_hash", how="left")

    # Mark posts that are themselves disclosures (so evaluators can mask them).
    is_disc = pd.Series(False, index=posts.index)
    for t in targets:
        col = f"disclosure_{t}"
        if col in posts.columns:
            is_disc = is_disc | (posts[col].fillna(0).astype(int) == 1)
    posts["is_disclosure_post"] = is_disc.astype(int)

    log.info(
        "disclosure_testset.posts",
        n_posts=len(posts),
        n_users=posts["author_hash"].nunique(),
        n_disclosure_posts=int(posts["is_disclosure_post"].sum()),
    )
    return posts.reset_index(drop=True)


def mark_held_out(corpus_df: pd.DataFrame, test_users: pd.DataFrame) -> pd.DataFrame:
    """Add `held_out_split` column to corpus. True iff the post is by a test user."""
    out = corpus_df.copy()
    if test_users.empty:
        out["held_out_split"] = False
        return out
    test_user_set = set(test_users["author_hash"].astype(str))
    out["held_out_split"] = out["author_hash"].astype(str).isin(test_user_set)
    log.info(
        "disclosure_testset.corpus_marked",
        n_posts=len(out),
        n_held_out=int(out["held_out_split"].sum()),
    )
    return out


# --------------------------------------------------------------------------- #
# User-level evaluation utility
# --------------------------------------------------------------------------- #


def evaluate_user_level(
    test_posts: pd.DataFrame,
    score_col: str,
    target: str,
    aggregation: str = "mean",
    mask_disclosure_posts: bool = False,
    threshold: float | None = None,
) -> dict:
    """Compute user-level metrics from per-post scores.

    Two evaluation modes you can run side-by-side (and should, for the thesis):
      - mask_disclosure_posts=False : evaluate on the user's full history
        including the disclosure utterance. Easier (regex-like cues present).
      - mask_disclosure_posts=True  : drop the disclosure posts before
        aggregating. Tests whether the model learned the *implicit* signal
        beyond the disclosure phrase itself.
    """
    import numpy as np

    from src.evaluation.metrics import best_threshold_f1, full_report

    user_col = f"user_{target}"
    if user_col not in test_posts.columns:
        raise ValueError(f"Missing column {user_col} in test posts")

    work = test_posts.copy()
    if mask_disclosure_posts and "is_disclosure_post" in work.columns:
        work = work[work["is_disclosure_post"] != 1]

    if aggregation == "mean":
        per_user = work.groupby("author_hash").agg(
            score=(score_col, "mean"),
            label=(user_col, "max"),
        )
    elif aggregation == "max":
        per_user = work.groupby("author_hash").agg(
            score=(score_col, "max"),
            label=(user_col, "max"),
        )
    elif aggregation == "topk_mean":
        # Mean of top-K scores per user (K=5 by default)
        K = 5

        def _topk_mean(s):
            arr = s.values
            if len(arr) <= K:
                return float(arr.mean()) if len(arr) else 0.0
            top = np.partition(arr, -K)[-K:]
            return float(top.mean())

        per_user = work.groupby("author_hash").agg(
            score=(score_col, _topk_mean),
            label=(user_col, "max"),
        )
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    y = per_user["label"].astype(int).to_numpy()
    s = per_user["score"].astype(float).to_numpy()
    if threshold is None:
        threshold, _ = best_threshold_f1(y, s)

    report = full_report(y, s, threshold=threshold, bootstrap=False)
    report.update({
        "n_users": int(len(per_user)),
        "n_positive_users": int(y.sum()),
        "aggregation": aggregation,
        "mask_disclosure_posts": bool(mask_disclosure_posts),
        "threshold": float(threshold),
    })
    return report
