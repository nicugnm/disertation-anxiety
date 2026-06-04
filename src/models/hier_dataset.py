"""User-sequence construction for the hierarchical user-level model.

Groups posts by author, orders chronologically, and builds fixed-width per-user
sequences (post indices + a post-level mask) plus user-level labels. The post
indices reference rows of the input DataFrame (and, equivalently, a precomputed
per-post embedding matrix aligned to those rows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def user_label_columns(df: pd.DataFrame, targets: list[str]) -> list[str]:
    """Prefer explicit user-level labels (`user_<t>`) else fall back to post labels."""
    return [f"user_{t}" if f"user_{t}" in df.columns else f"label_{t}" for t in targets]


def build_user_sequences(
    df: pd.DataFrame,
    targets: list[str],
    max_posts: int = 64,
    order: str = "recent",
    time_col: str = "created_utc",
    author_col: str = "author_hash",
    min_posts: int = 1,
):
    """Return (users, post_idx, post_mask, y).

    post_idx : (n_users, max_posts) int — row positions into `df` (pad = 0, masked).
    post_mask: (n_users, max_posts) bool — True for real posts.
    y        : (n_users, n_targets) float — user label = max over the user's posts.
    `order`: 'recent' keeps the most recent max_posts, 'chronological' the earliest;
    both return posts in chronological (ascending-time) order.
    """
    df = df.reset_index(drop=True)
    label_cols = user_label_columns(df, targets)
    has_time = time_col in df.columns
    users, post_idx, post_mask, y = [], [], [], []
    for author, g in df.groupby(author_col, sort=False):
        if author is None or (isinstance(author, float) and np.isnan(author)):
            continue
        gi = (g.sort_values(time_col).index.to_numpy() if has_time else g.index.to_numpy())
        if len(gi) < min_posts:
            continue
        if len(gi) > max_posts:
            gi = gi[-max_posts:] if order == "recent" else gi[:max_posts]
        n = len(gi)
        idx = np.zeros(max_posts, dtype=np.int64)
        idx[:n] = gi
        mask = np.zeros(max_posts, dtype=bool)
        mask[:n] = True
        yi = (g[label_cols].astype(float).fillna(0.0).to_numpy().max(axis=0) >= 0.5).astype(np.float32)
        users.append(author)
        post_idx.append(idx)
        post_mask.append(mask)
        y.append(yi)
    if not users:
        return [], np.zeros((0, max_posts), np.int64), np.zeros((0, max_posts), bool), np.zeros((0, len(targets)), np.float32)
    return users, np.stack(post_idx), np.stack(post_mask), np.stack(y)
