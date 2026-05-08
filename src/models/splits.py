"""Train/val/test splitting utilities. Stratified, group-aware."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split by `label_<target>` (binarized) when present and possible."""
    label_col = f"label_{target}"
    strat = None
    if stratify and label_col in df.columns:
        bin_labels = (df[label_col].astype(float).fillna(0.0) >= 0.5).astype(int)
        # Need at least 2 of each class for stratification to work
        if bin_labels.nunique() > 1 and bin_labels.value_counts().min() >= 2:
            strat = bin_labels

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )
    if val_size <= 0:
        return train_val.reset_index(drop=True), pd.DataFrame(), test.reset_index(drop=True)

    rel_val = val_size / (1.0 - test_size)
    strat_tv = None
    if stratify and label_col in train_val.columns:
        bin_labels = (train_val[label_col].astype(float).fillna(0.0) >= 0.5).astype(int)
        if bin_labels.nunique() > 1 and bin_labels.value_counts().min() >= 2:
            strat_tv = bin_labels
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=random_state, stratify=strat_tv
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def cross_subreddit_split(
    df: pd.DataFrame,
    held_out_subs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out entire subreddits for the cross-subreddit transfer experiment (RQ3)."""
    held = df["subreddit"].astype(str).str.lower().isin([s.lower() for s in held_out_subs])
    return df[~held].reset_index(drop=True), df[held].reset_index(drop=True)
