"""Temporal analysis hooks.

The dissertation's optional COVID chapter (RQ4) uses these to compare
pre-/during-/post-COVID windows. Designed so adding new windows or
new label dimensions takes one line.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

# Conventional COVID epoch boundaries — defensible defaults; adjust per the
# thesis's argument.
COVID_START = datetime(2020, 3, 11, tzinfo=timezone.utc).timestamp()  # WHO declaration
COVID_PEAK_END = datetime(2022, 5, 1, tzinfo=timezone.utc).timestamp()  # rough endemic transition


def add_period(df: pd.DataFrame, ts_col: str = "created_utc") -> pd.DataFrame:
    out = df.copy()
    ts = out[ts_col].astype(float)
    out["period"] = pd.cut(
        ts,
        bins=[-float("inf"), COVID_START, COVID_PEAK_END, float("inf")],
        labels=["pre_covid", "covid_peak", "post_peak"],
    )
    return out


def label_rates_by_period(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    work = add_period(df) if "period" not in df.columns else df
    rows = []
    for period, sub in work.groupby("period", observed=True):
        row = {"period": str(period), "n": len(sub)}
        for col in label_cols:
            if col in sub.columns:
                vals = sub[col].astype(float)
                row[f"{col}_rate"] = float((vals >= 0.5).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def label_rates_by_period_and_subreddit(
    df: pd.DataFrame, label_cols: list[str]
) -> pd.DataFrame:
    work = add_period(df) if "period" not in df.columns else df
    rows = []
    for (period, sub), grp in work.groupby(["period", "subreddit"], observed=True):
        row = {"period": str(period), "subreddit": sub, "n": len(grp)}
        for col in label_cols:
            if col in grp.columns:
                vals = grp[col].astype(float)
                row[f"{col}_rate"] = float((vals >= 0.5).mean())
        rows.append(row)
    return pd.DataFrame(rows)
