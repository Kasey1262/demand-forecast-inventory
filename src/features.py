"""
Step 3: Feature Engineering

A linear regression only learns a weighted sum of the columns you feed it.
It has NO built-in notion of "time", "yesterday", or "last week".
So our job is to TRANSLATE temporal patterns into numeric columns:

  1. calendar features - what day / month / quarter is it
  2. lag features      - what were sales N days ago
  3. rolling features  - recent average level and volatility

Two leakage traps (the most common time-series interview questions):
  A. lag / rolling MUST be computed within each (store, item) group,
     otherwise one series "borrows" values from an unrelated series.
  B. rolling MUST be shifted by 1 first, so "today" is never used to
     predict "today" (that would leak the answer into the features).
"""
from __future__ import annotations

import pandas as pd

GROUP = ["store", "item"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the date into model-readable columns."""
    df = df.copy()
    d = df["date"].dt
    df["dayofweek"] = d.dayofweek            # 0=Mon ... 6=Sun
    df["month"] = d.month                    # 1..12
    df["dayofyear"] = d.dayofyear            # 1..365
    df["weekofyear"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    df["quarter"] = d.quarter                # 1..4
    return df


def add_lag_features(df: pd.DataFrame, lags=(7, 14, 28, 365)) -> pd.DataFrame:
    """sales_lag_L = sales L days ago, computed WITHIN each (store, item)."""
    df = df.copy()
    g = df.groupby(GROUP)["sales"]
    for L in lags:
        df[f"sales_lag_{L}"] = g.shift(L)
    return df


def add_rolling_features(df: pd.DataFrame, windows=(7, 28)) -> pd.DataFrame:
    """Rolling mean / std of PAST sales (shift(1) first to exclude today)."""
    df = df.copy()
    # shift(1) within each group BEFORE rolling -> today is excluded
    df["_past"] = df.groupby(GROUP)["sales"].shift(1)
    g = df.groupby(GROUP)["_past"]
    for w in windows:
        df[f"sales_rollmean_{w}"] = g.transform(lambda s: s.rolling(w).mean())
        df[f"sales_rollstd_{w}"] = g.transform(lambda s: s.rolling(w).std())
    return df.drop(columns="_past")


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Run all three feature steps, then drop rows with NaN from lag/rolling."""
    df = df.sort_values(GROUP + ["date"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"[features] rows {before:,} -> {len(df):,} after dropping warm-up NaNs")
    return df


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data import load_sales

    df = load_sales()
    feat = build_feature_matrix(df)
    print(f"\nColumns ({len(feat.columns)}):")
    print(list(feat.columns))
    print("\nOne (store, item) series, first 5 rows of key features:")
    cols = ["date", "store", "item", "sales",
            "sales_lag_7", "sales_lag_365", "sales_rollmean_7", "sales_rollstd_7"]
    print(feat[feat["item"] == 1].head(5)[cols].to_string(index=False))
