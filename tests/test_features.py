"""Tests that the feature engineering is leakage-safe (the two documented traps)."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import add_lag_features, add_rolling_features


def _one_series(n=40, store=1, item=1, start=0):
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    sales = np.arange(start, start + n, dtype=float)  # 0,1,2,... easy to verify
    return pd.DataFrame({"date": dates, "store": store, "item": item, "sales": sales})


def test_lag_equals_value_n_days_earlier():
    df = _one_series(n=40)
    out = add_lag_features(df, lags=(7,))
    # row i's lag_7 must equal sales at row i-7 within the same series
    valid = out.dropna(subset=["sales_lag_7"])
    for _, row in valid.iterrows():
        earlier = out[out["date"] == row["date"] - pd.Timedelta(days=7)]
        assert earlier["sales"].iloc[0] == row["sales_lag_7"]


def test_rolling_excludes_today_no_leakage():
    # sales = 0,1,2,3,...  rollmean_7 at a row must be the mean of the 7 PAST days,
    # never including today's own value.
    df = _one_series(n=40)
    out = add_rolling_features(df, windows=(7,))
    out = out.dropna(subset=["sales_rollmean_7"]).reset_index(drop=True)
    row = out.iloc[0]  # first valid rolling row
    today = row["sales"]
    # mean of the 7 strictly-previous values (today excluded)
    expected = np.mean([today - k for k in range(1, 8)])
    assert np.isclose(row["sales_rollmean_7"], expected)
    assert not np.isclose(row["sales_rollmean_7"], np.mean([today - k for k in range(0, 7)]))


def test_lag_does_not_cross_series():
    # Two series with very different levels; lag of series B must never pull from A.
    a = _one_series(n=20, store=1, item=1, start=0)      # 0..19
    b = _one_series(n=20, store=2, item=2, start=1000)   # 1000..1019
    df = pd.concat([a, b], ignore_index=True)
    out = add_lag_features(df, lags=(7,))
    b_rows = out[(out["store"] == 2) & out["sales_lag_7"].notna()]
    # every non-null lag for series B must be a B-level value (>= 1000), never an A value
    assert (b_rows["sales_lag_7"] >= 1000).all()
