"""
Step 4: Forecasting Models

Three things happen here:
  1. Time-based train/test split  (NEVER random split for time series)
  2. Seasonal naive baseline      (the benchmark we must beat)
  3. Linear regression            (our first real model)

Metrics: MAE, RMSE, MAPE
  - MAE  = mean absolute error (in original sales units, easy to explain)
  - RMSE = penalises large errors more heavily than MAE
  - MAPE = percentage error (intuitive for business stakeholders)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Metrics ──────────────────────────────────────────────────────────────────

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE (returned as a percentage, range 0-200).

    Unlike MAPE, the denominator is (|actual| + |pred|) / 2, so a single
    zero-sales day no longer blows the whole metric up to thousands of %.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1.0, denom)   # avoid 0/0 when both are 0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def evaluate(name: str, y_true, y_pred) -> dict:
    """Return a results dict for one model."""
    return {
        "model": name,
        "MAE":   round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE":  round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "SMAPE": round(smape(y_true, y_pred), 2),
    }


# ── Train / test split ───────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, cutoff: str = "2017-10-01"):
    """
    Everything before cutoff  -> train
    Everything from cutoff on -> test  (last ~3 months of 2017)

    WHY NOT random split?
    If you randomly shuffle rows and pick 20% as test, rows from 2017 end up
    in train and rows from 2013 end up in test.  The model is trained on the
    FUTURE and tested on the PAST.  It looks great in evaluation but fails
    completely on real future data.  Always split by time.
    """
    train = df[df["date"] < cutoff].copy()
    test  = df[df["date"] >= cutoff].copy()
    print(f"[split] train={len(train):,} rows  test={len(test):,} rows  cutoff={cutoff}")
    return train, test


# ── Baseline: seasonal naive ─────────────────────────────────────────────────

def seasonal_naive_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """
    Prediction = sales from exactly 7 days ago (same day last week).

    This is the simplest sensible baseline for weekly-seasonal data.
    It requires zero training.  If our model can't beat this, it's useless.

    How it works:
      For each row in test, look up the sales value from (date - 7 days)
      in the combined train+test history.
    """
    history = pd.concat([train, test])[["date", "store", "item", "sales"]].copy()
    history = history.set_index(["date", "store", "item"])["sales"]

    preds = []
    for _, row in test.iterrows():
        lookup_date = row["date"] - pd.Timedelta(days=7)
        key = (lookup_date, row["store"], row["item"])
        preds.append(history.get(key, np.nan))
    return np.array(preds, dtype=float)


# ── Linear regression ────────────────────────────────────────────────────────

FEATURE_COLS = [
    "dayofweek", "month", "dayofyear", "weekofyear", "is_weekend", "quarter",
    "sales_lag_7", "sales_lag_14", "sales_lag_28", "sales_lag_365",
    "sales_rollmean_7", "sales_rollstd_7", "sales_rollmean_28", "sales_rollstd_28",
]

def linear_regression_predict(train: pd.DataFrame, test: pd.DataFrame):
    """
    Fit a LinearRegression on train features, predict on test features.
    Returns (model, predictions).
    """
    X_train = train[FEATURE_COLS]
    y_train = train["sales"]
    X_test  = test[FEATURE_COLS]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)   # sales can't be negative
    return model, preds


# ── Run everything ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data import load_sales
    from src.features import build_feature_matrix

    print("Building feature matrix (this takes ~30s)...")
    df   = load_sales()
    feat = build_feature_matrix(df)

    train, test = time_split(feat)

    print("\nRunning seasonal naive baseline...")
    naive_preds = seasonal_naive_predict(train, test)
    valid_mask  = ~np.isnan(naive_preds)

    print("Training linear regression...")
    model, lr_preds = linear_regression_predict(train, test)

    y_test = test["sales"].to_numpy()

    results = pd.DataFrame([
        evaluate("Seasonal Naive",     y_test[valid_mask], naive_preds[valid_mask]),
        evaluate("Linear Regression",  y_test,             lr_preds),
    ])
    print("\n── Model comparison ──────────────────────────────")
    print(results.to_string(index=False))

    print("\n── Linear regression top 5 feature weights ──────")
    weights = pd.Series(model.coef_, index=FEATURE_COLS).abs().sort_values(ascending=False)
    print(weights.head(5).round(3).to_string())
