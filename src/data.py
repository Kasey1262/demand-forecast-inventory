"""
Data layer: load real sales data, or generate synthetic data with the same structure.

Real data: Kaggle "Store Item Demand Forecasting Challenge"
    Download train.csv -> data/raw/train.csv
    Columns: date (YYYY-MM-DD), store (1..10), item (1..50), sales (int)
    ~913,000 rows (10 stores x 50 items x ~1826 days)

Until you have the real file, generate_synthetic_sales() produces data with the
same shape and columns, so you can build the whole pipeline before downloading.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "train.csv")


def generate_synthetic_sales(
    n_stores: int = 10,
    n_items: int = 50,
    start: str = "2013-01-01",
    end: str = "2017-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily sales with the same structure as Store Item Demand.

    Each (store, item) series is built as:
        sales ~ Poisson( base * store_mult * yearly * weekly * trend )
    a base demand level shaped by yearly + weekly seasonality and a slow upward
    trend, with Poisson noise on top.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()      # day of year: 1..365/366
    dow = dates.dayofweek.to_numpy()      # day of week: 0=Mon ... 6=Sun

    # Yearly seasonality: a sine wave over the year, peaking in summer
    yearly = 1.0 + 0.35 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    # Weekly seasonality: weekends are 25% higher
    weekly = np.where(dow >= 5, 1.25, 1.0)
    # Slow upward trend: +25% across the 5 years
    trend = 1.0 + np.linspace(0, 0.25, len(dates))

    frames = []
    for store in range(1, n_stores + 1):
        store_mult = rng.uniform(0.7, 1.4)            # each store runs bigger or smaller
        for item in range(1, n_items + 1):
            base = rng.uniform(8, 60)                 # each item has its own demand level
            lam = base * store_mult * yearly * weekly * trend
            sales = rng.poisson(np.clip(lam, 0.1, None))
            frames.append(pd.DataFrame({
                "date": dates, "store": store, "item": item, "sales": sales
            }))
    return pd.concat(frames, ignore_index=True)


def load_sales(path: str = RAW_PATH, fallback_synthetic: bool = True) -> pd.DataFrame:
    """Load sales data, or fall back to synthetic data if the file is missing."""
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        print(f"[data] Loaded real data: {path}  shape={df.shape}")
    elif fallback_synthetic:
        print("[data] train.csv not found -> using synthetic data.")
        df = generate_synthetic_sales()
        print(f"[data] Synthetic data shape={df.shape}")
    else:
        raise FileNotFoundError(f"{path} not found and synthetic fallback disabled.")
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    df = load_sales()
    print(df.head(10))
    print(f"\nDate range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Stores: {df['store'].nunique()}  Items: {df['item'].nunique()}")
    print(f"Total rows: {len(df):,}")
