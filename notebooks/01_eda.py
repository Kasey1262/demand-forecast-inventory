"""
Step 2: Exploratory Data Analysis (EDA)

Run from the project root:  python notebooks/01_eda.py
Figures are saved to outputs/.

What it does:
  1. Overview: shape, date range, #stores/#items, missing values
  2. Aggregate everything into ONE daily total-sales series
  3. Seasonal decomposition (trend / seasonal / residual)
  4. ADF stationarity test
  5. Weekly + yearly seasonality profiles
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# let this script import src/ when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_sales  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
sns.set_theme(style="whitegrid")


def overview(df):
    print("=" * 50 + "\n1) Overview\n" + "=" * 50)
    print(f"Shape          : {df.shape}")
    print(f"Date range     : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Stores / Items : {df['store'].nunique()} / {df['item'].nunique()}")
    print(f"Missing values : {int(df.isna().sum().sum())}")
    print(f"Sales summary  :\n{df['sales'].describe().round(2)}\n")


def daily_total_series(df):
    """Collapse all (store, item) series into one daily total-sales series."""
    s = df.groupby("date")["sales"].sum().sort_index()
    s = s.asfreq("D")  # force a regular daily frequency
    return s


def plot_daily(s):
    plt.figure(figsize=(12, 4))
    s.plot(color="#185FA5", lw=0.8)
    plt.title("Total daily sales")
    plt.xlabel("date"); plt.ylabel("sales")
    plt.tight_layout()
    p = os.path.join(OUT, "01_daily_sales.png")
    plt.savefig(p, dpi=120); plt.close()
    print(f"[plot] {p}")


def decompose(s):
    print("=" * 50 + "\n3) Seasonal decomposition (period=7)\n" + "=" * 50)
    res = seasonal_decompose(s.interpolate(), model="additive", period=7)
    fig = res.plot(); fig.set_size_inches(12, 8); plt.tight_layout()
    p = os.path.join(OUT, "02_decomposition.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[plot] {p}")
    print(f"Trend non-null points : {res.trend.notna().sum()}")
    print(f"Seasonal amplitude    : {res.seasonal.max() - res.seasonal.min():.1f}\n")


def adf_test(s):
    print("=" * 50 + "\n4) ADF stationarity test\n" + "=" * 50)
    stat, pval, *_ = adfuller(s.dropna())
    print(f"ADF statistic : {stat:.3f}")
    print(f"p-value       : {pval:.4f}")
    verdict = "stationary (reject unit root)" if pval < 0.05 else "non-stationary (consider differencing)"
    print(f"Verdict       : {verdict}\n")


def seasonality_profiles(df):
    print("=" * 50 + "\n5) Weekly + yearly seasonality\n" + "=" * 50)
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    dow = df.groupby("dow")["sales"].mean()
    mon = df.groupby("month")["sales"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    dow.plot(kind="bar", ax=axes[0], color="#1D9E75")
    axes[0].set_title("Avg sales by day of week (0=Mon)"); axes[0].set_xlabel("dayofweek")
    mon.plot(kind="bar", ax=axes[1], color="#D85A30")
    axes[1].set_title("Avg sales by month"); axes[1].set_xlabel("month")
    plt.tight_layout()
    p = os.path.join(OUT, "03_seasonality.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[plot] {p}")
    print(f"Weekend / weekday ratio : {dow[5:].mean() / dow[:5].mean():.2f}")
    print(f"Peak / trough month     : {mon.max() / mon.min():.2f}\n")


def main():
    df = load_sales()
    overview(df)
    s = daily_total_series(df)
    plot_daily(s)
    decompose(s)
    adf_test(s)
    seasonality_profiles(df)
    print("EDA done. Figures saved in outputs/")


if __name__ == "__main__":
    main()
