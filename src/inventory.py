"""
Step 5: Inventory Optimization (Operations Research layer)

This is where forecasts become DECISIONS. We take the demand forecast from
Step 4 and feed it into four classic inventory models:

  1. Safety Stock   - buffer to absorb demand uncertainty
  2. Reorder Point  - when to reorder
  3. EOQ            - how much to order each time (cost-optimal)
  4. Newsvendor     - optimal order for perishables (juice, fresh food)

KEY INSIGHT linking Step 4 and Step 5:
  Safety stock uses the standard deviation of FORECAST ERROR (not raw demand).
  A more accurate model -> smaller error std -> less safety stock -> lower cost.
  This is exactly how better forecasting translates into real money saved.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.stats import norm


# ── Core OR formulas ─────────────────────────────────────────────────────────

def safety_stock(error_std: float, lead_time: float, service_level: float = 0.95) -> float:
    """SS = z * sigma * sqrt(lead_time).  z comes from the target service level."""
    z = norm.ppf(service_level)            # 0.95 -> 1.645
    return z * error_std * math.sqrt(lead_time)


def reorder_point(avg_daily_demand: float, lead_time: float,
                  error_std: float, service_level: float = 0.95) -> float:
    """ROP = expected demand over lead time + safety stock."""
    return avg_daily_demand * lead_time + safety_stock(error_std, lead_time, service_level)


def eoq(annual_demand: float, order_cost: float, holding_cost: float) -> float:
    """EOQ = sqrt(2 * D * S / H).  The cost-minimising order quantity."""
    return math.sqrt(2 * annual_demand * order_cost / holding_cost)


def newsvendor_quantity(mean_demand: float, std_demand: float,
                        underage_cost: float, overage_cost: float) -> float:
    """
    Single-period optimal order for perishables.
        critical ratio = Cu / (Cu + Co)
        Q* = F^-1(critical ratio)  (assuming Normal demand)
    High overage cost (waste) -> low ratio -> order conservatively.
    """
    cr = underage_cost / (underage_cost + overage_cost)
    return mean_demand + norm.ppf(cr) * std_demand


# ── Link forecasts -> inventory policy ───────────────────────────────────────

def build_inventory_policy(test: pd.DataFrame, preds: np.ndarray,
                           lead_time: int = 7, service_level: float = 0.95,
                           order_cost: float = 50.0, holding_cost: float = 2.0,
                           underage_cost: float = 4.0, overage_cost: float = 3.0) -> pd.DataFrame:
    """
    For each (store, item): use the forecast mean as expected demand and the
    forecast-error std as uncertainty, then compute the full inventory policy.
    """
    df = test[["store", "item", "sales"]].copy()
    df["pred"] = preds
    df["error"] = df["sales"] - df["pred"]

    rows = []
    for (store, item), g in df.groupby(["store", "item"]):
        avg_demand = g["pred"].mean()          # expected daily demand
        error_std = g["error"].std()           # forecast uncertainty
        rows.append({
            "store": store,
            "item": item,
            "avg_daily_demand": round(avg_demand, 1),
            "forecast_error_std": round(error_std, 2),
            "safety_stock": round(safety_stock(error_std, lead_time, service_level), 1),
            "reorder_point": round(reorder_point(avg_demand, lead_time, error_std, service_level), 1),
            "eoq": round(eoq(avg_demand * 365, order_cost, holding_cost), 1),
            "newsvendor_qty": round(newsvendor_quantity(avg_demand, error_std,
                                                        underage_cost, overage_cost), 1),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data import load_sales
    from src.features import build_feature_matrix
    from src.models import time_split, linear_regression_predict

    print("Pipeline: data -> features -> forecast -> inventory policy")
    df = load_sales()
    feat = build_feature_matrix(df)
    train, test = time_split(feat)
    _, preds = linear_regression_predict(train, test)

    policy = build_inventory_policy(test, preds)
    print(f"\nInventory policy table: {policy.shape[0]} (store, item) combinations")
    print("\nFirst 8 rows:")
    print(policy.head(8).to_string(index=False))

    print("\n── Quick sanity check on store 1, item 1 ──")
    r = policy[(policy["store"] == 1) & (policy["item"] == 1)].iloc[0]
    print(f"Expected daily demand : {r['avg_daily_demand']}")
    print(f"Safety stock (95%)    : {r['safety_stock']}")
    print(f"Reorder point         : {r['reorder_point']}  <- reorder when stock hits this")
    print(f"EOQ (order size)      : {r['eoq']}")
    print(f"Newsvendor qty        : {r['newsvendor_qty']}  <- for a perishable version")
