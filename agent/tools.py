"""
Agent tool layer.

Wraps the existing forecasting + inventory pipeline (src/) as callable tools a
Claude agent can invoke. The linear-regression model is trained ONCE (lazily,
on first use) and held in memory; every tool then answers instantly from that
fitted state instead of retraining per question.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_sales
from src.features import build_feature_matrix
from src.models import time_split, linear_regression_predict
from src.inventory import (
    safety_stock, reorder_point, eoq, newsvendor_quantity, build_inventory_policy,
)

# OR parameters (kept in sync with inventory.build_inventory_policy defaults)
LEAD_TIME = 7
ORDER_COST = 50.0
HOLDING_COST = 2.0
UNDERAGE_COST = 4.0
OVERAGE_COST = 3.0

_WORLD = None


def _world():
    """Train once, cache the per-series inventory policy table in memory."""
    global _WORLD
    if _WORLD is None:
        print("[agent] Training model (one-time, ~30s)...")
        df = load_sales()
        feat = build_feature_matrix(df)
        train, test = time_split(feat)
        _, preds = linear_regression_predict(train, test)
        policy = build_inventory_policy(
            test, preds, lead_time=LEAD_TIME,
            order_cost=ORDER_COST, holding_cost=HOLDING_COST,
            underage_cost=UNDERAGE_COST, overage_cost=OVERAGE_COST,
        ).set_index(["store", "item"])
        _WORLD = {
            "policy": policy,
            "stores": sorted(test["store"].unique().tolist()),
            "items": sorted(test["item"].unique().tolist()),
        }
        print("[agent] Ready.")
    return _WORLD


def _lookup(store, item):
    w = _world()
    if store not in w["stores"] or item not in w["items"]:
        return None, (
            f"No data for store {store}, item {item}. "
            f"Valid stores {w['stores'][0]}-{w['stores'][-1]}, "
            f"items {w['items'][0]}-{w['items'][-1]}."
        )
    return w["policy"].loc[(store, item)], None


# ── Tool functions ───────────────────────────────────────────────────────────

def get_forecast(store: int, item: int) -> dict:
    row, err = _lookup(store, item)
    if err:
        return {"error": err}
    return {
        "store": store, "item": item,
        "forecasted_avg_daily_demand": round(float(row["avg_daily_demand"]), 1),
        "forecast_error_std": round(float(row["forecast_error_std"]), 2),
        "note": "Model-predicted average daily units for this series, with the "
                "standard deviation of forecast error (uncertainty).",
    }


def get_inventory_policy(store: int, item: int, service_level: float = 0.95) -> dict:
    row, err = _lookup(store, item)
    if err:
        return {"error": err}
    if not (0.5 <= service_level < 1.0):
        return {"error": "service_level must be between 0.5 and 0.999."}
    avg = float(row["avg_daily_demand"])
    e = float(row["forecast_error_std"])
    return {
        "store": store, "item": item,
        "service_level": service_level, "lead_time_days": LEAD_TIME,
        "avg_daily_demand": round(avg, 1),
        "forecast_error_std": round(e, 2),
        "safety_stock": round(float(safety_stock(e, LEAD_TIME, service_level)), 1),
        "reorder_point": round(float(reorder_point(avg, LEAD_TIME, e, service_level)), 1),
        "eoq_order_qty": round(float(eoq(avg * 365, ORDER_COST, HOLDING_COST)), 1),
        "newsvendor_qty": round(float(newsvendor_quantity(avg, e, UNDERAGE_COST, OVERAGE_COST)), 1),
    }


def check_stockout_risk(store: int, item: int, days: int = 7,
                        current_stock: float | None = None) -> dict:
    row, err = _lookup(store, item)
    if err:
        return {"error": err}
    avg = float(row["avg_daily_demand"])
    e = float(row["forecast_error_std"])
    exp = avg * days
    std = e * math.sqrt(days)
    out = {"store": store, "item": item, "horizon_days": days,
           "expected_demand": round(exp, 1), "demand_std": round(std, 2)}
    if current_stock is None:
        out["stock_for_95pct_service"] = round(exp + norm.ppf(0.95) * std, 1)
        out["stock_for_99pct_service"] = round(exp + norm.ppf(0.99) * std, 1)
        out["note"] = ("No current_stock given; showing units needed to cover "
                       "demand at 95% and 99% service.")
    else:
        p = float(1 - norm.cdf(current_stock, loc=exp, scale=std))
        out["current_stock"] = current_stock
        out["stockout_probability"] = round(p, 3)
        out["verdict"] = ("HIGH risk - reorder now" if p > 0.20
                          else "MODERATE risk" if p > 0.05 else "LOW risk")
    return out


# ── Anthropic tool schemas ───────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_forecast",
        "description": "Get the forecasted average daily demand and forecast-error "
                       "std for a store-item, from the trained model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store": {"type": "integer", "description": "Store id (1-10)"},
                "item": {"type": "integer", "description": "Item id (1-50)"},
            },
            "required": ["store", "item"],
        },
    },
    {
        "name": "get_inventory_policy",
        "description": "Compute inventory policy (safety stock, reorder point, EOQ, "
                       "newsvendor qty) for a store-item at a given service level. Use "
                       "when the user asks how much to stock/reorder, or what changes "
                       "at a different service level.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store": {"type": "integer", "description": "Store id (1-10)"},
                "item": {"type": "integer", "description": "Item id (1-50)"},
                "service_level": {"type": "number",
                                  "description": "Target service level 0.5-0.999 "
                                                 "(e.g. 0.95). Defaults to 0.95."},
            },
            "required": ["store", "item"],
        },
    },
    {
        "name": "check_stockout_risk",
        "description": "Estimate stockout risk for a store-item over a horizon. If "
                       "current_stock is given, returns P(demand exceeds it); "
                       "otherwise returns stock needed for 95%/99% service.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store": {"type": "integer", "description": "Store id (1-10)"},
                "item": {"type": "integer", "description": "Item id (1-50)"},
                "days": {"type": "integer", "description": "Horizon in days (default 7)"},
                "current_stock": {"type": "number",
                                  "description": "Units currently on hand (optional)"},
            },
            "required": ["store", "item"],
        },
    },
]

_DISPATCH = {
    "get_forecast": get_forecast,
    "get_inventory_policy": get_inventory_policy,
    "check_stockout_risk": check_stockout_risk,
}


def run_tool(name: str, tool_input: dict) -> dict:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**tool_input)
    except Exception as ex:  # never let a tool crash the agent loop
        return {"error": f"{type(ex).__name__}: {ex}"}


if __name__ == "__main__":
    import json
    print("=== self-test (no API key needed) ===")
    for call in [
        ("get_forecast", {"store": 1, "item": 1}),
        ("get_inventory_policy", {"store": 1, "item": 1, "service_level": 0.99}),
        ("check_stockout_risk", {"store": 1, "item": 1, "days": 7}),
        ("check_stockout_risk", {"store": 1, "item": 1, "days": 7, "current_stock": 300}),
        ("get_forecast", {"store": 99, "item": 99}),
    ]:
        print(f"\n>>> {call[0]}({call[1]})")
        print(json.dumps(run_tool(*call), indent=2))
