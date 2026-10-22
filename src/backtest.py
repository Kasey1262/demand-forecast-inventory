"""
Step 7: Inventory Policy Backtest (does the policy actually deliver its service level?)

Step 5 computes safety stock / reorder points from formulas. This step checks
those numbers against reality: we replay the test period day by day, run an
(s, Q) reorder policy against the ACTUAL sales, and measure what happens.

Honest-backtest rules:
  - Policy parameters (error std, EOQ) are estimated on a VALIDATION window
    (2017-07-01 .. 2017-09-30) that comes strictly before the test window.
    Estimating them on the test window and then scoring on it would be leakage.
  - Every reorder point s_t only uses information available at day t.

Two ways to size safety stock over a lead time of L days:
  - "iid"       : sigma_L = sigma_1day * sqrt(L)   (the textbook formula, Step 5)
                  Assumes daily forecast errors are independent.
  - "empirical" : sigma_L = std of the actual L-day lead-time-demand forecast
                  errors on the validation window. Captures autocorrelated
                  errors, which the sqrt(L) rule ignores.

Undershoot: the textbook ROP assumes we reorder exactly at s. With ~50 units
of demand per day, the inventory position usually jumps PAST s, so every order
starts with less protection than planned. Policies marked "+ undershoot" add
the renewal-theory correction (see inventory.reorder_point_with_undershoot).

Two ways to forecast lead-time demand:
  - "daily x L" : L * (tomorrow's one-step forecast)
  - "direct"    : a model trained to predict the L-day SUM directly

Metrics:
  - cycle service level (CSL): share of orders with NO lost sales while that
    order was in transit. This is exactly what the z-value in the safety stock
    formula promises, so target vs achieved CSL is the headline check.
  - fill rate: share of demand units actually served.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# Allow `python3 src/backtest.py` as well as `import src.backtest`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inventory import eoq, undershoot_moments
from src.models import FEATURE_COLS, linear_regression_predict

GROUP = ["store", "item"]


# ── Core simulator ───────────────────────────────────────────────────────────

def simulate_sq_policy(demand, reorder_points, order_qty: float, lead_time: int,
                       initial_on_hand: float, holding_cost: float = 2.0,
                       order_cost: float = 50.0, stockout_cost: float = 4.0) -> dict:
    """
    Replay a continuous-review (s, Q) policy with lost sales.

    Each day t:
      1. serve demand[t] from on-hand stock (unmet demand is lost)
      2. receive any order due today (arrives at end of day, after demand)
      3. if inventory position (on-hand + in transit) <= s_t, order Q
         (repeat until position > s_t); it arrives at end of day t + L,
         so it must cover demand on days t+1 .. t+L.

    holding_cost is per unit per YEAR (same unit as EOQ), charged daily on
    end-of-day stock. stockout_cost is per lost unit.
    """
    demand = np.asarray(demand, dtype=float)
    n = len(demand)
    s = np.broadcast_to(np.asarray(reorder_points, dtype=float), (n,))
    if order_qty <= 0:
        raise ValueError("order_qty must be positive")

    on_hand = float(initial_on_hand)
    arrivals = np.zeros(n + lead_time + 1)
    lost = np.zeros(n)
    on_hand_eod = np.zeros(n)
    orders = []                                   # placement days

    for t in range(n):
        served = min(on_hand, demand[t])
        lost[t] = demand[t] - served
        on_hand -= served
        on_hand += arrivals[t]
        on_hand_eod[t] = on_hand

        position = on_hand + arrivals[t + 1:].sum()
        while position <= s[t]:
            arrivals[t + lead_time] += order_qty
            orders.append(t)
            position += order_qty

    # An order "cycle" is the window it spends in transit: days t+1 .. t+L.
    # Only score orders whose whole window falls inside the horizon.
    scored = [t for t in orders if t + lead_time < n]
    cycle_ok = [lost[t + 1:t + lead_time + 1].sum() == 0 for t in scored]

    total_demand = demand.sum()
    total_lost = lost.sum()
    holding = on_hand_eod.sum() * holding_cost / 365
    ordering = len(orders) * order_cost
    stockout = total_lost * stockout_cost
    return {
        "days": n,
        "total_demand": total_demand,
        "lost_units": total_lost,
        "fill_rate": 1 - total_lost / total_demand if total_demand else 1.0,
        "n_orders": len(orders),
        "n_cycles": len(scored),
        "n_cycles_ok": int(sum(cycle_ok)),
        "cycle_service_level": float(np.mean(cycle_ok)) if scored else float("nan"),
        "stockout_days": int((lost > 0).sum()),
        "avg_on_hand": float(on_hand_eod.mean()),
        "holding_cost": holding,
        "ordering_cost": ordering,
        "stockout_cost": stockout,
        "total_cost": holding + ordering + stockout,
    }


# ── Forecasts used by the policies ───────────────────────────────────────────

def add_forward_sum(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """ltd_actual = sales over days t .. t+horizon-1 (NaN near each series' end)."""
    df = df.sort_values(GROUP + ["date"]).copy()
    g = df.groupby(GROUP)["sales"]
    df["ltd_actual"] = sum(g.shift(-k) for k in range(horizon))
    return df


def direct_ltd_predict(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """
    Direct lead-time-demand model: LinearRegression from FEATURE_COLS (all known
    at day t) to ltd_actual (the L-day sum starting at t).
    """
    tr = train.dropna(subset=["ltd_actual"])
    model = LinearRegression().fit(tr[FEATURE_COLS], tr["ltd_actual"])
    return np.clip(model.predict(target[FEATURE_COLS]), 0, None)


def lead_time_error_std(daily_errors, ltd_errors, lead_time: int, method: str) -> float:
    """sigma of lead-time demand: textbook sqrt(L) rule, or measured directly."""
    if method == "iid":
        return float(np.std(daily_errors, ddof=1) * math.sqrt(lead_time))
    if method == "empirical":
        return float(np.std(ltd_errors, ddof=1))
    raise ValueError(f"unknown method {method!r}")


# ── Policy definitions ───────────────────────────────────────────────────────
# name -> (lead-time-demand forecast column, sigma method, undershoot correction)

POLICIES = {
    "textbook (daily x L, iid sigma)":       ("ltd_daily_lr", "iid", False),
    "direct LTD, empirical sigma":           ("ltd_direct_lr", "empirical", False),
    "direct LTD, empirical + undershoot":    ("ltd_direct_lr", "empirical", True),
    "seasonal naive, empirical + undershoot": ("ltd_naive", "empirical", True),
}


def prepare_backtest_frame(feat: pd.DataFrame, lead_time: int = 7,
                           val_start: str = "2017-07-01",
                           test_start: str = "2017-10-01") -> pd.DataFrame:
    """
    Fit models on data before val_start, then attach every forecast column the
    policies need to the validation + test rows. Returns val+test rows only.
    """
    df = add_forward_sum(feat, lead_time)
    train = df[df["date"] < val_start]
    # The training target looks L-1 days ahead; drop rows whose target would
    # peek into the validation window.
    cutoff = pd.Timestamp(val_start) - pd.Timedelta(days=lead_time - 1)
    train = train[train["date"] < cutoff]
    hold = df[df["date"] >= val_start].copy()

    _, daily_pred = linear_regression_predict(train, hold)
    hold["pred_daily_lr"] = daily_pred
    hold["ltd_daily_lr"] = lead_time * daily_pred
    hold["ltd_direct_lr"] = direct_ltd_predict(train, hold)
    # Seasonal-naive LTD: "next L days look like the last 7 days" (rolling mean
    # already excludes today, so it is known at day t).
    hold["ltd_naive"] = lead_time * hold["sales_rollmean_7"]
    hold["pred_daily_naive"] = hold["sales_lag_7"]
    hold["split"] = np.where(hold["date"] < test_start, "val", "test")
    return hold


def run_backtest(hold: pd.DataFrame, policy: str, service_level: float = 0.95,
                 lead_time: int = 7, order_cost: float = 50.0,
                 holding_cost: float = 2.0, stockout_cost: float = 4.0,
                 series=None) -> pd.DataFrame:
    """
    Simulate one policy for every (store, item) in `hold` (or only `series`,
    a list of (store, item) tuples). Returns one row per series.
    """
    ltd_col, method, undershoot = POLICIES[policy]
    daily_col = "pred_daily_naive" if ltd_col == "ltd_naive" else "pred_daily_lr"
    z = norm.ppf(service_level)

    rows = []
    for (store, item), g in hold.groupby(GROUP):
        if series is not None and (store, item) not in series:
            continue
        val = g[g["split"] == "val"]
        test = g[g["split"] == "test"]
        val_ltd = val.dropna(subset=["ltd_actual"])

        sigma_l = lead_time_error_std(
            daily_errors=val["sales"] - val[daily_col],
            ltd_errors=val_ltd["ltd_actual"] - val_ltd[ltd_col],
            lead_time=lead_time, method=method,
        )
        q = eoq(val["sales"].mean() * 365, order_cost, holding_cost)
        mean_u, var_u = undershoot_moments(val["sales"]) if undershoot else (0.0, 0.0)
        buffer = mean_u + z * math.sqrt(sigma_l ** 2 + var_u)

        # The order placed at the END of day t covers days t+1 .. t+L, i.e. the
        # forecast row of day t+1 (whose features only use data up to day t).
        ltd = test[ltd_col].to_numpy()
        ltd_next = np.append(ltd[1:], ltd[-1])
        s = ltd_next + buffer

        res = simulate_sq_policy(
            test["sales"].to_numpy(), s, q, lead_time,
            initial_on_hand=s[0] + q, holding_cost=holding_cost,
            order_cost=order_cost, stockout_cost=stockout_cost,
        )
        rows.append({"store": store, "item": item, "policy": policy,
                     "target_csl": service_level, "sigma_lead_time": sigma_l,
                     "safety_buffer": buffer,
                     "order_qty": q, **res})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Pool cycles and units across all series (not a mean of per-series rates)."""
    g = results.groupby(["policy", "target_csl"], sort=False)
    out = pd.DataFrame({
        "achieved_csl": g["n_cycles_ok"].sum() / g["n_cycles"].sum(),
        "fill_rate": 1 - g["lost_units"].sum() / g["total_demand"].sum(),
        "avg_on_hand": g["avg_on_hand"].sum(),
        "total_cost": g["total_cost"].sum(),
    }).reset_index()
    out["csl_gap"] = out["achieved_csl"] - out["target_csl"]
    return out


# ── Run everything ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data import load_sales
    from src.features import build_feature_matrix

    print("Pipeline: data -> features -> forecasts -> day-by-day policy replay")
    feat = build_feature_matrix(load_sales())
    hold = prepare_backtest_frame(feat)

    levels = [0.90, 0.95, 0.99]
    all_results = pd.concat(
        [run_backtest(hold, p, sl) for p in POLICIES for sl in levels],
        ignore_index=True,
    )
    summary = summarize(all_results)

    pd.set_option("display.width", 140)
    print("\n── Backtest: target vs achieved service level (test = 2017 Q4) ──")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    os.makedirs("outputs", exist_ok=True)
    summary.to_csv("outputs/backtest_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for policy, g in summary.groupby("policy", sort=False):
        ax.plot(g["target_csl"], g["achieved_csl"], marker="o", label=policy)
    ax.plot(levels, levels, "k--", lw=1, label="target = achieved")
    ax.set_xlabel("Target cycle service level")
    ax.set_ylabel("Achieved cycle service level")
    ax.set_title("Inventory backtest: does the policy hit its target?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/04_backtest_service_level.png", dpi=120)
    print("\nSaved outputs/backtest_summary.csv and outputs/04_backtest_service_level.png")
