"""
MCP server: exposes the forecasting + inventory pipeline to any MCP client
(Claude Code, Claude Desktop, Cursor, ...), not just the bundled CLI agent.

Tools:
  get_forecast            forecasted daily demand + error std for a store-item
  get_inventory_policy    safety stock / ROP / EOQ / newsvendor at a service level
  check_stockout_risk     P(demand > current stock) over a horizon
  run_inventory_backtest  replay Q4 2017 day by day and report the service level
                          each reorder policy ACTUALLY delivered for a store-item

Every response carries a `provenance` block (data source, windows, caveats) so
the agent can cite where a number came from and what it assumes.

Run (stdio):
    pip install -r requirements.txt
    python3 agent/mcp_server.py

Register with Claude Code:
    claude mcp add demand-forecast -- python3 /abs/path/to/agent/mcp_server.py
"""
from __future__ import annotations

import contextlib
import functools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from agent import tools
from src.backtest import POLICIES, prepare_backtest_frame, run_backtest
from src.data import RAW_PATH, load_sales
from src.features import build_feature_matrix

DATA_SOURCE = "kaggle_train_csv" if os.path.exists(RAW_PATH) else "synthetic"

POLICY_PROVENANCE = {
    "data_source": DATA_SOURCE,
    "model": "LinearRegression on calendar/lag/rolling features",
    "train_window": "< 2017-10-01",
    "stats_window": "2017-10-01 .. 2017-12-31 (forecast mean and error std)",
    "assumptions": [
        "lead time 7 days, order cost 50, holding cost 2/unit/year",
        "safety stock uses sigma_1day * sqrt(L): assumes independent daily errors",
        "reorder point ignores undershoot; see run_inventory_backtest for the "
        "service level this actually achieves",
    ],
}

BACKTEST_PROVENANCE = {
    "data_source": DATA_SOURCE,
    "train_window": "< 2017-07-01",
    "validation_window": "2017-07-01 .. 2017-09-30 (sigma, EOQ, undershoot estimated here)",
    "test_window": "2017-10-01 .. 2017-12-31 (policy replayed against actual sales)",
    "assumptions": ["lost sales", "lead time 7 days", "continuous-review (s, Q) policy"],
    "caveat": "One store-item has only a handful of order cycles in Q4; per-series "
              "service levels are noisy. Pooled results are in outputs/backtest_summary.csv.",
}

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)

mcp = MCPServer(
    name="demand-forecast-inventory",
    instructions=(
        "Retail demand forecasting and inventory tools for 10 stores x 50 items. "
        "Store ids are 1-10, item ids 1-50. Cite the provenance block when "
        "reporting numbers. Before recommending a service level, use "
        "run_inventory_backtest to check what the policy actually delivers."
    ),
)


def _quiet(fn):
    """stdout is the MCP protocol channel on stdio; send pipeline prints to stderr."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)
    return wrapper


_HOLD = None


def _backtest_frame():
    """Train the backtest models once (validation + test forecasts), then cache."""
    global _HOLD
    if _HOLD is None:
        _HOLD = prepare_backtest_frame(build_feature_matrix(load_sales()))
    return _HOLD


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool(annotations=READ_ONLY)
@_quiet
def get_forecast(store: int, item: int) -> dict:
    """Forecasted average daily demand and forecast-error std for a store-item."""
    return {**tools.get_forecast(store, item), "provenance": POLICY_PROVENANCE}


@mcp.tool(annotations=READ_ONLY)
@_quiet
def get_inventory_policy(store: int, item: int, service_level: float = 0.95) -> dict:
    """
    Safety stock, reorder point, EOQ and newsvendor quantity for a store-item at
    a target service level (0.5-0.999). Use for "how much should I stock/reorder".
    """
    return {**tools.get_inventory_policy(store, item, service_level),
            "provenance": POLICY_PROVENANCE}


@mcp.tool(annotations=READ_ONLY)
@_quiet
def check_stockout_risk(store: int, item: int, days: int = 7,
                        current_stock: float | None = None) -> dict:
    """
    Probability that demand over the next `days` exceeds `current_stock`. Without
    current_stock, returns the stock needed for 95% / 99% service instead.
    """
    return {**tools.check_stockout_risk(store, item, days, current_stock),
            "provenance": POLICY_PROVENANCE}


@mcp.tool(annotations=READ_ONLY)
@_quiet
def run_inventory_backtest(store: int, item: int, service_level: float = 0.95) -> dict:
    """
    Replay Q4 2017 day by day for one store-item under each reorder policy and
    report target vs ACHIEVED cycle service level, fill rate, average stock and
    cost. Use to check whether a service-level promise actually holds.
    """
    if not (0.5 <= service_level < 1.0):
        return {"error": "service_level must be between 0.5 and 0.999."}
    hold = _backtest_frame()
    if not ((hold["store"] == store) & (hold["item"] == item)).any():
        return {"error": f"No data for store {store}, item {item}. "
                         "Valid stores 1-10, items 1-50."}

    results = []
    for policy in POLICIES:
        r = run_backtest(hold, policy, service_level, series=[(store, item)]).iloc[0]
        results.append({
            "policy": policy,
            "achieved_cycle_service_level": round(float(r["cycle_service_level"]), 3),
            "order_cycles": int(r["n_cycles"]),
            "fill_rate": round(float(r["fill_rate"]), 4),
            "stockout_days": int(r["stockout_days"]),
            "avg_on_hand": round(float(r["avg_on_hand"]), 1),
            "total_cost": round(float(r["total_cost"]), 2),
        })
    return {"store": store, "item": item, "target_service_level": service_level,
            "results": results, "provenance": BACKTEST_PROVENANCE}


if __name__ == "__main__":
    mcp.run()
