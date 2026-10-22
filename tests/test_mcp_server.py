"""Smoke test: the MCP server registers its tools and validates input (in-process)."""
import asyncio
import json
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

pytest.importorskip("mcp")
from mcp import Client

from agent.mcp_server import mcp


def test_server_lists_all_tools():
    async def go():
        async with Client(mcp) as c:
            return {t.name for t in (await c.list_tools()).tools}
    assert asyncio.run(go()) == {
        "get_forecast", "get_inventory_policy",
        "check_stockout_risk", "run_inventory_backtest",
    }


def test_backtest_rejects_bad_service_level_without_training():
    async def go():
        async with Client(mcp) as c:
            r = await c.call_tool("run_inventory_backtest",
                                  {"store": 1, "item": 1, "service_level": 1.5})
            return json.loads(r.content[0].text)
    assert "error" in asyncio.run(go())
