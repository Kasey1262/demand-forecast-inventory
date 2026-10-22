"""Tests for the (s, Q) policy simulator and the undershoot-corrected ROP."""
import math
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.backtest import simulate_sq_policy, lead_time_error_std
from src.inventory import undershoot_moments, reorder_point_with_undershoot


def test_constant_demand_with_enough_stock_never_stocks_out():
    r = simulate_sq_policy([10] * 60, reorder_points=40, order_qty=50,
                           lead_time=3, initial_on_hand=90)
    assert r["lost_units"] == 0
    assert r["fill_rate"] == 1.0
    assert r["cycle_service_level"] == 1.0
    assert r["n_orders"] > 0


def test_units_are_conserved():
    rng = np.random.default_rng(1)
    d = rng.poisson(20, 100)
    r = simulate_sq_policy(d, reorder_points=100, order_qty=150,
                           lead_time=5, initial_on_hand=50)
    assert math.isclose(r["total_demand"], d.sum())
    assert r["lost_units"] <= r["total_demand"]
    assert 0 <= r["fill_rate"] <= 1


def test_order_arrives_after_lead_time_and_shortfall_is_lost():
    # Day 0: serve 10 -> 10 left, position 10 <= s=20 -> order 30 (arrives end of day 2)
    # Day 1: serve 10 -> 0 left.  Day 2: demand 10 is LOST, then the 30 arrive.
    r = simulate_sq_policy([10, 10, 10], reorder_points=20, order_qty=30,
                           lead_time=2, initial_on_hand=20)
    assert r["n_orders"] == 1
    assert r["lost_units"] == 10
    assert r["stockout_days"] == 1
    assert r["cycle_service_level"] == 0.0


def test_zero_demand_places_no_orders():
    r = simulate_sq_policy([0] * 30, reorder_points=5, order_qty=10,
                           lead_time=2, initial_on_hand=20)
    assert r["n_orders"] == 0
    assert r["fill_rate"] == 1.0


def test_iid_sigma_uses_sqrt_lead_time():
    errs = np.array([-2.0, 1.0, 3.0, -1.0, 0.5])
    expected = np.std(errs, ddof=1) * math.sqrt(4)
    assert math.isclose(lead_time_error_std(errs, None, 4, "iid"), expected)


def test_undershoot_of_constant_demand_is_uniform_on_0_d():
    # Constant demand d -> undershoot ~ Uniform(0, d): mean d/2, var d^2/12.
    mean_u, var_u = undershoot_moments([30] * 100)
    assert math.isclose(mean_u, 15)
    assert math.isclose(var_u, 30 ** 2 / 12)


def test_undershoot_corrected_rop_hits_target_service_level():
    # Known iid demand, perfect forecast: the corrected ROP should deliver the
    # promised cycle service level (the plain textbook ROP falls well short).
    rng = np.random.default_rng(0)
    mu, sd, L, target = 50, 10, 7, 0.95
    sample = np.clip(rng.normal(mu, sd, 50_000), 0, None)
    s = reorder_point_with_undershoot(mu * L, sd * math.sqrt(L), sample, target)
    ok = n = 0
    for _ in range(100):
        d = np.clip(rng.normal(mu, sd, 400), 0, None)
        r = simulate_sq_policy(d, s, mu * 20, L, s + mu * 20)
        ok += r["n_cycles_ok"]
        n += r["n_cycles"]
    assert abs(ok / n - target) < 0.03
