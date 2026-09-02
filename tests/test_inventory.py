"""Unit tests for the inventory OR formulas (deterministic, hand-verifiable)."""
import math
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inventory import safety_stock, reorder_point, eoq, newsvendor_quantity


def test_eoq_matches_closed_form():
    # EOQ = sqrt(2 D S / H); D=1200, S=50, H=2 -> sqrt(60000) ~= 244.949
    assert math.isclose(eoq(1200, 50, 2), math.sqrt(60000), rel_tol=1e-9)


def test_safety_stock_uses_z_from_service_level():
    # At service level 0.95, z ~= 1.645. With error_std=10, lead_time=1 -> SS ~= 16.449
    ss = safety_stock(error_std=10, lead_time=1, service_level=0.95)
    assert math.isclose(ss, 1.6448536, rel_tol=1e-4) or math.isclose(ss, 16.448536, rel_tol=1e-4)


def test_safety_stock_scales_with_sqrt_lead_time():
    # Quadrupling lead time should double safety stock (sqrt relationship).
    ss1 = safety_stock(error_std=10, lead_time=1, service_level=0.95)
    ss4 = safety_stock(error_std=10, lead_time=4, service_level=0.95)
    assert math.isclose(ss4, 2 * ss1, rel_tol=1e-9)


def test_reorder_point_is_leadtime_demand_plus_safety_stock():
    ss = safety_stock(error_std=5, lead_time=7, service_level=0.95)
    rop = reorder_point(avg_daily_demand=20, lead_time=7, error_std=5, service_level=0.95)
    assert math.isclose(rop, 20 * 7 + ss, rel_tol=1e-9)


def test_newsvendor_symmetric_costs_gives_mean():
    # When underage == overage, critical ratio = 0.5, z = 0, so Q* == mean_demand.
    q = newsvendor_quantity(mean_demand=100, std_demand=15,
                            underage_cost=3, overage_cost=3)
    assert math.isclose(q, 100, abs_tol=1e-9)


def test_newsvendor_high_overage_orders_below_mean():
    # Expensive waste (overage) -> order conservatively, below mean.
    q = newsvendor_quantity(mean_demand=100, std_demand=15,
                            underage_cost=1, overage_cost=9)
    assert q < 100
