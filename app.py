"""
Step 6: Streamlit Dashboard

Run from the project root (NOTE: 'streamlit run', not 'python3'):
    streamlit run app.py

Pick a store + item to see the demand forecast vs actuals and the recommended
inventory policy. The service-level slider recomputes safety stock live.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_sales
from src.features import build_feature_matrix
from src.models import time_split, linear_regression_predict
from src.inventory import build_inventory_policy, safety_stock, reorder_point

LEAD_TIME = 7

st.set_page_config(page_title="Demand Forecasting & Inventory", layout="wide")


# Heavy work runs ONCE, then is cached. Without this, every click re-runs the
# whole 30s pipeline (load -> features -> train -> predict).
@st.cache_data(show_spinner="Building model (first load only, ~30s)...")
def prepare():
    df = load_sales()
    feat = build_feature_matrix(df)
    train, test = time_split(feat)
    _, preds = linear_regression_predict(train, test)
    test = test.copy()
    test["pred"] = preds.round(1)
    policy = build_inventory_policy(test, preds)
    return df, test, policy


df, test, policy = prepare()

st.title("Demand Forecasting & Inventory Optimization")
st.caption("Forecast-driven reorder points for 500 store-item combinations")

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.header("Controls")
store = st.sidebar.selectbox("Store", sorted(df["store"].unique()))
item = st.sidebar.selectbox("Item", sorted(df["item"].unique()))
service_level = st.sidebar.slider("Service level", 0.80, 0.99, 0.95, 0.01)

# ── Forecast vs actual chart ─────────────────────────────────────────────────
sub = test[(test["store"] == store) & (test["item"] == item)].sort_values("date")

st.subheader(f"Forecast vs actual — store {store}, item {item}")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(sub["date"], sub["sales"], label="Actual", color="#185FA5", lw=1.2)
ax.plot(sub["date"], sub["pred"], label="Forecast", color="#D85A30", lw=1.2, ls="--")
ax.set_xlabel("date"); ax.set_ylabel("sales"); ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

# ── Inventory policy (recomputed live from the slider) ───────────────────────
row = policy[(policy["store"] == store) & (policy["item"] == item)].iloc[0]
avg = row["avg_daily_demand"]
err = row["forecast_error_std"]
ss = safety_stock(err, LEAD_TIME, service_level)
rop = reorder_point(avg, LEAD_TIME, err, service_level)

st.subheader(f"Recommended inventory policy (service level = {service_level:.0%})")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg daily demand", f"{avg:.1f}")
c2.metric("Safety stock", f"{ss:.1f}")
c3.metric("Reorder point", f"{rop:.1f}")
c4.metric("EOQ (order size)", f"{row['eoq']:.0f}")

st.info(
    f"Reorder **item {item}** at **store {store}** when stock drops to "
    f"**{rop:.0f} units**; order **{row['eoq']:.0f} units** each time. "
    f"Higher service level → more safety stock → fewer stockouts, higher holding cost."
)

# ── Full policy table ────────────────────────────────────────────────────────
with st.expander("See full 500-row inventory policy table"):
    st.dataframe(policy, use_container_width=True)
