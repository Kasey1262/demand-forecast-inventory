# Demand Forecasting & Inventory Optimization

Forecasts daily sales for each store-item, then converts those forecasts into
operations-research inventory decisions (safety stock, reorder point, EOQ,
newsvendor) for 500 store-item combinations.

Dataset: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only)
(10 stores x 50 items x 5 years of daily sales, ~913,000 rows)

## Pipeline

```
data  ->  features  ->  forecast  ->  inventory policy  ->  dashboard
```

## Project structure

```
demand-forecast-inventory/
├── data/raw/            # put Kaggle train.csv here (auto-falls back to synthetic)
├── notebooks/01_eda.py  # Step 2: EDA (seasonal decomposition, ADF test)
├── src/
│   ├── data.py          # Step 1: load real data or generate synthetic
│   ├── features.py      # Step 3: calendar / lag / rolling features
│   ├── models.py        # Step 4: baseline + linear regression, metrics
│   └── inventory.py     # Step 5: safety stock / ROP / EOQ / newsvendor
├── app.py               # Step 6: Streamlit dashboard
└── requirements.txt
```

## Quick start

```bash
pip3 install -r requirements.txt

python3 notebooks/01_eda.py   # exploratory analysis -> figures in outputs/
python3 src/models.py         # forecast: compare baseline vs linear regression
python3 src/inventory.py      # 500-row inventory policy table
streamlit run app.py          # interactive dashboard
```

Works on synthetic data out of the box. For real data, drop the Kaggle
`train.csv` into `data/raw/` and rerun — `load_sales()` switches automatically.

## Key results (real Kaggle data)

Time-based split at 2017-10-01; test = last 3 months of 2017
(46,000 rows across 500 store-item series).

| Model | MAE | RMSE | SMAPE |
|-------|-------|--------|-------|
| Seasonal Naive (baseline) | 9.076 | 12.051 | 18.56 |
| Linear Regression | 6.996 | 9.167 | 14.82 |

Linear regression beats the seasonal-naive baseline by ~23% on MAE
(6.996 vs 9.076).

> Numbers above are on the real Kaggle `train.csv`. 
> pipeline falls back to synthetic data, which gives different results.

## AI Agent (natural-language interface)

An LLM agent (Claude, via tool use/function calling) that operates the
forecasting pipeline through conversation. The model decides which tool to
call and with what arguments — it is not a fixed script.

Three tools wrap the pipeline's real functions:

| Tool | What it does |
|------|--------------|
| `get_forecast` | Forecasted average daily demand + error std for a store-item |
| `get_inventory_policy` | Safety stock / reorder point / EOQ / newsvendor qty at a chosen service level |
| `check_stockout_risk` | Probability that demand exceeds current stock over a horizon |

The model is trained once at startup and held in memory, so tools answer
instantly. Example session:

## Notes/design decisions
- Time-based train/test split (never random — that leaks the future).
- Lag/rolling features computed within each (store, item) group.
- Rolling features shifted by 1 to exclude the current day (no leakage).
- Safety stock uses the std of forecast ERROR, so a better model directly
  lowers required inventory.
