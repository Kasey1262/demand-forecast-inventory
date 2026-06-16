# Demand Forecasting & Inventory Optimization

零售/快消场景下的需求预测与库存优化项目。先用时间序列方法预测各门店各商品的销量，
再用运筹学库存模型（安全库存 / 再订货点 / EOQ / 报童模型）把预测转化为补货决策。

数据集：[Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only)
（10 门店 × 50 商品 × 5 年日销量，约 913,000 行）

## 项目结构

```
demand-forecast-inventory/
├── data/
│   ├── raw/          # 把 Kaggle 的 train.csv 放这里
│   └── processed/    # 特征工程后的中间结果
├── notebooks/
│   └── 01_eda.py     # Phase 1：探索性分析
├── src/
│   ├── data.py       # 数据加载 + 合成数据生成
│   ├── features.py   # Phase 2：特征工程
│   ├── models.py     # Phase 3：预测模型 + 评估
│   └── inventory.py  # Phase 4：库存优化（运筹学）
├── app.py            # Phase 5：Streamlit dashboard
├── outputs/          # EDA 图表
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 不需要真实数据也能跑：未找到 train.csv 时自动用合成数据
python notebooks/01_eda.py
```

拿到真实数据后：把 Kaggle 的 `train.csv` 放进 `data/raw/`，再次运行同一条命令即可。

### 下载真实数据
需要 Kaggle 账号。两种方式：
1. 网页下载 `train.csv` 手动放进 `data/raw/`
2. Kaggle API：`kaggle competitions download -c demand-forecasting-kernels-only`

## 阶段路线图

| Phase | 内容 | 状态 |
|-------|------|------|
| 1 | EDA：季节性分解、ADF 检验、周/年季节性 | ✅ 可运行 |
| 2 | 特征工程：日历 / lag / 滚动统计 | 🚧 脚手架 |
| 3 | 预测：seasonal naive baseline → 线性回归 → Prophet | 🚧 脚手架 |
| 4 | 库存优化：安全库存 / ROP / EOQ / 报童模型 | ✅ 公式已实现 |
| 5 | Streamlit dashboard | 🚧 占位 |

## 注意事项
- 时间序列**按时间切分** train/test，禁止随机切分（数据泄漏）。
- lag / rolling 特征必须按 `(store, item)` 分组，且 rolling 前先 `shift(1)`。
# demand-forecast-inventory
