# Stock Predictor Thread Summary

Generated: 2026-06-03 17:58:26 CST

## Goal

Build a personal-use A-share short-term prediction tool under:

```text
/Users/tianyipeng/PythonProjects/PythonProject/think
```

Scope:

- Personal use only
- A-share stocks and indexes
- Short-term direction prediction
- Output up/down probability, risk tips, and estimated return
- Use AkShare, pandas, scikit-learn, and Streamlit

## Dependencies Installed

Installed with:

```bash
python -m pip install akshare pandas scikit-learn streamlit plotly joblib
```

Verified versions:

```text
akshare 1.18.64
pandas 3.0.3
sklearn 1.9.0
streamlit 1.58.0
plotly 6.7.0
```

Runtime warning observed but not blocking:

```text
RequestsDependencyWarning: urllib3 (2.5.0) or chardet (7.4.1)/charset_normalizer (3.4.4) doesn't match a supported version
```

## Files Created Or Updated

Main files:

```text
think/app.py
think/streamlit_app.py
think/README_stock_predictor.md
think/requirements_stock_predictor.txt
think/stock_predictor/__init__.py
think/stock_predictor/cli.py
think/stock_predictor/data_loader.py
think/stock_predictor/ml_features.py
think/stock_predictor/sklearn_model.py
think/stock_predictor/streamlit_app.py
think/stock_predictor/risk.py
think/stock_predictor/backtest.py
think/stock_predictor/features.py
think/stock_predictor/model.py
think/stock_predictor/demo_data.py
```

Notes:

- `ml_features.py` and `sklearn_model.py` are the current main sklearn/pandas pipeline.
- `features.py`, `model.py`, and `backtest.py` are the earlier lightweight numpy-only pipeline, kept for fallback/demo use.
- `README_stock_predictor.md` contains current user-facing usage docs.

## Current Implementation

Data:

- Uses AkShare to fetch daily bars.
- Stock data uses `stock_zh_a_hist`.
- Index data tries:
  - `stock_zh_index_daily_em`
  - `stock_zh_index_daily`
  - `index_zh_a_hist`
- `科创100`, `kcb100`, `kc100`, and `sh000698` are supported aliases for the index.
- There is a best-effort index spot append fallback, but AkShare spot data did not match 科创100 in testing, so it safely falls back to historical daily bars.

Features:

- 1/2/3/5/10/20 day returns
- 5/10/20/60 day moving average gaps
- 5/20 day volume ratios
- intraday range
- 5/20 day volatility
- 10/20 day drawdown
- amount log
- RSI 6/14
- MACD, MACD signal, MACD histogram

Models:

- Classification model predicts up probability.
- Regression model predicts future return.
- `model-type logistic` means:
  - classifier: `LogisticRegression`
  - regressor: `Ridge`
- `model-type random_forest` means:
  - classifier: `RandomForestClassifier`
  - regressor: `RandomForestRegressor`

Outputs:

- Direction: up/down
- Up probability
- Predicted return
- Predicted close
- Risk level and risk tips
- Backtest accuracy
- Signal win rate at threshold
- Return MAE

## Important Logic Fix

Earlier, `week-predict` incorrectly labeled target dates using the current system date. This was wrong when AkShare historical data lagged.

Fix made:

- `week-predict` now labels predicted dates from the latest data date.
- Added `--as-of YYYY-MM-DD` to truncate data for historical replay/backtesting.

Example:

```bash
python -m think.stock_predictor.cli week-predict \
  --symbol 科创100 \
  --asset-type index \
  --model-type logistic \
  --as-of 2026-06-02
```

This means: use data up to 2026-06-02 only, then predict the following trading days.

## Verified Commands

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/stock_predictor_pycache \
python -m py_compile think/stock_predictor/sklearn_model.py think/stock_predictor/cli.py think/stock_predictor/streamlit_app.py
```

Historical replay for 科创100:

```bash
python -m think.stock_predictor.cli week-predict \
  --symbol 科创100 \
  --asset-type index \
  --model-type logistic \
  --as-of 2026-06-02
```

Observed output:

```text
Symbol: 科创100
Latest data date: 2026-06-02
Current date: 2026-06-03
As-of cutoff: 2026-06-02

2026-06-03 forecast horizon=1: 上涨
Up probability: 67.02%
Predicted return: 1.03%
Predicted close: 1870.62
Risk level: 低 (score=0)
Backtest accuracy: 48.22%
Signal win rate at 55.00%: 53.85%
Return MAE: 1.59%
Risk tips:
- 未触发明显短线风险规则，但仍需结合大盘和板块环境

2026-06-04 forecast horizon=2: 上涨
Up probability: 60.17%
Predicted return: 1.62%
Predicted close: 1881.50
Risk level: 低 (score=0)
Backtest accuracy: 43.29%
Signal win rate at 55.00%: 57.63%
Return MAE: 2.21%
Risk tips:
- 未触发明显短线风险规则，但仍需结合大盘和板块环境

2026-06-05 forecast horizon=3: 上涨
Up probability: 57.01%
Predicted return: 1.55%
Predicted close: 1880.21
Risk level: 低 (score=0)
Backtest accuracy: 48.90%
Signal win rate at 55.00%: 60.34%
Return MAE: 2.54%
Risk tips:
- 未触发明显短线风险规则，但仍需结合大盘和板块环境
```

After AkShare later returned data through 2026-06-03, current prediction became:

```text
Latest data date: 2026-06-03

2026-06-04 forecast horizon=1: 上涨
Up probability: 61.80%
Risk level: 低
Backtest accuracy: 49.59%
Signal win rate at 55.00%: 55.32%

2026-06-05 forecast horizon=2: 上涨
Up probability: 55.26%
Risk level: 低
Backtest accuracy: 44.11%
Signal win rate at 55.00%: 56.67%
```

The exact current output may drift as AkShare data updates.

## How To Use

Fetch data:

```bash
python -m think.stock_predictor.cli fetch \
  --symbol 科创100 \
  --asset-type index \
  --output think/data/kc100.csv
```

Train:

```bash
python -m think.stock_predictor.cli train \
  --csv think/data/kc100.csv \
  --horizon 1 \
  --model think/models/kc100_h1.joblib \
  --model-type logistic
```

Predict:

```bash
python -m think.stock_predictor.cli predict \
  --csv think/data/kc100.csv \
  --model think/models/kc100_h1.joblib
```

One-command fetch/train/predict/backtest:

```bash
python -m think.stock_predictor.cli quick \
  --symbol 科创100 \
  --asset-type index \
  --horizon 1 \
  --model-type logistic \
  --threshold 0.55 \
  --csv-output think/data/kc100.csv
```

Week prediction:

```bash
python -m think.stock_predictor.cli week-predict \
  --symbol 科创100 \
  --asset-type index \
  --model-type logistic
```

Streamlit:

```bash
streamlit run think/streamlit_app.py
```

## Known Caveats

- The tool is a research assistant, not investment advice.
- Current model uses only daily technical features.
- It does not yet include:
  - constituent stock behavior
  - sector/market regime
  - macro/news/announcements
  - intraday data
  - transaction costs
  - walk-forward retraining
  - robust out-of-sample validation
- Backtest accuracy is weak, often around 48-50%.
- Signal win rate is only slightly above random in current 科创100 tests.
- Return prediction MAE is roughly 1.5-2.5 percentage points in observed tests, so predicted return should be treated as a rough estimate.

## Suggested Next Steps

1. Add a proper walk-forward validation command.
2. Add benchmark comparison against simple rules such as "tomorrow same as today" and moving average trend.
3. Add market context features from 上证指数/科创50/沪深300/创业板指.
4. Add 科创100 constituent aggregation if AkShare source is stable.
5. Persist fetched data locally and avoid repeated remote calls.
6. Fix or pin the `requests/urllib3/chardet` warning if it becomes noisy.
7. Add tests for:
   - date labeling
   - `--as-of` cutoff
   - predicted return output
   - model save/load compatibility

