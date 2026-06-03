from __future__ import annotations

import numpy as np


FEATURE_COLUMNS = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ma5_gap",
    "ma10_gap",
    "ma20_gap",
    "ma60_gap",
    "vol_ratio_5",
    "vol_ratio_20",
    "range_1",
    "volatility_5",
    "volatility_20",
    "drawdown_10",
    "drawdown_20",
    "amount_log",
    "rsi_6",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
]


def build_dataset(df, horizon: int = 1):
    data = df.copy()
    close = data["close"]
    volume = data["volume"]

    for window in (1, 2, 3, 5, 10, 20):
        data[f"ret_{window}"] = close.pct_change(window)
    for window in (5, 10, 20, 60):
        ma = close.rolling(window).mean()
        data[f"ma{window}_gap"] = close / ma - 1.0
    data["vol_ratio_5"] = volume / volume.rolling(5).mean()
    data["vol_ratio_20"] = volume / volume.rolling(20).mean()
    data["range_1"] = (data["high"] - data["low"]) / close
    data["volatility_5"] = close.pct_change().rolling(5).std()
    data["volatility_20"] = close.pct_change().rolling(20).std()
    data["drawdown_10"] = close / close.rolling(10).max() - 1.0
    data["drawdown_20"] = close / close.rolling(20).max() - 1.0
    data["amount_log"] = np.log1p(data["amount"].clip(lower=0))
    data["rsi_6"] = _rsi(close, 6)
    data["rsi_14"] = _rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    data["future_return"] = close.shift(-horizon) / close - 1.0
    data["label"] = (data["future_return"] > 0).astype(int)
    data.loc[data["future_return"].isna(), "label"] = np.nan

    feature_df = data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return feature_df


def latest_feature_frame(df, horizon: int = 1):
    return build_dataset(df, horizon=horizon).tail(1)


def _rsi(close, window: int):
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(window).mean()
    loss = (-diff.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

