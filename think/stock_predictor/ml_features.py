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
    "body_1",
    "upper_shadow_1",
    "lower_shadow_1",
    "gap_1",
    "up_days_5",
    "up_days_10",
    "new_high_20",
    "new_low_20",
    "volume_price_corr_10",
    "volume_price_corr_20",
    "trend_strength_20",
    "volatility_ratio_5_20",
    "range_rank_20",
]


def build_dataset(
    df,
    horizon: int = 1,
    target_return: float = 0.0,
    context_frames: dict[str, object] | None = None,
):
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
    data["body_1"] = (data["close"] - data["open"]) / close
    data["upper_shadow_1"] = (data["high"] - data[["open", "close"]].max(axis=1)) / close
    data["lower_shadow_1"] = (data[["open", "close"]].min(axis=1) - data["low"]) / close
    data["gap_1"] = data["open"] / close.shift(1) - 1.0
    data["up_days_5"] = (close.pct_change() > 0).rolling(5).mean()
    data["up_days_10"] = (close.pct_change() > 0).rolling(10).mean()
    data["new_high_20"] = (close >= close.rolling(20).max()).astype(float)
    data["new_low_20"] = (close <= close.rolling(20).min()).astype(float)
    data["volume_price_corr_10"] = close.pct_change().rolling(10).corr(volume.pct_change())
    data["volume_price_corr_20"] = close.pct_change().rolling(20).corr(volume.pct_change())
    data["trend_strength_20"] = close / close.rolling(20).mean() - close / close.rolling(60).mean()
    data["volatility_ratio_5_20"] = data["volatility_5"] / data["volatility_20"]
    data["range_rank_20"] = data["range_1"].rolling(20).rank(pct=True)

    feature_columns = list(FEATURE_COLUMNS)
    if context_frames:
        data, context_columns = _add_context_features(data, context_frames)
        feature_columns.extend(context_columns)

    zero_volume_columns = [
        "vol_ratio_5",
        "vol_ratio_20",
        "volume_price_corr_10",
        "volume_price_corr_20",
        "volatility_ratio_5_20",
    ]
    data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    for col in zero_volume_columns:
        if col in data.columns:
            data[col] = data[col].fillna(0.0)

    data["future_return"] = close.shift(-horizon) / close - 1.0
    data["label"] = (data["future_return"] > target_return).astype(int)
    data.loc[data["future_return"].isna(), "label"] = np.nan

    feature_df = data.dropna(subset=feature_columns).reset_index(drop=True)
    feature_df.attrs["feature_columns"] = feature_columns
    feature_df.attrs["target_return"] = target_return
    return feature_df


def latest_feature_frame(
    df,
    horizon: int = 1,
    target_return: float = 0.0,
    context_frames: dict[str, object] | None = None,
):
    return build_dataset(
        df,
        horizon=horizon,
        target_return=target_return,
        context_frames=context_frames,
    ).tail(1)


def _rsi(close, window: int):
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(window).mean()
    loss = (-diff.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _add_context_features(data, context_frames: dict[str, object]):
    merged = data.copy()
    merged["date"] = merged["date"].astype(str)
    added_columns: list[str] = []
    for raw_name, raw_context in context_frames.items():
        name = _safe_prefix(raw_name)
        context = raw_context.copy()
        context["date"] = context["date"].astype(str)
        context_close = context["close"]
        context_volume = context["volume"]
        context_feature_columns = []
        for window in (1, 3, 5, 10, 20):
            col = f"{name}_ret_{window}"
            context[col] = context_close.pct_change(window)
            context_feature_columns.append(col)
        context[f"{name}_ma20_gap"] = context_close / context_close.rolling(20).mean() - 1.0
        context[f"{name}_volatility_20"] = context_close.pct_change().rolling(20).std()
        context[f"{name}_volume_ratio_20"] = context_volume / context_volume.rolling(20).mean()
        context_feature_columns.extend(
            [f"{name}_ma20_gap", f"{name}_volatility_20", f"{name}_volume_ratio_20"]
        )
        keep = ["date"] + context_feature_columns
        merged = merged.merge(context[keep], on="date", how="left")
        added_columns.extend(context_feature_columns)
    return merged, added_columns


def _safe_prefix(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    out = "_".join(part for part in out.split("_") if part)
    return out or "context"

