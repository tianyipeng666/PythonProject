from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from .data_loader import PriceBar


FEATURE_NAMES = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ma5_gap",
    "ma10_gap",
    "ma20_gap",
    "vol_ratio_5",
    "range_1",
    "volatility_5",
    "drawdown_10",
    "turnover_amount_log",
]


@dataclass(frozen=True)
class FeatureRow:
    date: str
    features: list[float]
    close: float
    future_return: float | None
    label: int | None


def build_feature_rows(bars: list[PriceBar], horizon: int = 3) -> list[FeatureRow]:
    if len(bars) < 35 + horizon:
        raise ValueError("Not enough price bars. Need at least 35 rows plus forecast horizon.")

    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]
    rows: list[FeatureRow] = []
    start = 20
    end = len(bars)

    for i in range(start, end):
        close = closes[i]
        features = [
            _return(closes, i, 1),
            _return(closes, i, 3),
            _return(closes, i, 5),
            _return(closes, i, 10),
            _ma_gap(closes, i, 5),
            _ma_gap(closes, i, 10),
            _ma_gap(closes, i, 20),
            _safe_div(volumes[i], mean(volumes[i - 5 : i])),
            _safe_div(bars[i].high - bars[i].low, close),
            _volatility(closes, i, 5),
            _drawdown(closes, i, 10),
            _logish(bars[i].amount if bars[i].amount > 0 else bars[i].volume * close),
        ]
        if i + horizon < len(bars):
            future_return = _safe_div(closes[i + horizon] - close, close)
            label = 1 if future_return > 0 else 0
        else:
            future_return = None
            label = None
        rows.append(
            FeatureRow(
                date=bars[i].date,
                features=features,
                close=close,
                future_return=future_return,
                label=label,
            )
        )
    return rows


def _return(values: list[float], i: int, window: int) -> float:
    return _safe_div(values[i] - values[i - window], values[i - window])


def _ma_gap(values: list[float], i: int, window: int) -> float:
    ma = mean(values[i - window + 1 : i + 1])
    return _safe_div(values[i] - ma, ma)


def _volatility(values: list[float], i: int, window: int) -> float:
    returns = [_return(values, j, 1) for j in range(i - window + 1, i + 1)]
    return pstdev(returns) if len(returns) > 1 else 0.0


def _drawdown(values: list[float], i: int, window: int) -> float:
    recent_high = max(values[i - window + 1 : i + 1])
    return _safe_div(values[i] - recent_high, recent_high)


def _safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def _logish(value: float) -> float:
    x = max(value, 0.0)
    scale = 1.0
    while x >= 10.0:
        x /= 10.0
        scale += 1.0
    return scale + x / 10.0

