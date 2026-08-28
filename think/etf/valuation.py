from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd

from think.etf.models import MarketMove, MetricReference


STATUS_BANDS = (
    (20.0, "明显低估"),
    (35.0, "偏低"),
    (65.0, "合理"),
    (80.0, "偏高"),
    (100.0, "明显高估"),
)


def metric_reference(frame: pd.DataFrame, history_years: int) -> MetricReference:
    if history_years < 1:
        raise ValueError("history_years 必须大于等于 1")
    if frame.empty:
        raise ValueError("估值历史为空")
    latest_date = frame["date"].max()
    cutoff = latest_date - timedelta(days=366 * history_years)
    sample = frame[frame["date"] >= cutoff].copy()
    if len(sample) < 20:
        sample = frame.copy()
    values = sample["value"].astype(float)
    current = float(values.iloc[-1])
    percentile = float((values <= current).mean() * 100.0)
    return MetricReference(
        current=current,
        percentile=percentile,
        q20=float(values.quantile(0.20)),
        median=float(values.quantile(0.50)),
        q80=float(values.quantile(0.80)),
        sample_count=len(sample),
        start_date=sample["date"].iloc[0].date().isoformat(),
        end_date=sample["date"].iloc[-1].date().isoformat(),
    )


def composite_percentile(pe: MetricReference, pb: MetricReference | None) -> float:
    # PE是所有标的共有的主指标；有PB时，用30%权重补充验证。
    return pe.percentile if pb is None else 0.70 * pe.percentile + 0.30 * pb.percentile


def absolute_yield_score(earnings_yield_spread: float) -> float:
    """Map the earnings-yield spread to the same 0-cheap/100-expensive scale."""
    if earnings_yield_spread >= 8.0:
        return 15.0
    if earnings_yield_spread >= 6.0:
        return 25.0
    if earnings_yield_spread >= 4.0:
        return 45.0
    if earnings_yield_spread >= 2.0:
        return 65.0
    if earnings_yield_spread >= 1.0:
        return 80.0
    return 95.0


def combined_valuation_score(relative_percentile: float, earnings_yield_spread: float) -> float:
    # 历史分位回答“和自己相比贵不贵”，股债差回答“绝对收益补偿够不够”。
    return 0.65 * relative_percentile + 0.35 * absolute_yield_score(
        earnings_yield_spread
    )


def classify_valuation(percentile: float) -> str:
    for upper, label in STATUS_BANDS:
        if percentile <= upper:
            return label
    return "明显高估"


def calculate_market_move(frame: pd.DataFrame) -> MarketMove:
    if frame.empty:
        raise ValueError("价格历史为空")
    close = frame["close"].astype(float).reset_index(drop=True)
    returns = close.pct_change().dropna()

    def period_return(days: int) -> float | None:
        if len(close) <= days:
            return None
        return float(close.iloc[-1] / close.iloc[-days - 1] - 1.0)

    def drawdown(days: int) -> float | None:
        if len(close) < 2:
            return None
        window = close.tail(min(days, len(close)))
        peak = float(window.max())
        return float(close.iloc[-1] / peak - 1.0) if peak > 0 else None

    volatility = None
    if len(returns) >= 10:
        volatility = float(returns.tail(20).std(ddof=1) * math.sqrt(244))
    return MarketMove(
        latest_date=frame["date"].iloc[-1].date().isoformat(),
        close=float(close.iloc[-1]),
        return_1d=period_return(1),
        return_5d=period_return(5),
        return_20d=period_return(20),
        drawdown_60d=drawdown(60),
        drawdown_250d=drawdown(250),
        volatility_20d=volatility,
    )
