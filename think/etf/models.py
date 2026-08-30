from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MetricReference:
    current: float
    percentile: float
    q20: float
    median: float
    q80: float
    sample_count: int
    start_date: str
    end_date: str


@dataclass(frozen=True)
class MarketMove:
    latest_date: str
    close: float
    return_1d: float | None
    return_5d: float | None
    return_20d: float | None
    drawdown_60d: float | None
    drawdown_250d: float | None
    volatility_20d: float | None


@dataclass(frozen=True)
class ValuationResult:
    code: str
    name: str
    etf_code: str
    status: str
    confidence: str
    pe: MetricReference
    pb: MetricReference | None
    official_pe: float | None
    official_dividend_yield: float | None
    official_date: str | None
    analysis_pe: float
    analysis_pe_source: str
    history_source: str
    history_quality: str
    earnings_yield: float
    risk_free_rate: float
    earnings_yield_spread: float
    composite_percentile: float
    market: MarketMove
    base_multiplier: float
    dip_bonus: float
    suggested_multiplier: float
    reason: str
    note: str
    target_weight: float
    base_amount: float | None = None
    suggested_amount: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

