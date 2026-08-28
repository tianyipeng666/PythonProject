from __future__ import annotations

from think.etf.config import IndexSpec
from think.etf.data import DataProvider
from think.etf.models import ValuationResult
from think.etf.strategy import contribution_multiplier
from think.etf.valuation import (
    calculate_market_move,
    classify_valuation,
    combined_valuation_score,
    composite_percentile,
    metric_reference,
)


def analyze_index(
    spec: IndexSpec,
    provider: DataProvider,
    history_years: int,
    risk_free_rate: float,
) -> ValuationResult:
    official = provider.official_snapshot(spec)
    pe = metric_reference(provider.pe_history(spec), history_years)
    pb_frame = provider.pb_history(spec)
    pb = metric_reference(pb_frame, history_years) if pb_frame is not None else None
    market = calculate_market_move(provider.price_history(spec))
    relative_percentile = composite_percentile(pe, pb)
    earnings_yield = 100.0 / pe.current
    earnings_yield_spread = earnings_yield - risk_free_rate
    composite = combined_valuation_score(relative_percentile, earnings_yield_spread)
    status = classify_valuation(composite)
    base, dip_bonus, reason = contribution_multiplier(status, market)

    if provider.using_exact(spec.code):
        confidence = "精确（用户CSV）"
    else:
        confidence = "较高" if spec.valuation_quality == "exact" else "一般（代理）"
    notes = [spec.valuation_note] if spec.valuation_note else []
    if provider.using_exact(spec.code):
        notes = [f"历史估值使用数据库精确来源：{provider.exact_source(spec.code)}"]
    if official and official.pe:
        notes.append(
            f"中证指数快照 {official.date}: PE={official.pe:.2f}；"
            "快照与历史源口径/日期可能不同，仅作交叉核对。"
        )
    if spec.code in {"000001", "000300", "000015"}:
        notes.append("这些指数相互有较多成分股重叠，组合权重不能简单视为完全分散。")

    return ValuationResult(
        code=spec.code,
        name=spec.name,
        etf_code=spec.etf_code,
        status=status,
        confidence=confidence,
        pe=pe,
        pb=pb,
        official_pe=official.pe if official else None,
        official_dividend_yield=official.dividend_yield if official else None,
        official_date=official.date if official else None,
        earnings_yield=earnings_yield,
        risk_free_rate=risk_free_rate,
        earnings_yield_spread=earnings_yield_spread,
        composite_percentile=composite,
        market=market,
        base_multiplier=base,
        dip_bonus=dip_bonus,
        suggested_multiplier=min(2.0, base + dip_bonus),
        reason=reason,
        note=" ".join(notes),
        target_weight=spec.target_weight,
    )
