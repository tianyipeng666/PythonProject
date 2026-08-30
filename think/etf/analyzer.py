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
    # 历史分位必须保持统一历史口径；当前绝对估值优先采用指数公司官方PE。
    analysis_pe = official.pe if official and official.pe and official.pe > 0 else pe.current
    analysis_pe_source = official.source if official and official.pe else provider.history_source(spec.code)
    earnings_yield = 100.0 / analysis_pe
    earnings_yield_spread = earnings_yield - risk_free_rate
    composite = combined_valuation_score(relative_percentile, earnings_yield_spread)
    status = classify_valuation(composite)
    base, dip_bonus, reason = contribution_multiplier(status, market)

    history_quality = provider.history_quality(spec.code)
    history_source = provider.history_source(spec.code)
    current_is_official = bool(official and official.pe)
    if history_quality == "exact":
        confidence = "高（历史精确，当前官方）" if current_is_official else "高（历史精确）"
    elif history_quality == "index":
        confidence = "较高（历史指数级，当前官方）" if current_is_official else "较高（历史指数级）"
    elif current_is_official:
        confidence = "中等（当前官方，历史代理）"
    else:
        confidence = "较低（历史代理）"
    notes = [spec.valuation_note] if spec.valuation_note else []
    if provider.using_exact(spec.code):
        notes = [f"历史估值使用数据库精确来源：{provider.exact_source(spec.code)}"]
    if official and official.pe:
        notes.append(
            f"指数公司官方快照 {official.date}: PE={official.pe:.2f}；"
            "用于当前盈利收益率和股债差；历史分位仍使用统一历史源。"
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
        analysis_pe=analysis_pe,
        analysis_pe_source=analysis_pe_source or "unknown",
        history_source=history_source,
        history_quality=history_quality,
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
