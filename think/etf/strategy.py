from __future__ import annotations

from dataclasses import replace

from think.etf.models import MarketMove, ValuationResult


BASE_MULTIPLIER = {
    "明显低估": 1.40,
    "偏低": 1.20,
    "合理": 1.00,
    "偏高": 0.75,
    "明显高估": 0.50,
}


def decision_text(item: ValuationResult) -> str:
    multiplier = item.suggested_multiplier
    if multiplier >= 1.20:
        return "是：定投并适度加仓"
    if multiplier >= 0.90:
        return "是：按正常金额定投"
    if multiplier >= 0.60:
        return "是：降低金额定投"
    return "是：仅小额谨慎定投"


def contribution_multiplier(status: str, market: MarketMove) -> tuple[float, float, str]:
    base = BASE_MULTIPLIER[status]
    candidates = [0.0]
    reasons: list[str] = []

    if market.drawdown_60d is not None:
        dd = market.drawdown_60d
        if dd <= -0.18:
            candidates.append(0.60)
            reasons.append("距60日高点回撤至少18%")
        elif dd <= -0.12:
            candidates.append(0.40)
            reasons.append("距60日高点回撤至少12%")
        elif dd <= -0.08:
            candidates.append(0.25)
            reasons.append("距60日高点回撤至少8%")
        elif dd <= -0.05:
            candidates.append(0.10)
            reasons.append("距60日高点回撤至少5%")

    if market.return_5d is not None:
        ret5 = market.return_5d
        if ret5 <= -0.12:
            candidates.append(0.60)
            reasons.append("近5日下跌至少12%")
        elif ret5 <= -0.08:
            candidates.append(0.40)
            reasons.append("近5日下跌至少8%")
        elif ret5 <= -0.05:
            candidates.append(0.25)
            reasons.append("近5日下跌至少5%")

    if market.return_1d is not None and market.return_1d <= -0.07:
        candidates.append(0.25)
        reasons.append("单日下跌至少7%")

    dip_bonus = max(candidates)
    if status in {"偏高", "明显高估"}:
        dip_bonus = 0.0
        reasons = ["估值偏高，大跌不自动触发额外买入"]
    elif status == "合理":
        dip_bonus = min(dip_bonus, 0.25)
    suggested = min(2.0, base + dip_bonus)
    reason = "；".join(dict.fromkeys(reasons)) if reasons else "按估值执行常规定投"
    return base, dip_bonus, reason


def apply_budget(
    results: list[ValuationResult],
    weekly_budget: float | None,
    max_portfolio_multiplier: float,
) -> list[ValuationResult]:
    if weekly_budget is None:
        return results
    if weekly_budget <= 0:
        raise ValueError("weekly_budget 必须大于 0")
    if max_portfolio_multiplier < 1.0 or max_portfolio_multiplier > 3.0:
        raise ValueError("max_portfolio_multiplier 应在 1 到 3 之间")

    total_weight = sum(item.target_weight for item in results)
    if total_weight <= 0:
        raise ValueError("目标权重总和必须大于 0")
    provisional: list[tuple[ValuationResult, float, float]] = []
    for item in results:
        normalized_weight = item.target_weight / total_weight
        base_amount = weekly_budget * normalized_weight
        suggested = base_amount * item.suggested_multiplier
        provisional.append((item, base_amount, suggested))

    cap = weekly_budget * max_portfolio_multiplier
    total_suggested = sum(row[2] for row in provisional)
    scale = min(1.0, cap / total_suggested) if total_suggested > 0 else 1.0
    return [
        replace(item, base_amount=base, suggested_amount=suggested * scale)
        for item, base, suggested in provisional
    ]
