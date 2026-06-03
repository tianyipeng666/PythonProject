from __future__ import annotations

from statistics import mean

from .data_loader import PriceBar


def build_risk_report(bars: list[PriceBar], up_probability: float) -> dict:
    recent = bars[-20:]
    last = bars[-1]
    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]
    tips: list[str] = []
    score = 0

    ret_5 = _ret(closes, 5)
    ret_10 = _ret(closes, 10)
    range_1 = _safe_div(last.high - last.low, last.close)
    volume_ratio = _safe_div(last.volume, mean(volumes[-6:-1])) if len(volumes) >= 6 else 1.0
    drawdown_10 = _safe_div(last.close - max(closes[-10:]), max(closes[-10:]))

    if ret_5 > 0.12:
        score += 2
        tips.append("近5日涨幅较大，存在短线追高风险")
    if ret_10 > 0.20:
        score += 2
        tips.append("近10日涨幅过快，注意获利盘回吐")
    if ret_5 < -0.08 and volume_ratio > 1.5:
        score += 2
        tips.append("近5日下跌且成交量放大，可能有资金流出压力")
    if range_1 > 0.08:
        score += 1
        tips.append("单日振幅较大，短线波动风险偏高")
    if volume_ratio > 2.0:
        score += 1
        tips.append("成交量显著放大，需确认是否为异常放量")
    if drawdown_10 < -0.10:
        score += 1
        tips.append("价格相对近10日高点回撤较深，趋势可能转弱")
    if last.amount > 0 and last.amount < 50_000_000:
        score += 1
        tips.append("成交额偏低，流动性风险较高")
    if up_probability < 0.45:
        score += 1
        tips.append("模型给出的上涨概率偏低")
    if up_probability >= 0.65 and score >= 2:
        tips.append("上涨概率较高但风险项也较多，信号存在冲突")

    if not tips:
        tips.append("未触发明显短线风险规则，但仍需结合大盘和板块环境")

    level = "低"
    if score >= 4:
        level = "高"
    elif score >= 2:
        level = "中"

    return {
        "risk_level": level,
        "risk_score": score,
        "tips": tips,
        "metrics": {
            "ret_5": ret_5,
            "ret_10": ret_10,
            "range_1": range_1,
            "volume_ratio": volume_ratio,
            "drawdown_10": drawdown_10,
        },
    }


def build_risk_report_df(df, up_probability: float) -> dict:
    bars = [
        PriceBar(
            date=str(row["date"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            amount=float(row.get("amount", 0.0)),
        )
        for _, row in df.tail(40).iterrows()
    ]
    return build_risk_report(bars, up_probability)


def _ret(values: list[float], days: int) -> float:
    if len(values) <= days or values[-days - 1] == 0:
        return 0.0
    return (values[-1] - values[-days - 1]) / values[-days - 1]


def _safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b
