from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexSpec:
    code: str
    name: str
    price_symbol: str
    etf_code: str
    target_weight: float
    pe_source: str
    pe_symbol: str
    pb_symbol: str | None = None
    official_valuation_code: str | None = None
    valuation_quality: str = "exact"
    valuation_note: str = ""


# target_weight 只是研究用的示例配置，不是针对个人情况的投资建议。
# 上证指数、沪深300和上证红利重叠较多，因此示例中上证指数权重较低。
INDEX_SPECS: dict[str, IndexSpec] = {
    "000001": IndexSpec(
        code="000001",
        name="上证指数",
        price_symbol="sh000001",
        etf_code="510210",
        target_weight=0.05,
        pe_source="market",
        pe_symbol="上证",
        official_valuation_code="000001",
        valuation_quality="proxy",
        valuation_note="历史PE使用上交所市场平均PE，和上证综指成分范围接近但口径并非完全相同。",
    ),
    "000300": IndexSpec(
        code="000300",
        name="沪深300",
        price_symbol="sh000300",
        etf_code="510300",
        target_weight=0.40,
        pe_source="index",
        pe_symbol="沪深300",
        pb_symbol="沪深300",
        official_valuation_code="000300",
    ),
    "000015": IndexSpec(
        code="000015",
        name="上证红利",
        price_symbol="sh000015",
        etf_code="510880",
        target_weight=0.20,
        pe_source="index",
        pe_symbol="上证红利",
        pb_symbol="上证红利",
        official_valuation_code="000015",
    ),
    "399006": IndexSpec(
        code="399006",
        name="创业板指",
        price_symbol="sz399006",
        etf_code="159915",
        target_weight=0.20,
        pe_source="market",
        pe_symbol="创业板",
        valuation_quality="proxy",
        valuation_note="历史PE使用整个创业板市场平均PE代理，不等同于创业板指100只成分股的官方PE。",
    ),
    "000698": IndexSpec(
        code="000698",
        name="科创100",
        price_symbol="sh000698",
        etf_code="588030",
        target_weight=0.15,
        pe_source="market",
        pe_symbol="科创版",
        official_valuation_code="000698",
        valuation_quality="proxy",
        valuation_note="历史分位使用科创板市场PE代理；当前指数PE优先展示中证指数官网口径。",
    ),
}


def get_specs(codes: list[str] | None = None) -> list[IndexSpec]:
    if not codes:
        return list(INDEX_SPECS.values())
    unknown = [code for code in codes if code not in INDEX_SPECS]
    if unknown:
        raise ValueError(f"不支持的指数代码: {', '.join(unknown)}")
    return [INDEX_SPECS[code] for code in codes]
