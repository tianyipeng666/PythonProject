from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FundInfo:
    symbol: str
    name: str
    fund_type: str
    source: str


def fetch_fund_daily_df(symbol: str):
    import akshare as ak

    code = symbol.strip().zfill(6)
    info_map = fetch_fund_info_map()
    name, fund_type = info_map.get(code, (code, "fund"))
    attempts = [
        (
            "etf",
            lambda: _normalize_exchange_df(
                ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date="19700101",
                    end_date="22220101",
                    adjust="",
                )
            ),
        ),
        (
            "lof",
            lambda: _normalize_exchange_df(
                ak.fund_lof_hist_em(
                    symbol=code,
                    period="daily",
                    start_date="19700101",
                    end_date="22220101",
                    adjust="",
                )
            ),
        ),
        (
            "open_fund_nav",
            lambda: _normalize_nav_df(
                ak.fund_open_fund_info_em(
                    symbol=code,
                    indicator="单位净值走势",
                    period="成立来",
                )
            ),
        ),
    ]
    errors: list[str] = []
    for source, fetcher in attempts:
        try:
            df = fetcher()
            if len(df) >= 80:
                return (
                    df.sort_values("date").drop_duplicates("date").reset_index(drop=True),
                    FundInfo(symbol=code, name=name, fund_type=fund_type, source=source),
                )
            errors.append(f"{source}: only {len(df)} rows")
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Failed to fetch fund {code}. " + " | ".join(errors))


def fetch_fund_info_map() -> dict[str, tuple[str, str]]:
    import akshare as ak

    df = ak.fund_name_em()
    return {
        str(row["基金代码"]).zfill(6): (str(row["基金简称"]), str(row["基金类型"]))
        for _, row in df.iterrows()
    }


def _normalize_exchange_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
    }
    out = df.rename(columns={k: v for k, v in columns.items() if k in df.columns}).copy()
    required = ["date", "open", "high", "low", "close"]
    if any(col not in out.columns for col in required):
        raise ValueError(f"Unexpected exchange fund columns: {df.columns.tolist()}")
    if "volume" not in out.columns:
        out["volume"] = 0
    if "amount" not in out.columns:
        out["amount"] = 0
    return _finalize(out)


def _normalize_nav_df(df: pd.DataFrame) -> pd.DataFrame:
    date_col = "净值日期" if "净值日期" in df.columns else "date"
    nav_col = "单位净值" if "单位净值" in df.columns else "close"
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    out["close"] = pd.to_numeric(df[nav_col], errors="coerce")
    out["open"] = out["close"]
    out["high"] = out["close"]
    out["low"] = out["close"]
    out["volume"] = 0
    out["amount"] = 0
    return _finalize(out)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close"])
