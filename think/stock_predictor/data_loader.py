from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PriceBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0


_FIELD_ALIASES = {
    "date": ["date", "日期", "交易日期"],
    "open": ["open", "开盘", "开盘价"],
    "high": ["high", "最高", "最高价"],
    "low": ["low", "最低", "最低价"],
    "close": ["close", "收盘", "收盘价"],
    "volume": ["volume", "vol", "成交量"],
    "amount": ["amount", "成交额"],
}


def load_csv(path: str | Path) -> list[PriceBar]:
    """Load price bars from an English or AkShare-style Chinese CSV file."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    bars = [_row_to_bar(row) for row in rows]
    return sorted(bars, key=lambda x: x.date)


def save_csv(bars: Iterable[PriceBar], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "open", "high", "low", "close", "volume", "amount"]
        )
        writer.writeheader()
        for bar in bars:
            writer.writerow(bar.__dict__)


def fetch_akshare_daily(symbol: str, adjust: str = "qfq") -> list[PriceBar]:
    """Fetch daily A-share bars with AkShare when it is installed."""
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "AkShare is not installed. Install it with: python -m pip install akshare"
        ) from exc

    normalized_symbol = symbol.split(".")[0]
    df = ak.stock_zh_a_hist(
        symbol=normalized_symbol, period="daily", start_date="19900101", adjust=adjust
    )
    rows = df.to_dict(orient="records")
    return sorted((_row_to_bar(row) for row in rows), key=lambda x: x.date)


def load_price_df(path: str | Path):
    """Load a normalized price DataFrame with pandas."""
    import pandas as pd

    df = pd.read_csv(path)
    return normalize_price_df(df)


def save_price_df(df, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")


def fetch_akshare_price_df(
    symbol: str,
    asset_type: str = "stock",
    adjust: str = "qfq",
    start_date: str = "19900101",
):
    """Fetch normalized stock or index daily bars through AkShare."""
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "AkShare is not installed. Install it with: python -m pip install akshare"
        ) from exc

    normalized_symbol = normalize_symbol(symbol, asset_type=asset_type)
    if asset_type == "index":
        df = _fetch_index_df(ak, normalized_symbol, start_date=start_date)
    else:
        df = ak.stock_zh_a_hist(
            symbol=normalized_symbol,
            period="daily",
            start_date=start_date,
            adjust=adjust,
        )
    return normalize_price_df(df)


def fetch_akshare_index_spot_row(symbol: str):
    """Fetch the current index spot row and normalize it as a one-row DataFrame."""
    try:
        import akshare as ak  # type: ignore
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "AkShare and pandas are required. Install with: python -m pip install akshare pandas"
        ) from exc

    target = normalize_symbol(symbol, asset_type="index").replace("sh", "").replace("sz", "")
    for group in ("上证系列指数", "沪深重要指数", "指数成份", "中证系列指数"):
        try:
            spot_df = ak.stock_zh_index_spot_em(symbol=group)
        except Exception:
            continue
        matched = spot_df[spot_df["代码"].astype(str).str.zfill(6) == target.zfill(6)]
        if matched.empty:
            continue
        row = matched.iloc[0]
        out = pd.DataFrame(
            [
                {
                    "date": date.today().isoformat(),
                    "open": row.get("今开"),
                    "high": row.get("最高"),
                    "low": row.get("最低"),
                    "close": row.get("最新价"),
                    "volume": row.get("成交量", 0),
                    "amount": row.get("成交额", 0),
                }
            ]
        )
        normalized = normalize_price_df(out)
        if not normalized.empty and float(normalized["close"].iloc[0]) > 0:
            return normalized.iloc[0]
    raise RuntimeError(f"Failed to fetch spot index row for {symbol}")


def append_index_spot_if_newer(df, symbol: str):
    """Append today's spot index row if historical daily data is stale."""
    import pandas as pd

    if df.empty:
        return df
    last_date = pd.to_datetime(df["date"].iloc[-1]).date()
    today = date.today()
    if today.weekday() >= 5 or last_date >= today:
        return df
    try:
        spot = fetch_akshare_index_spot_row(symbol)
    except RuntimeError:
        return df
    spot_date = pd.to_datetime(spot["date"]).date()
    if spot_date <= last_date:
        return df
    return normalize_price_df(pd.concat([df, spot.to_frame().T], ignore_index=True))


def normalize_symbol(symbol: str, asset_type: str = "stock") -> str:
    raw = symbol.strip().lower()
    aliases = {
        "科创100": "sh000698",
        "kcb100": "sh000698",
        "kc100": "sh000698",
        "000698": "sh000698" if asset_type == "index" else "000698",
    }
    if raw in aliases:
        return aliases[raw]
    if asset_type == "stock":
        return raw.replace("sh", "").replace("sz", "").split(".")[0]
    if raw.startswith(("sh", "sz", "bj")):
        return raw
    if raw.startswith("6") or raw.startswith("9") or raw.startswith("000"):
        return f"sh{raw}"
    return f"sz{raw}"


def normalize_price_df(df):
    import pandas as pd

    rename_map = {}
    for canonical, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    out = df.rename(columns=rename_map).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing price columns: {missing}. Available columns: {list(df.columns)}")

    keep = ["date", "open", "high", "low", "close", "volume"]
    if "amount" not in out.columns:
        out["amount"] = out["volume"] * out["close"]
    keep.append("amount")
    out = out[keep]
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for col in keep[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    return out.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _fetch_index_df(ak, symbol: str, start_date: str = "19900101"):
    errors = []
    for func_name in ("stock_zh_index_daily_em", "stock_zh_index_daily"):
        func = getattr(ak, func_name, None)
        if func is None:
            continue
        try:
            if func_name == "stock_zh_index_daily_em":
                return func(symbol=symbol, start_date=start_date)
            return func(symbol=symbol)
        except Exception as exc:  # pragma: no cover - depends on remote provider.
            errors.append(f"{func_name}: {exc}")
    func = getattr(ak, "index_zh_a_hist", None)
    if func is not None:
        try:
            bare_code = symbol[2:] if symbol.startswith(("sh", "sz", "bj")) else symbol
            return func(
                symbol=bare_code,
                period="daily",
                start_date=start_date,
                end_date="22220101",
            )
        except Exception as exc:  # pragma: no cover - depends on remote provider.
            errors.append(f"index_zh_a_hist: {exc}")
    raise RuntimeError("Failed to fetch index data through AkShare. " + " | ".join(errors))


def _row_to_bar(row: dict) -> PriceBar:
    return PriceBar(
        date=_normalize_date(str(_pick(row, "date"))),
        open=_to_float(_pick(row, "open")),
        high=_to_float(_pick(row, "high")),
        low=_to_float(_pick(row, "low")),
        close=_to_float(_pick(row, "close")),
        volume=_to_float(_pick(row, "volume")),
        amount=_to_float(_pick(row, "amount", default=0.0)),
    )


def _pick(row: dict, canonical: str, default: object | None = None) -> object:
    for name in _FIELD_ALIASES[canonical]:
        if name in row and row[name] not in ("", None):
            return row[name]
    if default is not None:
        return default
    raise KeyError(f"Missing required field {canonical!r}. Available fields: {list(row)}")


def _to_float(value: object) -> float:
    if value is None:
        return 0.0
    return float(str(value).replace(",", "").strip())


def _normalize_date(value: str) -> str:
    value = value.strip()
    if "-" in value:
        return value[:10]
    for fmt in ("%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return value
