from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .db import (
    InstrumentSpec,
    delete_instruments,
    import_daily_df,
    list_instrument_symbols,
    load_daily_bars,
    load_instrument_info,
    latest_trade_date,
)
from .fund_data import fetch_fund_daily_df
from .risk import build_risk_report_df
from .sklearn_model import predict_latest, train_sklearn_model, walk_forward_evaluate


DEFAULT_CONFIG_PATH = Path("think/watchlist_funds.json")


@dataclass(frozen=True)
class WatchlistItem:
    symbol: str
    name: str | None = None
    asset_type: str = "fund"
    market: str = "CN_FUND"
    provider_symbol: str | None = None
    currency: str = "CNY"
    timezone: str = "Asia/Shanghai"


@dataclass(frozen=True)
class WatchlistSettings:
    horizon: int = 3
    target_return: float = 0.005
    threshold: float = 0.55
    model_type: str = "logistic"


def load_watchlist_config(path: str | Path = DEFAULT_CONFIG_PATH):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    default = raw.get("default", {})
    settings = WatchlistSettings(
        horizon=int(default.get("horizon", 3)),
        target_return=float(default.get("target_return", 0.005)),
        threshold=float(default.get("threshold", 0.55)),
        model_type=str(default.get("model_type", "logistic")),
    )
    items = []
    for item in raw.get("instruments", []):
        symbol = str(item["symbol"]).zfill(6)
        items.append(
            WatchlistItem(
                symbol=symbol,
                name=item.get("name"),
                asset_type=item.get("asset_type", "fund"),
                market=item.get("market", "CN_FUND"),
                provider_symbol=item.get("provider_symbol") or symbol,
                currency=item.get("currency", "CNY"),
                timezone=item.get("timezone", "Asia/Shanghai"),
            )
        )
    return settings, items


def sync_watchlist_to_db(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    prune: bool = True,
    smart_refresh: bool = True,
    database_url: str | None = None,
) -> list[dict]:
    _settings, items = load_watchlist_config(config_path)
    desired = {item.symbol for item in items}
    if prune:
        existing = list_instrument_symbols(database_url)
        delete_instruments(existing - desired, database_url)

    summaries = []
    for item in items:
        existing_latest = latest_trade_date(item.symbol, database_url)
        if smart_refresh and existing_latest is not None and existing_latest >= date.today():
            db_info = load_instrument_info(item.symbol, database_url) or {}
            summaries.append(
                {
                    "symbol": item.symbol,
                    "name": item.name or db_info.get("name") or item.symbol,
                    "fund_type": "cached",
                    "source": "postgres",
                    "rows": 0,
                    "status": "skipped",
                    "latest_date": existing_latest.isoformat(),
                }
            )
            continue
        df, info = fetch_fund_daily_df(item.provider_symbol or item.symbol)
        name = item.name or info.name
        count = import_daily_df(
            df,
            InstrumentSpec(
                symbol=item.symbol,
                name=name,
                asset_type=item.asset_type,
                market=item.market,
                provider="akshare",
                provider_symbol=item.provider_symbol or item.symbol,
                currency=item.currency,
                timezone=item.timezone,
            ),
            url=database_url,
            source=info.source,
        )
        summaries.append(
            {
                "symbol": item.symbol,
                "name": name,
                "fund_type": info.fund_type,
                "source": info.source,
                "rows": count,
                "status": "fetched",
                "latest_date": str(df["date"].iloc[-1]) if len(df) else None,
            }
        )
    return summaries


def predict_watchlist(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    refresh: bool = True,
    prune: bool = True,
    smart_refresh: bool = True,
    database_url: str | None = None,
) -> tuple[WatchlistSettings, list[dict], list[dict]]:
    settings, items = load_watchlist_config(config_path)
    imports = (
        sync_watchlist_to_db(
            config_path,
            prune=prune,
            smart_refresh=smart_refresh,
            database_url=database_url,
        )
        if refresh
        else []
    )
    results = []
    import_names = {row["symbol"]: row["name"] for row in imports}
    for item in items:
        df = load_daily_bars(item.symbol, database_url)
        db_info = load_instrument_info(item.symbol, database_url)
        name = import_names.get(item.symbol) or item.name or (db_info or {}).get("name") or item.symbol
        try:
            model, split_result, _metadata = train_sklearn_model(
                df,
                horizon=settings.horizon,
                model_type=settings.model_type,
                target_return=settings.target_return,
            )
            prediction = predict_latest(
                model,
                df,
                horizon=settings.horizon,
                target_return=settings.target_return,
            )
            forecast_days = _next_weekdays(
                date.fromisoformat(prediction["date"][:10]),
                settings.horizon,
            )
            risk = build_risk_report_df(df, prediction["up_probability"])
            row = {
                "symbol": item.symbol,
                "name": name,
                "latest_date": prediction["date"],
                "latest_close": prediction["close"],
                "forecast_start": forecast_days[0].isoformat(),
                "forecast_end": forecast_days[-1].isoformat(),
                "probability": prediction["up_probability"],
                "predicted_return": prediction["predicted_return"],
                "predicted_close": prediction["predicted_close"],
                "risk_level": risk["risk_level"],
                "split_accuracy": split_result.accuracy,
                "split_signal_win_rate": split_result.signal_win_rate,
                "split_avg_signal_return": split_result.avg_signal_return,
                "return_mae": split_result.return_mae,
            }
            if len(df) >= 820:
                wf = walk_forward_evaluate(
                    df,
                    horizon=settings.horizon,
                    model_type=settings.model_type,
                    threshold=settings.threshold,
                    target_return=settings.target_return,
                    min_train_rows=720,
                    step=60,
                )
                row.update(
                    {
                        "walk_forward_accuracy": wf.accuracy,
                        "walk_forward_signal_win_rate": wf.signal_win_rate,
                        "walk_forward_avg_signal_return": wf.avg_signal_return,
                    }
                )
            results.append(row)
        except Exception as exc:
            results.append(
                {
                    "symbol": item.symbol,
                    "name": name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return settings, imports, results


def _next_weekdays(latest: date, count: int) -> list[date]:
    days: list[date] = []
    current = latest + timedelta(days=1)
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days
