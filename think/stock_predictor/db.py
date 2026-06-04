from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg import sql

from .data_loader import load_price_df


DEFAULT_DATABASE_URL = "postgresql://postgres@localhost:5432/stock_predictor?connect_timeout=5"
SCHEMA_PATH = Path(__file__).with_name("db_schema.sql")


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    name: str
    asset_type: str
    market: str
    provider_symbol: str
    provider: str = "akshare"
    currency: str = "CNY"
    timezone: str = "Asia/Shanghai"


def database_url(explicit_url: str | None = None) -> str:
    return explicit_url or os.environ.get("STOCK_PREDICTOR_DATABASE_URL") or DEFAULT_DATABASE_URL


def connect(url: str | None = None):
    return psycopg.connect(database_url(url))


def init_schema(url: str | None = None) -> None:
    with connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def create_database(
    database: str = "stock_predictor",
    maintenance_url: str = "postgresql://postgres@localhost:5432/postgres",
) -> bool:
    with psycopg.connect(maintenance_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
            if cur.fetchone():
                return False
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
            return True


def upsert_instrument(conn, spec: InstrumentSpec) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO instruments (
                symbol, name, asset_type, market, provider, provider_symbol, currency, timezone
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE SET
                name = EXCLUDED.name,
                asset_type = EXCLUDED.asset_type,
                market = EXCLUDED.market,
                provider = EXCLUDED.provider,
                provider_symbol = EXCLUDED.provider_symbol,
                currency = EXCLUDED.currency,
                timezone = EXCLUDED.timezone,
                updated_at = now()
            RETURNING id
            """,
            (
                spec.symbol,
                spec.name,
                spec.asset_type,
                spec.market,
                spec.provider,
                spec.provider_symbol,
                spec.currency,
                spec.timezone,
            ),
        )
        return int(cur.fetchone()[0])


def import_daily_csv(
    csv_path: str | Path,
    spec: InstrumentSpec,
    url: str | None = None,
    source: str = "csv",
) -> int:
    df = load_price_df(csv_path)
    return import_daily_df(df, spec, url=url, source=source)


def import_daily_df(
    df,
    spec: InstrumentSpec,
    url: str | None = None,
    source: str = "dataframe",
) -> int:
    rows = [
        (
            row.date,
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
            float(row.amount),
            source,
        )
        for row in df.itertuples(index=False)
    ]
    with connect(url) as conn:
        instrument_id = upsert_instrument(conn, spec)
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO daily_bars (
                    instrument_id, trade_date, open, high, low, close, volume, amount, source
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (instrument_id, trade_date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    amount = EXCLUDED.amount,
                    source = EXCLUDED.source,
                    updated_at = now()
                """,
                [(instrument_id, *row) for row in rows],
            )
    return len(rows)


def list_instrument_symbols(url: str | None = None) -> set[str]:
    with connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol FROM instruments")
            return {str(row[0]) for row in cur.fetchall()}


def delete_instruments(symbols: set[str], url: str | None = None) -> int:
    if not symbols:
        return 0
    with connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM instruments WHERE symbol = ANY(%s)", (list(symbols),))
            return int(cur.rowcount)


def load_daily_bars(symbol: str, url: str | None = None):
    import pandas as pd

    with connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    b.trade_date AS date,
                    b.open,
                    b.high,
                    b.low,
                    b.close,
                    b.volume,
                    b.amount
                FROM daily_bars b
                JOIN instruments i ON i.id = b.instrument_id
                WHERE i.symbol = %s
                ORDER BY b.trade_date
                """,
                (symbol,),
            )
            rows = cur.fetchall()
    out = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume", "amount"],
    )
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_instrument_info(symbol: str, url: str | None = None) -> dict | None:
    with connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(
            """
            SELECT
                symbol, name, asset_type, market, provider, provider_symbol, currency, timezone
            FROM instruments
            WHERE symbol = %s
            """,
                (symbol,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    keys = ["symbol", "name", "asset_type", "market", "provider", "provider_symbol", "currency", "timezone"]
    return dict(zip(keys, row))


def latest_trade_date(symbol: str, url: str | None = None):
    with connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT max(b.trade_date)
                FROM daily_bars b
                JOIN instruments i ON i.id = b.instrument_id
                WHERE i.symbol = %s
                """,
                (symbol,),
            )
            row = cur.fetchone()
    return row[0] if row else None
