from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import unquote, urlparse

import pandas as pd


DATABASE_ENV = "ETF_DATABASE_URL"
MIN_EXACT_HISTORY_ROWS = 60
MIN_ACCUMULATED_OFFICIAL_ROWS = 250
MIN_EXACT_HISTORY_DAYS = 365 * 5


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"


def database_url(explicit_url: str | None = None) -> str:
    value = explicit_url or os.environ.get(DATABASE_ENV)
    if not value:
        raise RuntimeError(
            f"未配置数据库。请设置 {DATABASE_ENV}=mysql://用户:密码@127.0.0.1:3306/etf_data"
        )
    return value


def parse_mysql_url(url: str) -> MySQLConfig:
    normalized = url.replace("mysql+pymysql://", "mysql://", 1)
    parsed = urlparse(normalized)
    if parsed.scheme != "mysql":
        raise ValueError("当前ETF工具默认支持MySQL，URL应以 mysql:// 开头")
    database = parsed.path.lstrip("/")
    if not database or not re.fullmatch(r"[A-Za-z0-9_]+", database):
        raise ValueError("MySQL数据库名不能为空且只能包含字母、数字和下划线")
    if not parsed.username:
        raise ValueError("MySQL URL缺少用户名")
    return MySQLConfig(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=unquote(parsed.username),
        password=unquote(parsed.password or ""),
        database=database,
    )


class MySQLRepository:
    def __init__(self, url: str | None = None):
        self.config = parse_mysql_url(database_url(url))

    @staticmethod
    def _driver():
        try:
            import pymysql  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "缺少PyMySQL，请运行: python -m pip install pymysql"
            ) from exc
        return pymysql

    def ensure_database_and_schema(self) -> None:
        driver = self._driver()
        try:
            with self.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
        except driver.err.OperationalError as exc:
            if not exc.args or int(exc.args[0]) != 1049:
                raise
        else:
            self._init_schema()
            return
        server_conn = driver.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            charset=self.config.charset,
            autocommit=True,
            connect_timeout=5,
        )
        try:
            with server_conn.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.config.database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
        finally:
            server_conn.close()
        self._init_schema()

    def _init_schema(self) -> None:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                for statement in MYSQL_SCHEMA:
                    cursor.execute(statement)
            conn.commit()

    def check(self) -> dict:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT VERSION(), DATABASE(), NOW()")
                version, database, now = cursor.fetchone()
        return {"version": version, "database": database, "server_time": str(now)}

    @contextmanager
    def connection(self):
        driver = self._driver()
        conn = driver.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            charset=self.config.charset,
            autocommit=False,
            connect_timeout=5,
        )
        try:
            yield conn
        finally:
            conn.close()

    def upsert_index(self, code: str, name: str, etf_code: str) -> None:
        sql = """
        INSERT INTO etf_index_meta (index_code, index_name, etf_code)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            index_name = VALUES(index_name),
            etf_code = VALUES(etf_code),
            updated_at = CURRENT_TIMESTAMP
        """
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (code, name, etf_code))
            conn.commit()

    def upsert_valuations(
        self,
        code: str,
        name: str,
        etf_code: str,
        frame: pd.DataFrame,
        source: str,
        quality: str,
    ) -> int:
        if frame.empty:
            return 0
        self.upsert_index(code, name, etf_code)
        rows = []
        for row in frame.itertuples(index=False):
            rows.append(
                (
                    code,
                    _date_value(row.date),
                    source,
                    quality,
                    _nullable_number(getattr(row, "pe", None)),
                    _nullable_number(getattr(row, "pb", None)),
                    _nullable_number(getattr(row, "dividend_yield", None)),
                )
            )
        sql = """
        INSERT INTO etf_index_valuation (
            index_code, trade_date, source, quality, pe, pb, dividend_yield
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            quality = VALUES(quality), pe = VALUES(pe), pb = VALUES(pb),
            dividend_yield = VALUES(dividend_yield), fetched_at = CURRENT_TIMESTAMP
        """
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, rows)
            conn.commit()
        return len(rows)

    def upsert_prices(
        self,
        code: str,
        name: str,
        etf_code: str,
        frame: pd.DataFrame,
        source: str = "akshare_index",
    ) -> int:
        if frame.empty:
            return 0
        self.upsert_index(code, name, etf_code)
        rows = [
            (
                code,
                _date_value(row.date),
                source,
                _nullable_number(row.open),
                _nullable_number(row.high),
                _nullable_number(row.low),
                _nullable_number(row.close),
                _nullable_number(row.volume),
                _nullable_number(row.amount),
            )
            for row in frame.itertuples(index=False)
        ]
        sql = """
        INSERT INTO etf_index_price (
            index_code, trade_date, source, open_price, high_price, low_price,
            close_price, volume, amount
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open_price = VALUES(open_price), high_price = VALUES(high_price),
            low_price = VALUES(low_price), close_price = VALUES(close_price),
            volume = VALUES(volume), amount = VALUES(amount), fetched_at = CURRENT_TIMESTAMP
        """
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, rows)
            conn.commit()
        return len(rows)

    def upsert_bond_yields(
        self, frame: pd.DataFrame, source: str = "chinabond"
    ) -> int:
        rows = [
            (
                _date_value(row.date),
                str(row.tenor),
                source,
                _nullable_number(row.yield_pct),
            )
            for row in frame.itertuples(index=False)
        ]
        if not rows:
            return 0
        sql = """
        INSERT INTO etf_bond_yield (curve_date, tenor, source, yield_pct)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            yield_pct = VALUES(yield_pct), fetched_at = CURRENT_TIMESTAMP
        """
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, rows)
            conn.commit()
        return len(rows)

    def best_exact_source(
        self,
        code: str,
        min_rows: int = MIN_EXACT_HISTORY_ROWS,
        min_history_days: int = MIN_EXACT_HISTORY_DAYS,
    ) -> str | None:
        sql = """
        SELECT source, quality, COUNT(*) AS row_count,
               DATEDIFF(MAX(trade_date), MIN(trade_date)) AS history_days
        FROM etf_index_valuation
        WHERE index_code = %s
          AND quality IN ('exact', 'official_snapshot')
          AND pe IS NOT NULL
        GROUP BY source, quality
        HAVING ((quality = 'exact' AND COUNT(*) >= %s)
             OR (quality = 'official_snapshot' AND COUNT(*) >= %s))
           AND DATEDIFF(MAX(trade_date), MIN(trade_date)) >= %s
        ORDER BY row_count DESC
        LIMIT 1
        """
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (code, min_rows, MIN_ACCUMULATED_OFFICIAL_ROWS, min_history_days),
                )
                row = cursor.fetchone()
        return str(row[0]) if row else None

    def valuation_source_stats(self, code: str) -> list[dict]:
        sql = """
        SELECT source, quality, COUNT(*) AS row_count,
               MIN(trade_date) AS start_date, MAX(trade_date) AS end_date,
               DATEDIFF(MAX(trade_date), MIN(trade_date)) AS history_days
        FROM etf_index_valuation
        WHERE index_code = %s AND pe IS NOT NULL
        GROUP BY source, quality
        ORDER BY quality, row_count DESC
        """
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (code,))
                rows = cursor.fetchall()
        return [
            {
                "source": row[0], "quality": row[1], "row_count": int(row[2]),
                "start_date": row[3].isoformat(), "end_date": row[4].isoformat(),
                "history_days": int(row[5] or 0),
            }
            for row in rows
        ]

    def load_valuations(self, code: str, source: str) -> pd.DataFrame:
        sql = """
        SELECT trade_date AS date, pe, pb, dividend_yield, quality
        FROM etf_index_valuation
        WHERE index_code = %s AND source = %s
        ORDER BY trade_date
        """
        return self._query_frame(sql, (code, source))

    def latest_official_valuation(self, code: str) -> dict | None:
        sql = """
        SELECT trade_date, pe, pb, dividend_yield, source
        FROM etf_index_valuation
        WHERE index_code = %s AND quality = 'official_snapshot'
        ORDER BY trade_date DESC, fetched_at DESC
        LIMIT 1
        """
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (code,))
                row = cursor.fetchone()
        if not row:
            return None
        return {
            "date": row[0].isoformat(), "pe": _float_or_none(row[1]),
            "pb": _float_or_none(row[2]),
            "dividend_yield": _float_or_none(row[3]), "source": row[4],
        }

    def load_prices(self, code: str, source: str = "akshare_index") -> pd.DataFrame:
        sql = """
        SELECT trade_date AS date, open_price AS open, high_price AS high,
               low_price AS low, close_price AS close, volume, amount
        FROM etf_index_price
        WHERE index_code = %s AND source = %s
        ORDER BY trade_date
        """
        return self._query_frame(sql, (code, source))

    def latest_bond_yield(self, tenor: str = "10Y") -> tuple[float, str] | None:
        sql = """
        SELECT yield_pct, curve_date
        FROM etf_bond_yield
        WHERE tenor = %s
        ORDER BY curve_date DESC, fetched_at DESC
        LIMIT 1
        """
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (tenor,))
                row = cursor.fetchone()
        return (float(row[0]), row[1].isoformat()) if row else None

    def save_analysis_run(
        self,
        risk_free_rate: float,
        risk_free_date: str,
        results: Iterable,
        report_path: str | None,
    ) -> int:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO etf_analysis_run (risk_free_rate, risk_free_date, report_path)
                    VALUES (%s, %s, %s)
                    """,
                    (risk_free_rate, risk_free_date, report_path),
                )
                run_id = int(cursor.lastrowid)
                rows = [
                    (
                        run_id, item.code, item.status, item.confidence,
                        item.pe.current, item.pb.current if item.pb else None,
                        item.earnings_yield, item.earnings_yield_spread,
                        item.composite_percentile, item.suggested_multiplier,
                        item.suggested_amount, item.reason,
                    )
                    for item in results
                ]
                cursor.executemany(
                    """
                    INSERT INTO etf_analysis_result (
                        run_id, index_code, valuation_status, confidence, pe, pb,
                        earnings_yield, earnings_yield_spread, composite_score,
                        suggested_multiplier, suggested_amount, reason
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
            conn.commit()
        return run_id

    def _query_frame(self, sql: str, params: tuple) -> pd.DataFrame:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                columns = [item[0] for item in cursor.description]
        frame = pd.DataFrame(rows, columns=columns)
        for column in frame.columns:
            if column not in {"date", "quality"}:
                try:
                    frame[column] = pd.to_numeric(frame[column])
                except (TypeError, ValueError):
                    pass
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        return frame


def _date_value(value):
    return pd.Timestamp(value).date()


def _nullable_number(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _float_or_none(value):
    return None if value is None else float(value)


MYSQL_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS etf_index_meta (
        index_code VARCHAR(16) PRIMARY KEY,
        index_name VARCHAR(128) NOT NULL,
        etf_code VARCHAR(16) NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS etf_index_valuation (
        index_code VARCHAR(16) NOT NULL,
        trade_date DATE NOT NULL,
        source VARCHAR(64) NOT NULL,
        quality VARCHAR(32) NOT NULL,
        pe DECIMAL(20,8) NULL,
        pb DECIMAL(20,8) NULL,
        dividend_yield DECIMAL(20,8) NULL,
        fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (index_code, trade_date, source),
        INDEX idx_etf_valuation_lookup (index_code, quality, trade_date),
        CONSTRAINT fk_etf_valuation_meta FOREIGN KEY (index_code)
            REFERENCES etf_index_meta(index_code) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    ALTER TABLE etf_index_valuation
        MODIFY quality VARCHAR(32) NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS etf_index_price (
        index_code VARCHAR(16) NOT NULL,
        trade_date DATE NOT NULL,
        source VARCHAR(64) NOT NULL,
        open_price DECIMAL(24,8) NULL,
        high_price DECIMAL(24,8) NULL,
        low_price DECIMAL(24,8) NULL,
        close_price DECIMAL(24,8) NOT NULL,
        volume DECIMAL(30,4) NULL,
        amount DECIMAL(30,4) NULL,
        fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (index_code, trade_date, source),
        INDEX idx_etf_price_date (trade_date),
        CONSTRAINT fk_etf_price_meta FOREIGN KEY (index_code)
            REFERENCES etf_index_meta(index_code) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS etf_bond_yield (
        curve_date DATE NOT NULL,
        tenor VARCHAR(16) NOT NULL,
        source VARCHAR(64) NOT NULL,
        yield_pct DECIMAL(20,8) NOT NULL,
        fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (curve_date, tenor, source)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS etf_analysis_run (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        risk_free_rate DECIMAL(20,8) NOT NULL,
        risk_free_date DATE NOT NULL,
        report_path VARCHAR(1024) NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS etf_analysis_result (
        run_id BIGINT NOT NULL,
        index_code VARCHAR(16) NOT NULL,
        valuation_status VARCHAR(32) NOT NULL,
        confidence VARCHAR(64) NOT NULL,
        pe DECIMAL(20,8) NULL,
        pb DECIMAL(20,8) NULL,
        earnings_yield DECIMAL(20,8) NULL,
        earnings_yield_spread DECIMAL(20,8) NULL,
        composite_score DECIMAL(20,8) NULL,
        suggested_multiplier DECIMAL(20,8) NOT NULL,
        suggested_amount DECIMAL(20,4) NULL,
        reason VARCHAR(512) NULL,
        PRIMARY KEY (run_id, index_code),
        CONSTRAINT fk_etf_analysis_run FOREIGN KEY (run_id)
            REFERENCES etf_analysis_run(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]
