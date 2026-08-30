from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from think.etf.config import IndexSpec
from think.etf.database import MIN_EXACT_HISTORY_ROWS, MySQLRepository
from think.stock_predictor.data_loader import fetch_akshare_price_df


LIXINGER_TOKEN_ENV = "LIXINGER_TOKEN"
LIXINGER_SOURCE = "lixinger_index_mcw"
LIXINGER_FUNDAMENTAL_URL = "https://open.lixinger.com/api/cn/index/fundamental"


@dataclass(frozen=True)
class OfficialSnapshot:
    date: str
    pe: float | None
    dividend_yield: float | None
    pb: float | None = None
    source: str | None = None


class DataProvider:
    """Fetch remote data, persist it, then read the analysis input from MySQL."""

    def __init__(
        self,
        repository: MySQLRepository | None,
        offline: bool = False,
        lixinger_token: str | None = None,
    ):
        self.repository = repository
        self.offline = offline
        self.lixinger_token = lixinger_token or os.environ.get(LIXINGER_TOKEN_ENV)
        self.warnings: list[str] = []
        self._exact_sources: dict[str, str] = {}
        self._history_sources: dict[str, str] = {}
        self._history_qualities: dict[str, str] = {}
        self._memory: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _akshare():
        try:
            import akshare as ak  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "缺少AkShare，请运行: python -m pip install akshare pandas"
            ) from exc
        return ak

    def pe_history(self, spec: IndexSpec) -> pd.DataFrame:
        self._sync_lixinger_history(spec)
        exact_source = self._best_exact_source(spec)
        if exact_source:
            self._exact_sources[spec.code] = exact_source
            self._history_sources[spec.code] = exact_source
            self._history_qualities[spec.code] = "exact"
            raw = self._load_exact_valuations(spec.code, exact_source)
            metric = self._metric_column(raw, "pe")
            return self._append_latest_official_metric(spec.code, metric, "pe")

        source = "legulegu_index" if spec.pe_source == "index" else "legulegu_market_proxy"
        quality = "index" if spec.pe_source == "index" else "proxy"
        self._history_sources[spec.code] = source
        self._history_qualities[spec.code] = quality

        def fetch() -> pd.DataFrame:
            ak = self._akshare()
            if spec.pe_source == "index":
                frame = ak.stock_index_pe_lg(symbol=spec.pe_symbol)
                return pd.DataFrame(
                    {
                        "date": frame["日期"],
                        "pe": pd.to_numeric(frame["滚动市盈率"], errors="coerce"),
                        "pb": pd.NA,
                        "dividend_yield": pd.NA,
                    }
                )
            frame = ak.stock_market_pe_lg(symbol=spec.pe_symbol)
            column = "市盈率" if "市盈率" in frame.columns else "平均市盈率"
            return pd.DataFrame(
                {
                    "date": frame["日期"],
                    "pe": pd.to_numeric(frame[column], errors="coerce"),
                    "pb": pd.NA,
                    "dividend_yield": pd.NA,
                }
            )

        raw = self._sync_valuation(spec, source, quality, fetch)
        return self._metric_column(raw, "pe")

    def pb_history(self, spec: IndexSpec) -> pd.DataFrame | None:
        exact_source = self._exact_sources.get(spec.code) or self._best_exact_source(spec)
        if exact_source:
            raw = self._load_exact_valuations(spec.code, exact_source)
            metric = self._metric_column(raw, "pb")
            metric = self._append_latest_official_metric(spec.code, metric, "pb")
            return metric if len(metric) >= 20 else None
        if not spec.pb_symbol:
            return None

        source = "legulegu_index_pb"

        def fetch() -> pd.DataFrame:
            ak = self._akshare()
            frame = ak.stock_index_pb_lg(symbol=spec.pb_symbol)
            return pd.DataFrame(
                {
                    "date": frame["日期"],
                    "pe": pd.NA,
                    "pb": pd.to_numeric(frame["市净率"], errors="coerce"),
                    "dividend_yield": pd.NA,
                }
            )

        raw = self._sync_valuation(spec, source, "index", fetch)
        metric = self._metric_column(raw, "pb")
        return metric if len(metric) >= 20 else None

    def official_snapshot(self, spec: IndexSpec) -> OfficialSnapshot | None:
        if spec.code == "399006":
            self._sync_cnindex_snapshot(spec)
        elif spec.official_valuation_code:
            self._sync_csindex_snapshot(spec)
        if self.repository:
            row = self.repository.latest_official_valuation(spec.code)
            if row:
                return OfficialSnapshot(
                    date=row["date"], pe=row["pe"], pb=row["pb"],
                    dividend_yield=row["dividend_yield"], source=row["source"],
                )
        frame = self._memory.get(f"official:{spec.code}")
        if frame is not None and not frame.empty:
            row = frame.sort_values("date").iloc[-1]
            return OfficialSnapshot(
                date=pd.Timestamp(row["date"]).date().isoformat(),
                pe=_float_or_none(row.get("pe")),
                pb=_float_or_none(row.get("pb")),
                dividend_yield=_float_or_none(row.get("dividend_yield")),
                source="official_live",
            )
        return None

    def price_history(self, spec: IndexSpec, years: int = 3) -> pd.DataFrame:
        source = "akshare_index"
        if not self.offline:
            start = (date.today() - timedelta(days=366 * years)).strftime("%Y%m%d")
            try:
                frame = fetch_akshare_price_df(
                    symbol=spec.price_symbol,
                    asset_type="index",
                    adjust="",
                    start_date=start,
                )
                frame = self._normalize_price_frame(frame)
                if self.repository:
                    self.repository.upsert_prices(
                        spec.code, spec.name, spec.etf_code, frame, source=source
                    )
                self._memory[f"price:{spec.code}"] = frame
            except Exception as exc:
                self.warnings.append(f"{spec.code}实时行情拉取失败，尝试读取数据库：{exc}")
        if self.repository:
            frame = self.repository.load_prices(spec.code, source=source)
            if not frame.empty:
                return self._normalize_price_frame(frame)
        frame = self._memory.get(f"price:{spec.code}")
        if frame is not None and not frame.empty:
            return frame
        raise RuntimeError(f"{spec.code}没有可用行情数据")

    def risk_free_rate(self, override: float | None = None) -> tuple[float, str]:
        if override is not None:
            if override < 0 or override > 20:
                raise ValueError("无风险利率应以百分数输入，例如1.70")
            return override, "用户指定"
        if not self.offline:
            end = date.today()
            start = end - timedelta(days=20)
            try:
                raw = self._akshare().bond_china_yield(
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                )
                frame = pd.DataFrame(
                    {
                        "date": pd.to_datetime(raw["日期"], errors="coerce"),
                        "tenor": "10Y",
                        "yield_pct": pd.to_numeric(raw["10年"], errors="coerce"),
                    }
                ).dropna(subset=["date", "yield_pct"])
                if self.repository:
                    self.repository.upsert_bond_yields(frame)
                self._memory["bond:10Y"] = frame
            except Exception as exc:
                self.warnings.append(f"国债收益率拉取失败，尝试读取数据库：{exc}")
        if self.repository:
            value = self.repository.latest_bond_yield("10Y")
            if value:
                return value
        frame = self._memory.get("bond:10Y")
        if frame is not None and not frame.empty:
            row = frame.sort_values("date").iloc[-1]
            return float(row["yield_pct"]), pd.Timestamp(row["date"]).date().isoformat()
        raise RuntimeError("没有可用的10年期国债收益率")

    def using_exact(self, code: str) -> bool:
        return code in self._exact_sources

    def exact_source(self, code: str) -> str | None:
        return self._exact_sources.get(code)

    def history_source(self, code: str) -> str:
        return self._history_sources.get(code, "unknown")

    def history_quality(self, code: str) -> str:
        return self._history_qualities.get(code, "unknown")

    def _sync_valuation(self, spec, source, quality, fetcher) -> pd.DataFrame:
        if not self.offline:
            try:
                frame = self._normalize_valuation_frame(fetcher())
                if self.repository:
                    self.repository.upsert_valuations(
                        spec.code, spec.name, spec.etf_code, frame, source, quality
                    )
                self._memory[f"valuation:{spec.code}:{source}"] = frame
            except Exception as exc:
                self.warnings.append(
                    f"{spec.code} {source}拉取失败，尝试读取数据库：{exc}"
                )
        if self.repository:
            frame = self.repository.load_valuations(spec.code, source)
            if not frame.empty:
                return self._normalize_valuation_frame(frame)
        frame = self._memory.get(f"valuation:{spec.code}:{source}")
        if frame is not None and not frame.empty:
            return frame
        raise RuntimeError(f"{spec.code}数据库中没有{source}数据")

    def _sync_csindex_snapshot(self, spec: IndexSpec) -> None:
        if self.offline:
            return
        try:
            raw = self._akshare().stock_zh_index_value_csindex(
                symbol=spec.official_valuation_code
            )
            frame = pd.DataFrame(
                {
                    "date": raw["日期"],
                    "pe": pd.to_numeric(raw["市盈率1"], errors="coerce"),
                    "pb": pd.NA,
                    "dividend_yield": pd.to_numeric(raw["股息率1"], errors="coerce"),
                }
            )
            frame = self._normalize_valuation_frame(frame)
            if self.repository:
                self.repository.upsert_valuations(
                    spec.code, spec.name, spec.etf_code, frame,
                    source="csindex_official", quality="official_snapshot",
                )
            self._memory[f"official:{spec.code}"] = frame
        except Exception as exc:
            self.warnings.append(f"{spec.code}中证官方快照拉取失败：{exc}")

    def _sync_cnindex_snapshot(self, spec: IndexSpec) -> None:
        if self.offline:
            return
        try:
            import requests

            response = requests.post(
                "https://www.cnindex.com.cn/index/search",
                data={"content": spec.code, "rows": 20, "pageNum": 1},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("data", {}).get("rows", [])
            matched = next(row for row in rows if str(row.get("indexcode")) == spec.code)
            day_response = requests.get(
                "https://www.cnindex.com.cn/index/queryDay", timeout=15
            )
            day_response.raise_for_status()
            trade_date = pd.to_datetime(
                day_response.json().get("data"), unit="ms", errors="coerce"
            )
            if pd.isna(trade_date):
                trade_date = pd.Timestamp(date.today())
            frame = pd.DataFrame(
                [
                    {
                        "date": trade_date,
                        "pe": matched.get("peDynamic"),
                        "pb": matched.get("pb"),
                        "dividend_yield": pd.NA,
                    }
                ]
            )
            frame = self._normalize_valuation_frame(frame)
            if self.repository:
                self.repository.upsert_valuations(
                    spec.code, spec.name, spec.etf_code, frame,
                    source="cnindex_official", quality="official_snapshot",
                )
            self._memory[f"official:{spec.code}"] = frame
        except Exception as exc:
            self.warnings.append(f"{spec.code}国证官方快照拉取失败：{exc}")

    def _sync_lixinger_history(self, spec: IndexSpec) -> None:
        if self.offline or not self.lixinger_token:
            return
        try:
            import requests

            start_date = date.today() - timedelta(days=3652)
            has_complete_history = False
            if spec.launch_date:
                start_date = max(start_date, date.fromisoformat(spec.launch_date))
            if self.repository:
                stats = next(
                    (
                        row for row in self.repository.valuation_source_stats(spec.code)
                        if row["source"] == LIXINGER_SOURCE
                    ),
                    None,
                )
                if (
                    stats
                    and stats["row_count"] >= MIN_EXACT_HISTORY_ROWS
                    and stats["history_days"] >= spec.minimum_exact_history_days()
                ):
                    has_complete_history = True
                    start_date = max(
                        start_date,
                        date.fromisoformat(stats["end_date"]) - timedelta(days=7),
                    )
            response = requests.post(
                LIXINGER_FUNDAMENTAL_URL,
                json={
                    "startDate": start_date.isoformat(),
                    "endDate": date.today().isoformat(),
                    "stockCodes": [spec.code],
                    "metricsList": ["pe_ttm.mcw", "pb.mcw", "dyr.mcw"],
                    "token": self.lixinger_token,
                },
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 1:
                raise RuntimeError(payload.get("message") or "理杏仁返回未知错误")
            frame = self._lixinger_frame(payload.get("data") or [])
            minimum_rows = 1 if has_complete_history else 20
            if frame.empty or frame["pe"].notna().sum() < minimum_rows:
                raise RuntimeError(f"理杏仁返回的有效PE记录少于{minimum_rows}条")
            if self.repository:
                self.repository.upsert_valuations(
                    spec.code, spec.name, spec.etf_code, frame,
                    source=LIXINGER_SOURCE, quality="exact",
                )
            self._memory[f"valuation:{spec.code}:{LIXINGER_SOURCE}"] = frame
        except Exception as exc:
            self.warnings.append(f"{spec.code}理杏仁指数估值拉取失败，使用备用来源：{exc}")

    @staticmethod
    def _lixinger_frame(rows: list[dict]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["date", "pe", "pb", "dividend_yield"])
        metrics = frame.reindex(
            columns=["date", "pe_ttm.mcw", "pb.mcw", "dyr.mcw"]
        )
        result = pd.DataFrame(
            {
                "date": metrics["date"],
                "pe": metrics["pe_ttm.mcw"],
                "pb": metrics["pb.mcw"],
                # The API returns a ratio (0.027 means 2.7%). MySQL stores percent.
                "dividend_yield": pd.to_numeric(
                    metrics["dyr.mcw"], errors="coerce"
                ) * 100.0,
            }
        )
        return DataProvider._normalize_valuation_frame(result)

    def _load_exact_valuations(self, code: str, source: str) -> pd.DataFrame:
        if self.repository:
            return self.repository.load_valuations(code, source)
        frame = self._memory.get(f"valuation:{code}:{source}")
        if frame is None:
            return pd.DataFrame(columns=["date", "pe", "pb", "dividend_yield"])
        return self._normalize_valuation_frame(frame)

    def _best_exact_source(self, spec: IndexSpec) -> str | None:
        if not self.repository:
            frame = self._memory.get(f"valuation:{spec.code}:{LIXINGER_SOURCE}")
            if frame is not None and len(frame) >= MIN_EXACT_HISTORY_ROWS:
                return LIXINGER_SOURCE
            return None
        return self.repository.best_exact_source(
            spec.code,
            MIN_EXACT_HISTORY_ROWS,
            spec.minimum_exact_history_days(),
        )

    def _append_latest_official_metric(
        self, code: str, metric: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        if not self.repository:
            return metric
        row = self.repository.latest_official_valuation(code)
        if not row or row.get(column) is None:
            return metric
        latest_date = pd.Timestamp(row["date"])
        if not metric.empty and latest_date <= metric["date"].max():
            return metric
        appended = pd.concat(
            [metric, pd.DataFrame([{"date": latest_date, "value": row[column]}])],
            ignore_index=True,
        )
        return self._normalize_metric_frame(appended)

    @staticmethod
    def _metric_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        if column not in frame:
            return pd.DataFrame(columns=["date", "value"])
        return DataProvider._normalize_metric_frame(
            frame[["date", column]].rename(columns={column: "value"})
        )

    @staticmethod
    def _normalize_valuation_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for column in ("pe", "pb", "dividend_yield"):
            if column not in out:
                out[column] = pd.NA
            out[column] = pd.to_numeric(out[column], errors="coerce")
        return (
            out.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )

    @staticmethod
    def _normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for column in ("open", "high", "low", "close", "volume", "amount"):
            out[column] = pd.to_numeric(out[column], errors="coerce")
        return (
            out.dropna(subset=["date", "close"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )

    @staticmethod
    def _normalize_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return (
            out.dropna(subset=["date", "value"])
            .query("value > 0")
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )


def _float_or_none(value):
    if value is None or pd.isna(value):
        return None
    return float(value)
