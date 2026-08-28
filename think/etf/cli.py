from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from think.etf.analyzer import analyze_index
from think.etf.config import get_specs
from think.etf.data import DataProvider
from think.etf.database import MySQLRepository
from think.etf.models import ValuationResult
from think.etf.strategy import apply_budget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A股指数估值与每周定投参考工具（仅供研究，不自动交易）"
    )
    parser.add_argument(
        "--index",
        action="append",
        dest="indices",
        help="只分析指定代码，可重复传入，例如 --index 000300 --index 000015",
    )
    parser.add_argument("--history-years", type=int, default=10, help="历史估值分位窗口")
    parser.add_argument(
        "--weekly-budget", type=float, default=None, help="每周基础预算，单位：元"
    )
    parser.add_argument(
        "--max-weekly-multiplier",
        type=float,
        default=1.50,
        help="组合单周建议金额上限，相对于基础预算的倍数",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=None,
        help="手工指定10年期国债收益率百分数，例如 1.70；默认联网获取",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="MySQL URL；默认读取ETF_DATABASE_URL环境变量",
    )
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="仅用于排障：不写数据库并直接使用本次内存数据",
    )
    parser.add_argument("--db-check", action="store_true", help="建表、检查连接后退出")
    parser.add_argument(
        "--import-valuation-csv",
        default=None,
        help="将外部指数级历史估值CSV直接导入MySQL后退出",
    )
    parser.add_argument(
        "--import-index",
        default=None,
        help="与--import-valuation-csv配套使用的指数代码",
    )
    parser.add_argument(
        "--import-source",
        default="licensed_import",
        help="历史估值导入来源名称，如wind、choice、ifind",
    )
    parser.add_argument(
        "--output-dir",
        default="think/etf/output",
        help="Excel报告输出目录",
    )
    parser.add_argument("--no-excel", action="store_true", help="不生成Excel报告")
    parser.add_argument("--verbose", action="store_true", help="终端同时显示完整明细表")
    parser.add_argument("--offline", action="store_true", help="不联网，只读取MySQL已有数据")
    parser.add_argument("--json", action="store_true", help="输出JSON而不是表格")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = None
    database_label = "未启用（排障模式）"
    try:
        if not args.no_database:
            repository = MySQLRepository(args.database_url)
            repository.ensure_database_and_schema()
            db_info = repository.check()
            database_label = (
                f"MySQL {repository.config.host}:{repository.config.port}/"
                f"{repository.config.database}"
            )
            if args.db_check:
                print(
                    f"MySQL连接正常：database={db_info['database']} "
                    f"version={db_info['version']} server_time={db_info['server_time']}"
                )
                return 0
        elif args.offline:
            raise ValueError("--offline必须连接MySQL，不能与--no-database同时使用")

        specs = get_specs(args.indices)
        if args.import_valuation_csv:
            if repository is None:
                raise ValueError("导入历史估值必须启用MySQL")
            if not args.import_index:
                raise ValueError("--import-valuation-csv必须同时指定--import-index")
            _import_valuation_csv(
                repository,
                specs=get_specs([args.import_index]),
                csv_path=Path(args.import_valuation_csv),
                source=args.import_source,
            )
            return 0

        provider = DataProvider(repository=repository, offline=args.offline)
        risk_free_rate, risk_free_date = provider.risk_free_rate(args.risk_free_rate)
    except Exception as exc:
        raise SystemExit(f"初始化失败: {exc}") from exc

    results: list[ValuationResult] = []
    failures: list[str] = []
    for spec in specs:
        try:
            results.append(
                analyze_index(
                    spec=spec,
                    provider=provider,
                    history_years=args.history_years,
                    risk_free_rate=risk_free_rate,
                )
            )
        except Exception as exc:
            failures.append(f"{spec.code} {spec.name}: {exc}")

    results = apply_budget(
        results,
        weekly_budget=args.weekly_budget,
        max_portfolio_multiplier=args.max_weekly_multiplier,
    )
    excel_path = None
    if not args.no_excel:
        try:
            from think.etf.report import default_report_path, export_excel_report

            excel_path = export_excel_report(
                results=results,
                output_path=default_report_path(args.output_dir),
                risk_free_rate=risk_free_rate,
                risk_free_date=risk_free_date,
                warnings=provider.warnings,
                failures=failures,
                database_label=database_label,
            )
        except ImportError as exc:
            raise SystemExit(
                "生成Excel需要 openpyxl，请运行: python -m pip install openpyxl"
            ) from exc
        except Exception as exc:
            raise SystemExit(f"生成Excel失败: {exc}") from exc
    if repository is not None:
        repository.save_analysis_run(
            risk_free_rate=risk_free_rate,
            risk_free_date=risk_free_date,
            results=results,
            report_path=str(excel_path) if excel_path else None,
        )
    if args.json:
        payload = {
            "risk_free_rate": risk_free_rate,
            "risk_free_date": risk_free_date,
            "results": [item.to_dict() for item in results],
            "warnings": provider.warnings,
            "failures": failures,
            "excel_path": str(excel_path) if excel_path else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_summary(results, excel_path, failures)
        if args.verbose:
            print()
            _print_report(results, risk_free_rate, risk_free_date, provider.warnings, failures)
    return 0 if results else 2


def _import_valuation_csv(
    repository: MySQLRepository,
    specs,
    csv_path: Path,
    source: str,
) -> None:
    if not csv_path.exists():
        raise ValueError(f"CSV不存在: {csv_path}")
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"date", "pe"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"估值CSV缺少字段: {', '.join(missing)}")
    for column in ("pb", "dividend_yield"):
        if column not in frame:
            frame[column] = pd.NA
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("pe", "pb", "dividend_yield"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "pe"]).sort_values("date")
    if len(frame) < 20:
        raise ValueError("估值CSV少于20条有效PE记录，无法形成可靠历史分位")
    spec = specs[0]
    count = repository.upsert_valuations(
        spec.code, spec.name, spec.etf_code, frame,
        source=source, quality="exact",
    )
    print(f"已导入MySQL：{spec.code} {spec.name}，{count}条，source={source}")


def _print_summary(
    results: list[ValuationResult], excel_path: Path | None, failures: list[str]
) -> None:
    from think.etf.strategy import decision_text

    print("本周定投结论")
    print("=" * 72)
    for item in results:
        amount = ""
        if item.suggested_amount is not None:
            amount = f"，建议金额 {item.suggested_amount:.2f} 元"
        print(
            f"{item.etf_code} {item.name}：{decision_text(item)}｜"
            f"倍率 {item.suggested_multiplier:.2f}x{amount}｜{item.reason}"
        )
    if failures:
        print("\n以下指数分析失败：")
        for failure in failures:
            print(f"- {failure}")
    if excel_path:
        print(f"\nExcel报告：{excel_path}")
    print("执行建议：每周固定一天运行一次；只有触发明显大跌时，才在当周临时重跑一次。")


def _print_report(
    results: list[ValuationResult],
    risk_free_rate: float,
    risk_free_date: str,
    warnings: list[str],
    failures: list[str],
) -> None:
    print(f"10年期国债收益率: {risk_free_rate:.2f}%（{risk_free_date}）")
    print("估值状态综合近10年PE/PB分位与当前股债收益差；PE越低通常越便宜。\n")
    rows = []
    for item in results:
        row = {
            "代码": item.code,
            "指数": item.name,
            "ETF": item.etf_code,
            "状态": item.status,
            "可信度": item.confidence,
            "PE": item.pe.current,
            "官方PE快照": item.official_pe,
            "PE分位": item.pe.percentile,
            "PE20%": item.pe.q20,
            "PE中位": item.pe.median,
            "PE80%": item.pe.q80,
            "盈利收益率": item.earnings_yield,
            "股债差": item.earnings_yield_spread,
            "综合评分": item.composite_percentile,
            "近5日": _percent_or_none(item.market.return_5d),
            "60日回撤": _percent_or_none(item.market.drawdown_60d),
            "建议倍数": item.suggested_multiplier,
        }
        if item.pb is not None:
            row["PB"] = item.pb.current
            row["PB分位"] = item.pb.percentile
        if item.official_dividend_yield is not None:
            row["股息率快照"] = item.official_dividend_yield
        if item.suggested_amount is not None:
            row["基础额"] = item.base_amount
            row["建议额"] = item.suggested_amount
        rows.append(row)
    if rows:
        frame = pd.DataFrame(rows)
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(frame.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    for item in results:
        print(
            f"\n[{item.code} {item.name}] {item.reason}；"
            f"估值样本 {item.pe.start_date}~{item.pe.end_date}（{item.pe.sample_count}期）。"
        )
        if item.official_pe is not None:
            print(
                f"  中证指数快照：PE {item.official_pe:.2f}，"
                f"股息率 {_format_optional(item.official_dividend_yield, '%')}，"
                f"日期 {item.official_date}。"
            )
        if item.note:
            print(f"  口径说明：{item.note}")

    if warnings:
        print("\n数据警告：")
        for warning in warnings:
            print(f"- {warning}")
    if failures:
        print("\n分析失败：")
        for failure in failures:
            print(f"- {failure}")
    print(
        "\n规则提醒：单日大跌不是自动抄底信号；只有估值不高时才启用回撤加仓，"
        "单指数最高2倍，组合单周金额还受 --max-weekly-multiplier 限制。"
    )


def _percent_or_none(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def _format_optional(value: float | None, suffix: str = "") -> str:
    return "无" if value is None else f"{value:.2f}{suffix}"


if __name__ == "__main__":
    main()
