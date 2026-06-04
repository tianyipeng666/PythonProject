from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .backtest import evaluate_predictions
from .data_loader import (
    fetch_akshare_daily,
    append_index_spot_if_newer,
    fetch_akshare_price_df,
    load_csv,
    load_price_df,
    save_csv,
    save_price_df,
)
from .demo_data import make_demo_bars
from .db import (
    InstrumentSpec,
    import_daily_csv,
    import_daily_df,
    init_schema,
    connect,
    create_database,
)
from .features import FEATURE_NAMES, build_feature_rows
from .model import LogisticModel, train_logistic_regression
from .risk import build_risk_report, build_risk_report_df
from .sklearn_model import (
    evaluate_sklearn_model,
    load_model,
    predict_latest,
    save_model,
    train_sklearn_model,
    walk_forward_evaluate,
)
from .watchlist import DEFAULT_CONFIG_PATH, predict_watchlist


DEFAULT_MODEL_PATH = Path("think/models/short_term_logistic.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal A-share short-term predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo-data", help="Create a local demo CSV")
    demo.add_argument("--output", default="think/data/demo_stock.csv")
    demo.add_argument("--days", type=int, default=260)

    fetch = sub.add_parser("fetch", help="Fetch A-share daily data with AkShare")
    fetch.add_argument("--symbol", required=True, help="Example: 600519 or 000001")
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--asset-type", default="stock", choices=["stock", "index", "hk_index"])
    fetch.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])

    train = sub.add_parser("train", help="Train the sklearn short-term probability model")
    train.add_argument("--csv", required=True)
    train.add_argument("--horizon", type=int, default=1)
    train.add_argument("--model", default="think/models/short_term_sklearn.joblib")
    train.add_argument("--model-type", default="logistic", choices=["logistic", "random_forest"])
    train.add_argument("--target-return", type=float, default=0.0)
    train.add_argument("--context-csv", action="append", default=[])

    predict = sub.add_parser("predict", help="Predict latest up probability and risk")
    predict.add_argument("--csv", required=True)
    predict.add_argument("--model", default="think/models/short_term_sklearn.joblib")
    predict.add_argument("--context-csv", action="append", default=[])

    bt = sub.add_parser("backtest", help="Evaluate historical prediction behavior")
    bt.add_argument("--csv", required=True)
    bt.add_argument("--model", default="think/models/short_term_sklearn.joblib")
    bt.add_argument("--threshold", type=float, default=0.55)
    bt.add_argument("--context-csv", action="append", default=[])

    wf = sub.add_parser("walk-forward", help="Run rolling time-series evaluation")
    wf.add_argument("--csv", required=True)
    wf.add_argument("--horizon", type=int, default=1)
    wf.add_argument("--model-type", default="logistic", choices=["logistic", "random_forest"])
    wf.add_argument("--threshold", type=float, default=0.55)
    wf.add_argument("--target-return", type=float, default=0.0)
    wf.add_argument("--min-train-rows", type=int, default=720)
    wf.add_argument("--step", type=int, default=20)
    wf.add_argument("--context-csv", action="append", default=[])

    db_create = sub.add_parser("db-create", help="Create PostgreSQL database")
    db_create.add_argument("--database", default="stock_predictor")
    db_create.add_argument(
        "--maintenance-url",
        default="postgresql://postgres@localhost:5432/postgres",
    )

    db_init = sub.add_parser("db-init", help="Create PostgreSQL schema")
    db_init.add_argument("--database-url", default=None)

    db_import = sub.add_parser("db-import-csv", help="Import normalized daily CSV into PostgreSQL")
    db_import.add_argument("--csv", required=True)
    db_import.add_argument("--symbol", required=True)
    db_import.add_argument("--name", required=True)
    db_import.add_argument("--asset-type", required=True)
    db_import.add_argument("--market", required=True)
    db_import.add_argument("--provider-symbol", required=True)
    db_import.add_argument("--provider", default="akshare")
    db_import.add_argument("--currency", default="CNY")
    db_import.add_argument("--timezone", default="Asia/Shanghai")
    db_import.add_argument("--source", default="csv")
    db_import.add_argument("--database-url", default=None)

    db_fetch = sub.add_parser("db-fetch", help="Fetch daily bars and write them into PostgreSQL")
    db_fetch.add_argument("--symbol", required=True)
    db_fetch.add_argument("--name", required=True)
    db_fetch.add_argument("--asset-type", required=True, choices=["stock", "index", "hk_index"])
    db_fetch.add_argument("--market", required=True)
    db_fetch.add_argument("--provider-symbol", required=True)
    db_fetch.add_argument("--provider", default="akshare")
    db_fetch.add_argument("--currency", default="CNY")
    db_fetch.add_argument("--timezone", default="Asia/Shanghai")
    db_fetch.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])
    db_fetch.add_argument("--database-url", default=None)

    db_list = sub.add_parser("db-list", help="List instruments in PostgreSQL")
    db_list.add_argument("--database-url", default=None)

    watch = sub.add_parser("predict-watchlist", help="Refresh configured funds and predict them")
    watch.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    watch.add_argument("--database-url", default=None)
    watch.add_argument("--no-refresh", action="store_true")
    watch.add_argument("--no-prune", action="store_true")
    watch.add_argument("--force-refresh", action="store_true")

    quick = sub.add_parser("quick", help="Fetch, train, predict, and backtest in one command")
    quick.add_argument("--symbol", required=True)
    quick.add_argument("--asset-type", default="stock", choices=["stock", "index", "hk_index"])
    quick.add_argument("--horizon", type=int, default=1)
    quick.add_argument("--model-type", default="logistic", choices=["logistic", "random_forest"])
    quick.add_argument("--threshold", type=float, default=0.55)
    quick.add_argument("--target-return", type=float, default=0.0)
    quick.add_argument("--context-csv", action="append", default=[])
    quick.add_argument("--csv-output", default=None)

    week = sub.add_parser("week-predict", help="Predict this week's remaining trading days")
    week.add_argument("--symbol", default="科创100")
    week.add_argument("--asset-type", default="index", choices=["stock", "index", "hk_index"])
    week.add_argument("--model-type", default="logistic", choices=["logistic", "random_forest"])
    week.add_argument("--threshold", type=float, default=0.55)
    week.add_argument("--target-return", type=float, default=0.0)
    week.add_argument("--context-csv", action="append", default=[])
    week.add_argument(
        "--as-of",
        default=None,
        help="Only use rows up to this date, for example 2026-06-02.",
    )

    sub.add_parser("streamlit", help="Print the Streamlit startup command")

    args = parser.parse_args()

    if args.command == "demo-data":
        bars = make_demo_bars(days=args.days)
        save_csv(bars, args.output)
        print(f"Demo data written: {args.output}")
    elif args.command == "fetch":
        df = fetch_akshare_price_df(
            args.symbol, asset_type=args.asset_type, adjust=args.adjust
        )
        save_price_df(df, args.output)
        print(f"Fetched {len(df)} rows: {args.output}")
    elif args.command == "train":
        _train_sklearn(args.csv, args.model, args.horizon, args.model_type, args.target_return, args.context_csv)
    elif args.command == "predict":
        _predict_sklearn(args.csv, args.model, args.context_csv)
    elif args.command == "backtest":
        _backtest_sklearn(args.csv, args.model, args.threshold, args.context_csv)
    elif args.command == "walk-forward":
        _walk_forward(args)
    elif args.command == "db-create":
        _db_create(args)
    elif args.command == "db-init":
        _db_init(args)
    elif args.command == "db-import-csv":
        _db_import_csv(args)
    elif args.command == "db-fetch":
        _db_fetch(args)
    elif args.command == "db-list":
        _db_list(args)
    elif args.command == "predict-watchlist":
        _predict_watchlist(args)
    elif args.command == "quick":
        _quick(args)
    elif args.command == "week-predict":
        _week_predict(args)
    elif args.command == "streamlit":
        print("streamlit run think/streamlit_app.py")


def _train(csv_path: str, model_path: str, horizon: int) -> None:
    bars = load_csv(csv_path)
    rows = build_feature_rows(bars, horizon=horizon)
    labeled = [r for r in rows if r.label is not None]
    split = int(len(labeled) * 0.75)
    train_rows = labeled[:split]
    test_rows = labeled[split:]

    x_train = np.array([r.features for r in train_rows], dtype=float)
    y_train = np.array([r.label for r in train_rows], dtype=float)
    model = train_logistic_regression(x_train, y_train)
    model.save(model_path, FEATURE_NAMES, horizon)

    result = evaluate_predictions(model, test_rows)
    print(f"Model written: {model_path}")
    print(f"Horizon: {horizon} trading days")
    print(f"Train rows: {len(train_rows)}, test rows: {len(test_rows)}")
    print(f"Test accuracy: {_pct(result.accuracy)}")
    print(f"Signals at 55%: {result.signals}")
    print(f"Signal win rate: {_pct(result.signal_win_rate)}")
    print(f"Average signal return: {_pct(result.avg_signal_return)}")
    print(f"Signal max drawdown: {_pct(result.max_drawdown)}")
    if result.return_mae is not None:
        print(f"Return MAE: {_pct(result.return_mae)}")


def _predict(csv_path: str, model_path: str) -> None:
    model, payload = LogisticModel.load(model_path)
    horizon = int(payload.get("horizon", 3))
    bars = load_csv(csv_path)
    latest = build_feature_rows(bars, horizon=horizon)[-1]
    probability = float(model.predict_proba(np.array([latest.features], dtype=float))[0])
    risk = build_risk_report(bars, probability)

    print(f"Date: {latest.date}")
    print(f"Close: {latest.close:.2f}")
    print(f"Forecast horizon: {horizon} trading days")
    print(f"Up probability: {_pct(probability)}")
    print(f"Risk level: {risk['risk_level']} (score={risk['risk_score']})")
    print("Risk tips:")
    for tip in risk["tips"]:
        print(f"- {tip}")


def _backtest(csv_path: str, model_path: str, threshold: float) -> None:
    model, payload = LogisticModel.load(model_path)
    horizon = int(payload.get("horizon", 3))
    rows = build_feature_rows(load_csv(csv_path), horizon=horizon)
    result = evaluate_predictions(model, rows, threshold=threshold)
    print(f"Samples: {result.samples}")
    print(f"Signals at {_pct(threshold)}: {result.signals}")
    print(f"Accuracy: {_pct(result.accuracy)}")
    print(f"Signal win rate: {_pct(result.signal_win_rate)}")
    print(f"Average signal return: {_pct(result.avg_signal_return)}")
    print(f"Signal max drawdown: {_pct(result.max_drawdown)}")


def _train_sklearn(
    csv_path: str,
    model_path: str,
    horizon: int,
    model_type: str,
    target_return: float,
    context_specs: list[str],
) -> None:
    df = load_price_df(csv_path)
    context_frames = _load_context_frames(context_specs)
    model, result, metadata = train_sklearn_model(
        df,
        horizon=horizon,
        model_type=model_type,
        target_return=target_return,
        context_frames=context_frames,
    )
    save_model(model, metadata, model_path)
    print(f"Model written: {model_path}")
    print(f"Horizon: {horizon} trading day(s)")
    print(f"Model type: {model_type}")
    print(f"Target return: {_pct(target_return)}")
    print(f"Feature count: {len(metadata['feature_columns'])}")
    print(f"Train rows: {result.train_rows}, test rows: {result.test_rows}")
    print(f"Test accuracy: {_pct(result.accuracy)}")
    print(f"Signals at 55%: {result.signal_count}")
    print(f"Signal win rate: {_pct(result.signal_win_rate)}")
    print(f"Average signal return: {_pct(result.avg_signal_return)}")
    print(f"Signal max drawdown: {_pct(result.max_drawdown)}")


def _predict_sklearn(csv_path: str, model_path: str, context_specs: list[str]) -> None:
    df = load_price_df(csv_path)
    context_frames = _load_context_frames(context_specs)
    model, metadata = load_model(model_path)
    prediction = predict_latest(
        model,
        df,
        horizon=int(metadata["horizon"]),
        target_return=float(metadata.get("target_return", 0.0)),
        context_frames=context_frames,
    )
    risk = build_risk_report_df(df, prediction["up_probability"])
    _print_prediction(prediction, risk)


def _backtest_sklearn(
    csv_path: str,
    model_path: str,
    threshold: float,
    context_specs: list[str],
) -> None:
    from .ml_features import build_dataset

    df = load_price_df(csv_path)
    context_frames = _load_context_frames(context_specs)
    model, metadata = load_model(model_path)
    dataset = build_dataset(
        df,
        horizon=int(metadata["horizon"]),
        target_return=float(metadata.get("target_return", 0.0)),
        context_frames=context_frames,
    ).dropna(
        subset=["label", "future_return"]
    )
    result = evaluate_sklearn_model(model, dataset, threshold=threshold)
    _print_backtest(result, threshold)


def _walk_forward(args: argparse.Namespace) -> None:
    df = load_price_df(args.csv)
    context_frames = _load_context_frames(args.context_csv)
    result = walk_forward_evaluate(
        df,
        horizon=args.horizon,
        model_type=args.model_type,
        threshold=args.threshold,
        target_return=args.target_return,
        min_train_rows=args.min_train_rows,
        step=args.step,
        context_frames=context_frames,
    )
    print("Walk-forward backtest:")
    print(f"Horizon: {args.horizon} trading day(s)")
    print(f"Model type: {args.model_type}")
    print(f"Target return: {_pct(args.target_return)}")
    print(f"Min train rows: {args.min_train_rows}")
    print(f"Step: {args.step}")
    _print_backtest(result, args.threshold)


def _quick(args: argparse.Namespace) -> None:
    df = fetch_akshare_price_df(args.symbol, asset_type=args.asset_type)
    if args.asset_type == "index":
        df = append_index_spot_if_newer(df, args.symbol)
    if args.csv_output:
        save_price_df(df, args.csv_output)
        print(f"Data written: {args.csv_output}")
    context_frames = _load_context_frames(args.context_csv)
    model, result, metadata = train_sklearn_model(
        df,
        horizon=args.horizon,
        model_type=args.model_type,
        target_return=args.target_return,
        context_frames=context_frames,
    )
    prediction = predict_latest(
        model,
        df,
        horizon=args.horizon,
        target_return=args.target_return,
        context_frames=context_frames,
    )
    risk = build_risk_report_df(df, prediction["up_probability"])
    print(f"Symbol: {args.symbol}")
    _print_prediction(prediction, risk)
    print("Backtest:")
    _print_backtest(result, args.threshold)


def _week_predict(args: argparse.Namespace) -> None:
    df = fetch_akshare_price_df(args.symbol, asset_type=args.asset_type)
    if args.asset_type == "index":
        df = append_index_spot_if_newer(df, args.symbol)
    if args.as_of:
        df = df[df["date"] <= args.as_of].reset_index(drop=True)
    latest_data_date = _parse_iso_date(str(df["date"].iloc[-1]))
    remaining = _remaining_weekdays(latest_data_date)
    if not remaining:
        print("最新数据日期所在周没有可预测的后续工作日。")
        return
    print(f"Symbol: {args.symbol}")
    print(f"Latest data date: {df['date'].iloc[-1]}")
    print(f"Current date: {date.today().isoformat()}")
    if args.as_of:
        print(f"As-of cutoff: {args.as_of}")
    for idx, day in enumerate(remaining, start=1):
        model, result, _metadata = train_sklearn_model(
            df,
            horizon=idx,
            model_type=args.model_type,
            target_return=args.target_return,
            context_frames=_load_context_frames(args.context_csv),
        )
        prediction = predict_latest(
            model,
            df,
            horizon=idx,
            target_return=args.target_return,
            context_frames=_load_context_frames(args.context_csv),
        )
        risk = build_risk_report_df(df, prediction["up_probability"])
        direction = "上涨" if prediction["up_probability"] >= 0.5 else "下跌"
        print("")
        print(f"{day.isoformat()} forecast horizon={idx}: {direction}")
        print(f"Up probability: {_pct(prediction['up_probability'])}")
        if prediction.get("predicted_return") is not None:
            print(f"Predicted return: {_pct(prediction['predicted_return'])}")
            print(f"Predicted close: {prediction['predicted_close']:.2f}")
        print(f"Risk level: {risk['risk_level']} (score={risk['risk_score']})")
        print(f"Backtest accuracy: {_pct(result.accuracy)}")
        print(f"Signal win rate at {_pct(args.threshold)}: {_pct(result.signal_win_rate)}")
        if result.return_mae is not None:
            print(f"Return MAE: {_pct(result.return_mae)}")
        print("Risk tips:")
        for tip in risk["tips"]:
            print(f"- {tip}")


def _remaining_weekdays(today: date) -> list[date]:
    days: list[date] = []
    current = today + timedelta(days=1)
    while current.weekday() < 5:
        days.append(current)
        current += timedelta(days=1)
    return days


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def _print_prediction(prediction: dict, risk: dict) -> None:
    print(f"Date: {prediction['date']}")
    print(f"Close: {prediction['close']:.2f}")
    print(f"Forecast horizon: {prediction['horizon']} trading day(s)")
    print(f"Target return: {_pct(prediction.get('target_return', 0.0))}")
    print(f"Up probability: {_pct(prediction['up_probability'])}")
    if prediction.get("predicted_return") is not None:
        print(f"Predicted return: {_pct(prediction['predicted_return'])}")
        print(f"Predicted close: {prediction['predicted_close']:.2f}")
    print(f"Risk level: {risk['risk_level']} (score={risk['risk_score']})")
    print("Risk tips:")
    for tip in risk["tips"]:
        print(f"- {tip}")


def _print_backtest(result, threshold: float) -> None:
    print(f"Test rows: {result.test_rows}")
    print(f"Target return: {_pct(getattr(result, 'target_return', 0.0))}")
    print(f"Accuracy: {_pct(result.accuracy)}")
    print(f"Signals at {_pct(threshold)}: {result.signal_count}")
    print(f"Signal win rate: {_pct(result.signal_win_rate)}")
    print(f"Average signal return: {_pct(result.avg_signal_return)}")
    print(f"Signal max drawdown: {_pct(result.max_drawdown)}")
    if getattr(result, "return_mae", None) is not None:
        print(f"Return MAE: {_pct(result.return_mae)}")


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _load_context_frames(specs: list[str]) -> dict[str, object]:
    frames = {}
    for spec in specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            path = spec
            name = Path(path).stem
        frames[name.strip()] = load_price_df(path.strip())
    return frames


def _db_init(args: argparse.Namespace) -> None:
    init_schema(args.database_url)
    print("PostgreSQL schema initialized.")


def _db_create(args: argparse.Namespace) -> None:
    created = create_database(args.database, args.maintenance_url)
    if created:
        print(f"Database created: {args.database}")
    else:
        print(f"Database already exists: {args.database}")


def _db_import_csv(args: argparse.Namespace) -> None:
    count = import_daily_csv(
        args.csv,
        InstrumentSpec(
            symbol=args.symbol,
            name=args.name,
            asset_type=args.asset_type,
            market=args.market,
            provider=args.provider,
            provider_symbol=args.provider_symbol,
            currency=args.currency,
            timezone=args.timezone,
        ),
        url=args.database_url,
        source=args.source,
    )
    print(f"Imported {count} daily bars for {args.symbol}.")


def _db_fetch(args: argparse.Namespace) -> None:
    df = fetch_akshare_price_df(
        args.provider_symbol,
        asset_type=args.asset_type,
        adjust=args.adjust,
    )
    if args.asset_type == "index":
        df = append_index_spot_if_newer(df, args.provider_symbol)
    count = import_daily_df(
        df,
        InstrumentSpec(
            symbol=args.symbol,
            name=args.name,
            asset_type=args.asset_type,
            market=args.market,
            provider=args.provider,
            provider_symbol=args.provider_symbol,
            currency=args.currency,
            timezone=args.timezone,
        ),
        url=args.database_url,
        source=args.provider,
    )
    print(f"Fetched and imported {count} daily bars for {args.symbol}.")


def _db_list(args: argparse.Namespace) -> None:
    try:
        with connect(args.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        i.symbol,
                        i.name,
                        i.asset_type,
                        i.market,
                        i.provider_symbol,
                        count(b.trade_date) AS bars,
                        min(b.trade_date) AS start_date,
                        max(b.trade_date) AS end_date
                    FROM instruments i
                    LEFT JOIN daily_bars b ON b.instrument_id = i.id
                    GROUP BY i.id
                    ORDER BY i.symbol
                    """
                )
                rows = cur.fetchall()
    except Exception as exc:
        print(f"Failed to connect to PostgreSQL: {exc}")
        return
    if not rows:
        print("No instruments found.")
        return
    for row in rows:
        print(
            f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | "
            f"provider_symbol={row[4]} | bars={row[5]} | {row[6]}..{row[7]}"
        )


def _predict_watchlist(args: argparse.Namespace) -> None:
    settings, imports, results = predict_watchlist(
        args.config,
        refresh=not args.no_refresh,
        prune=not args.no_prune,
        smart_refresh=not args.force_refresh,
        database_url=args.database_url,
    )
    print(
        f"Watchlist: horizon={settings.horizon}, "
        f"target_return={_pct(settings.target_return)}, "
        f"threshold={_pct(settings.threshold)}, model={settings.model_type}"
    )
    if imports:
        print("Refreshed:")
        for row in imports:
            print(
                f"- {row['symbol']} {row['name']} "
                f"status={row.get('status', 'fetched')} "
                f"source={row['source']} rows={row['rows']} "
                f"latest={row.get('latest_date')}"
            )
    print("Predictions:")
    for row in results:
        if "error" in row:
            print(f"- {row['symbol']} {row['name']}: FAILED {row['error']}")
            continue
        wf = ""
        if "walk_forward_accuracy" in row:
            wf = (
                f", 滚动准确率={_pct(row['walk_forward_accuracy'])}, "
                f"滚动信号胜率={_pct(row['walk_forward_signal_win_rate'])}"
            )
        print(
            f"- {row['symbol']} {row['name']}: "
            f"{row['forecast_start']}..{row['forecast_end']}, "
            f"涨超目标概率={_pct(row['probability'])}, "
            f"预测收益={_pct(row['predicted_return'])}, "
            f"预测净值={row['predicted_close']:.4f}, "
            f"风险={row['risk_level']}, "
            f"固定准确率={_pct(row['split_accuracy'])}, "
            f"固定信号胜率={_pct(row['split_signal_win_rate'])}"
            f"{wf}"
        )


if __name__ == "__main__":
    main()
