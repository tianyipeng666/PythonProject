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
from .features import FEATURE_NAMES, build_feature_rows
from .model import LogisticModel, train_logistic_regression
from .risk import build_risk_report, build_risk_report_df
from .sklearn_model import (
    evaluate_sklearn_model,
    load_model,
    predict_latest,
    save_model,
    train_sklearn_model,
)


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
    fetch.add_argument("--asset-type", default="stock", choices=["stock", "index"])
    fetch.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])

    train = sub.add_parser("train", help="Train the sklearn short-term probability model")
    train.add_argument("--csv", required=True)
    train.add_argument("--horizon", type=int, default=1)
    train.add_argument("--model", default="think/models/short_term_sklearn.joblib")
    train.add_argument("--model-type", default="logistic", choices=["logistic", "random_forest"])

    predict = sub.add_parser("predict", help="Predict latest up probability and risk")
    predict.add_argument("--csv", required=True)
    predict.add_argument("--model", default="think/models/short_term_sklearn.joblib")

    bt = sub.add_parser("backtest", help="Evaluate historical prediction behavior")
    bt.add_argument("--csv", required=True)
    bt.add_argument("--model", default="think/models/short_term_sklearn.joblib")
    bt.add_argument("--threshold", type=float, default=0.55)

    quick = sub.add_parser("quick", help="Fetch, train, predict, and backtest in one command")
    quick.add_argument("--symbol", required=True)
    quick.add_argument("--asset-type", default="stock", choices=["stock", "index"])
    quick.add_argument("--horizon", type=int, default=1)
    quick.add_argument("--model-type", default="logistic", choices=["logistic", "random_forest"])
    quick.add_argument("--threshold", type=float, default=0.55)
    quick.add_argument("--csv-output", default=None)

    week = sub.add_parser("week-predict", help="Predict this week's remaining trading days")
    week.add_argument("--symbol", default="科创100")
    week.add_argument("--asset-type", default="index", choices=["stock", "index"])
    week.add_argument("--model-type", default="logistic", choices=["logistic", "random_forest"])
    week.add_argument("--threshold", type=float, default=0.55)
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
        _train_sklearn(args.csv, args.model, args.horizon, args.model_type)
    elif args.command == "predict":
        _predict_sklearn(args.csv, args.model)
    elif args.command == "backtest":
        _backtest_sklearn(args.csv, args.model, args.threshold)
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


def _train_sklearn(csv_path: str, model_path: str, horizon: int, model_type: str) -> None:
    df = load_price_df(csv_path)
    model, result, metadata = train_sklearn_model(df, horizon=horizon, model_type=model_type)
    save_model(model, metadata, model_path)
    print(f"Model written: {model_path}")
    print(f"Horizon: {horizon} trading day(s)")
    print(f"Model type: {model_type}")
    print(f"Train rows: {result.train_rows}, test rows: {result.test_rows}")
    print(f"Test accuracy: {_pct(result.accuracy)}")
    print(f"Signals at 55%: {result.signal_count}")
    print(f"Signal win rate: {_pct(result.signal_win_rate)}")
    print(f"Average signal return: {_pct(result.avg_signal_return)}")
    print(f"Signal max drawdown: {_pct(result.max_drawdown)}")


def _predict_sklearn(csv_path: str, model_path: str) -> None:
    df = load_price_df(csv_path)
    model, metadata = load_model(model_path)
    prediction = predict_latest(model, df, horizon=int(metadata["horizon"]))
    risk = build_risk_report_df(df, prediction["up_probability"])
    _print_prediction(prediction, risk)


def _backtest_sklearn(csv_path: str, model_path: str, threshold: float) -> None:
    from .ml_features import build_dataset

    df = load_price_df(csv_path)
    model, metadata = load_model(model_path)
    dataset = build_dataset(df, horizon=int(metadata["horizon"])).dropna(
        subset=["label", "future_return"]
    )
    result = evaluate_sklearn_model(model, dataset, threshold=threshold)
    _print_backtest(result, threshold)


def _quick(args: argparse.Namespace) -> None:
    df = fetch_akshare_price_df(args.symbol, asset_type=args.asset_type)
    if args.asset_type == "index":
        df = append_index_spot_if_newer(df, args.symbol)
    if args.csv_output:
        save_price_df(df, args.csv_output)
        print(f"Data written: {args.csv_output}")
    model, result, metadata = train_sklearn_model(
        df, horizon=args.horizon, model_type=args.model_type
    )
    prediction = predict_latest(model, df, horizon=args.horizon)
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
            df, horizon=idx, model_type=args.model_type
        )
        prediction = predict_latest(model, df, horizon=idx)
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
    print(f"Accuracy: {_pct(result.accuracy)}")
    print(f"Signals at {_pct(threshold)}: {result.signal_count}")
    print(f"Signal win rate: {_pct(result.signal_win_rate)}")
    print(f"Average signal return: {_pct(result.avg_signal_return)}")
    print(f"Signal max drawdown: {_pct(result.max_drawdown)}")
    if getattr(result, "return_mae", None) is not None:
        print(f"Return MAE: {_pct(result.return_mae)}")


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


if __name__ == "__main__":
    main()
