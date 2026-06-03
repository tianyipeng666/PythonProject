from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .ml_features import FEATURE_COLUMNS, build_dataset


@dataclass(frozen=True)
class TrainResult:
    train_rows: int
    test_rows: int
    accuracy: float
    signal_count: int
    signal_win_rate: float
    avg_signal_return: float
    max_drawdown: float
    return_mae: float | None = None


def train_sklearn_model(df, horizon: int = 1, model_type: str = "logistic"):
    dataset = build_dataset(df, horizon=horizon).dropna(subset=["label"])
    if len(dataset) < 80:
        raise ValueError("训练样本不足，至少需要约 80 条有效日线记录。")

    split = max(1, int(len(dataset) * 0.75))
    train_df = dataset.iloc[:split]
    test_df = dataset.iloc[split:]
    model = _make_classifier(model_type)
    return_model = _make_regressor(model_type)
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"].astype(int))
    return_model.fit(train_df[FEATURE_COLUMNS], train_df["future_return"].astype(float))
    eval_result = evaluate_sklearn_model(model, test_df)
    return_mae = evaluate_return_model(return_model, test_df)
    result = TrainResult(
        train_rows=len(train_df),
        test_rows=eval_result.test_rows,
        accuracy=eval_result.accuracy,
        signal_count=eval_result.signal_count,
        signal_win_rate=eval_result.signal_win_rate,
        avg_signal_return=eval_result.avg_signal_return,
        max_drawdown=eval_result.max_drawdown,
        return_mae=return_mae,
    )
    metadata = {
        "horizon": horizon,
        "model_type": model_type,
        "feature_columns": FEATURE_COLUMNS,
        "last_train_date": str(train_df["date"].iloc[-1]),
        "last_test_date": str(test_df["date"].iloc[-1]) if len(test_df) else None,
    }
    return {"classifier": model, "regressor": return_model}, result, metadata


def evaluate_sklearn_model(model, dataset, threshold: float = 0.55) -> TrainResult:
    classifier, _regressor = _split_model(model)
    data = dataset.dropna(subset=["label", "future_return"])
    if data.empty:
        raise ValueError("没有可回测的数据。")
    proba = classifier.predict_proba(data[FEATURE_COLUMNS])[:, 1]
    pred = (proba >= 0.5).astype(int)
    labels = data["label"].astype(int)
    signals = proba >= threshold
    signal_returns = data.loc[signals, "future_return"]
    return TrainResult(
        train_rows=0,
        test_rows=len(data),
        accuracy=float(accuracy_score(labels, pred)),
        signal_count=int(signals.sum()),
        signal_win_rate=float((signal_returns > 0).mean()) if len(signal_returns) else 0.0,
        avg_signal_return=float(signal_returns.mean()) if len(signal_returns) else 0.0,
        max_drawdown=_max_drawdown(signal_returns),
    )


def predict_latest(model, df, horizon: int):
    classifier, regressor = _split_model(model)
    dataset = build_dataset(df, horizon=horizon)
    latest = dataset.tail(1).iloc[0]
    latest_x = latest[FEATURE_COLUMNS].to_frame().T
    probability = float(classifier.predict_proba(latest_x)[:, 1][0])
    predicted_return = float(regressor.predict(latest_x)[0]) if regressor is not None else None
    return {
        "date": str(latest["date"]),
        "close": float(latest["close"]),
        "horizon": horizon,
        "up_probability": probability,
        "predicted_return": predicted_return,
        "predicted_close": (
            float(latest["close"]) * (1.0 + predicted_return)
            if predicted_return is not None
            else None
        ),
        "features": latest[FEATURE_COLUMNS].to_dict(),
    }


def save_model(model, metadata: dict, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, out)


def load_model(path: str | Path):
    payload = joblib.load(path)
    return payload["model"], payload["metadata"]


def evaluate_return_model(model, dataset) -> float:
    data = dataset.dropna(subset=["future_return"])
    if data.empty:
        return 0.0
    pred = model.predict(data[FEATURE_COLUMNS])
    return float(mean_absolute_error(data["future_return"].astype(float), pred))


def _make_classifier(model_type: str):
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=8,
            random_state=42,
            class_weight="balanced_subsample",
        )
    if model_type != "logistic":
        raise ValueError("model_type must be 'logistic' or 'random_forest'.")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _make_regressor(model_type: str):
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=8,
            random_state=42,
        )
    if model_type != "logistic":
        raise ValueError("model_type must be 'logistic' or 'random_forest'.")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]
    )


def _split_model(model):
    if isinstance(model, dict):
        return model["classifier"], model.get("regressor")
    return model, None


def _max_drawdown(returns: pd.Series) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in returns:
        equity *= 1.0 + float(ret)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
    return float(max_dd)
