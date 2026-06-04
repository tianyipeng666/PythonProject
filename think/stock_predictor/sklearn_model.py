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
    target_return: float = 0.0


def train_sklearn_model(
    df,
    horizon: int = 1,
    model_type: str = "logistic",
    target_return: float = 0.0,
    context_frames: dict[str, object] | None = None,
):
    dataset = build_dataset(
        df,
        horizon=horizon,
        target_return=target_return,
        context_frames=context_frames,
    ).dropna(subset=["label"])
    if len(dataset) < 80:
        raise ValueError("Training sample is too small; at least about 80 valid daily rows are required.")

    feature_columns = _feature_columns_from_dataset(dataset)
    split = max(1, int(len(dataset) * 0.75))
    train_df = dataset.iloc[:split]
    test_df = dataset.iloc[split:]

    classifier = _make_classifier(model_type)
    regressor = _make_regressor(model_type)
    classifier.fit(train_df[feature_columns], train_df["label"].astype(int))
    regressor.fit(train_df[feature_columns], train_df["future_return"].astype(float))

    model_bundle = {
        "classifier": classifier,
        "regressor": regressor,
        "feature_columns": feature_columns,
        "target_return": target_return,
    }
    eval_result = evaluate_sklearn_model(model_bundle, test_df)
    return_mae = evaluate_return_model(model_bundle, test_df)
    result = TrainResult(
        train_rows=len(train_df),
        test_rows=eval_result.test_rows,
        accuracy=eval_result.accuracy,
        signal_count=eval_result.signal_count,
        signal_win_rate=eval_result.signal_win_rate,
        avg_signal_return=eval_result.avg_signal_return,
        max_drawdown=eval_result.max_drawdown,
        return_mae=return_mae,
        target_return=target_return,
    )
    metadata = {
        "horizon": horizon,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "target_return": target_return,
        "last_train_date": str(train_df["date"].iloc[-1]),
        "last_test_date": str(test_df["date"].iloc[-1]) if len(test_df) else None,
    }
    return model_bundle, result, metadata


def evaluate_sklearn_model(model, dataset, threshold: float = 0.55) -> TrainResult:
    classifier, _regressor, feature_columns, target_return = _split_model(model)
    data = dataset.dropna(subset=["label", "future_return"])
    if data.empty:
        raise ValueError("No rows are available for evaluation.")

    proba = classifier.predict_proba(data[feature_columns])[:, 1]
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
        target_return=target_return,
    )


def evaluate_return_model(model, dataset) -> float:
    _classifier, regressor, feature_columns, _target_return = _split_model(model)
    if regressor is None:
        return 0.0
    data = dataset.dropna(subset=["future_return"])
    if data.empty:
        return 0.0
    pred = regressor.predict(data[feature_columns])
    return float(mean_absolute_error(data["future_return"].astype(float), pred))


def walk_forward_evaluate(
    df,
    horizon: int = 1,
    model_type: str = "logistic",
    threshold: float = 0.55,
    target_return: float = 0.0,
    min_train_rows: int = 720,
    step: int = 20,
    context_frames: dict[str, object] | None = None,
) -> TrainResult:
    dataset = build_dataset(
        df,
        horizon=horizon,
        target_return=target_return,
        context_frames=context_frames,
    ).dropna(subset=["label", "future_return"])
    feature_columns = _feature_columns_from_dataset(dataset)
    if len(dataset) < min_train_rows + step:
        raise ValueError("Not enough rows for walk-forward evaluation.")

    probabilities: list[float] = []
    labels: list[int] = []
    future_returns: list[float] = []
    return_predictions: list[float] = []
    for start in range(min_train_rows, len(dataset), step):
        train_df = dataset.iloc[:start]
        test_df = dataset.iloc[start : min(start + step, len(dataset))]
        if test_df.empty:
            continue
        classifier = _make_classifier(model_type)
        regressor = _make_regressor(model_type)
        classifier.fit(train_df[feature_columns], train_df["label"].astype(int))
        regressor.fit(train_df[feature_columns], train_df["future_return"].astype(float))
        probabilities.extend(classifier.predict_proba(test_df[feature_columns])[:, 1].tolist())
        return_predictions.extend(regressor.predict(test_df[feature_columns]).tolist())
        labels.extend(test_df["label"].astype(int).tolist())
        future_returns.extend(test_df["future_return"].astype(float).tolist())

    result_df = pd.DataFrame(
        {
            "probability": probabilities,
            "label": labels,
            "future_return": future_returns,
            "return_prediction": return_predictions,
        }
    )
    pred = (result_df["probability"] >= 0.5).astype(int)
    signals = result_df["probability"] >= threshold
    signal_returns = result_df.loc[signals, "future_return"]
    return TrainResult(
        train_rows=min_train_rows,
        test_rows=len(result_df),
        accuracy=float(accuracy_score(result_df["label"].astype(int), pred)),
        signal_count=int(signals.sum()),
        signal_win_rate=float((signal_returns > 0).mean()) if len(signal_returns) else 0.0,
        avg_signal_return=float(signal_returns.mean()) if len(signal_returns) else 0.0,
        max_drawdown=_max_drawdown(signal_returns),
        return_mae=float(
            mean_absolute_error(result_df["future_return"], result_df["return_prediction"])
        ),
        target_return=target_return,
    )


def predict_latest(
    model,
    df,
    horizon: int,
    target_return: float | None = None,
    context_frames: dict[str, object] | None = None,
):
    classifier, regressor, feature_columns, model_target_return = _split_model(model)
    if target_return is None:
        target_return = model_target_return
    dataset = build_dataset(
        df,
        horizon=horizon,
        target_return=target_return,
        context_frames=context_frames,
    )
    latest = dataset.tail(1).iloc[0]
    latest_x = latest[feature_columns].to_frame().T
    probability = float(classifier.predict_proba(latest_x)[:, 1][0])
    predicted_return = float(regressor.predict(latest_x)[0]) if regressor is not None else None
    return {
        "date": str(latest["date"]),
        "close": float(latest["close"]),
        "horizon": horizon,
        "up_probability": probability,
        "target_return": target_return,
        "predicted_return": predicted_return,
        "predicted_close": (
            float(latest["close"]) * (1.0 + predicted_return)
            if predicted_return is not None
            else None
        ),
        "features": latest[feature_columns].to_dict(),
    }


def save_model(model, metadata: dict, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, out)


def load_model(path: str | Path):
    payload = joblib.load(path)
    return payload["model"], payload["metadata"]


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
        return (
            model["classifier"],
            model.get("regressor"),
            list(model.get("feature_columns", FEATURE_COLUMNS)),
            float(model.get("target_return", 0.0)),
        )
    return model, None, FEATURE_COLUMNS, 0.0


def _feature_columns_from_dataset(dataset) -> list[str]:
    return list(dataset.attrs.get("feature_columns") or FEATURE_COLUMNS)


def _max_drawdown(returns: pd.Series) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in returns:
        equity *= 1.0 + float(ret)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
    return float(max_dd)
