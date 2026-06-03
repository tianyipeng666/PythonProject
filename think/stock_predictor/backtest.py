from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FeatureRow
from .model import LogisticModel


@dataclass(frozen=True)
class BacktestResult:
    samples: int
    signals: int
    accuracy: float
    signal_win_rate: float
    avg_signal_return: float
    max_drawdown: float


def evaluate_predictions(
    model: LogisticModel,
    rows: list[FeatureRow],
    threshold: float = 0.55,
) -> BacktestResult:
    usable = [r for r in rows if r.label is not None and r.future_return is not None]
    if not usable:
        raise ValueError("No labeled feature rows available for backtest.")

    x = np.array([r.features for r in usable], dtype=float)
    y = np.array([r.label for r in usable], dtype=int)
    future_returns = np.array([r.future_return for r in usable], dtype=float)
    prob = model.predict_proba(x)
    pred = (prob >= 0.5).astype(int)
    signals = prob >= threshold
    signal_returns = future_returns[signals]

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for ret in signal_returns:
        equity *= 1.0 + float(ret)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)

    if len(signal_returns) == 0:
        signal_win_rate = 0.0
        avg_signal_return = 0.0
    else:
        signal_win_rate = float((signal_returns > 0).mean())
        avg_signal_return = float(signal_returns.mean())

    return BacktestResult(
        samples=len(usable),
        signals=int(signals.sum()),
        accuracy=float((pred == y).mean()),
        signal_win_rate=signal_win_rate,
        avg_signal_return=avg_signal_return,
        max_drawdown=float(max_drawdown),
    )

