from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LogisticModel:
    weights: np.ndarray
    bias: float
    means: np.ndarray
    stds: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.means) / self.stds) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    def save(self, path: str | Path, feature_names: list[str], horizon: int) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": feature_names,
            "horizon": horizon,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> tuple["LogisticModel", dict]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            weights=np.array(payload["weights"], dtype=float),
            bias=float(payload["bias"]),
            means=np.array(payload["means"], dtype=float),
            stds=np.array(payload["stds"], dtype=float),
        )
        return model, payload


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.08,
    epochs: int = 1200,
    l2: float = 0.01,
) -> LogisticModel:
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1.0
    xs = (x - means) / stds

    weights = np.zeros(xs.shape[1], dtype=float)
    bias = 0.0
    n = len(y)

    for _ in range(epochs):
        logits = xs @ weights + bias
        pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        error = pred - y
        grad_w = (xs.T @ error) / n + l2 * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return LogisticModel(weights=weights, bias=bias, means=means, stds=stds)

