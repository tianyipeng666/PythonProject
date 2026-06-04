from __future__ import annotations

import os
from pathlib import Path

from stock_predictor.watchlist import predict_watchlist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "think" / "watchlist_funds.json"
DEFAULT_DATABASE_URL = "postgresql://postgres:8888@localhost:5432/stock_predictor?connect_timeout=5"


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def close_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def verdict(row: dict) -> str:
    probability = row["probability"]
    predicted_return = row["predicted_return"]
    walk_forward_win = row.get("walk_forward_signal_win_rate")
    if probability >= 0.65 and predicted_return > 0 and (walk_forward_win is None or walk_forward_win >= 0.55):
        return "偏强"
    if probability >= 0.55 and predicted_return > 0:
        return "偏强但需确认"
    if probability <= 0.45 and predicted_return < 0:
        return "偏弱"
    if predicted_return < 0:
        return "中性偏弱"
    return "中性"


def main() -> None:
    os.environ.setdefault("STOCK_PREDICTOR_DATABASE_URL", DEFAULT_DATABASE_URL)
    settings, refresh_rows, predictions = predict_watchlist(
        CONFIG_PATH,
        refresh=True,
        prune=True,
        smart_refresh=True,
    )

    print("基金观察池预测")
    print("=" * 72)
    print(
        f"配置: horizon={settings.horizon} 个交易日, "
        f"目标收益={pct(settings.target_return)}, "
        f"信号阈值={pct(settings.threshold)}, "
        f"模型={settings.model_type}"
    )
    print("")

    print("数据刷新")
    print("-" * 72)
    for row in refresh_rows:
        status = "跳过" if row.get("status") == "skipped" else "已刷新"
        print(
            f"{row['symbol']} {row['name']} | {status} | "
            f"source={row['source']} | rows={row['rows']} | latest={row.get('latest_date')}"
        )
    print("")

    print("预测结果")
    print("-" * 72)
    for row in predictions:
        if "error" in row:
            print(f"{row['symbol']} {row['name']} | 预测失败 | {row['error']}")
            continue
        wf = ""
        if "walk_forward_signal_win_rate" in row:
            wf = (
                f" | 滚动准确率={pct(row['walk_forward_accuracy'])}"
                f" | 滚动信号胜率={pct(row['walk_forward_signal_win_rate'])}"
            )
        print(
            f"{row['symbol']} {row['name']} | {verdict(row)} | "
            f"{row['forecast_start']}..{row['forecast_end']} | "
            f"涨超目标概率={pct(row['probability'])} | "
            f"预测收益={pct(row['predicted_return'])} | "
            f"预测净值={close_value(row['predicted_close'])} | "
            f"固定准确率={pct(row['split_accuracy'])} | "
            f"固定信号胜率={pct(row['split_signal_win_rate'])}"
            f"{wf}"
        )

    print("")
    print("说明")
    print("-" * 72)
    print("偏强: 概率、预测收益和历史信号质量相对一致。")
    print("偏强但需确认: 概率和收益偏多，但历史滚动信号或稳定性一般。")
    print("中性/中性偏弱/偏弱: 信号不够强，或者概率与预测收益不一致。")
    print("本结果仅为量化辅助，不构成投资建议。")


if __name__ == "__main__":
    main()
