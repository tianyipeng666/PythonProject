from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data_loader import fetch_akshare_price_df, save_price_df
from .risk import build_risk_report_df
from .sklearn_model import evaluate_sklearn_model, predict_latest, train_sklearn_model


st.set_page_config(page_title="A股短线预测", layout="wide")


def main() -> None:
    st.title("A股短线涨跌概率预测")
    with st.sidebar:
        asset_type = st.selectbox("标的类型", ["stock", "index"], index=0)
        symbol = st.text_input("代码", value="600519" if asset_type == "stock" else "科创100")
        horizon = st.selectbox("预测周期", [1, 2, 3, 5], index=0)
        model_type = st.selectbox("模型", ["logistic", "random_forest"], index=0)
        threshold = st.slider("信号阈值", 0.50, 0.80, 0.55, 0.01)
        run = st.button("拉取并预测", type="primary")

    if not run:
        st.info("选择标的后点击左侧按钮。")
        return

    df = fetch_akshare_price_df(symbol, asset_type=asset_type)
    cache_path = Path("think/data") / f"{asset_type}_{symbol}_latest.csv"
    save_price_df(df, cache_path)
    model, _result, _metadata = train_sklearn_model(df, horizon=horizon, model_type=model_type)
    prediction = predict_latest(model, df, horizon=horizon)
    risk = build_risk_report_df(df, prediction["up_probability"])
    full_dataset = evaluate_sklearn_model(model, _drop_unlabeled_for_eval(df, horizon), threshold)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最新日期", prediction["date"])
    c2.metric("收盘价", f"{prediction['close']:.2f}")
    c3.metric("上涨概率", f"{prediction['up_probability'] * 100:.2f}%")
    c4.metric(
        "预计涨跌幅",
        f"{prediction['predicted_return'] * 100:.2f}%"
        if prediction["predicted_return"] is not None
        else "N/A",
    )
    c5.metric("风险等级", f"{risk['risk_level']} ({risk['risk_score']})")

    st.subheader("风险提示")
    for tip in risk["tips"]:
        st.write(f"- {tip}")

    st.subheader("回测指标")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("准确率", f"{full_dataset.accuracy * 100:.2f}%")
    m2.metric("信号数", str(full_dataset.signal_count))
    m3.metric("信号胜率", f"{full_dataset.signal_win_rate * 100:.2f}%")
    m4.metric("信号最大回撤", f"{full_dataset.max_drawdown * 100:.2f}%")

    st.subheader("价格走势")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["date"].tail(160),
                open=df["open"].tail(160),
                high=df["high"].tail(160),
                low=df["low"].tail(160),
                close=df["close"].tail(160),
            )
        ]
    )
    fig.update_layout(height=480, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"数据已缓存到 {cache_path}")


def _drop_unlabeled_for_eval(df: pd.DataFrame, horizon: int):
    from .ml_features import build_dataset

    return build_dataset(df, horizon=horizon).dropna(subset=["label", "future_return"])


if __name__ == "__main__":
    main()
