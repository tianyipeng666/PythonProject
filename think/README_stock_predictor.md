# A股短线涨跌概率预测

这是个人自用的 A 股/指数短线预测工具。当前版本使用：

- `akshare` 拉取 A 股股票和指数日线
- `pandas` 做数据清洗和技术指标特征
- `scikit-learn` 训练上涨概率模型和收益率回归模型
- `streamlit` 提供可视化界面
- `plotly` 展示 K 线

输出包括：

- 未来 1/2/3/5 个交易日上涨概率
- 预计涨跌幅和预计目标点位
- 风险等级和风险提示
- 简单时间切分回测指标
- 本周剩余工作日方向预测

## 预测依据

当前主流程只使用日线行情数据：

- 日期
- 开盘价、最高价、最低价、收盘价
- 成交量、成交额

这些数据由 `akshare` 拉取。股票默认使用前复权日线，指数使用 AkShare 指数日线接口。系统不会直接把原始价格丢给模型，而是先用 `pandas` 生成技术指标特征，包括：

- 1/2/3/5/10/20 日收益率
- 5/10/20/60 日均线偏离
- 成交量相对 5/20 日均量的倍数
- 单日振幅
- 5/20 日波动率
- 10/20 日回撤
- RSI 6/14
- MACD、MACD signal、MACD hist

方向预测使用分类模型，回答“涨的概率多大”。涨跌幅预测使用回归模型，回答“模型估计大约涨跌多少”。

## 安装依赖

```bash
cd /Users/tianyipeng/PythonProjects/PythonProject
python -m pip install akshare pandas scikit-learn streamlit plotly joblib
```

## 命令行使用

### 1. 拉取真实数据

股票示例：

```bash
python -m think.stock_predictor.cli fetch \
  --symbol 600519 \
  --asset-type stock \
  --output think/data/600519.csv
```

指数示例，科创100：

```bash
python -m think.stock_predictor.cli fetch \
  --symbol 科创100 \
  --asset-type index \
  --output think/data/kc100.csv
```

科创100也可以写成：

```bash
python -m think.stock_predictor.cli fetch --symbol sh000698 --asset-type index --output think/data/kc100.csv
```

### 2. 训练模型

训练未来 1 个交易日上涨概率模型：

```bash
python -m think.stock_predictor.cli train \
  --csv think/data/kc100.csv \
  --horizon 1 \
  --model think/models/kc100_h1.joblib \
  --model-type logistic
```

可选模型：

- `logistic`：更稳、更容易解释，适合作为默认
- `random_forest`：非线性更强，但小样本时更容易过拟合

### 3. 预测最新数据

```bash
python -m think.stock_predictor.cli predict \
  --csv think/data/kc100.csv \
  --model think/models/kc100_h1.joblib
```

### 4. 回测

```bash
python -m think.stock_predictor.cli backtest \
  --csv think/data/kc100.csv \
  --model think/models/kc100_h1.joblib \
  --threshold 0.55
```

### 5. 一条命令完成拉取、训练、预测、回测

```bash
python -m think.stock_predictor.cli quick \
  --symbol 科创100 \
  --asset-type index \
  --horizon 1 \
  --model-type logistic \
  --threshold 0.55 \
  --csv-output think/data/kc100.csv
```

### 6. 预测本周剩余工作日

```bash
python -m think.stock_predictor.cli week-predict \
  --symbol 科创100 \
  --asset-type index \
  --model-type logistic
```

说明：如果今天是周三，命令会输出周四和周五两个结果。第一个结果是 `horizon=1`，第二个结果是 `horizon=2`。第二个表示“基于当前最新收盘数据，未来两个交易日后的累计涨跌方向”，不是拿到周四数据后再预测周五。

也可以用 `--as-of` 截断数据做复盘，例如只使用 2026-06-02 及以前的数据来预测后续交易日：

```bash
python -m think.stock_predictor.cli week-predict \
  --symbol 科创100 \
  --asset-type index \
  --model-type logistic \
  --as-of 2026-06-02
```

## Streamlit 界面

启动界面：

```bash
streamlit run think/streamlit_app.py
```

然后在左侧选择：

- 标的类型：`stock` 或 `index`
- 代码：股票如 `600519`，科创100可填 `科创100` 或 `sh000698`
- 预测周期：1/2/3/5 个交易日
- 模型：`logistic` 或 `random_forest`

## 指标解释

- `Up probability`：模型估计未来 horizon 个交易日后上涨的概率
- `Predicted return`：回归模型估计的未来 horizon 个交易日累计涨跌幅
- `Predicted close`：用当前收盘价乘以预计涨跌幅得到的目标点位
- `Risk level`：规则型风险等级，低/中/高
- `Risk tips`：触发的风险原因
- `Accuracy`：测试集方向预测准确率
- `Signal win rate`：当上涨概率超过阈值时，历史上真实上涨的比例
- `Average signal return`：触发信号后的平均 horizon 日收益
- `Signal max drawdown`：只按信号交易时的简化最大回撤
- `Return MAE`：回归模型在测试集上的平均绝对误差；例如 1.50% 表示历史测试中预计涨跌幅平均偏差约 1.50 个百分点

## 代码结构

```text
think/
  app.py
  streamlit_app.py
  stock_predictor/
    cli.py
    data_loader.py
    ml_features.py
    sklearn_model.py
    streamlit_app.py
    risk.py
    backtest.py
    features.py
    model.py
```

`features.py/model.py/backtest.py` 是上一版的轻量 numpy 实现，保留用于无依赖 demo。当前主流程使用 `ml_features.py/sklearn_model.py`。

## 注意

这是研究工具，不是投资建议。当前模型只使用日线技术特征，没有纳入实时盘口、成分股变化、板块轮动、公告、宏观事件、涨跌停、停牌和交易成本。短线模型尤其容易过拟合，正式参考前应持续做滚动验证和风险控制。
