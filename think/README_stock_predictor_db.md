# Stock Predictor PostgreSQL Storage

## Connection

Set `STOCK_PREDICTOR_DATABASE_URL` before running DB commands.

Example:

```powershell
$env:STOCK_PREDICTOR_DATABASE_URL = "postgresql://postgres:<password>@localhost:5432/stock_predictor"
```

If this variable is not set, the CLI tries:

```text
postgresql://postgres@localhost:5432/stock_predictor
```

## Schema

The schema lives in:

```text
think/stock_predictor/db_schema.sql
```

Tables:

- `instruments`: one row per index, ETF, or stock-like target.
- `daily_bars`: normalized OHLCV daily bars keyed by `(instrument_id, trade_date)`.
- `prediction_runs`: metadata for a model run.
- `prediction_results`: forecast results tied to a run and instrument.

This layout supports A-share indices, HK indices, and ETFs in one store.

## Initialize

Create the database first:

```powershell
D:\tools\Anaconda\Anaconda3\python.exe -m think.stock_predictor.cli db-create `
  --maintenance-url "postgresql://postgres:<password>@localhost:5432/postgres"
```

Then initialize tables:

```powershell
D:\tools\Anaconda\Anaconda3\python.exe -m think.stock_predictor.cli db-init
```

## Import Current CSV Files

```powershell
D:\tools\Anaconda\Anaconda3\python.exe -m think.stock_predictor.cli db-import-csv `
  --csv think\data\cyb399006_akshare.csv `
  --symbol 399006 `
  --name 创业板指 `
  --asset-type index `
  --market CN_SZ `
  --provider-symbol sz399006

D:\tools\Anaconda\Anaconda3\python.exe -m think.stock_predictor.cli db-import-csv `
  --csv think\data\sh000001.csv `
  --symbol 000001 `
  --name 上证指数 `
  --asset-type index `
  --market CN_SH `
  --provider-symbol sh000001

D:\tools\Anaconda\Anaconda3\python.exe -m think.stock_predictor.cli db-import-csv `
  --csv think\data\sz399001.csv `
  --symbol 399001 `
  --name 深证成指 `
  --asset-type index `
  --market CN_SZ `
  --provider-symbol sz399001

D:\tools\Anaconda\Anaconda3\python.exe -m think.stock_predictor.cli db-import-csv `
  --csv think\data\sh000300.csv `
  --symbol 000300 `
  --name 沪深300 `
  --asset-type index `
  --market CN_SH `
  --provider-symbol sh000300
```

List imported instruments:

```powershell
D:\tools\Anaconda\Anaconda3\python.exe -m think.stock_predictor.cli db-list
```

## Planned Symbols

Suggested normalized symbols:

- `399006`: 创业板指
- `000001`: 上证指数
- `000698`: 科创100
- `000016`: 上证50
- `000300`: 沪深300
- `HSI`: 恒生指数
- `HSTECH`: 恒生科技

ETF targets should use the exchange code once confirmed, for example:

- 有色矿业 ETF: confirm exact fund code before import.
- 电力 ETF: confirm exact fund code before import.
