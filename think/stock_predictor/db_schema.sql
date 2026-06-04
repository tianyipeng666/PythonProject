CREATE TABLE IF NOT EXISTS instruments (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    market TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'akshare',
    provider_symbol TEXT NOT NULL,
    currency TEXT NOT NULL DEFAULT 'CNY',
    timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (symbol),
    UNIQUE (provider, provider_symbol)
);

CREATE TABLE IF NOT EXISTS daily_bars (
    instrument_id BIGINT NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    trade_date DATE NOT NULL,
    open NUMERIC(18, 6) NOT NULL,
    high NUMERIC(18, 6) NOT NULL,
    low NUMERIC(18, 6) NOT NULL,
    close NUMERIC(18, 6) NOT NULL,
    volume NUMERIC(24, 4) NOT NULL DEFAULT 0,
    amount NUMERIC(24, 4) NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'csv',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (instrument_id, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_bars_trade_date ON daily_bars (trade_date);

CREATE TABLE IF NOT EXISTS prediction_runs (
    id BIGSERIAL PRIMARY KEY,
    model_type TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    target_return NUMERIC(12, 8) NOT NULL DEFAULT 0,
    threshold NUMERIC(12, 8) NOT NULL DEFAULT 0.55,
    feature_count INTEGER,
    train_start_date DATE,
    train_end_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS prediction_results (
    run_id BIGINT NOT NULL REFERENCES prediction_runs(id) ON DELETE CASCADE,
    instrument_id BIGINT NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,
    forecast_date DATE NOT NULL,
    up_probability NUMERIC(12, 8) NOT NULL,
    predicted_return NUMERIC(12, 8),
    predicted_close NUMERIC(18, 6),
    risk_level TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, instrument_id, as_of_date, forecast_date)
);
