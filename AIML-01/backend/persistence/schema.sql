-- Reference schema for PostgreSQL deployments.
-- Local SQLite development uses the equivalent bootstrap in database.py.

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    step INTEGER NOT NULL,
    transaction_type TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    sender_id TEXT NOT NULL,
    sender_balance_before DOUBLE PRECISION NOT NULL,
    sender_balance_after DOUBLE PRECISION NOT NULL,
    receiver_id TEXT NOT NULL,
    receiver_balance_before DOUBLE PRECISION NOT NULL,
    receiver_balance_after DOUBLE PRECISION NOT NULL,
    event_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    transaction_id TEXT PRIMARY KEY REFERENCES transactions(transaction_id),
    event_id TEXT UNIQUE,
    model_name TEXT NOT NULL,
    fraud_probability DOUBLE PRECISION NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    is_alert BOOLEAN NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    recommended_action TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    transaction_id TEXT PRIMARY KEY REFERENCES predictions(transaction_id),
    event_id TEXT UNIQUE,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS cases (
    transaction_id TEXT PRIMARY KEY REFERENCES alerts(transaction_id),
    case_status TEXT NOT NULL,
    analyst_feedback TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    id BIGSERIAL PRIMARY KEY,
    transaction_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
