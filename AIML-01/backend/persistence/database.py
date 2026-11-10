"""Database-backed persistence for scored transaction events and analyst cases."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fraud_detection.config import PROJECT_ROOT
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

DEFAULT_DATABASE_URL = (
    f"sqlite:///{PROJECT_ROOT / 'data' / 'processed' / 'fraud_detection.db'}"
)
LEGACY_CASES_PATH = PROJECT_ROOT / "data" / "processed" / "cases.json"

SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY,
        step INTEGER NOT NULL,
        transaction_type TEXT NOT NULL,
        amount REAL NOT NULL,
        sender_id TEXT NOT NULL,
        sender_balance_before REAL NOT NULL,
        sender_balance_after REAL NOT NULL,
        receiver_id TEXT NOT NULL,
        receiver_balance_before REAL NOT NULL,
        receiver_balance_after REAL NOT NULL,
        event_timestamp TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        transaction_id TEXT PRIMARY KEY REFERENCES transactions(transaction_id),
        event_id TEXT UNIQUE,
        model_name TEXT NOT NULL,
        fraud_probability REAL NOT NULL,
        risk_score REAL NOT NULL,
        is_alert INTEGER NOT NULL,
        threshold REAL NOT NULL,
        recommended_action TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS alerts (
        transaction_id TEXT PRIMARY KEY REFERENCES predictions(transaction_id),
        event_id TEXT UNIQUE,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cases (
        transaction_id TEXT PRIMARY KEY REFERENCES alerts(transaction_id),
        case_status TEXT NOT NULL,
        analyst_feedback TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
)

POSTGRES_AUDIT_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS audit_events (
        id BIGSERIAL PRIMARY KEY,
        transaction_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    )
    """


def get_engine() -> Engine:
    """Create an engine from DATABASE_URL or the local SQLite fallback."""
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    if database_url.startswith("sqlite:///"):
        Path(database_url.removeprefix("sqlite:///")).parent.mkdir(
            parents=True, exist_ok=True
        )
    return create_engine(database_url, pool_pre_ping=True, future=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _execute(connection: Any, statement: str, parameters: tuple[Any, ...] | None = None):
    """Execute SQL with the parameter style expected by the active driver."""
    if connection.engine.dialect.name == "postgresql":
        statement = statement.replace("?", "%s")
    if parameters is None:
        return connection.exec_driver_sql(statement)
    return connection.exec_driver_sql(statement, parameters)


def _execute_schema(engine: Engine) -> None:
    with engine.begin() as connection:
        statements = SCHEMA_SQL
        if engine.dialect.name == "postgresql":
            statements = SCHEMA_SQL[:-1] + (POSTGRES_AUDIT_SCHEMA_SQL,)
        for statement in statements:
            _execute(connection, statement)


def initialize_database(
    engine: Engine | None = None,
    *,
    bootstrap_legacy: bool | None = None,
) -> Engine:
    """Create tables and import legacy cases once when present."""
    active_engine = engine or get_engine()
    _execute_schema(active_engine)
    if bootstrap_legacy is None:
        bootstrap_legacy = engine is None
    if bootstrap_legacy:
        _bootstrap_legacy_cases(active_engine)
    return active_engine


def _bootstrap_legacy_cases(engine: Engine) -> None:
    if not LEGACY_CASES_PATH.exists():
        return
    with engine.begin() as connection:
        existing = _execute(connection, "SELECT COUNT(*) FROM cases").scalar_one()
        if existing:
            return
        legacy_cases = json.loads(LEGACY_CASES_PATH.read_text(encoding="utf-8"))
        for case in legacy_cases:
            transaction = case.get("transaction_details", {})
            transaction_id = case.get("transaction_id")
            if not transaction_id:
                continue
            now = case.get("updated_at") or _now()
            _execute(
                connection,
                """
                INSERT INTO transactions
                (transaction_id, step, transaction_type, amount, sender_id,
                 sender_balance_before, sender_balance_after, receiver_id,
                 receiver_balance_before, receiver_balance_after, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(transaction_id) DO NOTHING
                """,
                (
                    transaction_id,
                    transaction.get("step", 0),
                    transaction.get("type", "UNKNOWN"),
                    transaction.get("amount", 0),
                    transaction.get("nameOrig", "UNKNOWN"),
                    transaction.get("oldbalanceOrg", 0),
                    transaction.get("newbalanceOrig", 0),
                    transaction.get("nameDest", "UNKNOWN"),
                    transaction.get("oldbalanceDest", 0),
                    transaction.get("newbalanceDest", 0),
                    now,
                ),
            )
            _execute(
                connection,
                """
                INSERT INTO predictions
                (transaction_id, model_name, fraud_probability, risk_score,
                 is_alert, threshold, recommended_action, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(transaction_id) DO NOTHING
                """,
                (
                    transaction_id,
                    case.get("model_name", "legacy"),
                    case.get("fraud_probability", 0),
                    case.get("risk_score", 0),
                    int(case.get("is_alert", True)),
                    case.get("threshold", 0),
                    case.get("recommended_action", "Review"),
                    now,
                ),
            )
            _execute(
                connection,
                """
                INSERT INTO alerts
                (transaction_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(transaction_id) DO NOTHING
                """,
                (transaction_id, case.get("case_status", "Open"), now, now),
            )
            _execute(
                connection,
                """
                INSERT INTO cases
                (transaction_id, case_status, analyst_feedback, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(transaction_id) DO NOTHING
                """,
                (
                    transaction_id,
                    case.get("case_status", "Open"),
                    case.get("analyst_feedback"),
                    case.get("created_at", now),
                    now,
                ),
            )


def persist_scored_event(
    engine: Engine,
    event_id: str | None,
    event_timestamp: str | None,
    transaction: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """Persist a scored event idempotently and create an alert case when flagged."""
    transaction_id = transaction["transaction_id"]
    now = _now()
    with engine.begin() as connection:
        _execute(
            connection,
            """
            INSERT INTO transactions
            (transaction_id, step, transaction_type, amount, sender_id,
             sender_balance_before, sender_balance_after, receiver_id,
             receiver_balance_before, receiver_balance_after, event_timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(transaction_id) DO NOTHING
            """,
            (
                transaction_id,
                transaction["step"],
                transaction["type"],
                transaction["amount"],
                transaction["nameOrig"],
                transaction["oldbalanceOrg"],
                transaction["newbalanceOrig"],
                transaction["nameDest"],
                transaction["oldbalanceDest"],
                transaction["newbalanceDest"],
                event_timestamp,
                now,
            ),
        )
        _execute(
            connection,
            """
            INSERT INTO predictions
            (transaction_id, event_id, model_name, fraud_probability, risk_score,
             is_alert, threshold, recommended_action, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(transaction_id) DO NOTHING
            """,
            (
                transaction_id,
                event_id,
                prediction["model_name"],
                prediction["fraud_probability"],
                prediction["risk_score"],
                int(prediction["is_alert"]),
                prediction["threshold"],
                prediction["recommended_action"],
                now,
            ),
        )
        if prediction["is_alert"]:
            _execute(
                connection,
                """
                INSERT INTO alerts
                (transaction_id, event_id, status, created_at, updated_at)
                VALUES (?, ?, 'Open', ?, ?)
                ON CONFLICT(transaction_id) DO NOTHING
                """,
                (transaction_id, event_id, now, now),
            )
            _execute(
                connection,
                """
                INSERT INTO cases
                (transaction_id, case_status, created_at, updated_at)
                VALUES (?, 'Open', ?, ?)
                ON CONFLICT(transaction_id) DO NOTHING
                """,
                (transaction_id, now, now),
            )
    return get_case(engine, transaction_id) or prediction


def list_cases(engine: Engine) -> list[dict[str, Any]]:
    """Return alert cases with transaction and prediction details."""
    query = """
        SELECT c.transaction_id, c.case_status, c.analyst_feedback,
               c.created_at, c.updated_at, p.fraud_probability, p.risk_score,
               p.is_alert, p.model_name, p.threshold, p.recommended_action,
               t.step, t.transaction_type, t.amount, t.sender_id,
               t.sender_balance_before, t.sender_balance_after, t.receiver_id,
               t.receiver_balance_before, t.receiver_balance_after
        FROM cases c
        JOIN predictions p ON p.transaction_id = c.transaction_id
        JOIN transactions t ON t.transaction_id = c.transaction_id
        ORDER BY c.updated_at DESC
    """
    with engine.connect() as connection:
        rows = _execute(connection, query).mappings().all()
    return [
        {
            "transaction_id": row["transaction_id"],
            "fraud_probability": row["fraud_probability"],
            "risk_score": row["risk_score"],
            "is_alert": bool(row["is_alert"]),
            "model_name": row["model_name"],
            "threshold": row["threshold"],
            "recommended_action": row["recommended_action"],
            "case_status": row["case_status"],
            "analyst_feedback": row["analyst_feedback"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "transaction_details": {
                "transaction_id": row["transaction_id"],
                "step": row["step"],
                "type": row["transaction_type"],
                "amount": row["amount"],
                "nameOrig": row["sender_id"],
                "oldbalanceOrg": row["sender_balance_before"],
                "newbalanceOrig": row["sender_balance_after"],
                "nameDest": row["receiver_id"],
                "oldbalanceDest": row["receiver_balance_before"],
                "newbalanceDest": row["receiver_balance_after"],
            },
        }
        for row in rows
    ]


def runtime_counts(engine: Engine) -> dict[str, int]:
    """Return counts derived from persisted runtime events."""
    with engine.connect() as connection:
        transaction_count = _execute(
            connection,
            "SELECT COUNT(*) FROM transactions"
        ).scalar_one()
        alert_count = _execute(
            connection,
            "SELECT COUNT(*) FROM alerts"
        ).scalar_one()
    return {
        "total_transactions": int(transaction_count),
        "alert_count": int(alert_count),
    }


def list_scored_events(engine: Engine) -> list[dict[str, Any]]:
    """Return persisted predictions in the legacy dashboard event shape."""
    query = """
        SELECT t.transaction_id, t.step, t.transaction_type, t.amount,
               t.sender_id, t.sender_balance_before, t.sender_balance_after,
               t.receiver_id, t.receiver_balance_before, t.receiver_balance_after,
               p.fraud_probability, p.risk_score, p.is_alert, p.model_name,
               p.threshold, p.recommended_action, p.created_at
        FROM transactions t
        JOIN predictions p ON p.transaction_id = t.transaction_id
        ORDER BY p.created_at DESC
    """
    with engine.connect() as connection:
        rows = _execute(connection, query).mappings().all()
    return [
        {
            "processed_at": row["created_at"],
            "transaction": {
                "transaction_id": row["transaction_id"],
                "step": row["step"],
                "type": row["transaction_type"],
                "amount": row["amount"],
                "nameOrig": row["sender_id"],
                "oldbalanceOrg": row["sender_balance_before"],
                "newbalanceOrig": row["sender_balance_after"],
                "nameDest": row["receiver_id"],
                "oldbalanceDest": row["receiver_balance_before"],
                "newbalanceDest": row["receiver_balance_after"],
            },
            "prediction": {
                "transaction_id": row["transaction_id"],
                "fraud_probability": row["fraud_probability"],
                "risk_score": row["risk_score"],
                "is_alert": bool(row["is_alert"]),
                "model_name": row["model_name"],
                "threshold": row["threshold"],
                "recommended_action": row["recommended_action"],
            },
        }
        for row in rows
    ]


def get_case(engine: Engine, transaction_id: str) -> dict[str, Any] | None:
    """Return one case by transaction ID."""
    return next(
        (case for case in list_cases(engine) if case["transaction_id"] == transaction_id),
        None,
    )


def update_case(
    engine: Engine,
    transaction_id: str,
    case_status: str,
    analyst_feedback: str | None,
) -> dict[str, Any] | None:
    """Update a case and record an audit event."""
    now = _now()
    with engine.begin() as connection:
        result = _execute(
            connection,
            """
            UPDATE cases
            SET case_status = ?, analyst_feedback = COALESCE(?, analyst_feedback),
                updated_at = ?
            WHERE transaction_id = ?
            """,
            (case_status, analyst_feedback, now, transaction_id),
        )
        if result.rowcount == 0:
            return None
        _execute(
            connection,
            """
            UPDATE alerts SET status = ?, updated_at = ? WHERE transaction_id = ?
            """,
            (case_status, now, transaction_id),
        )
        _execute(
            connection,
            """
            INSERT INTO audit_events (transaction_id, event_type, payload, created_at)
            VALUES (?, 'case_updated', ?, ?)
            """,
            (
                transaction_id,
                json.dumps(
                    {"case_status": case_status, "analyst_feedback": analyst_feedback}
                ),
                now,
            ),
        )
    return get_case(engine, transaction_id)
