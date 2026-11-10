"""Tests for idempotent scored-event and analyst-case persistence."""

from persistence.database import (
    initialize_database,
    list_cases,
    persist_scored_event,
    runtime_counts,
    update_case,
)
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


def event_payload() -> tuple[dict, dict]:
    transaction = {
        "transaction_id": "txn_persistence_001",
        "step": 1,
        "type": "TRANSFER",
        "amount": 100.0,
        "nameOrig": "C100",
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C200",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 100.0,
    }
    prediction = {
        "transaction_id": transaction["transaction_id"],
        "fraud_probability": 0.99,
        "risk_score": 99.0,
        "is_alert": True,
        "model_name": "test-model",
        "threshold": 0.1,
        "recommended_action": "Send to analyst review queue",
    }
    return transaction, prediction


def test_scored_event_creates_alert_case_idempotently():
    engine = initialize_database(
        create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    )
    transaction, prediction = event_payload()

    persist_scored_event(engine, "event_001", "2026-11-07T10:00:00+00:00", transaction, prediction)
    persist_scored_event(engine, "event_001", "2026-11-07T10:00:00+00:00", transaction, prediction)

    assert runtime_counts(engine) == {"total_transactions": 1, "alert_count": 1}
    cases = list_cases(engine)
    assert len(cases) == 1
    assert cases[0]["case_status"] == "Open"


def test_case_update_writes_status_and_feedback():
    engine = initialize_database(
        create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    )
    transaction, prediction = event_payload()
    persist_scored_event(engine, "event_002", None, transaction, prediction)

    updated = update_case(engine, transaction["transaction_id"], "Resolved", "Confirmed")

    assert updated is not None
    assert updated["case_status"] == "Resolved"
    assert updated["analyst_feedback"] == "Confirmed"


def test_non_alert_event_is_persisted_without_creating_case():
    engine = initialize_database(
        create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    )
    transaction, prediction = event_payload()
    prediction["is_alert"] = False
    prediction["fraud_probability"] = 0.01
    prediction["risk_score"] = 1.0

    persisted = persist_scored_event(
        engine,
        "event_non_alert",
        None,
        transaction,
        prediction,
    )

    assert persisted["is_alert"] is False
    assert runtime_counts(engine) == {"total_transactions": 1, "alert_count": 0}
    assert list_cases(engine) == []
