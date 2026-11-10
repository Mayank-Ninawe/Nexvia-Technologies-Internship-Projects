"""Tests for Kafka streaming event validation and persistence helpers."""

import json

import pytest
from streaming.consumer import (
    append_json_record,
    load_json_records,
    process_message,
    validate_event,
)


def valid_event() -> dict:
    """Return a valid transaction event for testing."""
    return {
        "event_id": "event_test_001",
        "event_type": "transaction",
        "event_timestamp": "2026-10-11T10:00:00+00:00",
        "transaction": {
            "transaction_id": "txn_test_001",
            "step": 1,
            "type": "TRANSFER",
            "amount": 100.0,
            "nameOrig": "C100",
            "oldbalanceOrg": 100.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C200",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 100.0,
        },
    }


def test_validate_event_returns_transaction():
    event = valid_event()

    transaction = validate_event(event)

    assert transaction["transaction_id"] == "txn_test_001"
    assert transaction["type"] == "TRANSFER"


def test_validate_event_rejects_unsupported_event_type():
    event = valid_event()
    event["event_type"] = "kafka_smoke_test"

    with pytest.raises(ValueError, match="Unsupported event_type"):
        validate_event(event)


def test_validate_event_rejects_missing_transaction_field():
    event = valid_event()
    del event["transaction"]["amount"]

    with pytest.raises(ValueError, match="Transaction is missing fields"):
        validate_event(event)


def test_json_record_persistence(tmp_path):
    output_path = tmp_path / "records.json"
    record = {"transaction_id": "txn_test_001", "risk_score": 99.5}

    append_json_record(output_path, record)

    assert output_path.exists()
    assert load_json_records(output_path) == [record]

    with open(output_path, "r", encoding="utf-8") as file:
        assert json.load(file) == [record]


def test_jsonl_record_persistence(tmp_path):
    output_path = tmp_path / "records.jsonl"
    record1 = {"transaction_id": "txn_test_001", "risk_score": 99.5}
    record2 = {"transaction_id": "txn_test_002", "risk_score": 12.3}

    append_json_record(output_path, record1)
    append_json_record(output_path, record2)

    assert output_path.exists()
    loaded = load_json_records(output_path)
    assert len(loaded) == 2
    assert loaded[0] == record1
    assert loaded[1] == record2


def test_process_message_sends_configured_api_key(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "transaction_id": "txn_stream_api_key",
                "fraud_probability": 0.1,
                "risk_score": 10.0,
                "is_alert": False,
                "model_name": "test-model",
                "threshold": 0.2,
                "recommended_action": "Allow transaction",
            }

    def fake_post(*args, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setenv("API_KEY", "test-stream-key")
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(
        "streaming.consumer.persist_scored_event",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("streaming.consumer.initialize_database", lambda: None)

    process_message(json.dumps(valid_event()).encode())

    assert captured["headers"] == {"X-API-Key": "test-stream-key"}