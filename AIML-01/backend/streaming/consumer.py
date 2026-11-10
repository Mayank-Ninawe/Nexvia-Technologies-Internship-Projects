"""Consume Kafka transaction events and score them through FastAPI."""

from __future__ import annotations

import json
import os
import signal
import time
from datetime import datetime, timezone

import requests
from confluent_kafka import Consumer, KafkaError
from persistence.database import initialize_database, persist_scored_event

from streaming.config import (
    CONSUMER_ERRORS_PATH,
    FASTAPI_PREDICT_URL,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP,
    KAFKA_TOPIC,
)

RUNNING = True


def stop_consumer(_signal_number, _frame) -> None:
    """Handle Ctrl+C gracefully."""
    global RUNNING
    RUNNING = False


def load_json_records(path) -> list[dict]:
    """Load records from JSONL (line-delimited) or legacy JSON list file."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        # Check if alternative extension exists
        alt = p.with_suffix(".json") if p.suffix == ".jsonl" else p.with_suffix(".jsonl")
        if alt.exists():
            p = alt
        else:
            return []

    with open(p, "r", encoding="utf-8") as file:
        content = file.read().strip()
        if not content:
            return []
        if content.startswith("["):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        records = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def append_json_record(path, record: dict) -> None:
    """Append one record to a local append-only JSONL or JSON file with O(1) write."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix == ".jsonl":
        with open(p, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
    else:
        # Fallback for explicit legacy .json extension
        records = load_json_records(p)
        records.append(record)
        with open(p, "w", encoding="utf-8") as file:
            json.dump(records, file, indent=2)


def validate_event(event: dict) -> dict:
    """Validate basic Kafka event structure and return transaction payload."""
    if event.get("event_type") != "transaction":
        raise ValueError("Unsupported event_type. Expected 'transaction'.")

    transaction = event.get("transaction")
    if not isinstance(transaction, dict):
        raise TypeError("Event must contain a transaction object.")

    required_fields = {
        "transaction_id",
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
    }

    missing_fields = required_fields - set(transaction)
    if missing_fields:
        raise ValueError(
            f"Transaction is missing fields: {sorted(missing_fields)}"
        )

    return transaction


def process_message(message_value: bytes) -> dict:
    """Decode, validate, score, and persist a Kafka transaction event."""
    event = json.loads(message_value.decode("utf-8"))
    transaction = validate_event(event)

    retry_count = max(int(os.getenv("PREDICT_RETRY_COUNT", "3")), 1)
    response = None
    for attempt in range(retry_count):
        try:
            headers = {}
            api_key = os.getenv("API_KEY", "").strip()
            if api_key:
                headers["X-API-Key"] = api_key
            response = requests.post(
                FASTAPI_PREDICT_URL,
                json=transaction,
                timeout=15,
                headers=headers,
            )
            response.raise_for_status()
            break
        except requests.exceptions.RequestException:
            if attempt == retry_count - 1:
                raise
            time.sleep(min(2**attempt, 8))

    if response is None:
        raise RuntimeError("Prediction service returned no response.")

    prediction = response.json()

    processed_record = {
        "event_id": event.get("event_id"),
        "event_timestamp": event.get("event_timestamp"),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "transaction": transaction,
        "prediction": prediction,
    }

    persist_scored_event(
        initialize_database(),
        event_id=processed_record["event_id"],
        event_timestamp=processed_record["event_timestamp"],
        transaction=transaction,
        prediction=prediction,
    )
    return processed_record


def run_consumer() -> None:
    """Continuously consume and score events until Ctrl+C."""
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id": KAFKA_CONSUMER_GROUP,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )

    signal.signal(signal.SIGINT, stop_consumer)
    signal.signal(signal.SIGTERM, stop_consumer)

    consumer.subscribe([KAFKA_TOPIC])

    print(
        f"Consumer started. Listening to topic '{KAFKA_TOPIC}'. "
        "Press Ctrl+C to stop."
    )

    try:
        while RUNNING:
            message = consumer.poll(1.0)

            if message is None:
                continue

            if message.error():
                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"Kafka consumer error: {message.error()}")
                continue

            try:
                result = process_message(message.value())
                consumer.commit(message=message, asynchronous=False)
                prediction = result["prediction"]
                print(
                    f"Processed {prediction['transaction_id']}: "
                    f"risk={prediction['risk_score']}%, "
                    f"alert={prediction['is_alert']}"
                )
            except (
                KeyError,
                OSError,
                requests.exceptions.RequestException,
                TypeError,
                ValueError,
            ) as error:
                error_record = {
                    "received_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(error),
                    "raw_message": message.value().decode(
                        "utf-8", errors="replace"
                    ),
                }
                append_json_record(CONSUMER_ERRORS_PATH, error_record)
                consumer.commit(message=message, asynchronous=False)
                print(f"Failed to process message: {error}")

    finally:
        consumer.close()
        print("Consumer stopped.")


if __name__ == "__main__":
    run_consumer()