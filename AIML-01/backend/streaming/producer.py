"""Replay PaySim transactions as Kafka JSON events."""

from __future__ import annotations

import argparse
import json
import uuid
import time
from datetime import datetime, timezone

import pandas as pd
from confluent_kafka import Producer

from streaming.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    RAW_DATA_PATH,
)

PAYLOAD_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
]


def delivery_report(error, message) -> None:
    """Report asynchronous Kafka delivery failures."""
    if error is not None:
        print(f"Delivery failed: {error}")
    else:
        print(
            f"Delivered transaction event to {message.topic()} "
            f"[partition {message.partition()}] at offset {message.offset()}"
        )


def transaction_to_event(row: pd.Series, sequence_number: int) -> dict:
    """Convert a PaySim CSV row into an API-compatible Kafka event."""
    transaction = {
        column: (
            int(row[column])
            if column == "step"
            else float(row[column])
            if column
            in {
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
            }
            else str(row[column])
        )
        for column in PAYLOAD_COLUMNS
    }

    transaction["transaction_id"] = (
        f"paysim_{int(row.name)}_{sequence_number}_{uuid.uuid4().hex[:8]}"
    )
    return {
        "event_id": f"event_{transaction['transaction_id']}",
        "event_type": "transaction",
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
        "transaction": transaction,
    }


def replay_transactions(
    limit: int,
    delay_seconds: float,
    fraud_only: bool,
) -> None:
    """Read PaySim records and publish them to the Kafka topic."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at: {RAW_DATA_PATH}. "
            "Download it before running the producer."
        )

    producer = Producer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "client.id": "paysim-replay-producer",
        }
    )

    use_columns = PAYLOAD_COLUMNS + ["isFraud"]
    dataframe = pd.read_csv(RAW_DATA_PATH, usecols=use_columns)

    if fraud_only:
        dataframe = dataframe[dataframe["isFraud"] == 1]

    replay_dataframe = dataframe.head(limit)

    if replay_dataframe.empty:
        raise ValueError("No records matched the requested replay filter.")

    print(
        f"Replaying {len(replay_dataframe)} transaction(s) to "
        f"Kafka topic '{KAFKA_TOPIC}'. Fraud-only mode: {fraud_only}"
    )

    for sequence_number, (_, row) in enumerate(replay_dataframe.iterrows(), start=1):
        event = transaction_to_event(row, sequence_number)

        producer.produce(
            topic=KAFKA_TOPIC,
            key=event["transaction"]["transaction_id"],
            value=json.dumps(event).encode("utf-8"),
            callback=delivery_report,
        )
        producer.poll(0)

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    remaining_messages = producer.flush(timeout=30)
    if remaining_messages != 0:
        raise RuntimeError(
            f"Kafka producer could not deliver {remaining_messages} message(s)."
        )

    print("Replay completed successfully.")


def parse_arguments() -> argparse.Namespace:
    """Parse replay command-line options."""
    parser = argparse.ArgumentParser(
        description="Replay PaySim transactions to Kafka."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of transaction events to send.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.2,
        help="Delay between events, in seconds.",
    )
    parser.add_argument(
        "--fraud-only",
        action="store_true",
        help="Replay only records labelled as fraud in the historical dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    replay_transactions(
        limit=arguments.limit,
        delay_seconds=arguments.delay_seconds,
        fraud_only=arguments.fraud_only,
    )