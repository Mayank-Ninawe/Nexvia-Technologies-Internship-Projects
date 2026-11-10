"""Configuration for local Kafka transaction streaming."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "fraud-transactions"
KAFKA_CONSUMER_GROUP = "fraud-scoring-consumer-group"

FASTAPI_PREDICT_URL = "https://sentinelgraph-api-n090.onrender.com/predict"

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

STREAMED_TRANSACTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "streamed_transactions.jsonl"
)

CONSUMER_ERRORS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "consumer_errors.jsonl"
)