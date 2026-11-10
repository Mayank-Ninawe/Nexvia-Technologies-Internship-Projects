"""Feast entities, data source, and feature view definitions."""

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.value_type import ValueType

transaction = Entity(
    name="transaction_id",
    join_keys=["transaction_id"],
    value_type=ValueType.STRING,
    description="Unique PaySim transaction event identifier.",
)

transaction_features_source = FileSource(
    name="transaction_features_source",
    path="data/transaction_features.parquet",
    timestamp_field="event_timestamp",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=None,
    schema=[
        Field(name="amount", dtype=Float32),
        Field(name="oldbalance_org", dtype=Float32),
        Field(name="oldbalance_dest", dtype=Float32),
        Field(name="is_transfer_or_cash_out", dtype=Int64),
        Field(name="sender_balance_before_zero", dtype=Int64),
        Field(name="receiver_balance_before_zero", dtype=Int64),
    ],
    source=transaction_features_source,
    online=True,
    description=(
        "Realtime-safe transaction features derived from PaySim "
        "pre-transaction fields."
    ),
)