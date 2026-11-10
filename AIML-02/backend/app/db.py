"""Small SQLite persistence layer for human review records."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.app.config import PROJECT_ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    reference TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('approve', 'reject')),
    model_artifact TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_reference ON reviews(reference);
"""


def database_path() -> Path:
    """Resolve the configured SQLite database path."""
    configured = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    prefix = "sqlite:///"
    if not configured.startswith(prefix):
        raise ValueError("Only sqlite DATABASE_URL values are supported locally.")
    path = Path(configured[len(prefix) :])
    return path if path.is_absolute() else PROJECT_ROOT / path


def initialize_database() -> None:
    """Create the review schema if it does not exist."""
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)


def create_review(reference: str, action: str, model_artifact: str) -> dict[str, str]:
    """Persist one accepted review action and return its public fields."""
    if action not in {"approve", "reject"}:
        raise ValueError("Review action must be 'approve' or 'reject'.")
    if not reference.strip() or not model_artifact.strip():
        raise ValueError("Review reference and model artifact are required.")

    record = {
        "id": str(uuid4()),
        "reference": reference,
        "action": action,
        "model_artifact": model_artifact,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    initialize_database()
    with sqlite3.connect(database_path()) as connection:
        connection.execute(
            """
            INSERT INTO reviews (id, reference, action, model_artifact, created_at)
            VALUES (:id, :reference, :action, :model_artifact, :created_at)
            """,
            record,
        )
    return record


def list_reviews(reference: str) -> list[dict[str, str]]:
    """Return persisted reviews for one generated reference."""
    initialize_database()
    with sqlite3.connect(database_path()) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, reference, action, model_artifact, created_at
            FROM reviews
            WHERE reference = ?
            ORDER BY created_at ASC
            """,
            (reference,),
        ).fetchall()
    return [dict(row) for row in rows]
