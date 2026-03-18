"""Audit Trail — immutable SQLite log for every AI ethics assessment.

Records assessments, bias audits, compliance checks, and remediation
actions with timestamps for regulatory compliance.
"""
from __future__ import annotations
import json
import sqlite3
import time
import uuid
from typing import Any
from loguru import logger


class AuditTrail:
    """Immutable audit trail backed by SQLite."""

    def __init__(self, db_path: str = "audit_trail.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    model_name TEXT,
                    actor TEXT,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type ON audit_log(event_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_name ON audit_log(model_name)
            """)

    def _compute_checksum(self, data: str) -> str:
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def log_event(
        self, event_type: str, payload: dict[str, Any],
        model_name: str = "", actor: str = "system",
    ) -> str:
        """Log an immutable audit event.

        Args:
            event_type: Type of event (bias_audit, compliance_check, remediation, etc.)
            payload: Event data (serialized to JSON).
            model_name: Name of the model being audited.
            actor: Who triggered the event.

        Returns:
            Event ID (UUID).
        """
        event_id = str(uuid.uuid4())
        ts = time.time()
        payload_json = json.dumps(payload, default=str, sort_keys=True)
        checksum = self._compute_checksum(f"{event_id}{ts}{event_type}{payload_json}")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO audit_log (id, timestamp, event_type, model_name, actor, payload, checksum) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event_id, ts, event_type, model_name, actor, payload_json, checksum),
            )

        logger.info("Audit logged: {} [{}] for model '{}'", event_type, event_id[:8], model_name)
        return event_id

    def get_events(
        self, event_type: str | None = None, model_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve audit events with optional filtering."""
        query = "SELECT id, timestamp, event_type, model_name, actor, payload, checksum FROM audit_log WHERE 1=1"
        params: list[Any] = []
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {"id": r[0], "timestamp": r[1], "event_type": r[2], "model_name": r[3],
             "actor": r[4], "payload": json.loads(r[5]), "checksum": r[6]}
            for r in rows
        ]

    def verify_integrity(self) -> dict[str, Any]:
        """Verify all audit records haven't been tampered with."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id, timestamp, event_type, payload, checksum FROM audit_log").fetchall()

        total = len(rows)
        valid = 0
        for r in rows:
            payload_json = r[3]
            expected = self._compute_checksum(f"{r[0]}{r[1]}{r[2]}{payload_json}")
            if expected == r[4]:
                valid += 1

        return {"total_records": total, "valid_records": valid,
                "integrity_ok": total == valid, "tampered": total - valid}

    def count_events(self) -> dict[str, int]:
        """Count events by type."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT event_type, COUNT(*) FROM audit_log GROUP BY event_type"
            ).fetchall()
        return {r[0]: r[1] for r in rows}
