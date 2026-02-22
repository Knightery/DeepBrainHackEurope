from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any


def _db_path() -> Path:
    configured = os.getenv("PITCH_DB_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path("data/pitches/pitches.db")


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pitches (
                pitch_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                thesis TEXT NOT NULL DEFAULT '',
                time_horizon TEXT,
                tickers_json TEXT NOT NULL DEFAULT '[]',
                source_urls_json TEXT NOT NULL DEFAULT '[]',
                supporting_notes TEXT NOT NULL DEFAULT '',
                one_shot_mode INTEGER NOT NULL DEFAULT 0,
                overall_score REAL,
                allocation_usd INTEGER,
                decision TEXT,
                validation_outcome TEXT,
                result_json TEXT
            );

            CREATE TABLE IF NOT EXISTS pitch_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pitch_id TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (pitch_id) REFERENCES pitches (pitch_id)
            );

            CREATE INDEX IF NOT EXISTS idx_pitches_status ON pitches(status);
            CREATE INDEX IF NOT EXISTS idx_pitches_score ON pitches(overall_score DESC);
            CREATE INDEX IF NOT EXISTS idx_pitches_updated ON pitches(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_messages_pitch_time
                ON pitch_messages(pitch_id, timestamp_utc ASC);
            """
        )
        cols = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(pitches)").fetchall()
            if isinstance(row["name"], str)
        }
        if "supporting_notes" not in cols:
            conn.execute("ALTER TABLE pitches ADD COLUMN supporting_notes TEXT NOT NULL DEFAULT ''")


def upsert_pitch_snapshot(draft: dict[str, Any], now_iso: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO pitches (
                pitch_id, created_at, updated_at, status, thesis, time_horizon,
                tickers_json, source_urls_json, supporting_notes, one_shot_mode
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pitch_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                status = excluded.status,
                thesis = excluded.thesis,
                time_horizon = excluded.time_horizon,
                tickers_json = excluded.tickers_json,
                source_urls_json = excluded.source_urls_json,
                supporting_notes = excluded.supporting_notes,
                one_shot_mode = excluded.one_shot_mode
            """,
            (
                str(draft.get("pitch_id", "")),
                str(draft.get("created_at", now_iso)),
                now_iso,
                str(draft.get("status", "draft")),
                str(draft.get("thesis", "")),
                draft.get("time_horizon"),
                json.dumps(draft.get("tickers", []), ensure_ascii=True),
                json.dumps(draft.get("source_urls", []), ensure_ascii=True),
                str(draft.get("supporting_notes", "")),
                1 if bool(draft.get("one_shot_mode")) else 0,
            ),
        )


def append_pitch_message(pitch_id: str, timestamp_utc: str, role: str, content: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO pitch_messages (pitch_id, timestamp_utc, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (pitch_id, timestamp_utc, role, content),
        )


def save_pitch_result(
    pitch_id: str,
    updated_at: str,
    completed_at: str | None,
    result: dict[str, Any],
) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE pitches
            SET updated_at = ?,
                completed_at = ?,
                overall_score = ?,
                allocation_usd = ?,
                decision = ?,
                validation_outcome = ?,
                result_json = ?
            WHERE pitch_id = ?
            """,
            (
                updated_at,
                completed_at,
                result.get("overall_score"),
                result.get("allocation_usd"),
                result.get("decision"),
                result.get("validation_outcome"),
                json.dumps(result, ensure_ascii=True),
                pitch_id,
            ),
        )


def _safe_json_list(raw: Any) -> list[str]:
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _base_pitch_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "pitch_id": row["pitch_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
        "status": row["status"],
        "thesis": row["thesis"],
        "time_horizon": row["time_horizon"],
        "tickers": _safe_json_list(row["tickers_json"]),
        "source_urls": _safe_json_list(row["source_urls_json"]),
        "supporting_notes": row["supporting_notes"],
        "one_shot_mode": bool(row["one_shot_mode"]),
        "overall_score": row["overall_score"],
        "allocation_usd": row["allocation_usd"],
        "decision": row["decision"],
        "validation_outcome": row["validation_outcome"],
        "message_count": int(row["message_count"] or 0),
    }


def list_pitches(limit: int = 200, status: str | None = None, completed_only: bool = False) -> list[dict[str, Any]]:
    init_db()
    where_parts: list[str] = []
    params: list[Any] = []
    if status:
        where_parts.append("p.status = ?")
        params.append(status)
    if completed_only:
        where_parts.append("p.completed_at IS NOT NULL")
    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    params.append(max(1, min(int(limit), 1000)))

    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT
                p.*,
                (SELECT COUNT(*) FROM pitch_messages m WHERE m.pitch_id = p.pitch_id) AS message_count
            FROM pitches p
            {where_sql}
            ORDER BY
                CASE WHEN p.overall_score IS NULL THEN 1 ELSE 0 END,
                p.overall_score DESC,
                p.updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_base_pitch_record(row) for row in rows]


def get_pitch(pitch_id: str) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                p.*,
                (SELECT COUNT(*) FROM pitch_messages m WHERE m.pitch_id = p.pitch_id) AS message_count
            FROM pitches p
            WHERE p.pitch_id = ?
            """,
            (pitch_id,),
        ).fetchone()
    if row is None:
        return None
    return _base_pitch_record(row)


def get_pitch_messages(pitch_id: str, limit: int = 500) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT timestamp_utc, role, content
            FROM pitch_messages
            WHERE pitch_id = ?
            ORDER BY timestamp_utc ASC, id ASC
            LIMIT ?
            """,
            (pitch_id, max(1, min(int(limit), 5000))),
        ).fetchall()
    return [
        {
            "timestamp_utc": row["timestamp_utc"],
            "role": row["role"],
            "content": row["content"],
        }
        for row in rows
    ]

