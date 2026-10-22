"""Optional, anonymous SQLite log of screenings (off by default).

Only the answers, the risk level and the model probability are stored: no names,
IP addresses or other identifiers. Turn it on with ``storage.enabled: true``,
and only with a consent and retention policy in place.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS screenings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    answers_json TEXT NOT NULL,
    level TEXT NOT NULL,
    probability REAL,
    model_version TEXT
)
"""


class ScreeningStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def save(self, answers: dict, level: str, probability: float | None, model_version: str | None) -> int | None:
        try:
            with self._lock, self._connect() as con:
                cur = con.execute(
                    "INSERT INTO screenings (created_utc, answers_json, level, probability, model_version) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (datetime.now(timezone.utc).isoformat(timespec="seconds"), json.dumps(answers), level,
                     probability, model_version))
                return int(cur.lastrowid)
        except sqlite3.Error as exc:  # never fail a screening because logging failed
            log.warning("Could not store screening: %s", exc)
            return None

    def count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM screenings").fetchone()[0])
