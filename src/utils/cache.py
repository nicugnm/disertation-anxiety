"""SQLite-backed cache for expensive function calls (LLM API mostly)."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import orjson


class SqliteCache:
    """A tiny key-value cache. Keys are hashed; values are JSON-serialized.

    Used to make LLM labeling and zero-shot evaluation idempotent: re-running
    the same prompt against the same text returns the cached response.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)"
        )
        self._conn.commit()

    @staticmethod
    def make_key(*parts: Any) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(orjson.dumps(p, option=orjson.OPT_SORT_KEYS))
            h.update(b"\x1e")
        return h.hexdigest()

    def get(self, key: str) -> Any | None:
        row = self._conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return orjson.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, orjson.dumps(value)),
        )
        self._conn.commit()

    def __contains__(self, key: str) -> bool:
        return (
            self._conn.execute("SELECT 1 FROM cache WHERE key = ?", (key,)).fetchone()
            is not None
        )

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
