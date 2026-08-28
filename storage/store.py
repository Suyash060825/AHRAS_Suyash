"""
AHRAS Storage Layer
--------------------
Unified insert/query API that works in two modes:

  DEV_MODE=True  → SQLite with JSON columns  (zero dependencies)
  DEV_MODE=False → MongoDB                   (production)

Usage (identical in both modes):
    from storage.store import get_store

    store = get_store()
    store.insert("events", event_dict)
    results = store.query("events", {"ocsf_class": "network_activity"}, limit=10)
    count   = store.count("events", {})
    store.close()
"""

import json
import sqlite3
import threading
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dev: SQLite store
# ─────────────────────────────────────────────────────────────────────────────

class _SQLiteStore:
    """
    Single-file SQLite store. Each collection becomes a table with:
      id TEXT PRIMARY KEY, data TEXT (JSON), created_at TEXT
    Thread-safe via a per-instance lock.
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads
        self._conn.execute("PRAGMA synchronous=NORMAL")
        log.info(f"[STORE] SQLite dev store: {db_path}")

    def _ensure_table(self, collection: str) -> None:
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{collection}" (
                id         TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                ocsf_class TEXT,
                severity   INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{collection}_class '
            f'ON "{collection}" (ocsf_class)'
        )
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{collection}_time '
            f'ON "{collection}" (created_at)'
        )
        self._conn.commit()

    def insert(self, collection: str, doc: dict) -> None:
        with self._lock:
            self._ensure_table(collection)
            doc_id = doc.get("event_id", doc.get("_id", ""))
            self._conn.execute(
                f'INSERT OR REPLACE INTO "{collection}" '
                f'(id, data, ocsf_class, severity, created_at) VALUES (?,?,?,?,?)',
                (
                    doc_id,
                    json.dumps(doc, default=str),
                    doc.get("ocsf_class", ""),
                    doc.get("severity_id", 1),
                    doc.get("time", ""),
                )
            )
            self._conn.commit()

    def query(self, collection: str, filters: dict = None,
              limit: int = 100, order_by: str = "created_at DESC") -> list[dict]:
        with self._lock:
            self._ensure_table(collection)
            rows = self._conn.execute(
                f'SELECT data FROM "{collection}" ORDER BY {order_by} LIMIT ?',
                (limit,)
            ).fetchall()
        results = [json.loads(r[0]) for r in rows]
        # Apply simple filter client-side (good enough for dev)
        if filters:
            for k, v in filters.items():
                results = [r for r in results if r.get(k) == v]
        return results

    def count(self, collection: str, filters: dict = None) -> int:
        with self._lock:
            self._ensure_table(collection)
            n = self._conn.execute(
                f'SELECT COUNT(*) FROM "{collection}"'
            ).fetchone()[0]
        return n

    def aggregate_classes(self, collection: str) -> list[dict]:
        """Return event counts grouped by ocsf_class."""
        with self._lock:
            self._ensure_table(collection)
            rows = self._conn.execute(
                f'SELECT ocsf_class, COUNT(*) FROM "{collection}" GROUP BY ocsf_class'
            ).fetchall()
        return [{"_id": r[0], "count": r[1]} for r in rows]

    def close(self) -> None:
        if hasattr(self, "_conn") and self._conn:
            self._conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Production: MongoDB store
# ─────────────────────────────────────────────────────────────────────────────

class _MongoStore:
    def __init__(self, uri: str, db_name: str):
        from pymongo import MongoClient
        client = MongoClient(uri)
        self._db = client[db_name]
        log.info(f"[STORE] MongoDB connected: {uri}/{db_name}")

    def insert(self, collection: str, doc: dict) -> None:
        col = self._db[collection]
        col.insert_one({**doc, "_id": doc.get("event_id", doc.get("_id"))})

    def query(self, collection: str, filters: dict = None,
              limit: int = 100, order_by: str = "time") -> list[dict]:
        col = self._db[collection]
        cursor = col.find(filters or {}).sort(order_by, -1).limit(limit)
        return list(cursor)

    def count(self, collection: str, filters: dict = None) -> int:
        return self._db[collection].count_documents(filters or {})

    def aggregate_classes(self, collection: str) -> list[dict]:
        pipeline = [{"$group": {"_id": "$ocsf_class", "count": {"$sum": 1}}}]
        return list(self._db[collection].aggregate(pipeline))

    def close(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────────────────────────────────────

_store_instance = None
_store_lock = threading.Lock()


def get_store():
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            from config.settings import DEV_MODE, SQLITE_PATH, MONGO_URI, MONGO_DB
            if DEV_MODE:
                _store_instance = _SQLiteStore(SQLITE_PATH)
            else:
                _store_instance = _MongoStore(MONGO_URI, MONGO_DB)
    return _store_instance
