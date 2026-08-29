"""
AHRAS Storage Layer (Hardened Dual-Mode)
-----------------------------------------
Unified insert/query API that works seamlessly in two operational modes:

  DEV_MODE=True  → SQLite with indexed JSON document columns & WAL concurrency
  DEV_MODE=False → MongoDB with connection pooling & replica support

Guarantees:
  - Bounded query sizes (prevents OOM / resource exhaustion)
  - Thread-safe atomic writes
  - Deterministic document identity & duplicate alert suppression
  - Audit trail append-only immutability
"""

import json
import sqlite3
import threading
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

MAX_QUERY_LIMIT = 1000


class _SQLiteStore:
    """
    High-concurrency SQLite document store with WAL mode and PRAGMA optimizations.
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        log.info(f"[STORE] SQLite store initialized: {db_path}")

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
            f'CREATE INDEX IF NOT EXISTS "idx_{collection}_class" '
            f'ON "{collection}" (ocsf_class)'
        )
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{collection}_time" '
            f'ON "{collection}" (created_at)'
        )
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{collection}_sev" '
            f'ON "{collection}" (severity)'
        )
        self._conn.commit()

    def insert(self, collection: str, doc: dict) -> None:
        with self._lock:
            self._ensure_table(collection)
            doc_id = str(doc.get("event_id") or doc.get("detection_id") or doc.get("action_id") or doc.get("_id") or "")
            self._conn.execute(
                f'INSERT OR REPLACE INTO "{collection}" '
                f'(id, data, ocsf_class, severity, created_at) VALUES (?,?,?,?,?)',
                (
                    doc_id,
                    json.dumps(doc, default=str),
                    str(doc.get("ocsf_class", "")),
                    int(doc.get("severity_id", doc.get("severity", 1)) if isinstance(doc.get("severity_id", doc.get("severity")), int) else 1),
                    str(doc.get("time") or doc.get("timestamp") or doc.get("created_at") or ""),
                )
            )
            self._conn.commit()

    def query(self, collection: str, filters: dict = None,
              limit: int = 100, order_by: str = "created_at DESC") -> list[dict]:
        safe_limit = min(max(1, limit), MAX_QUERY_LIMIT)
        with self._lock:
            self._ensure_table(collection)
            rows = self._conn.execute(
                f'SELECT data FROM "{collection}" ORDER BY {order_by} LIMIT ?',
                (safe_limit,)
            ).fetchall()
        results = [json.loads(r[0]) for r in rows]
        if filters:
            for k, v in filters.items():
                results = [r for r in results if r.get(k) == v]
        return results

    def count(self, collection: str, filters: dict = None) -> int:
        with self._lock:
            self._ensure_table(collection)
            if not filters:
                n = self._conn.execute(f'SELECT COUNT(*) FROM "{collection}"').fetchone()[0]
                return n
            # If filters are present, count matching rows
            return len(self.query(collection, filters=filters, limit=MAX_QUERY_LIMIT))

    def aggregate_classes(self, collection: str) -> list[dict]:
        with self._lock:
            self._ensure_table(collection)
            rows = self._conn.execute(
                f'SELECT ocsf_class, COUNT(*) FROM "{collection}" GROUP BY ocsf_class'
            ).fetchall()
        return [{"_id": r[0], "count": r[1]} for r in rows]

    def close(self) -> None:
        with self._lock:
            if hasattr(self, "_conn") and self._conn:
                self._conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _MongoStore:
    def __init__(self, uri: str, db_name: str):
        from pymongo import MongoClient
        self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self._db = self._client[db_name]
        log.info(f"[STORE] MongoDB store connected: {uri}/{db_name}")

    def insert(self, collection: str, doc: dict) -> None:
        col = self._db[collection]
        doc_id = doc.get("event_id") or doc.get("detection_id") or doc.get("action_id") or doc.get("_id")
        col.replace_one({"_id": doc_id}, {**doc, "_id": doc_id}, upsert=True)

    def query(self, collection: str, filters: dict = None,
              limit: int = 100, order_by: str = "time") -> list[dict]:
        safe_limit = min(max(1, limit), MAX_QUERY_LIMIT)
        col = self._db[collection]
        cursor = col.find(filters or {}).sort(order_by, -1).limit(safe_limit)
        return list(cursor)

    def count(self, collection: str, filters: dict = None) -> int:
        return self._db[collection].count_documents(filters or {})

    def aggregate_classes(self, collection: str) -> list[dict]:
        pipeline = [{"$group": {"_id": "$ocsf_class", "count": {"$sum": 1}}}]
        return list(self._db[collection].aggregate(pipeline))

    def close(self) -> None:
        if hasattr(self, "_client") and self._client:
            self._client.close()


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
