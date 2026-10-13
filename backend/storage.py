"""
G.O.A.T. Shield Storage Engine - SQLite WAL Persistence Layer
Provides centralized, thread-safe, and ACID-compliant persistence using stdlib sqlite3.
NOTE: This is the ONLY module in the backend allowed to import sqlite3.
"""

import os
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Thread-local storage for DB connections
_local = threading.local()


def get_db_path() -> str:
    """Returns absolute path to SQLite database configured by GOAT_DB_PATH or defaults to backend/goat.db."""
    env_path = os.environ.get("GOAT_DB_PATH")
    if env_path:
        return str(Path(env_path).resolve())
    return str(Path(__file__).resolve().parent / "goat.db")


def get_connection() -> sqlite3.Connection:
    """
    Returns thread-local SQLite connection with WAL mode and synchronous=NORMAL.
    Never share connection objects across threads or into ProcessPoolExecutor workers.
    """
    conn = getattr(_local, "connection", None)
    if conn is None:
        db_path = get_db_path()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        _local.connection = conn
    return conn


def close_connection() -> None:
    """Closes thread-local connection if open."""
    conn = getattr(_local, "connection", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.connection = None


def init() -> None:
    """
    Initializes database schema and auto-imports data if table is empty.
    Creates records table with kind, id, data JSON, and seq for deterministic ordering.
    """
    conn = get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                kind TEXT NOT NULL,
                id TEXT NOT NULL,
                data TEXT NOT NULL,
                seq INTEGER DEFAULT 0,
                PRIMARY KEY (kind, id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_kind_seq ON records(kind, seq)")

    cursor = conn.execute("SELECT COUNT(*) AS cnt FROM records")
    row = cursor.fetchone()
    count = row["cnt"] if isinstance(row, sqlite3.Row) else row[0]
    if count == 0:
        auto_import_json_data()


def put(kind: str, id: str, obj: Any, seq: int = 0) -> None:
    """Inserts or updates a single record with parameterized query inside transaction."""
    conn = get_connection()
    data_str = json.dumps(obj)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO records (kind, id, data, seq) VALUES (?, ?, ?, ?)",
            (kind, str(id), data_str, seq)
        )


def put_many(kind: str, items: Union[Dict[str, Any], List[Any]]) -> None:
    """
    Inserts or updates multiple records within a single transaction.
    Supports:
      - dict: {id: obj, ...}
      - list of dicts with 'id' key: [{id: ..., ...}]
      - list of tuples: [(id, obj, seq), ...] or [(id, obj), ...]
    Preserves sequential ordering.
    """
    conn = get_connection()
    rows = []
    if isinstance(items, dict):
        for seq, (item_id, obj) in enumerate(items.items()):
            rows.append((kind, str(item_id), json.dumps(obj), seq))
    elif isinstance(items, list):
        for seq, elem in enumerate(items):
            if isinstance(elem, dict) and "id" in elem:
                item_id = str(elem["id"])
                rows.append((kind, item_id, json.dumps(elem), seq))
            elif isinstance(elem, (list, tuple)) and len(elem) >= 2:
                item_id = str(elem[0])
                obj = elem[1]
                s = elem[2] if len(elem) >= 3 else seq
                rows.append((kind, item_id, json.dumps(obj), s))
            else:
                rows.append((kind, str(seq), json.dumps(elem), seq))
    else:
        raise ValueError(f"Unsupported items type for put_many: {type(items)}")

    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO records (kind, id, data, seq) VALUES (?, ?, ?, ?)",
            rows
        )


def get(kind: str, id: str, default: Any = None) -> Any:
    """Retrieves and deserializes single JSON record by kind and id."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT data FROM records WHERE kind = ? AND id = ?",
        (kind, str(id))
    )
    row = cursor.fetchone()
    if row is None:
        return default
    data_str = row["data"] if isinstance(row, sqlite3.Row) else row[0]
    return json.loads(data_str)


def all_(kind: str) -> Dict[str, Any]:
    """
    Retrieves all records for a given kind ordered by seq ASC.
    Returns {id: deserialized_obj, ...} preserving insertion/original sequence.
    """
    conn = get_connection()
    cursor = conn.execute(
        "SELECT id, data FROM records WHERE kind = ? ORDER BY seq ASC",
        (kind,)
    )
    result = {}
    for row in cursor.fetchall():
        row_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
        data_str = row["data"] if isinstance(row, sqlite3.Row) else row[1]
        result[row_id] = json.loads(data_str)
    return result


def delete(kind: str, id: str) -> bool:
    """Deletes a single record by kind and id. Returns True if deleted."""
    conn = get_connection()
    with conn:
        cursor = conn.execute(
            "DELETE FROM records WHERE kind = ? AND id = ?",
            (kind, str(id))
        )
        return cursor.rowcount > 0


def auto_import_json_data() -> Dict[str, int]:
    """
    Imports core JSON data files into SQLite if they exist.
    Idempotent and safe to run multiple times.
    """
    backend_dir = Path(__file__).resolve().parent
    data_dir = backend_dir / "data"
    counts = {}

    # 1. banks.json
    banks_path = data_dir / "banks.json"
    if banks_path.exists():
        with open(banks_path, "r", encoding="utf-8") as f:
            banks = json.load(f)
            if isinstance(banks, list):
                put_many("banks", banks)
                counts["banks"] = len(banks)

    # 2. transactions.json
    tx_path = data_dir / "transactions.json"
    if tx_path.exists():
        with open(tx_path, "r", encoding="utf-8") as f:
            txs = json.load(f)
            if isinstance(txs, list):
                put_many("transactions", txs)
                counts["transactions"] = len(txs)

    # 3. alerts.json
    alerts_path = data_dir / "alerts.json"
    if alerts_path.exists():
        with open(alerts_path, "r", encoding="utf-8") as f:
            alerts = json.load(f)
            if isinstance(alerts, list):
                put_many("alerts", alerts)
                counts["alerts"] = len(alerts)

    # 4. optimization_experiment_results.json (if present)
    opt_path = backend_dir / "optimization_experiment_results.json"
    if opt_path.exists():
        with open(opt_path, "r", encoding="utf-8") as f:
            opt_data = json.load(f)
            put("benchmarks", "optimization_experiment_results", opt_data, seq=0)
            counts["optimization_experiment_results"] = 1

    return counts
