"""
Unit & Integration Tests for G.O.A.T. Shield SQLite Storage Engine (backend/storage.py)
Tests CRUD operations, sequence ordering, auto-import, round-trip fidelity, and concurrent writes.
"""

import os
import json
import tempfile
import threading
import sys
from pathlib import Path
import pytest

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import storage


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """Isolates each test in an independent temporary SQLite database."""
    test_db = tmp_path / "test_goat.db"
    monkeypatch.setenv("GOAT_DB_PATH", str(test_db))
    storage.close_connection()
    storage.init()
    yield test_db
    storage.close_connection()


def test_put_and_get():
    """Test basic put and get operations with JSON serialization."""
    payload = {"account": "ACC-001", "risk": 85.5, "flags": ["R1", "R2"]}
    storage.put("accounts", "ACC-001", payload, seq=1)

    retrieved = storage.get("accounts", "ACC-001")
    assert retrieved == payload
    assert storage.get("accounts", "NON_EXISTENT", default=None) is None


def test_put_many_and_all_ordering():
    """Test put_many and ensure all_() preserves exact sequence ordering (ORDER BY seq)."""
    items = [
        {"id": "item-c", "val": 300},
        {"id": "item-a", "val": 100},
        {"id": "item-b", "val": 200},
    ]
    storage.put_many("test_kind", items)

    all_items = storage.all_("test_kind")
    assert list(all_items.keys()) == ["item-c", "item-a", "item-b"]
    assert list(all_items.values()) == items


def test_delete():
    """Test record deletion."""
    storage.put("test_kind", "del-1", {"val": "hello"})
    assert storage.get("test_kind", "del-1") is not None

    deleted = storage.delete("test_kind", "del-1")
    assert deleted is True
    assert storage.get("test_kind", "del-1") is None

    # Deleting non-existent returns False
    assert storage.delete("test_kind", "del-1") is False


def test_concurrent_writes(isolated_db):
    """Test concurrent thread writes under WAL mode without locking issues."""
    errors = []
    num_threads = 8
    writes_per_thread = 50

    def worker(thread_id):
        try:
            for i in range(writes_per_thread):
                key = f"t{thread_id}_{i}"
                storage.put("concurrent", key, {"thread": thread_id, "idx": i}, seq=i)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors occurred during concurrent writes: {errors}"
    records = storage.all_("concurrent")
    assert len(records) == num_threads * writes_per_thread


def test_round_trip_json_fidelity():
    """
    Round-trip test asserting original JSON files match storage.all_(kind).
    """
    backend_dir = Path(__file__).resolve().parent.parent
    data_dir = backend_dir / "data"

    # Re-run auto import for this fresh isolated DB
    storage.auto_import_json_data()

    # 1. Banks
    banks_path = data_dir / "banks.json"
    if banks_path.exists():
        with open(banks_path, "r", encoding="utf-8") as f:
            expected_banks = json.load(f)
        actual_banks = list(storage.all_("banks").values())
        assert actual_banks == expected_banks, "Migrated banks do not match banks.json"

    # 2. Transactions
    tx_path = data_dir / "transactions.json"
    if tx_path.exists():
        with open(tx_path, "r", encoding="utf-8") as f:
            expected_txs = json.load(f)
        actual_txs = list(storage.all_("transactions").values())
        assert actual_txs == expected_txs, "Migrated transactions do not match transactions.json"

    # 3. Alerts
    alerts_path = data_dir / "alerts.json"
    if alerts_path.exists():
        with open(alerts_path, "r", encoding="utf-8") as f:
            expected_alerts = json.load(f)
        actual_alerts = list(storage.all_("alerts").values())
        assert actual_alerts == expected_alerts, "Migrated alerts do not match alerts.json"
