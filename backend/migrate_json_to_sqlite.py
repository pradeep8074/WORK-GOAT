"""
G.O.A.T. Shield - JSON to SQLite Data Migration Script
Migrates banks.json, transactions.json, alerts.json, and benchmark JSON into SQLite storage.
Preserves original JSON files. Idempotent and safe to rerun.
"""

import sys
import json
from pathlib import Path

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import storage


def run_migration():
    print("=" * 60)
    print("   G.O.A.T. SHIELD — JSON TO SQLITE MIGRATION PIPELINE")
    print("=" * 60)

    db_path = storage.get_db_path()
    print(f"Target SQLite Database: {db_path}")

    # Ensure schema exists
    storage.init()

    data_dir = backend_dir / "data"
    total_migrated = 0

    # 1. Banks
    banks_path = data_dir / "banks.json"
    if banks_path.exists():
        with open(banks_path, "r", encoding="utf-8") as f:
            banks = json.load(f)
        storage.put_many("banks", banks)
        count = len(banks)
        total_migrated += count
        print(f"[OK] Migrated {banks_path.name} -> kind='banks': {count} records")
    else:
        print(f"[WARN] {banks_path} not found.")

    # 2. Transactions
    tx_path = data_dir / "transactions.json"
    if tx_path.exists():
        with open(tx_path, "r", encoding="utf-8") as f:
            txs = json.load(f)
        storage.put_many("transactions", txs)
        count = len(txs)
        total_migrated += count
        print(f"[OK] Migrated {tx_path.name} -> kind='transactions': {count} records")
    else:
        print(f"[WARN] {tx_path} not found.")

    # 3. Alerts
    alerts_path = data_dir / "alerts.json"
    if alerts_path.exists():
        with open(alerts_path, "r", encoding="utf-8") as f:
            alerts = json.load(f)
        storage.put_many("alerts", alerts)
        count = len(alerts)
        total_migrated += count
        print(f"[OK] Migrated {alerts_path.name} -> kind='alerts': {count} records")
    else:
        print(f"[WARN] {alerts_path} not found.")

    # 4. Optimization experiment results (if exists)
    opt_path = backend_dir / "optimization_experiment_results.json"
    if opt_path.exists():
        with open(opt_path, "r", encoding="utf-8") as f:
            opt_data = json.load(f)
        storage.put("benchmarks", "optimization_experiment_results", opt_data, seq=0)
        total_migrated += 1
        print(f"[OK] Migrated {opt_path.name} -> kind='benchmarks' id='optimization_experiment_results': 1 document")
    else:
        print(f"[INFO] {opt_path.name} not present; skipping benchmark cache migration.")

    print("-" * 60)
    print(f"Migration completed successfully. Total entities stored: {total_migrated}")
    print("Original JSON files preserved.")
    print("=" * 60)


if __name__ == "__main__":
    run_migration()
