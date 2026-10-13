import json
import sys
from pathlib import Path
from main import app
from fastapi.testclient import TestClient

def main():
    client = TestClient(app)
    base_dir = Path(__file__).resolve().parent / "baseline_responses"

    endpoints = [
        "/api/health",
        "/api/banks",
        "/api/transactions",
        "/api/alerts",
        "/api/v2/federated/benchmark",
    ]

    all_matched = True
    print("=" * 70)
    print("API RESPONSE REGRESSION VERIFICATION (BASELINE vs SQLITE STORAGE)")
    print("=" * 70)

    for ep in endpoints:
        filename = ep.replace("/", "_").strip("_") + ".json"
        baseline_file = base_dir / filename
        assert baseline_file.exists(), f"Baseline file {baseline_file} missing!"

        with open(baseline_file, "r", encoding="utf-8") as f:
            expected_data = json.load(f)

        resp = client.get(ep)
        actual_data = resp.json()

        if expected_data == actual_data:
            print(f"[IDENTICAL] {ep:<30} -> Exact match with baseline ({len(actual_data)} entities/keys)")
        else:
            print(f"[DIFF FAIL] {ep:<30} -> Response differs from baseline!")
            all_matched = False

    print("=" * 70)
    if all_matched:
        print("ALL 5 ENDPOINTS RETURN 100% IDENTICAL RESPONSES TO BASELINE!")
        sys.exit(0)
    else:
        print("ERROR: One or more endpoints returned differing data.")
        sys.exit(1)

if __name__ == "__main__":
    main()
