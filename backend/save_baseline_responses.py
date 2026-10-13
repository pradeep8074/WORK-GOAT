import os
import json
from pathlib import Path
from main import app
from fastapi.testclient import TestClient

def main():
    client = TestClient(app)
    out_dir = Path(__file__).resolve().parent / "baseline_responses"
    out_dir.mkdir(exist_ok=True)

    endpoints = [
        "/api/health",
        "/api/banks",
        "/api/transactions",
        "/api/alerts",
        "/api/v2/federated/benchmark",
    ]

    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code == 200, f"Endpoint {ep} returned {resp.status_code}"
        filename = ep.replace("/", "_").strip("_") + ".json"
        target_path = out_dir / filename
        target_path.write_text(json.dumps(resp.json(), indent=2), encoding="utf-8")
        print(f"Captured {ep} -> {filename} ({len(resp.json())} items/keys)")

if __name__ == "__main__":
    main()
