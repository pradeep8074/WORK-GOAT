import urllib.request
import json

def test_system():
    print("=== Testing Backend (FastAPI on :8000) ===")
    endpoints = {
        "/": ["project", "version", "status"],
        "/api/health": ["status"],
        "/api/banks": None,
        "/api/transactions": None,
        "/api/alerts": None
    }
    
    for ep, expected_keys in endpoints.items():
        url = f"http://127.0.0.1:8000{ep}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            status = resp.status
            assert status == 200, f"Expected 200, got {status}"
            if expected_keys:
                for k in expected_keys:
                    assert k in data, f"Key {k} missing in {ep}"
            
            # Verify INR currency consistency
            if ep == "/api/transactions":
                assert len(data) > 0, "No transactions returned"
                for txn in data:
                    assert txn.get("currency") == "INR", f"Expected INR currency, got {txn.get('currency')}"
                    assert txn.get("amount") > 0, "Transaction amount must be positive"
                print(f"  [PASS] All {len(data)} transactions verified with INR currency")
            
            if ep == "/api/alerts":
                for alert in data:
                    details = alert.get("details", "")
                    assert "$" not in details, f"Found dollar in alert details: {details}"
                print(f"  [PASS] All {len(data)} alerts verified with no dollar symbols")

            print(f"  [PASS] {ep} returned {status} (Payload size: {len(data)})")
            
    print("\n=== Testing Frontend (Vite on :5173) ===")
    with urllib.request.urlopen("http://127.0.0.1:5173/") as resp:
        content = resp.read().decode()
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        assert "G.O.A.T. Shield" in content, "Project title missing in index.html"
        assert '<div id="root">' in content, "Root element missing in index.html"
        assert '/src/main.jsx' in content, "Main entry point script missing in index.html"
        print(f"  [PASS] http://127.0.0.1:5173/ returned 200 with HTML shell and bundle entry")

    print("\n=== Verification Successful: All tests passed! ===")

if __name__ == "__main__":
    test_system()
