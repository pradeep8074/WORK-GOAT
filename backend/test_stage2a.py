"""
G.O.A.T. Shield Stage 2A - Complete Automated Verification Suite
Verifies all Stage 1 and Stage 2A endpoints, synthetic datasets, NetworkX graph,
rule-based detector, ground-truth metrics, and frontend connectivity.
"""

import urllib.request
import json
import sys

BASE_API = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:5173"

passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}: {detail}")
        failed += 1


def fetch_json(endpoint):
    url = f"{BASE_API}{endpoint}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "GOAT-Shield-Test", "Connection": "close"}
    )
    with urllib.request.urlopen(req, timeout=20) as res:
        return res.status, json.loads(res.read().decode("utf-8"))


print("==================================================")
print("  G.O.A.T. Shield Stage 2A Verification Suite")
print("==================================================")

# 1. Health & Stage 1 Endpoints
print("\n1. Verifying Stage 1 Endpoints (Backward Compatibility):")
try:
    status, root = fetch_json("/")
    test("Root endpoint status 200", status == 200)
    test("Root stage is Stage 2A", root.get("stage") == "Stage 2A")

    status, health = fetch_json("/api/health")
    test("Health check 200 healthy", status == 200 and health.get("status") == "healthy")

    status, banks = fetch_json("/api/banks")
    test("Banks endpoint 200 (3 banks)", status == 200 and len(banks) == 3)
    bank_names = [b["name"] for b in banks]
    test("Bank names match", any("Apex" in n for n in bank_names) and any("Beacon" in n for n in bank_names))

    status, txns = fetch_json("/api/transactions")
    test("Stage 1 transactions 200 (12 txns)", status == 200 and len(txns) == 12)
    test("Stage 1 transactions use INR", all(t.get("currency") == "INR" for t in txns))

    status, alerts = fetch_json("/api/alerts")
    test("Stage 1 alerts 200 (6 alerts)", status == 200 and len(alerts) == 6)
except Exception as e:
    test("Stage 1 API connectivity", False, str(e))

# 2. Stage 2A Simulation Endpoint
print("\n2. Verifying Stage 2A Simulation Engine (/api/v2/simulate):")
try:
    status, sim_full = fetch_json("/api/v2/simulate?limit=0")
    test("Simulate endpoint status 200", status == 200)
    test("Total dataset is exactly 500 transactions", sim_full.get("total") == 500)

    txns_500 = sim_full.get("transactions", [])
    test("Returned transactions length is 500", len(txns_500) == 500)

    # Validate transaction schema & INR currency
    first_txn = txns_500[0]
    required_fields = ["id", "bank", "sender", "receiver", "amount", "currency", "type", "is_cross_bank", "is_mule", "risk_score", "status", "triggered_rules", "explanation"]
    test("Transaction schema contains all required fields", all(k in first_txn for k in required_fields))
    test("All 500 transactions have currency == 'INR'", all(t.get("currency") == "INR" for t in txns_500))

    # Validate ground truth & mule rings
    mule_txns = [t for t in txns_500 if t.get("is_mule")]
    test("Planted mule transactions present (>50)", len(mule_txns) >= 50, f"Found {len(mule_txns)}")

    rings = set(t.get("mule_ring_id") for t in mule_txns if t.get("mule_ring_id"))
    test("All 4 planted mule rings present (RING-001..RING-004)", {"RING-001", "RING-002", "RING-003", "RING-004"}.issubset(rings))

    # Test pagination
    status, sim_page = fetch_json("/api/v2/simulate?limit=25&offset=50")
    test("Pagination limit=25 returns 25 records", len(sim_page.get("transactions", [])) == 25)
    test("Pagination total metadata is 500", sim_page.get("total") == 500)

    # Test filtering by mule ring
    status, sim_r1 = fetch_json("/api/v2/simulate?ring_id=RING-001")
    test("Filter ring_id=RING-001 returns only RING-001", all(t.get("mule_ring_id") == "RING-001" for t in sim_r1.get("transactions", [])))
except Exception as e:
    test("Simulation API tests", False, str(e))

# 3. Stage 2A NetworkX Graph Endpoint
print("\n3. Verifying Stage 2A NetworkX Graph Engine (/api/v2/graph):")
try:
    status, graph = fetch_json("/api/v2/graph")
    test("Graph endpoint status 200", status == 200)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    stats = graph.get("stats", {})

    test("Graph contains 150 account nodes", len(nodes) == 150)
    test("Graph contains 500 transaction edges", len(edges) == 500)
    test("Cross-bank edges computed (>200)", stats.get("cross_bank_edges", 0) > 200)
    test("Mule nodes identified (>40)", stats.get("mule_nodes", 0) > 40)
    test("Top inbound hubs (fan-in) computed", len(stats.get("top_inbound_hubs", [])) == 5)
    test("Top outbound sources (fan-out) computed", len(stats.get("top_outbound_sources", [])) == 5)

    # Test cross-bank filter
    status, graph_cross = fetch_json("/api/v2/graph?cross_bank_only=true")
    test("Graph cross_bank_only filter returns only cross-bank edges", all(e.get("is_cross_bank") for e in graph_cross.get("edges", [])))
except Exception as e:
    test("Graph API tests", False, str(e))

# 4. Stage 2A Detection Engine Endpoint
print("\n4. Verifying Stage 2A Rule-Based Detector (/api/v2/detect):")
try:
    status, detect = fetch_json("/api/v2/detect")
    test("Detect endpoint status 200", status == 200)

    rule_defs = detect.get("rule_definitions", {})
    test("Contains all 6 deterministic rules (R1..R6)", len(rule_defs) == 6 and set(rule_defs.keys()) == {"R1", "R2", "R3", "R4", "R5", "R6"})

    freqs = detect.get("rule_frequencies", {})
    test("Rule trigger frequencies computed for all rules", all(k in freqs for k in ["R1", "R2", "R3", "R4", "R5", "R6"]))
    test("Total evaluated is 500", detect.get("total_evaluated") == 500)
except Exception as e:
    test("Detection API tests", False, str(e))

# 5. Stage 2A Metrics Endpoint
print("\n5. Verifying Stage 2A Ground-Truth Evaluation Metrics (/api/v2/metrics):")
try:
    status, metrics = fetch_json("/api/v2/metrics")
    test("Metrics endpoint status 200", status == 200)

    m = metrics.get("metrics", {})
    cm = metrics.get("confusion_matrix", {})
    gt = metrics.get("ground_truth", {})

    test("Ground truth mule count recorded (>50)", gt.get("mule_transactions", 0) >= 50)
    test("Ground truth normal count recorded (>400)", gt.get("normal_transactions", 0) >= 400)
    test("True Positives (TP) > 0", cm.get("true_positives", 0) > 0)
    test("True Negatives (TN) > 0", cm.get("true_negatives", 0) > 0)
    test("Precision is genuine calculated value (>0.8)", m.get("precision", 0) >= 0.8)
    test("Recall is genuine calculated value (>0.7)", m.get("recall", 0) >= 0.7)
    test("F1 score is genuine calculated value (>0.75)", m.get("f1_score", 0) >= 0.75)
    test("Accuracy is genuine calculated value (>0.9)", m.get("accuracy", 0) >= 0.9)

    ring_bd = metrics.get("per_ring_breakdown", [])
    test("Per-ring breakdown contains all 4 rings", len(ring_bd) == 4)
    test("All rings achieve >50% detection rate", all(r.get("total_coverage_rate", 0) >= 50 for r in ring_bd))
except Exception as e:
    test("Metrics API tests", False, str(e))

# 6. Frontend Dev Server Connectivity
print("\n6. Verifying Frontend Dev Server Connectivity:")
try:
    req = urllib.request.Request(FRONTEND_URL, headers={"User-Agent": "GOAT-Shield-Test"})
    with urllib.request.urlopen(req, timeout=20) as res:
        test(f"Frontend server {FRONTEND_URL} returned 200", res.status == 200)
except Exception as e:
    test("Frontend server connectivity", False, str(e))

# Summary
print("\n==================================================")
print(f"  VERIFICATION RESULTS: {passed} PASSED, {failed} FAILED")
print("==================================================")

if failed > 0:
    sys.exit(1)
else:
    print("\nALL STAGE 2A TESTS PASSED WITH 100% SUCCESS!")

