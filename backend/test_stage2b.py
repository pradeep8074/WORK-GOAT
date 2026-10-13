import sys
import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000"

def fetch_json(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def run_tests():
    print("=================================================================")
    print("       G.O.A.T. SHIELD — STAGE 2B END-TO-END VERIFICATION       ")
    print("=================================================================\n")

    # 1. Test /api/v2/gnn/predict
    print("[TEST 1/5] Testing /api/v2/gnn/predict ...")
    pred = fetch_json("/api/v2/gnn/predict")
    assert pred["total_accounts"] == 150
    assert "2-Layer GraphSAGE" in pred["model_architecture"]
    assert len(pred["predictions"]) == 150

    # Ensure no data leakage in predictions
    for node in pred["predictions"]:
        assert "is_mule" not in node, f"Leakage detected: is_mule in {node['account_id']}"
        assert "mule_ring_id" not in node, f"Leakage detected: mule_ring_id in {node['account_id']}"
        assert 0.0 <= node["gnn_mule_probability"] <= 1.0
        assert 0 <= node["gnn_risk_score"] <= 100
        assert node["gnn_status"] in ("High Risk", "Review", "Normal")
        assert "observable_stats" in node
    print("  --> PASS: 150 nodes scored, proper schema, zero ground-truth leakage verified.")

    # 2. Test /api/v2/gnn/metrics
    print("\n[TEST 2/5] Testing /api/v2/gnn/metrics ...")
    metrics = fetch_json("/api/v2/gnn/metrics")
    assert metrics["test_node_count"] == 30
    assert metrics["test_ground_truth"]["mule_nodes"] + metrics["test_ground_truth"]["normal_nodes"] == 30

    cm = metrics["confusion_matrix"]
    tp = cm["true_positives"]
    fp = cm["false_positives"]
    tn = cm["true_negatives"]
    fn = cm["false_negatives"]
    assert tp + fp + tn + fn == 30

    m = metrics["metrics"]
    calc_acc = round((tp + tn) / 30 * 100, 2)
    calc_prec = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
    calc_rec = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
    calc_f1 = round(2 * calc_prec * calc_rec / (calc_prec + calc_rec), 2) if (calc_prec + calc_rec) > 0 else 0.0

    assert abs(m["accuracy_percent"] - calc_acc) < 0.05
    assert abs(m["precision_percent"] - calc_prec) < 0.05
    assert abs(m["recall_percent"] - calc_rec) < 0.05
    assert abs(m["f1_percent"] - calc_f1) < 0.05
    print(f"  --> PASS: Test Accuracy={m['accuracy_percent']}%, Precision={m['precision_percent']}%, Recall={m['recall_percent']}%, F1={m['f1_percent']}%")

    # 3. Test /api/v2/gnn/compare
    print("\n[TEST 3/5] Testing /api/v2/gnn/compare ...")
    comp = fetch_json("/api/v2/gnn/compare")
    assert comp["test_node_count"] == 30
    assert "rule_baseline" in comp
    assert "graphsage_gnn" in comp
    assert "comparison_summary" in comp

    rule_f1 = comp["rule_baseline"]["f1_percent"]
    gnn_f1 = comp["graphsage_gnn"]["f1_percent"]
    summary = comp["comparison_summary"]
    delta_f1 = summary["f1_delta"]
    agreement = summary["consensus_percentage"]
    print(f"  --> PASS: Rule Baseline Test F1={rule_f1}%, GraphSAGE GNN Test F1={gnn_f1}%, Delta F1={delta_f1}%, Agreement={agreement}%")

    # 4. Regression Test: Stage 2A Endpoints
    print("\n[TEST 4/5] Regression testing Stage 2A endpoints ...")
    sim = fetch_json("/api/v2/simulate?limit=10")
    assert sim["total"] == 500
    assert len(sim["transactions"]) == 10

    graph = fetch_json("/api/v2/graph")
    assert len(graph["nodes"]) == 150
    assert len(graph["edges"]) == 500

    detect = fetch_json("/api/v2/detect")
    assert detect["total_evaluated"] == 500

    metrics_2a = fetch_json("/api/v2/metrics")
    assert metrics_2a["total_transactions"] == 500
    print("  --> PASS: Stage 2A simulation, graph, detect, and metrics endpoints all healthy.")

    # 5. Regression Test: Stage 1 Endpoints
    print("\n[TEST 5/5] Regression testing Stage 1 endpoints ...")
    health = fetch_json("/api/health")
    assert health["status"] == "healthy"

    banks = fetch_json("/api/banks")
    assert len(banks) == 3

    txns = fetch_json("/api/transactions")
    assert len(txns) > 0

    alerts = fetch_json("/api/alerts")
    assert len(alerts) > 0
    print("  --> PASS: Stage 1 health, banks, transactions, and alerts endpoints all healthy.")

    print("\n=================================================================")
    print("  SUCCESS: ALL 5/5 STAGE 2B & REGRESSION VERIFICATIONS PASSED!   ")
    print("=================================================================")

if __name__ == "__main__":
    run_tests()
