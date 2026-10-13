"""
G.O.A.T. Shield Stage 2A - Ground-Truth Evaluation & Metrics Engine
Computes genuine Precision, Recall, F1 score, Accuracy, Confusion Matrix,
and Ring-Level breakdown by comparing rule-based detection results against ground-truth labels.
"""

from typing import List, Dict, Any


def calculate_metrics(scored_transactions: List[Dict[str, Any]], high_risk_threshold: int = 65) -> Dict[str, Any]:
    """
    Computes genuine mathematical metrics from the generated dataset.
    Positive condition: risk_score >= high_risk_threshold (High Risk)
    Ground truth positive: is_mule == True
    """
    total = len(scored_transactions)
    if total == 0:
        return {
            "total_transactions": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "accuracy": 0.0,
            "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        }

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    mule_count = 0
    normal_count = 0

    high_risk_count = 0
    review_count = 0
    normal_classified_count = 0

    ring_stats: Dict[str, Dict[str, int]] = {
        "RING-001": {"total": 0, "high_risk": 0, "review": 0, "normal": 0},
        "RING-002": {"total": 0, "high_risk": 0, "review": 0, "normal": 0},
        "RING-003": {"total": 0, "high_risk": 0, "review": 0, "normal": 0},
        "RING-004": {"total": 0, "high_risk": 0, "review": 0, "normal": 0},
    }

    rule_frequencies: Dict[str, int] = {
        "R1": 0,
        "R2": 0,
        "R3": 0,
        "R4": 0,
        "R5": 0,
        "R6": 0
    }

    for t in scored_transactions:
        is_mule = t.get("is_mule", False)
        score = t.get("risk_score", 0)
        status = t.get("status", "Normal")
        ring_id = t.get("mule_ring_id")
        rules = t.get("triggered_rules", [])

        # Count rule triggers
        for r in rules:
            if r in rule_frequencies:
                rule_frequencies[r] += 1

        if is_mule:
            mule_count += 1
            if ring_id and ring_id in ring_stats:
                ring_stats[ring_id]["total"] += 1
                if status == "High Risk":
                    ring_stats[ring_id]["high_risk"] += 1
                elif status == "Review":
                    ring_stats[ring_id]["review"] += 1
                else:
                    ring_stats[ring_id]["normal"] += 1
        else:
            normal_count += 1

        # Prediction: High Risk is considered positive detection
        is_predicted_positive = (score >= high_risk_threshold)

        if is_predicted_positive:
            high_risk_count += 1
            if is_mule:
                tp += 1
            else:
                fp += 1
        else:
            if status == "Review":
                review_count += 1
            else:
                normal_classified_count += 1

            if is_mule:
                fn += 1
            else:
                tn += 1

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
    accuracy = round((tp + tn) / total, 4)

    # Calculate per-ring coverage rates
    rings_formatted = []
    ring_names = {
        "RING-001": "Fan-In Funneling (Collector Drainage)",
        "RING-002": "Fan-Out Dispersion (Smurfing)",
        "RING-003": "Multi-Hop Cross-Bank Layering Chains",
        "RING-004": "Circular Round-Trip Loops"
    }

    for rid, rdata in ring_stats.items():
        rtotal = rdata["total"]
        rhr = rdata["high_risk"]
        rrev = rdata["review"]
        cov = round((rhr + rrev) / rtotal * 100, 1) if rtotal > 0 else 0.0
        hr_rate = round(rhr / rtotal * 100, 1) if rtotal > 0 else 0.0

        rings_formatted.append({
            "ring_id": rid,
            "name": ring_names.get(rid, rid),
            "total_transactions": rtotal,
            "detected_high_risk": rhr,
            "detected_review": rrev,
            "high_risk_detection_rate": hr_rate,
            "total_coverage_rate": cov
        })

    return {
        "total_transactions": total,
        "ground_truth": {
            "mule_transactions": mule_count,
            "normal_transactions": normal_count,
            "mule_percentage": round((mule_count / total) * 100, 2)
        },
        "detection_summary": {
            "high_risk": high_risk_count,
            "review": review_count,
            "normal": normal_classified_count,
            "threshold_used": high_risk_threshold
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "precision_percent": round(precision * 100, 2),
            "recall_percent": round(recall * 100, 2),
            "f1_percent": round(f1 * 100, 2),
            "accuracy_percent": round(accuracy * 100, 2)
        },
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        },
        "per_ring_breakdown": rings_formatted,
        "rule_frequencies": rule_frequencies
    }
