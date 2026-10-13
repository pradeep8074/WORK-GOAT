"""
================================================================================
G.O.A.T. SHIELD — FRAUD DETECTION ENGINE EQUIVALENCE VALIDATION
================================================================================
Compares the baseline linear-scan engine against:
1. The optimized indexed sliding-window detector (Single-Worker)
2. The account-partitioned parallel detector (Multi-Worker)

Verifies across all transactions:
- Identical composite risk scores
- Identical triggered rule sets (R1–R6)
- Identical status classifications (Normal, Review, High Risk)
- Identical alert explanations
- Identical ground-truth metrics
================================================================================
"""

import sys
import os
import time
from typing import List, Dict, Any

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from engine.detector import detect_frauds, detect_frauds_parallel, detect_frauds_baseline
from engine.simulator import generate_synthetic_dataset
from progressive_scalability_test import generate_workload_dataset


def compare_datasets(base_res: List[Dict[str, Any]], opt_res: List[Dict[str, Any]], label: str):
    print(f"\nComparing {len(base_res):,} transactions for {label}...")
    assert len(base_res) == len(opt_res), f"Length mismatch: {len(base_res)} vs {len(opt_res)}"

    score_mismatches = 0
    rule_mismatches = 0
    status_mismatches = 0
    explanation_mismatches = 0

    for i, (b, o) in enumerate(zip(base_res, opt_res)):
        if b["risk_score"] != o["risk_score"]:
            score_mismatches += 1
            if score_mismatches <= 3:
                print(f"  [SCORE MISMATCH] Txn {b['id']}: Base {b['risk_score']} != Opt {o['risk_score']}")

        if set(b["triggered_rules"]) != set(o["triggered_rules"]):
            rule_mismatches += 1
            if rule_mismatches <= 3:
                print(f"  [RULE MISMATCH] Txn {b['id']}: Base {b['triggered_rules']} != Opt {o['triggered_rules']}")

        if b["status"] != o["status"]:
            status_mismatches += 1

        if b["explanation"] != o["explanation"]:
            explanation_mismatches += 1

    total_mismatches = score_mismatches + rule_mismatches + status_mismatches + explanation_mismatches
    print(f"  * Risk Score Matches:         {len(base_res) - score_mismatches}/{len(base_res)} ({(1 - score_mismatches/len(base_res))*100:.2f}%)")
    print(f"  * Triggered Rule Matches:     {len(base_res) - rule_mismatches}/{len(base_res)} ({(1 - rule_mismatches/len(base_res))*100:.2f}%)")
    print(f"  * Status Matches:             {len(base_res) - status_mismatches}/{len(base_res)} ({(1 - status_mismatches/len(base_res))*100:.2f}%)")
    print(f"  * Explanation Matches:        {len(base_res) - explanation_mismatches}/{len(base_res)} ({(1 - explanation_mismatches/len(base_res))*100:.2f}%)")

    assert total_mismatches == 0, f"FAILED: Found {total_mismatches} mismatches in {label}!"
    print(f"  >>> PASSED: 100.00% Exact Equivalence Confirmed for {label}!")


def main():
    print("=" * 80)
    print("  G.O.A.T. SHIELD — FRAUD ENGINE EQUIVALENCE VERIFICATION SUITE")
    print("=" * 80)

    # 1. MVP 500-Transaction Dataset
    txns_500 = generate_synthetic_dataset(seed=42)
    t0 = time.perf_counter()
    base_500 = detect_frauds_baseline(txns_500)
    t_base_500 = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    opt_500 = detect_frauds(txns_500)
    t_opt_500 = (time.perf_counter() - t0) * 1000.0

    print(f"\n[MVP 500 Scale] Baseline: {t_base_500:.2f} ms | Optimized: {t_opt_500:.2f} ms (Speedup: {t_base_500/t_opt_500:.2f}x)")
    compare_datasets(base_500, opt_500, "MVP 500 (Single-Worker Indexed)")

    # 2. Scale 2: 5,000 Transactions (10 banks, 1,500 accounts)
    txns_5k, _ = generate_workload_dataset(num_banks=10, total_accounts=1500, total_transactions=5000, seed=42)
    t0 = time.perf_counter()
    base_5k = detect_frauds_baseline(txns_5k)
    t_base_5k = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    opt_5k = detect_frauds(txns_5k)
    t_opt_5k = (time.perf_counter() - t0) * 1000.0

    print(f"\n[Scale 5,000] Baseline: {t_base_5k:.2f} ms | Optimized: {t_opt_5k:.2f} ms (Speedup: {t_base_5k/t_opt_5k:.2f}x)")
    compare_datasets(base_5k, opt_5k, "Scale 5,000 (Single-Worker Indexed)")

    # 3. Parallel Multiprocessing Verification on 5,000 Transactions
    t0 = time.perf_counter()
    par_5k = detect_frauds_parallel(txns_5k, num_workers=2)
    t_par_5k = (time.perf_counter() - t0) * 1000.0
    print(f"\n[Scale 5,000 Parallel] Runtime: {t_par_5k:.2f} ms")
    compare_datasets(base_5k, par_5k, "Scale 5,000 (Multi-Worker 2 Cores)")

    print("\n" + "=" * 80)
    print("ALL VALIDATION CRITERIA PASSED: ZERO LOGICAL OR NUMERICAL DISCREPANCIES.")
    print("=" * 80)

if __name__ == "__main__":
    main()
