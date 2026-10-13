"""
Profiling script for G.O.A.T. Shield end-to-end pipeline at 5k, 25k, and 100k transactions.
Uses cProfile and tracemalloc. Saves pstats/text profiles to results/profile_*.txt.
Measures per-stage timings:
1. data load / generation
2. graph build
3. feature extraction
4. rule engine
5. GNN inference
6. ensemble
7. alert creation
"""

import sys
import os
import time
import cProfile
import pstats
import io
import tracemalloc
import gc
from typing import Dict, Any, List, Tuple
import numpy as np

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from progressive_scalability_test import generate_workload_dataset
from engine.detector import detect_frauds
from engine.graph_builder import build_graph
from engine.gnn_detector import extract_node_features, build_adjacency_matrix, GraphSAGEEngine


def run_pipeline_with_stage_timing(
    num_banks: int,
    num_accounts: int,
    num_transactions: int,
    seed: int = 42
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Executes the 7-stage pipeline and returns exact stage timings in milliseconds.
    Discards initial warmup if needed.
    """
    timings = {}
    
    # Stage 1: Data load / generation
    t0 = time.perf_counter()
    transactions, accounts_dict = generate_workload_dataset(
        num_banks=num_banks,
        total_accounts=num_accounts,
        total_transactions=num_transactions,
        seed=seed
    )
    timings["data_load"] = (time.perf_counter() - t0) * 1000.0

    # Stage 2: Rule Engine
    t0 = time.perf_counter()
    scored_txns = detect_frauds(transactions)
    timings["rule_engine"] = (time.perf_counter() - t0) * 1000.0

    # Stage 3: Graph Build
    t0 = time.perf_counter()
    G = build_graph(scored_txns, accounts_dict=accounts_dict)
    timings["graph_build"] = (time.perf_counter() - t0) * 1000.0

    # Stage 4: Feature Extraction
    t0 = time.perf_counter()
    all_accounts, X_raw, obs_stats = extract_node_features(transactions, accounts_dict)
    acc_idx_map = {acc: i for i, acc in enumerate(all_accounts)}
    mean = np.mean(X_raw, axis=0, keepdims=True)
    std = np.std(X_raw, axis=0, keepdims=True) + 1e-6
    X_norm = (X_raw - mean) / std
    timings["feature_extraction"] = (time.perf_counter() - t0) * 1000.0

    # Stage 5: GNN Inference
    # For large graphs, build adjacency and run 1 forward pass
    t0 = time.perf_counter()
    # If nodes <= 5000, exact dense matrix; otherwise chunked / sampled
    if len(all_accounts) <= 5000:
        A_norm = build_adjacency_matrix(transactions, acc_idx_map)
        model = GraphSAGEEngine(in_features=10, hidden_dim=32, num_classes=2, seed=seed)
        probs, _, _, _ = model.forward(X_norm, A_norm)
        gnn_scores = probs[:, 1]
    else:
        # Avoid 20000x20000 1.6GB dense matrix in baseline: compute row-normalized aggregation in CSR/subgraph
        # For fair baseline measurement, measure exact adjacency build on sampled or full
        # Let's see: np.eye(20000) takes 1.6GB which fits in RAM (machine has 16GB)
        A_norm = build_adjacency_matrix(transactions, acc_idx_map)
        model = GraphSAGEEngine(in_features=10, hidden_dim=32, num_classes=2, seed=seed)
        probs, _, _, _ = model.forward(X_norm, A_norm)
        gnn_scores = probs[:, 1]
    timings["gnn_inference"] = (time.perf_counter() - t0) * 1000.0

    # Stage 6: Ensemble
    t0 = time.perf_counter()
    # Map max rule score per node
    node_rule_scores = np.zeros(len(all_accounts), dtype=np.float32)
    for t in scored_txns:
        s = t["sender"]
        r = t["receiver"]
        score = float(t.get("risk_score", 0)) / 100.0
        if s in acc_idx_map:
            node_rule_scores[acc_idx_map[s]] = max(node_rule_scores[acc_idx_map[s]], score)
        if r in acc_idx_map:
            node_rule_scores[acc_idx_map[r]] = max(node_rule_scores[acc_idx_map[r]], score)
    
    ensemble_scores = 0.5 * gnn_scores + 0.5 * node_rule_scores
    timings["ensemble"] = (time.perf_counter() - t0) * 1000.0

    # Stage 7: Alert Creation
    t0 = time.perf_counter()
    alerts = []
    for t in scored_txns:
        if t.get("risk_score", 0) >= 65:
            alerts.append({
                "alert_id": f"ALT-{t['id']}",
                "transaction_id": t["id"],
                "account": t["sender"],
                "bank": t["bank"],
                "risk_score": t["risk_score"],
                "rules": t.get("triggered_rules", []),
                "explanation": t.get("explanation", "")
            })
    timings["alert_creation"] = (time.perf_counter() - t0) * 1000.0

    total_pipeline_ms = sum(timings.values())
    timings["total_pipeline"] = total_pipeline_ms

    meta = {
        "num_transactions": len(transactions),
        "num_accounts": len(all_accounts),
        "alert_count": len(alerts),
        "edge_count": G.number_of_edges()
    }
    return timings, meta


def profile_scale(scale_name: str, num_banks: int, num_accounts: int, num_transactions: int):
    print(f"\n>>> PROFILING SCALE {scale_name}: {num_transactions:,} txns, {num_accounts:,} accounts, {num_banks} banks")
    
    # 1. Warmup run (discarded)
    print("  Running warmup run...")
    _ = run_pipeline_with_stage_timing(num_banks, num_accounts, min(500, num_transactions), seed=999)
    
    # 2. cProfile execution
    print("  Executing with cProfile & tracemalloc...")
    gc.collect()
    tracemalloc.start()
    
    pr = cProfile.Profile()
    pr.enable()
    
    timings, meta = run_pipeline_with_stage_timing(num_banks, num_accounts, num_transactions, seed=42)
    
    pr.disable()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)
    timings["peak_memory_mb"] = peak_mb
    
    # Format and save profile report
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(40)
    
    profile_txt_path = os.path.join(RESULTS_DIR, f"profile_{scale_name.lower()}.txt")
    with open(profile_txt_path, "w", encoding="utf-8") as f:
        f.write(f"G.O.A.T. SHIELD CPROFILE DUMP — {scale_name.upper()} ({num_transactions:,} TXNS, {num_accounts:,} NODES)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Pipeline Time: {timings['total_pipeline']:.2f} ms ({timings['total_pipeline']/1000:.2f} s)\n")
        f.write(f"Peak Memory: {peak_mb:.2f} MB\n")
        f.write(f"Alerts Generated: {meta['alert_count']}\n")
        f.write("\nSTAGE BREAKDOWN:\n")
        for stage in ["data_load", "rule_engine", "graph_build", "feature_extraction", "gnn_inference", "ensemble", "alert_creation"]:
            pct = (timings[stage] / timings["total_pipeline"]) * 100.0
            f.write(f"  - {stage:<20}: {timings[stage]:10.2f} ms ({pct:5.1f}%)\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 40 CUMULATIVE TIME CALLS:\n")
        f.write("=" * 80 + "\n")
        f.write(s.getvalue())
        
    print(f"  Saved profile to {profile_txt_path}")
    print(f"  Total Time: {timings['total_pipeline']:.2f} ms | Peak RAM: {peak_mb:.1f} MB | Alerts: {meta['alert_count']}")
    for stage in ["data_load", "rule_engine", "graph_build", "feature_extraction", "gnn_inference", "ensemble", "alert_creation"]:
        pct = (timings[stage] / timings["total_pipeline"]) * 100.0
        print(f"    * {stage:<20}: {timings[stage]:8.2f} ms ({pct:5.1f}%)")
    
    return timings, meta


def main():
    print("=" * 80)
    print("G.O.A.T. SHIELD — DETAILED PIPELINE STAGE PROFILER")
    print("Scales: 5k (10 banks), 25k (25 banks), 100k (50 banks)")
    print("=" * 80)
    
    results = {}
    # 5k
    t5k, m5k = profile_scale("5k", num_banks=10, num_accounts=1500, num_transactions=5000)
    results["5k"] = (t5k, m5k)
    
    # 25k
    t25k, m25k = profile_scale("25k", num_banks=25, num_accounts=5000, num_transactions=25000)
    results["25k"] = (t25k, m25k)
    
    # 100k
    t100k, m100k = profile_scale("100k", num_banks=50, num_accounts=20000, num_transactions=100000)
    results["100k"] = (t100k, m100k)
    
    print("\n" + "=" * 90)
    print(f"{'Stage':<22} | {'5k Time (ms)':<14} | {'25k Time (ms)':<14} | {'100k Time (ms)':<14} | {'Growth (100k vs 5k)'}")
    print("-" * 90)
    for stage in ["data_load", "rule_engine", "graph_build", "feature_extraction", "gnn_inference", "ensemble", "alert_creation", "total_pipeline"]:
        v5 = results["5k"][0][stage]
        v25 = results["25k"][0][stage]
        v100 = results["100k"][0][stage]
        factor = v100 / max(1e-4, v5)
        print(f"{stage:<22} | {v5:14.2f} | {v25:14.2f} | {v100:14.2f} | {factor:10.2f}x (vs 20x data)")
    print("=" * 90)


if __name__ == "__main__":
    main()
