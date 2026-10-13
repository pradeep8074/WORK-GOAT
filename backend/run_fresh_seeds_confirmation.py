"""
G.O.A.T. Shield — Fresh-Seed Confirmation Runner (Seeds 5–9)
Evaluates the final Hybrid model on fresh, unpeeked seeds 5, 6, 7, 8, 9
Strictly follows the 60/20/20 stratified protocol with validation tuning only.
Saves empirical output to results/fresh_seeds_results.json.
"""

import sys
import os
import json
import time
import numpy as np

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from run_ablation_study import evaluate_single_seed

def run_fresh_seeds():
    seeds = [5, 6, 7, 8, 9]
    print(f"Starting Fresh Seeds Confirmation Run for seeds {seeds}...")
    
    seed_results = []
    
    for s in seeds:
        t0 = time.time()
        print(f"Evaluating Seed {s}...")
        variant_metrics, sweep_results, val_choice = evaluate_single_seed(s)
        h = variant_metrics["hybrid_tuned"]
        elapsed = time.time() - t0
        print(f"Seed {s} finished in {elapsed:.1f}s | Val F1: {val_choice['val_f1']*100:.2f}% | Test F1: {h['f1']*100:.2f}%, P: {h['precision']*100:.2f}%, R: {h['recall']*100:.2f}%, FPR: {h['fpr']*100:.2f}%")
        
        seed_results.append({
            "seed": s,
            "chosen_beta": val_choice["chosen_beta"],
            "chosen_threshold": val_choice["chosen_threshold"],
            "val_f1": val_choice["val_f1"],
            "precision": h["precision"],
            "recall": h["recall"],
            "f1": h["f1"],
            "accuracy": h["accuracy"],
            "fpr": h["fpr"],
            "tp": h["tp"],
            "fp": h["fp"],
            "fn": h["fn"],
            "tn": h["tn"],
            "pr_auc": h["pr_auc"],
            "roc_auc": h["roc_auc"]
        })

    # Compute mean and std over all 5 fresh seeds
    summary = {
        "precision": {
            "mean": float(np.mean([x["precision"] for x in seed_results])),
            "std": float(np.std([x["precision"] for x in seed_results]))
        },
        "recall": {
            "mean": float(np.mean([x["recall"] for x in seed_results])),
            "std": float(np.std([x["recall"] for x in seed_results]))
        },
        "f1": {
            "mean": float(np.mean([x["f1"] for x in seed_results])),
            "std": float(np.std([x["f1"] for x in seed_results]))
        },
        "accuracy": {
            "mean": float(np.mean([x["accuracy"] for x in seed_results])),
            "std": float(np.std([x["accuracy"] for x in seed_results]))
        },
        "fpr": {
            "mean": float(np.mean([x["fpr"] for x in seed_results])),
            "std": float(np.std([x["fpr"] for x in seed_results]))
        },
        "seeds_evaluated": seeds,
        "per_seed_runs": seed_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    out_file = os.path.join(RESULTS_DIR, "fresh_seeds_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUCCESS] Fresh seeds results written to {out_file}!")
    print(f"Summary (5 Seeds: 5–9):")
    print(f"  Precision: {summary['precision']['mean']*100:.2f}% ± {summary['precision']['std']*100:.2f}%")
    print(f"  Recall:    {summary['recall']['mean']*100:.2f}% ± {summary['recall']['std']*100:.2f}%")
    print(f"  F1 Score:  {summary['f1']['mean']*100:.2f}% ± {summary['f1']['std']*100:.2f}%")
    print(f"  Accuracy:  {summary['accuracy']['mean']*100:.2f}% ± {summary['accuracy']['std']*100:.2f}%")
    print(f"  FPR:       {summary['fpr']['mean']*100:.2f}% ± {summary['fpr']['std']*100:.2f}%")

if __name__ == "__main__":
    run_fresh_seeds()
