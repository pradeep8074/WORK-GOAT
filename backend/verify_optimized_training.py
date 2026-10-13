"""
Verify Genuine FL Optimization Training across 10 rounds.
No hardcoded metrics. All values computed dynamically from model forward pass.
"""

import time
import numpy as np
from engine.fl_coordinator import FLCoordinator
from engine.simulator import generate_accounts, generate_synthetic_dataset

def main():
    print("=" * 70, flush=True)
    print("  GENUINE FL OPTIMIZATION VERIFICATION (10 ROUNDS, E=5, 2.0x MW, tau=0.55)", flush=True)
    print("=" * 70, flush=True)

    accounts_dict = generate_accounts()
    transactions = generate_synthetic_dataset(seed=42)

    coord = FLCoordinator(seed=42, eval_threshold=0.55, mule_weight_multiplier=2.0)
    coord.initialize(transactions, accounts_dict)

    # Round 0
    r0 = coord._evaluate_model(coord.global_model)
    print(f"Round 00 (Cold Start): Loss={r0['loss']:.4f} | Acc={r0['accuracy']*100:.2f}% | Prec={r0['precision']*100:.2f}% | Rec={r0['recall']*100:.2f}% | F1={r0['f1']*100:.2f}% (TP={r0['tp']}, FP={r0['fp']}, TN={r0['tn']}, FN={r0['fn']})", flush=True)

    for r in range(1, 11):
        t0 = time.perf_counter()
        res = coord.execute_round(local_epochs=5, lr=0.02)
        elapsed = (time.perf_counter() - t0) * 1000
        m = res["metrics"]
        print(f"Round {r:02d}: Loss={m['loss']:.4f} | Acc={m['accuracy']*100:.2f}% | Prec={m['precision']*100:.2f}% | Rec={m['recall']*100:.2f}% | F1={m['f1']*100:.2f}% (TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}) | Latency={elapsed:.1f}ms", flush=True)

    # Final summary check
    r10 = coord._evaluate_model(coord.global_model)
    print("\n" + "=" * 70, flush=True)
    print("  FINAL MEASURED ROUND-10 EVALUATION ON HELD-OUT 30 TEST ACCOUNTS", flush=True)
    print("=" * 70, flush=True)
    print(f"Accuracy:        {r10['accuracy']*100:.2f}% ({r10['tp'] + r10['tn']}/30)", flush=True)
    print(f"Precision:       {r10['precision']*100:.2f}% ({r10['tp']}/{r10['tp'] + r10['fp']})", flush=True)
    print(f"Recall:          {r10['recall']*100:.2f}% ({r10['tp']}/{r10['tp'] + r10['fn']})", flush=True)
    print(f"F1-Score:        {r10['f1']*100:.2f}%", flush=True)
    print(f"Mules Detected:  {r10['tp']}/{r10['tp'] + r10['fn']}", flush=True)
    print(f"Confusion Matrix: TP={r10['tp']}, FP={r10['fp']}, TN={r10['tn']}, FN={r10['fn']}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
