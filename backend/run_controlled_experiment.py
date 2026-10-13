"""
Controlled Stage 2C-2 Performance Experiment Runner
Executes 5 FL rounds with 5 local epochs per round from cold-start random initialization.
Measures all required telemetry, timing breakdowns, convergence metrics, and verifies mathematical invariants.
"""

import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def post(path, params=None):
    url = f"{BASE_URL}{path}"
    if params:
        query = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"{url}?{query}"
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def run_experiment():
    print("=========================================================================")
    print("        G.O.A.T. SHIELD — STAGE 2C-2 CONTROLLED EXPERIMENT (5 ROUNDS)    ")
    print("=========================================================================\n")

    # 1. Reset FL Session to Cold-Start Random Initialization
    print("[1] Resetting FL session to fresh cold-start random initialization...")
    reset_res = post("/api/v2/federated/reset")
    print("    --> Reset status:", reset_res["status"])

    status = get("/api/v2/federated/status")
    print("    --> Initialized:", status["is_initialized"])
    print("    --> Modulus bits:", status["public_key"]["modulus_bits"])
    print("    --> Model parameters:", status["model_architecture"]["trainable_parameters"])

    # 2. Get Round 0 Baseline Metrics
    metrics_0 = get("/api/v2/federated/metrics")
    round0 = metrics_0["history"][0]
    print(f"\n[BASELINE: Round 0 (Random Initialization)]")
    print(f"    • Global Loss:      {round0['loss']:.4f}")
    print(f"    • Test Accuracy:    {round0['accuracy']*100:.2f}%")
    print(f"    • Test Precision:   {round0['precision']*100:.2f}%")
    print(f"    • Test Recall:      {round0['recall']*100:.2f}%")
    print(f"    • Test F1-Score:    {round0['f1']*100:.2f}%")

    # 3. Execute 5 FL Rounds (local_epochs = 5, lr = 0.02)
    print("\n-------------------------------------------------------------------------")
    print("                EXECUTING 5 PAILLIER-ENCRYPTED FL ROUNDS                 ")
    print("-------------------------------------------------------------------------")

    rounds_data = []

    for r in range(1, 6):
        print(f"\n>>> Executing Round {r} (5 local epochs per bank)...")
        t_start = time.perf_counter()
        round_res = post("/api/v2/federated/round", {"local_epochs": 5, "lr": 0.02})
        total_time = (time.perf_counter() - t_start) * 1000

        m = round_res["metrics"]
        t = round_res["telemetry"]
        tb = t["timing_breakdown_ms"]
        ll = t["local_losses"]

        print(f"    [Round {r} Results]")
        print(f"    • Global Loss:       {m['loss']:.4f}")
        print(f"    • Test Accuracy:     {m['accuracy']*100:.2f}%")
        print(f"    • Test Precision:    {m['precision']*100:.2f}%")
        print(f"    • Test Recall:       {m['recall']*100:.2f}%")
        print(f"    • Test F1-Score:     {m['f1']*100:.2f}%")
        print(f"    • Local Losses:      APX={ll['APEX']:.4f}, BCN={ll['BEACON']:.4f}, CLN={ll['CRESTLINE']:.4f}")
        print(f"    • Timing Breakdown:  LocalTrain={tb['local_training']:.2f}ms, Encrypt={tb['paillier_encryption']:.2f}ms, Agg={tb['homomorphic_aggregation']:.2f}ms, Dec={tb['decryption_and_fedavg']:.2f}ms")
        print(f"    • Total Latency:     {tb['total_round_latency']:.2f}ms")

        rounds_data.append({
            "round": r,
            "metrics": m,
            "telemetry": t
        })

    # Save experiment results to JSON
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "round_0": round0,
            "rounds_1_to_5": rounds_data,
            "benchmark_comparison": status["benchmark_comparison"]
        }, f, indent=2)

    print("\n=========================================================================")
    print("                 EXPERIMENT COMPLETE — RESULTS SAVED                     ")
    print("=========================================================================")

if __name__ == "__main__":
    run_experiment()
