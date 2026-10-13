"""
================================================================================
  G.O.A.T. SHIELD — SELF-CONTAINED SCALABILITY BENCHMARK & AUDIT SUITE
================================================================================
This is a 100% self-contained Python benchmark script. It has ZERO project-internal
dependencies and requires only standard Python 3 and NumPy.

It contains:
1. Synthetic Multi-Bank Transaction Simulator (MVP, S1, S2, S3 with 4 mule typologies)
2. Transparent Rule-Based Fraud Detection Engine (6 Regulatory Rules)
3. Graph Feature Extraction Engine (10 Behavioral Graph Features)
4. Graph Topology & Normalized Adjacency Builder
5. GraphSAGE 2-Hop GNN Neural Architecture (Forward Pass)
6. 5 Warm-Up + 10 Measured Iterations per Scale
7. Throughput, P50, P95, Latency Statistics, Memory Tracking, and Edge Reconciliation
================================================================================
"""

import sys
import os
import time
import math
import random
import tracemalloc
import gc
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import numpy as np


# ==============================================================================
# SECTION 1: SYNTHETIC DATA SIMULATOR (SELF-CONTAINED)
# ==============================================================================

BANKS = [
    {"name": "Apex Bank", "code": "APX", "prefix": "ACC-APX-"},
    {"name": "Beacon Bank", "code": "BCN", "prefix": "ACC-BCN-"},
    {"name": "Crestline Bank", "code": "CLN", "prefix": "ACC-CLN-"},
]

def get_bank_for_account(acc_id: str) -> str:
    if "-APX-" in acc_id:
        return "Apex Bank"
    elif "-BCN-" in acc_id:
        return "Beacon Bank"
    elif "-CLN-" in acc_id:
        return "Crestline Bank"
    return "Apex Bank"


def generate_scaled_accounts(scale_txns: int) -> Dict[str, List[str]]:
    """Generates account pools scaled proportionally with transaction volume."""
    if scale_txns <= 500:
        accts_per_bank = 50
    elif scale_txns <= 2500:
        accts_per_bank = 250
    elif scale_txns <= 5000:
        accts_per_bank = 500
    elif scale_txns <= 10000:
        accts_per_bank = 1000
    else:
        accts_per_bank = max(50, scale_txns // 10)

    accounts = {}
    for bank in BANKS:
        fmt = f"{bank['prefix']}{{:03d}}" if accts_per_bank < 1000 else f"{bank['prefix']}{{:04d}}"
        bank_accounts = [fmt.format(i) for i in range(1, accts_per_bank + 1)]
        accounts[bank["name"]] = bank_accounts
    return accounts


def generate_scaled_dataset(scale_txns: int, seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Generates synthetic transaction dataset for a given scale preserving
    the exact 4 planted mule ring typologies and legitimate transaction distribution.
    """
    random.seed(seed)
    accounts = generate_scaled_accounts(scale_txns)
    all_accounts = []
    for bank_accs in accounts.values():
        all_accounts.extend(bank_accs)

    base_time = datetime(2026, 9, 16, 8, 0, 0)
    transactions: List[Dict[str, Any]] = []
    txn_counter = 1

    # Number of planted ring clusters scales proportionally (MVP = 1 cluster = 4 rings)
    ring_mult = max(1, scale_txns // 500)

    for m in range(ring_mult):
        ring_offset_time = base_time + timedelta(hours=m * 2)

        # ----------------------------------------------------------------------
        # Ring 1: Fan-In Multi-Source Concentration & Drainage (7 txns)
        # ----------------------------------------------------------------------
        collector = random.choice(accounts["Beacon Bank"])
        fan_in_senders = [
            random.choice(accounts["Apex Bank"]),
            random.choice(accounts["Crestline Bank"]),
            random.choice(accounts["Apex Bank"]),
            random.choice(accounts["Crestline Bank"]),
            random.choice(accounts["Apex Bank"]),
            random.choice(accounts["Beacon Bank"]),
        ]
        for i, sender in enumerate(fan_in_senders):
            amt = round(random.uniform(865000.0, 985000.0), 2)
            t_offset = timedelta(minutes=i * 6 + random.randint(1, 3))
            txn_time = ring_offset_time + t_offset
            s_bank = get_bank_for_account(sender)
            r_bank = get_bank_for_account(collector)
            transactions.append({
                "id": f"TXN-{scale_txns}-{txn_counter:07d}",
                "bank": s_bank,
                "sender": sender,
                "receiver": collector,
                "amount": amt,
                "currency": "INR",
                "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S IST"),
                "iso_timestamp": txn_time.isoformat(),
                "type": "IMPS",
                "is_cross_bank": s_bank != r_bank,
                "is_mule": True,
                "mule_ring_id": f"RING-001-{m}",
                "pattern_type": "Fan-In Funnel",
            })
            txn_counter += 1

        drain_rec = random.choice(accounts["Crestline Bank"])
        drain_time = ring_offset_time + timedelta(minutes=48)
        transactions.append({
            "id": f"TXN-{scale_txns}-{txn_counter:07d}",
            "bank": "Beacon Bank",
            "sender": collector,
            "receiver": drain_rec,
            "amount": 5250000.0,
            "currency": "INR",
            "timestamp": drain_time.strftime("%Y-%m-%d %H:%M:%S IST"),
            "iso_timestamp": drain_time.isoformat(),
            "type": "RTGS",
            "is_cross_bank": True,
            "is_mule": True,
            "mule_ring_id": f"RING-001-{m}",
            "pattern_type": "Fan-In Drainage",
        })
        txn_counter += 1

        # ----------------------------------------------------------------------
        # Ring 2: Fan-Out Dispersion / Smurfing (6 txns)
        # ----------------------------------------------------------------------
        distributor = random.choice(accounts["Apex Bank"])
        fan_out_recipients = [
            random.choice(accounts["Beacon Bank"]),
            random.choice(accounts["Crestline Bank"]),
            random.choice(accounts["Beacon Bank"]),
            random.choice(accounts["Crestline Bank"]),
            random.choice(accounts["Beacon Bank"]),
            random.choice(accounts["Crestline Bank"]),
        ]
        for i, rec in enumerate(fan_out_recipients):
            amt = round(random.uniform(830000.0, 960000.0), 2)
            t_offset = timedelta(minutes=i * 5 + random.randint(1, 2))
            txn_time = ring_offset_time + timedelta(hours=3) + t_offset
            s_bank = get_bank_for_account(distributor)
            r_bank = get_bank_for_account(rec)
            transactions.append({
                "id": f"TXN-{scale_txns}-{txn_counter:07d}",
                "bank": s_bank,
                "sender": distributor,
                "receiver": rec,
                "amount": amt,
                "currency": "INR",
                "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S IST"),
                "iso_timestamp": txn_time.isoformat(),
                "type": "RTGS",
                "is_cross_bank": s_bank != r_bank,
                "is_mule": True,
                "mule_ring_id": f"RING-002-{m}",
                "pattern_type": "Fan-Out Dispersion",
            })
            txn_counter += 1

        # ----------------------------------------------------------------------
        # Ring 3: Multi-Hop Layering Chains (8 txns)
        # ----------------------------------------------------------------------
        for ch in range(2):
            chain_nodes = [
                random.choice(accounts["Apex Bank"]),
                random.choice(accounts["Beacon Bank"]),
                random.choice(accounts["Crestline Bank"]),
                random.choice(accounts["Apex Bank"]),
                random.choice(accounts["Beacon Bank"]),
            ]
            c_amt = 950000.0
            for hop in range(len(chain_nodes) - 1):
                s = chain_nodes[hop]
                r = chain_nodes[hop + 1]
                t_offset = timedelta(minutes=hop * 15 + random.randint(2, 4))
                txn_time = ring_offset_time + timedelta(hours=6 + ch) + t_offset
                s_bank = get_bank_for_account(s)
                r_bank = get_bank_for_account(r)
                hop_amt = round(c_amt * random.uniform(0.92, 0.96), 2)
                c_amt = hop_amt
                transactions.append({
                    "id": f"TXN-{scale_txns}-{txn_counter:07d}",
                    "bank": s_bank,
                    "sender": s,
                    "receiver": r,
                    "amount": hop_amt,
                    "currency": "INR",
                    "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S IST"),
                    "iso_timestamp": txn_time.isoformat(),
                    "type": "IMPS",
                    "is_cross_bank": s_bank != r_bank,
                    "is_mule": True,
                    "mule_ring_id": f"RING-003-{m}-{ch}",
                    "pattern_type": "Multi-Hop Layering Chain",
                })
                txn_counter += 1

        # ----------------------------------------------------------------------
        # Ring 4: Circular Round-Trip Loop (6 txns)
        # ----------------------------------------------------------------------
        for cyc in range(2):
            na = random.choice(accounts["Crestline Bank"])
            nb = random.choice(accounts["Apex Bank"])
            nc = random.choice(accounts["Beacon Bank"])
            c_amt = 890000.0
            trip_hops = [(na, nb), (nb, nc), (nc, na)]
            for h_idx, (s, r) in enumerate(trip_hops):
                t_offset = timedelta(minutes=h_idx * 14 + random.randint(1, 3))
                txn_time = ring_offset_time + timedelta(hours=10 + cyc) + t_offset
                s_bank = get_bank_for_account(s)
                r_bank = get_bank_for_account(r)
                h_amt = round(c_amt * (0.94 ** h_idx), 2)
                transactions.append({
                    "id": f"TXN-{scale_txns}-{txn_counter:07d}",
                    "bank": s_bank,
                    "sender": s,
                    "receiver": r,
                    "amount": h_amt,
                    "currency": "INR",
                    "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S IST"),
                    "iso_timestamp": txn_time.isoformat(),
                    "type": "IMPS",
                    "is_cross_bank": s_bank != r_bank,
                    "is_mule": True,
                    "mule_ring_id": f"RING-004-{m}-{cyc}",
                    "pattern_type": "Circular Round-Trip Loop",
                })
                txn_counter += 1

    planted_count = len(transactions)
    normal_needed = max(0, scale_txns - planted_count)
    normal_archetypes = [
        {"type": "NACH", "min": 15000.0, "max": 120000.0},
        {"type": "UPI", "min": 250.0, "max": 25000.0},
        {"type": "NEFT", "min": 10000.0, "max": 350000.0},
        {"type": "RTGS", "min": 200000.0, "max": 650000.0},
        {"type": "IMPS", "min": 2000.0, "max": 180000.0},
    ]

    for i in range(normal_needed):
        arch = random.choice(normal_archetypes)
        amt = round(random.uniform(arch["min"], arch["max"]), 2)
        sender = random.choice(all_accounts)
        receiver = random.choice(all_accounts)
        while receiver == sender:
            receiver = random.choice(all_accounts)

        random_minute = random.randint(0, 72 * 60)
        txn_time = base_time + timedelta(minutes=random_minute, seconds=random.randint(0, 59))
        s_bank = get_bank_for_account(sender)
        r_bank = get_bank_for_account(receiver)

        transactions.append({
            "id": f"TXN-{scale_txns}-{txn_counter:07d}",
            "bank": s_bank,
            "sender": sender,
            "receiver": receiver,
            "amount": amt,
            "currency": "INR",
            "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S IST"),
            "iso_timestamp": txn_time.isoformat(),
            "type": arch["type"],
            "is_cross_bank": s_bank != r_bank,
            "is_mule": False,
            "mule_ring_id": None,
            "pattern_type": "Legitimate Transaction",
        })
        txn_counter += 1

    transactions.sort(key=lambda x: x["iso_timestamp"])
    for idx, t in enumerate(transactions, start=1):
        t["id"] = f"TXN-{scale_txns}-{idx:07d}"

    return transactions, accounts


# ==============================================================================
# SECTION 2: RULE-BASED FRAUD DETECTION ENGINE (SELF-CONTAINED)
# ==============================================================================

RULE_DEFINITIONS = {
    "R1": {"name": "FIU-IND Threshold Proximity", "weight": 20},
    "R2": {"name": "Cross-Bank Boundary Crossing", "weight": 15},
    "R3": {"name": "Fan-In Multi-Source Concentration", "weight": 20},
    "R4": {"name": "Fan-Out High Velocity Dispersion", "weight": 20},
    "R5": {"name": "Rapid Pass-Through Velocity", "weight": 20},
    "R6": {"name": "Rapid Structuring & Chain Burst", "weight": 15}
}

def parse_time(iso_str: str) -> datetime:
    return datetime.fromisoformat(iso_str)


def detect_frauds(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluates transactions against the 6 deterministic regulatory rules."""
    sorted_txns = sorted(transactions, key=lambda x: x["iso_timestamp"])
    
    account_inbounds: Dict[str, List[Tuple[datetime, float, str]]] = {}
    account_outbounds: Dict[str, List[Tuple[datetime, float, str]]] = {}
    account_all: Dict[str, List[Tuple[datetime, float, str]]] = {}

    for t in sorted_txns:
        t_time = parse_time(t["iso_timestamp"])
        amt = t["amount"]
        t_id = t["id"]
        sender = t["sender"]
        receiver = t["receiver"]

        if sender not in account_outbounds:
            account_outbounds[sender] = []
        account_outbounds[sender].append((t_time, amt, t_id))

        if receiver not in account_inbounds:
            account_inbounds[receiver] = []
        account_inbounds[receiver].append((t_time, amt, t_id))

        if sender not in account_all:
            account_all[sender] = []
        account_all[sender].append((t_time, amt, t_id))

        if receiver not in account_all:
            account_all[receiver] = []
        account_all[receiver].append((t_time, amt, t_id))

    enriched_transactions: List[Dict[str, Any]] = []

    for t in sorted_txns:
        t_time = parse_time(t["iso_timestamp"])
        amt = t["amount"]
        sender = t["sender"]
        receiver = t["receiver"]
        is_cross = t.get("is_cross_bank", False)
        
        triggered: List[str] = []

        # R1: FIU-IND Threshold Proximity (Rs 7.5L to Rs 9.99L)
        if 750000.0 <= amt <= 999999.0:
            triggered.append("R1")

        # R2: Cross-Bank Boundary Crossing
        if is_cross:
            triggered.append("R2")

        # R3: Fan-In Multi-Source Concentration (Receiver side)
        inbounds = account_inbounds.get(receiver, [])
        in_window_inbounds = [x for x in inbounds if abs((x[0] - t_time).total_seconds()) <= 3600]
        if len(in_window_inbounds) >= 3:
            triggered.append("R3")

        # R4: Fan-Out Dispersion Velocity (Sender side)
        outbounds = account_outbounds.get(sender, [])
        in_window_outbounds = [x for x in outbounds if abs((x[0] - t_time).total_seconds()) <= 3600]
        if len(in_window_outbounds) >= 3:
            triggered.append("R4")

        # R5: Rapid Pass-Through Velocity
        sender_inbounds = account_inbounds.get(sender, [])
        recent_inbounds_to_sender = [x for x in sender_inbounds if 0 <= (t_time - x[0]).total_seconds() <= 3600]
        pass_through = False
        for _, in_amt, _ in recent_inbounds_to_sender:
            if amt >= 0.70 * in_amt or in_amt >= 0.70 * amt:
                pass_through = True
                break
        
        if not pass_through:
            receiver_outbounds = account_outbounds.get(receiver, [])
            subsequent_outbounds = [x for x in receiver_outbounds if 0 <= (x[0] - t_time).total_seconds() <= 3600]
            for _, out_amt, _ in subsequent_outbounds:
                if out_amt >= 0.70 * amt or amt >= 0.70 * out_amt:
                    pass_through = True
                    break

        if pass_through:
            triggered.append("R5")

        # R6: Rapid Structuring & Chain Burst
        sender_events = account_all.get(sender, [])
        burst_events = [x for x in sender_events if abs((x[0] - t_time).total_seconds()) <= 1800]
        total_burst_amt = sum(x[1] for x in burst_events)
        if len(burst_events) >= 2 and total_burst_amt >= 400000.0:
            triggered.append("R6")

        unique_triggered = list(dict.fromkeys(triggered))
        score = min(100, sum(RULE_DEFINITIONS[r]["weight"] for r in unique_triggered))

        enriched = dict(t)
        enriched["risk_score"] = score
        enriched["triggered_rules"] = unique_triggered
        enriched_transactions.append(enriched)

    return enriched_transactions


# ==============================================================================
# SECTION 3: GRAPH FEATURE EXTRACTION & TOPOLOGY (SELF-CONTAINED)
# ==============================================================================

def extract_node_features(
    transactions: List[Dict[str, Any]],
    accounts_dict: Dict[str, List[str]]
) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, Any]]]:
    """Extracts 10 observable behavioral graph features for all accounts."""
    all_accounts = []
    for bank_name, accs in accounts_dict.items():
        all_accounts.extend(accs)
    all_accounts = sorted(all_accounts)

    acc_idx_map = {acc: i for i, acc in enumerate(all_accounts)}
    num_nodes = len(all_accounts)

    in_counts = np.zeros(num_nodes, dtype=np.float32)
    out_counts = np.zeros(num_nodes, dtype=np.float32)
    in_volumes = np.zeros(num_nodes, dtype=np.float32)
    out_volumes = np.zeros(num_nodes, dtype=np.float32)
    cross_bank_counts = np.zeros(num_nodes, dtype=np.float32)
    counterparties = [set() for _ in range(num_nodes)]
    account_timestamps = [[] for _ in range(num_nodes)]

    for t in transactions:
        s = t["sender"]
        r = t["receiver"]
        amt = float(t["amount"])
        is_cross = bool(t.get("is_cross_bank", False))
        t_time = parse_time(t["iso_timestamp"])

        if s in acc_idx_map:
            s_idx = acc_idx_map[s]
            out_counts[s_idx] += 1.0
            out_volumes[s_idx] += amt
            counterparties[s_idx].add(r)
            account_timestamps[s_idx].append(t_time)
            if is_cross:
                cross_bank_counts[s_idx] += 1.0

        if r in acc_idx_map:
            r_idx = acc_idx_map[r]
            in_counts[r_idx] += 1.0
            in_volumes[r_idx] += amt
            counterparties[r_idx].add(s)
            account_timestamps[r_idx].append(t_time)
            if is_cross:
                cross_bank_counts[r_idx] += 1.0

    features = np.zeros((num_nodes, 10), dtype=np.float32)
    observable_stats = {}

    for i, acc in enumerate(all_accounts):
        tot_in_cnt = in_counts[i]
        tot_out_cnt = out_counts[i]
        tot_tx_cnt = tot_in_cnt + tot_out_cnt
        tot_in_vol = in_volumes[i]
        tot_out_vol = out_volumes[i]

        features[i, 0] = tot_in_cnt
        features[i, 1] = tot_out_cnt
        features[i, 2] = math.log10(1.0 + tot_in_vol)
        features[i, 3] = math.log10(1.0 + tot_out_vol)
        features[i, 4] = (tot_in_vol / max(1.0, tot_in_cnt)) / 100000.0
        features[i, 5] = (tot_out_vol / max(1.0, tot_out_cnt)) / 100000.0
        denom = tot_in_vol + tot_out_vol + 1e-6
        features[i, 6] = (tot_in_vol - tot_out_vol) / denom
        features[i, 7] = cross_bank_counts[i] / max(1.0, tot_tx_cnt)
        features[i, 8] = float(len(counterparties[i]))

        ts_list = sorted(account_timestamps[i])
        max_burst = 0
        for idx_a, t_a in enumerate(ts_list):
            burst = sum(1 for t_b in ts_list[idx_a:] if (t_b - t_a).total_seconds() <= 3600)
            if burst > max_burst:
                max_burst = burst
        features[i, 9] = float(max_burst)

        observable_stats[acc] = {
            "in_degree": int(tot_in_cnt),
            "out_degree": int(tot_out_cnt),
            "total_inflow": round(float(tot_in_vol), 2),
            "total_outflow": round(float(tot_out_vol), 2),
            "cross_bank_ratio": round(float(features[i, 7]), 3),
            "unique_counterparties": int(features[i, 8]),
            "burst_velocity_60m": int(features[i, 9])
        }

    return all_accounts, features, observable_stats


def build_adjacency_matrix(
    transactions: List[Dict[str, Any]],
    acc_idx_map: Dict[str, int]
) -> np.ndarray:
    """Constructs row-normalized bidirectional adjacency matrix with self-loops."""
    num_nodes = len(acc_idx_map)
    adj = np.eye(num_nodes, dtype=np.float32)

    for t in transactions:
        u = t["sender"]
        v = t["receiver"]
        if u in acc_idx_map and v in acc_idx_map:
            u_idx = acc_idx_map[u]
            v_idx = acc_idx_map[v]
            adj[u_idx, v_idx] = 1.0
            adj[v_idx, u_idx] = 1.0

    row_sum = np.sum(adj, axis=1, keepdims=True)
    adj_norm = adj / np.maximum(1.0, row_sum)
    return adj_norm


# ==============================================================================
# SECTION 4: GRAPHSAGE GNN ARCHITECTURE (SELF-CONTAINED)
# ==============================================================================

class GraphSAGEEngine:
    """GraphSAGE 2-Hop Message Passing Classifier with Mean Neighborhood Aggregation."""
    def __init__(self, in_features: int = 10, hidden_dim: int = 32, num_classes: int = 2, seed: int = 42):
        rng = np.random.RandomState(seed)
        
        # Layer 1: in_features -> hidden_dim (32)
        scale1 = math.sqrt(2.0 / (in_features + hidden_dim))
        self.W_self1 = rng.randn(in_features, hidden_dim).astype(np.float32) * scale1
        self.W_neigh1 = rng.randn(in_features, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)

        # Layer 2: hidden_dim (32) -> 16
        scale2 = math.sqrt(2.0 / (hidden_dim + 16))
        self.W_self2 = rng.randn(hidden_dim, 16).astype(np.float32) * scale2
        self.W_neigh2 = rng.randn(hidden_dim, 16).astype(np.float32) * scale2
        self.b2 = np.zeros((1, 16), dtype=np.float32)

        # Classifier Head: 16 -> num_classes (2)
        scale_cls = math.sqrt(2.0 / (16 + num_classes))
        self.W_cls = rng.randn(16, num_classes).astype(np.float32) * scale_cls
        self.b_cls = np.zeros((1, num_classes), dtype=np.float32)

    def forward(self, X: np.ndarray, A_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 1-Hop Message Passing & Aggregation
        X_neigh = np.dot(A_norm, X)
        Z1 = np.dot(X, self.W_self1) + np.dot(X_neigh, self.W_neigh1) + self.b1
        H1 = np.maximum(0.0, Z1) # ReLU

        # 2-Hop Message Passing & Aggregation
        H1_neigh = np.dot(A_norm, H1)
        Z2 = np.dot(H1, self.W_self2) + np.dot(H1_neigh, self.W_neigh2) + self.b2
        H2 = np.maximum(0.0, Z2) # ReLU

        # Output Softmax Probabilities
        logits = np.dot(H2, self.W_cls) + self.b_cls
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs, H2, H1, Z1


# ==============================================================================
# SECTION 5: CONTROLLED BENCHMARK & AUDIT RUNNER
# ==============================================================================

def run_controlled_profile(scale_name: str, target_txns: int) -> Dict[str, Any]:
    print(f"\n================================================================================", flush=True)
    print(f"  PROFILING SCALE: {scale_name} ({target_txns:,} Transactions)", flush=True)
    print(f"================================================================================", flush=True)

    gc.collect()

    # 1. Dataset Generation (5 warmups + 10 measured runs)
    gen_times = []
    for _ in range(5):
        _ = generate_scaled_dataset(target_txns, seed=42)
    for _ in range(10):
        t0 = time.perf_counter()
        txns, _ = generate_scaled_dataset(target_txns, seed=42)
        gen_times.append((time.perf_counter() - t0) * 1000)

    transactions, raw_pool_accounts = generate_scaled_dataset(target_txns, seed=42)
    actual_txns = len(transactions)

    # --------------------------------------------------------------------------
    # DERIVE NODES & EDGES DIRECTLY FROM THE GENERATED TRANSACTION DATASET
    # Formula:
    # Nodes = |unique(sender_account UNION receiver_account)|
    # Edges = number of unique directed (sender, receiver) pairs
    # Transactions = total generated transaction records
    # Cross-Bank Edges = unique directed edges where sender_bank != receiver_bank
    # --------------------------------------------------------------------------
    senders = [t["sender"] for t in transactions]
    receivers = [t["receiver"] for t in transactions]
    unique_nodes = sorted(list(set(senders).union(set(receivers))))
    num_nodes = len(unique_nodes)
    acc_idx_map = {acc: i for i, acc in enumerate(unique_nodes)}

    # Derived accounts dict partitioned by bank for feature extraction
    derived_accounts_dict = {
        bank["name"]: [acc for acc in unique_nodes if get_bank_for_account(acc) == bank["name"]]
        for bank in BANKS
    }

    # Unique directed edges
    unique_directed_edges_set = set((t["sender"], t["receiver"]) for t in transactions)
    unique_directed_edges = len(unique_directed_edges_set)
    unique_undirected_edges = len(set(tuple(sorted(p)) for p in unique_directed_edges_set))

    # Cross-bank unique directed edges
    cross_bank_edges_set = set(
        (u, v) for (u, v) in unique_directed_edges_set
        if get_bank_for_account(u) != get_bank_for_account(v)
    )
    cross_bank_edges = len(cross_bank_edges_set)
    intra_bank_edges = unique_directed_edges - cross_bank_edges

    # 2. Feature Extraction (5 warmups + 10 runs)
    print("  -> Running Graph Feature Extraction (5 warmups + 10 measured iterations)...", flush=True)
    feat_times = []
    for _ in range(5):
        _ = extract_node_features(transactions, derived_accounts_dict)
    for _ in range(10):
        t0 = time.perf_counter()
        _ = extract_node_features(transactions, derived_accounts_dict)
        feat_times.append((time.perf_counter() - t0) * 1000)

    _, X_raw, obs_stats = extract_node_features(transactions, derived_accounts_dict)
    mean = np.mean(X_raw, axis=0, keepdims=True)
    std = np.std(X_raw, axis=0, keepdims=True) + 1e-6
    X_norm = (X_raw - mean) / std

    # 3. Adjacency Matrix Construction (5 warmups + 10 runs)
    print("  -> Running Graph Adjacency Matrix Builder (5 warmups + 10 measured iterations)...", flush=True)
    adj_times = []
    for _ in range(5):
        _ = build_adjacency_matrix(transactions, acc_idx_map)
    for _ in range(10):
        t0 = time.perf_counter()
        _ = build_adjacency_matrix(transactions, acc_idx_map)
        adj_times.append((time.perf_counter() - t0) * 1000)

    A_norm = build_adjacency_matrix(transactions, acc_idx_map)

    # 4. Rule Detection (5 warm-ups + 10 measured iterations)
    print("  -> Running Rule Engine (5 warmups + 10 measured iterations)...", flush=True)
    for _ in range(5):
        _ = detect_frauds(transactions)

    rule_times_ms = []
    rule_throughputs = []
    for r_i in range(10):
        t0 = time.perf_counter()
        _ = detect_frauds(transactions)
        t_el_ms = (time.perf_counter() - t0) * 1000.0
        rule_times_ms.append(t_el_ms)
        rule_throughputs.append(actual_txns / (t_el_ms / 1000.0))

    # 5. GNN Forward Inference (5 warm-ups + 10 measured iterations)
    print("  -> Running GNN Forward Inference (5 warmups + 10 measured iterations)...", flush=True)
    model = GraphSAGEEngine(in_features=10, hidden_dim=32, num_classes=2, seed=42)
    for _ in range(5):
        _ = model.forward(X_norm, A_norm)

    gnn_times_ms = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = model.forward(X_norm, A_norm)
        gnn_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # 6. Peak Memory Measurement (Isolated per scale)
    tracemalloc.start()
    _dummy_adj = build_adjacency_matrix(transactions, acc_idx_map)
    _dummy_probs, _, _, _ = model.forward(X_norm, _dummy_adj)
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = round(peak_mem_bytes / (1024 * 1024), 2)

    def compute_stats(arr: List[float]) -> Dict[str, float]:
        return {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "samples": [round(x, 3) for x in arr]
        }

    rule_stats = compute_stats(rule_times_ms)
    tp_stats = compute_stats(rule_throughputs)
    gnn_stats = compute_stats(gnn_times_ms)
    feat_stats = compute_stats(feat_times)
    adj_stats = compute_stats(adj_times)
    gen_stats = compute_stats(gen_times)

    print(f"  [PROFILED {scale_name}]")
    print(f"  * Transactions:           {actual_txns:,}")
    print(f"  * Unique Nodes:           {num_nodes:,}")
    print(f"  * Total Directed Edges:   {unique_directed_edges:,}")
    print(f"  * Cross-Bank Edges:       {cross_bank_edges:,} (Intra-Bank: {intra_bank_edges:,})")
    print(f"  * Unique Undirected Edges: {unique_undirected_edges:,}")
    print(f"  * Rule Throughput (tx/s): Median={tp_stats['median']:,.1f} | Mean={tp_stats['mean']:,.1f} | Min={tp_stats['min']:,.1f} | Max={tp_stats['max']:,.1f}")
    print(f"  * Rule Latency (ms):      Median={rule_stats['median']}ms | Mean={rule_stats['mean']}ms | Std={rule_stats['std']}ms")
    print(f"  * GNN Inference (ms):     Median(P50)={gnn_stats['p50']}ms | P95={gnn_stats['p95']}ms | Std={gnn_stats['std']}ms")
    print(f"  * Peak Memory:            {peak_mem_mb} MB")

    return {
        "scale": scale_name,
        "target_txns": target_txns,
        "actual_txns": actual_txns,
        "num_nodes": num_nodes,
        "unique_directed_edges": unique_directed_edges,
        "cross_bank_edges": cross_bank_edges,
        "intra_bank_edges": intra_bank_edges,
        "unique_undirected_edges": unique_undirected_edges,
        "peak_mem_mb": peak_mem_mb,
        "gen_stats": gen_stats,
        "feat_stats": feat_stats,
        "adj_stats": adj_stats,
        "rule_stats": rule_stats,
        "throughput_stats": tp_stats,
        "gnn_stats": gnn_stats
    }


def main():
    print("=" * 80, flush=True)
    print("     G.O.A.T. SHIELD - COMPLETE SELF-CONTAINED SCALABILITY BENCHMARK     ", flush=True)
    print("=" * 80, flush=True)

    # Evaluated across all 4 scales: MVP (500), S1 (2,500), S2 (5,000), S3 (10,000)
    scales = [
        ("MVP", 500),
        ("S1", 2500),
        ("S2", 5000),
        ("S3", 10000),
    ]

    all_results = {}
    for name, tx in scales:
        res = run_controlled_profile(name, tx)
        all_results[name] = res

    # --------------------------------------------------------------------------
    # 1. FINAL COMPARATIVE TABLE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115, flush=True)
    print("                              FINAL MEASURED SCALABILITY BENCHMARK TABLE                             ", flush=True)
    print("=" * 115, flush=True)
    print(f"{'Scale':<6} | {'Transactions':<12} | {'Nodes':<8} | {'Total Directed Edges':<20} | {'Cross-Bank Edges':<16} | {'Throughput (tx/s)':<18} | {'GNN P50':<9} | {'GNN P95':<9} | {'Peak RAM':<10}", flush=True)
    print("-" * 115, flush=True)

    for name in ["MVP", "S1", "S2", "S3"]:
        r = all_results[name]
        tx_str = f"{r['actual_txns']:,}"
        nd_str = f"{r['num_nodes']:,}"
        ed_str = f"{r['unique_directed_edges']:,}"
        cb_str = f"{r['cross_bank_edges']:,}"
        tp_str = f"{r['throughput_stats']['median']:,.1f}"
        p50_str = f"{r['gnn_stats']['p50']} ms"
        p95_str = f"{r['gnn_stats']['p95']} ms"
        ram_str = f"{r['peak_mem_mb']} MB"
        print(f"{name:<6} | {tx_str:<12} | {nd_str:<8} | {ed_str:<20} | {cb_str:<16} | {tp_str:<18} | {p50_str:<9} | {p95_str:<9} | {ram_str:<10}", flush=True)

    print("=" * 115, flush=True)

    # --------------------------------------------------------------------------
    # 2. RAW TIMING ARRAYS (10 SAMPLES PER SCALE)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("                    RAW MEASURED TIMING ARRAYS (10 SAMPLES)              ", flush=True)
    print("=" * 80, flush=True)
    for name in ["MVP", "S1", "S2", "S3"]:
        r = all_results[name]
        print(f"\n[{name} - {r['actual_txns']:,} Transactions]", flush=True)
        print(f"  * Rule Latency (ms):  {r['rule_stats']['samples']}", flush=True)
        print(f"  * Throughput (tx/s):  {r['throughput_stats']['samples']}", flush=True)
        print(f"  * GNN Inference (ms): {r['gnn_stats']['samples']}", flush=True)
        print(f"  * Feature Extraction: {r['feat_stats']['samples']}", flush=True)
        print(f"  * Graph Construction: {r['adj_stats']['samples']}", flush=True)

    # --------------------------------------------------------------------------
    # 3. METHODOLOGY & CALCULATION SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("                       CALCULATION METHODOLOGY SUMMARY                   ", flush=True)
    print("=" * 80, flush=True)
    print("1. Throughput Calculation:  Throughput = Transaction_Count / Rule_Latency_Seconds (Median over 10 runs)", flush=True)
    print("2. GNN Inference Latency:   Forward pass through 2-hop GraphSAGE (P50 and P95 over 10 measured runs)", flush=True)
    print("3. Warm-Up Protocol:        5 warm-up iterations executed prior to timing to eliminate JIT/allocator warm-up", flush=True)
    print("4. Node Derivation:         Nodes = |unique(sender_account UNION receiver_account)| directly from transactions", flush=True)
    print("5. Edge Accounting:         Total directed edges count unique (u, v) pairs; cross-bank edges count bank(u) != bank(v)", flush=True)
    print("6. Reproducibility:         Seed = 42 across all scales; 100% deterministic and standalone.", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
