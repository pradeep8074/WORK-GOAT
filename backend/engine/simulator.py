"""
G.O.A.T. Shield Stage 2A - Synthetic Data Simulator
Generates 500 realistic synthetic INR transactions with 4 planted mule-ring topologies
and ground-truth labels across 3 fictional banks (Apex Bank, Beacon Bank, Crestline Bank).
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

BANKS = [
    {"name": "Apex Bank", "code": "APX", "prefix": "ACC-APX-"},
    {"name": "Beacon Bank", "code": "BCN", "prefix": "ACC-BCN-"},
    {"name": "Crestline Bank", "code": "CLN", "prefix": "ACC-CLN-"},
]

PAYMENT_TYPES = ["IMPS", "RTGS", "NEFT", "UPI", "NACH"]


def get_bank_for_account(acc_id: str) -> str:
    if "-APX-" in acc_id:
        return "Apex Bank"
    elif "-BCN-" in acc_id:
        return "Beacon Bank"
    elif "-CLN-" in acc_id:
        return "Crestline Bank"
    parts = acc_id.split("-")
    if len(parts) >= 2:
        return f"Bank {parts[1]}"
    return "Apex Bank"


def generate_accounts() -> Dict[str, List[str]]:
    """Generates 50 accounts per bank (150 total)."""
    accounts = {}
    for bank in BANKS:
        bank_accounts = [f"{bank['prefix']}{i:03d}" for i in range(1, 51)]
        accounts[bank["name"]] = bank_accounts
    return accounts


def generate_synthetic_dataset(seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generates 500 total synthetic transactions:
    - ~78 planted mule ring transactions (4 distinct topologies) with ground truth is_mule=True
    - ~422 normal transactions with ground truth is_mule=False
    All amounts are in INR.
    """
    random.seed(seed)
    accounts = generate_accounts()
    all_accounts = []
    for bank_accs in accounts.values():
        all_accounts.extend(bank_accs)

    base_time = datetime(2026, 9, 16, 8, 0, 0)
    transactions: List[Dict[str, Any]] = []
    txn_counter = 1

    # -------------------------------------------------------------
    # 1. Planted Mule Ring 1: Fan-In Funneling (RING-001)
    # 6 senders send structured amounts (Rs 8.65L - Rs 9.85L) to 1 collector (ACC-BCN-008)
    # within 45 minutes, followed by a rapid drainage to ACC-CLN-049. (7 txns)
    # -------------------------------------------------------------
    collector_1 = "ACC-BCN-008"
    fan_in_senders = ["ACC-APX-003", "ACC-CLN-014", "ACC-APX-025", "ACC-CLN-031", "ACC-APX-044", "ACC-BCN-018"]
    ring1_time = base_time + timedelta(hours=4, minutes=10)

    for i, sender in enumerate(fan_in_senders):
        amt = round(random.uniform(865000.0, 985000.0), 2)
        t_offset = timedelta(minutes=i * 6 + random.randint(1, 3))
        txn_time = ring1_time + t_offset
        s_bank = get_bank_for_account(sender)
        r_bank = get_bank_for_account(collector_1)

        transactions.append({
            "id": f"TXN-S2A-{txn_counter:04d}",
            "bank": s_bank,
            "sender": sender,
            "receiver": collector_1,
            "amount": amt,
            "currency": "INR",
            "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S IST"),
            "iso_timestamp": txn_time.isoformat(),
            "type": "IMPS",
            "is_cross_bank": s_bank != r_bank,
            "is_mule": True,
            "mule_ring_id": "RING-001",
            "pattern_type": "Fan-In Funnel",
            "note": "Planted Fan-In funneling into central collector account"
        })
        txn_counter += 1

    drain_time = ring1_time + timedelta(minutes=48)
    transactions.append({
        "id": f"TXN-S2A-{txn_counter:04d}",
        "bank": "Beacon Bank",
        "sender": collector_1,
        "receiver": "ACC-CLN-049",
        "amount": 5250000.0,
        "currency": "INR",
        "timestamp": drain_time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "iso_timestamp": drain_time.isoformat(),
        "type": "RTGS",
        "is_cross_bank": True,
        "is_mule": True,
        "mule_ring_id": "RING-001",
        "pattern_type": "Fan-In Drainage",
        "note": "Planted Fan-In collector rapid high-value outbound drainage"
    })
    txn_counter += 1

    # -------------------------------------------------------------
    # 2. Planted Mule Ring 2: Fan-Out Dispersion / Smurfing (RING-002)
    # 1 distributor in Apex Bank disperses structured amounts (Rs 8.25L - Rs 9.6L)
    # to 6 recipient accounts in Crestline & Beacon within 35 minutes. (6 txns)
    # -------------------------------------------------------------
    distributor_2 = "ACC-APX-015"
    fan_out_recipients = ["ACC-BCN-021", "ACC-CLN-011", "ACC-BCN-039", "ACC-CLN-027", "ACC-BCN-045", "ACC-CLN-042"]
    ring2_time = base_time + timedelta(hours=11, minutes=15)

    for i, rec in enumerate(fan_out_recipients):
        amt = round(random.uniform(830000.0, 960000.0), 2)
        t_offset = timedelta(minutes=i * 5 + random.randint(1, 2))
        txn_time = ring2_time + t_offset
        s_bank = get_bank_for_account(distributor_2)
        r_bank = get_bank_for_account(rec)

        transactions.append({
            "id": f"TXN-S2A-{txn_counter:04d}",
            "bank": s_bank,
            "sender": distributor_2,
            "receiver": rec,
            "amount": amt,
            "currency": "INR",
            "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S IST"),
            "iso_timestamp": txn_time.isoformat(),
            "type": "RTGS",
            "is_cross_bank": s_bank != r_bank,
            "is_mule": True,
            "mule_ring_id": "RING-002",
            "pattern_type": "Fan-Out Dispersion",
            "note": "Planted Fan-Out structuring dispersion to multiple mules"
        })
        txn_counter += 1

    # -------------------------------------------------------------
    # 3. Planted Mule Ring 3: Multi-Hop Cross-Bank Layering Chains (RING-003)
    # 8 chains of 5 nodes crossing APX -> BCN -> CLN -> APX -> BCN (32 transactions)
    # -------------------------------------------------------------
    chains = [
        ["ACC-APX-022", "ACC-BCN-033", "ACC-CLN-019", "ACC-APX-041", "ACC-BCN-050"],
        ["ACC-CLN-002", "ACC-APX-018", "ACC-BCN-029", "ACC-CLN-036", "ACC-APX-047"],
        ["ACC-BCN-007", "ACC-CLN-016", "ACC-APX-035", "ACC-BCN-041", "ACC-CLN-048"],
        ["ACC-APX-011", "ACC-BCN-024", "ACC-CLN-033", "ACC-APX-038", "ACC-CLN-012"],
        ["ACC-CLN-020", "ACC-APX-007", "ACC-BCN-017", "ACC-CLN-044", "ACC-APX-049"],
        ["ACC-BCN-014", "ACC-CLN-028", "ACC-APX-016", "ACC-BCN-037", "ACC-CLN-007"],
        ["ACC-APX-005", "ACC-CLN-023", "ACC-BCN-047", "ACC-APX-034", "ACC-CLN-017"],
        ["ACC-CLN-038", "ACC-BCN-010", "ACC-APX-026", "ACC-CLN-043", "ACC-BCN-049"],
    ]

    for c_idx, chain in enumerate(chains):
        ring3_time = base_time + timedelta(hours=15 + c_idx * 2, minutes=10)
        current_amt = 960000.0 - (c_idx % 4) * 15000.0

        for hop in range(len(chain) - 1):
            sender = chain[hop]
            receiver = chain[hop + 1]
            t_offset = timedelta(minutes=hop * 16 + random.randint(2, 4))
            txn_time = ring3_time + t_offset
            s_bank = get_bank_for_account(sender)
            r_bank = get_bank_for_account(receiver)

            hop_amt = round(current_amt * random.uniform(0.92, 0.96), 2)
            current_amt = hop_amt

            transactions.append({
                "id": f"TXN-S2A-{txn_counter:04d}",
                "bank": s_bank,
                "sender": sender,
                "receiver": receiver,
                "amount": hop_amt,
                "currency": "INR",
                "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S IST"),
                "iso_timestamp": txn_time.isoformat(),
                "type": "IMPS",
                "is_cross_bank": s_bank != r_bank,
                "is_mule": True,
                "mule_ring_id": "RING-003",
                "pattern_type": "Multi-Hop Layering Chain",
                "note": f"Planted Layering Chain hop {hop+1} crossing {s_bank} -> {r_bank}"
            })
            txn_counter += 1

    # -------------------------------------------------------------
    # 4. Planted Mule Ring 4: Circular Round-Trip / Smurfing Loop (RING-004)
    # 11 cycles * 3 hops across all 3 banks (33 transactions)
    # -------------------------------------------------------------
    cycles = [
        ("ACC-CLN-005", "ACC-APX-030", "ACC-BCN-012"),
        ("ACC-BCN-015", "ACC-CLN-022", "ACC-APX-009"),
        ("ACC-APX-028", "ACC-BCN-004", "ACC-CLN-037"),
        ("ACC-CLN-010", "ACC-APX-032", "ACC-BCN-046"),
        ("ACC-BCN-031", "ACC-CLN-040", "ACC-APX-006"),
        ("ACC-APX-040", "ACC-BCN-023", "ACC-CLN-015"),
        ("ACC-CLN-025", "ACC-APX-019", "ACC-BCN-038"),
        ("ACC-BCN-048", "ACC-CLN-009", "ACC-APX-036"),
        ("ACC-APX-014", "ACC-CLN-030", "ACC-BCN-002"),
        ("ACC-CLN-004", "ACC-BCN-035", "ACC-APX-027"),
        ("ACC-BCN-026", "ACC-APX-043", "ACC-CLN-046"),
    ]

    for cyc_idx, (node_a, node_b, node_c) in enumerate(cycles):
        ring4_time = base_time + timedelta(hours=18 + cyc_idx * 2, minutes=5)
        c_amt = 885000.0 + (cyc_idx % 4) * 25000.0

        trip_hops = [(node_a, node_b), (node_b, node_c), (node_c, node_a)]
        for h_idx, (s, r) in enumerate(trip_hops):
            t_offset = timedelta(minutes=h_idx * 14 + random.randint(1, 3))
            txn_time = ring4_time + t_offset
            s_bank = get_bank_for_account(s)
            r_bank = get_bank_for_account(r)
            h_amt = round(c_amt * (0.94 ** h_idx), 2)

            transactions.append({
                "id": f"TXN-S2A-{txn_counter:04d}",
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
                "mule_ring_id": "RING-004",
                "pattern_type": "Circular Round-Trip Loop",
                "note": f"Planted Circular Loop hop {h_idx+1} ({s} -> {r})"
            })
            txn_counter += 1

    planted_count = len(transactions)

    # -------------------------------------------------------------
    # 5. Normal Legitimate Transactions (up to exactly 500 total)
    # -------------------------------------------------------------
    target_total = 500
    normal_needed = target_total - planted_count

    normal_archetypes = [
        {"type": "NACH", "min": 15000.0, "max": 120000.0, "desc": "Monthly Payroll and Salary Batch"},
        {"type": "UPI", "min": 250.0, "max": 25000.0, "desc": "Peer-to-Peer and Retail UPI Payment"},
        {"type": "NEFT", "min": 10000.0, "max": 350000.0, "desc": "Vendor Invoice Settlement"},
        {"type": "RTGS", "min": 200000.0, "max": 650000.0, "desc": "Commercial Operating Expenditure"},
        {"type": "IMPS", "min": 2000.0, "max": 180000.0, "desc": "Routine Interbank Transfer"},
    ]

    for i in range(normal_needed):
        arch = random.choice(normal_archetypes)
        amt = round(random.uniform(arch["min"], arch["max"]), 2)

        sender = random.choice(all_accounts)
        receiver = random.choice(all_accounts)
        while receiver == sender:
            receiver = random.choice(all_accounts)

        random_minute = random.randint(0, 48 * 60)
        txn_time = base_time + timedelta(minutes=random_minute, seconds=random.randint(0, 59))
        s_bank = get_bank_for_account(sender)
        r_bank = get_bank_for_account(receiver)

        transactions.append({
            "id": f"TXN-S2A-{txn_counter:04d}",
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
            "note": f"{arch['desc']} (Verified Standard Profile)"
        })
        txn_counter += 1

    transactions.sort(key=lambda x: x["iso_timestamp"])

    for idx, t in enumerate(transactions, start=1):
        t["id"] = f"TXN-S2A-{idx:04d}"

    return transactions
