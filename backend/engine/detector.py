"""
G.O.A.T. Shield Stage 2A - Transparent Rule-Based Fraud Detection Engine
Evaluates transactions against 6 explainable rules and produces transparent risk scores,
status classifications, and audit explanations.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple


RULE_DEFINITIONS = {
    "R1": {
        "name": "FIU-IND Threshold Proximity",
        "weight": 20,
        "description": "Amount structured between ?7,50,000 and ?9,99,999 near regulatory reporting threshold."
    },
    "R2": {
        "name": "Cross-Bank Boundary Crossing",
        "weight": 15,
        "description": "Fund displacement traverses institutional boundaries between distinct banking entities."
    },
    "R3": {
        "name": "Fan-In Multi-Source Concentration",
        "weight": 20,
        "description": "Beneficiary account receives >= 3 inbound transfers within a 60-minute moving window."
    },
    "R4": {
        "name": "Fan-Out High Velocity Dispersion",
        "weight": 20,
        "description": "Originating account executes >= 3 outbound transfers within a 60-minute moving window."
    },
    "R5": {
        "name": "Rapid Pass-Through Velocity",
        "weight": 20,
        "description": "Account receives funds and drains >= 70% outbound within a 60-minute temporal window."
    },
    "R6": {
        "name": "Rapid Structuring & Chain Burst",
        "weight": 15,
        "description": "Account participates in rapid sequential transfers within 30 minutes totaling > ?4,00,000."
    }
}


import os
import bisect
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional


def parse_time(iso_str: str) -> datetime:
    return datetime.fromisoformat(iso_str)


def _score_transaction_chunk(
    chunk: List[Dict[str, Any]],
    chunk_timestamps: List[float],
    in_times: Dict[str, List[float]],
    in_amts: Dict[str, List[float]],
    out_times: Dict[str, List[float]],
    out_amts: Dict[str, List[float]],
    all_times: Dict[str, List[float]],
    all_amts: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    """
    Scores a slice of transactions using indexed account timelines in O(log K) per rule.
    """
    enriched: List[Dict[str, Any]] = []

    for idx, t in enumerate(chunk):
        ts = chunk_timestamps[idx]
        amt = float(t["amount"])
        sender = t["sender"]
        receiver = t["receiver"]
        is_cross = bool(t.get("is_cross_bank", False))

        triggered: List[str] = []
        reasons: List[str] = []

        # R1: FIU-IND Threshold Proximity
        if 750000.0 <= amt <= 999999.0:
            triggered.append("R1")
            reasons.append(f"Amount ?{amt:,.2f} near ?10L threshold (+20)")

        # R2: Cross-Bank Boundary Crossing
        if is_cross:
            triggered.append("R2")
            reasons.append(f"Cross-bank transfer from {t['bank']} (+15)")

        # R3: Fan-In Multi-Source Concentration (Receiver side: 60m window |event - ts| <= 3600)
        rec_in_t = in_times.get(receiver)
        if rec_in_t is not None:
            left_r3 = bisect.bisect_left(rec_in_t, ts - 3600.0)
            right_r3 = bisect.bisect_right(rec_in_t, ts + 3600.0)
            count_r3 = right_r3 - left_r3
            if count_r3 >= 3:
                triggered.append("R3")
                reasons.append(f"Receiver concentration: {count_r3} inbound transfers in 60m (+20)")

        # R4: Fan-Out Dispersion Velocity (Sender side: 60m window |event - ts| <= 3600)
        snd_out_t = out_times.get(sender)
        if snd_out_t is not None:
            left_r4 = bisect.bisect_left(snd_out_t, ts - 3600.0)
            right_r4 = bisect.bisect_right(snd_out_t, ts + 3600.0)
            count_r4 = right_r4 - left_r4
            if count_r4 >= 3:
                triggered.append("R4")
                reasons.append(f"Sender dispersion: {count_r4} outbound transfers in 60m (+20)")

        # R5: Rapid Pass-Through Velocity (Recent inbounds to sender OR subsequent outbounds from receiver)
        pass_through_detected = False
        snd_in_t = in_times.get(sender)
        if snd_in_t is not None:
            # 0 <= (ts - event) <= 3600 => ts - 3600 <= event <= ts
            left_r5 = bisect.bisect_left(snd_in_t, ts - 3600.0)
            right_r5 = bisect.bisect_right(snd_in_t, ts)
            snd_in_a = in_amts[sender]
            for in_amt in snd_in_a[left_r5:right_r5]:
                if amt >= 0.70 * in_amt or in_amt >= 0.70 * amt:
                    pass_through_detected = True
                    break

        if not pass_through_detected:
            rec_out_t = out_times.get(receiver)
            if rec_out_t is not None:
                # 0 <= (event - ts) <= 3600 => ts <= event <= ts + 3600
                left_r5_sub = bisect.bisect_left(rec_out_t, ts)
                right_r5_sub = bisect.bisect_right(rec_out_t, ts + 3600.0)
                rec_out_a = out_amts[receiver]
                for out_amt in rec_out_a[left_r5_sub:right_r5_sub]:
                    if out_amt >= 0.70 * amt or amt >= 0.70 * out_amt:
                        pass_through_detected = True
                        break

        if pass_through_detected:
            triggered.append("R5")
            reasons.append("Rapid pass-through: >=70% funds drained within 60m (+20)")

        # R6: Rapid Structuring & Chain Burst (30m window: |event - ts| <= 1800)
        snd_all_t = all_times.get(sender)
        if snd_all_t is not None:
            left_r6 = bisect.bisect_left(snd_all_t, ts - 1800.0)
            right_r6 = bisect.bisect_right(snd_all_t, ts + 1800.0)
            count_r6 = right_r6 - left_r6
            if count_r6 >= 2:
                snd_all_a = all_amts[sender]
                total_burst_amt = sum(snd_all_a[left_r6:right_r6])
                if total_burst_amt >= 400000.0:
                    triggered.append("R6")
                    reasons.append(f"Structuring/chain burst: {count_r6} events totaling ?{total_burst_amt:,.2f} in 30m (+15)")

        unique_triggered: List[str] = []
        for r in triggered:
            if r not in unique_triggered:
                unique_triggered.append(r)

        score = min(100, sum(RULE_DEFINITIONS[r]["weight"] for r in unique_triggered))

        if score >= 65:
            status = "High Risk"
        elif score >= 40:
            status = "Review"
        else:
            status = "Normal"

        if not unique_triggered:
            rule_details = []
            explanation = "Standard behavioral profile; no anomalous cross-bank or velocity rules triggered."
        else:
            rule_details = [
                {
                    "rule_id": r,
                    "name": RULE_DEFINITIONS[r]["name"],
                    "points": RULE_DEFINITIONS[r]["weight"],
                    "description": RULE_DEFINITIONS[r]["description"]
                }
                for r in unique_triggered
            ]
            explanation = ". ".join(reasons) + f". Composite Score: {score}/100 ({status})."

        enriched_t = dict(t)
        enriched_t["risk_score"] = score
        enriched_t["status"] = status
        enriched_t["triggered_rules"] = unique_triggered
        enriched_t["rule_details"] = rule_details
        enriched_t["explanation"] = explanation

        enriched.append(enriched_t)

    return enriched


def detect_frauds(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optimized Fraud Detection Engine:
    - Pre-indexes account inbound, outbound, and global timelines once in O(N).
    - Employs binary-search sliding windows (bisect_left/bisect_right) in O(log K).
    - Preserves 100% mathematical and logical equivalence with baseline R1–R6.
    """
    if not transactions:
        return []

    # Fast-path timestamp extraction and order validation
    if transactions and "epoch_timestamp" in transactions[0]:
        raw_ts = [t.get("epoch_timestamp", 0.0) for t in transactions]
    else:
        raw_ts = [parse_time(t["iso_timestamp"]).timestamp() for t in transactions]

    # Check if already sorted in linear scan
    is_ordered = all(raw_ts[i] <= raw_ts[i+1] for i in range(len(raw_ts) - 1))
    if is_ordered:
        sorted_txns = transactions
        timestamps = raw_ts
    else:
        sorted_indices = sorted(range(len(transactions)), key=lambda i: transactions[i]["iso_timestamp"])
        sorted_txns = [transactions[i] for i in sorted_indices]
        timestamps = [raw_ts[i] for i in sorted_indices]

    # Pre-index account event streams
    in_times: Dict[str, List[float]] = {}
    in_amts: Dict[str, List[float]] = {}
    out_times: Dict[str, List[float]] = {}
    out_amts: Dict[str, List[float]] = {}
    all_times: Dict[str, List[float]] = {}
    all_amts: Dict[str, List[float]] = {}

    for idx, t in enumerate(sorted_txns):
        ts = timestamps[idx]
        amt = float(t["amount"])
        sender = t["sender"]
        receiver = t["receiver"]

        if sender not in out_times:
            out_times[sender] = []
            out_amts[sender] = []
        out_times[sender].append(ts)
        out_amts[sender].append(amt)

        if receiver not in in_times:
            in_times[receiver] = []
            in_amts[receiver] = []
        in_times[receiver].append(ts)
        in_amts[receiver].append(amt)

        if sender not in all_times:
            all_times[sender] = []
            all_amts[sender] = []
        all_times[sender].append(ts)
        all_amts[sender].append(amt)

        if receiver not in all_times:
            all_times[receiver] = []
            all_amts[receiver] = []
        all_times[receiver].append(ts)
        all_amts[receiver].append(amt)

    return _score_transaction_chunk(
        chunk=sorted_txns,
        chunk_timestamps=timestamps,
        in_times=in_times,
        in_amts=in_amts,
        out_times=out_times,
        out_amts=out_amts,
        all_times=all_times,
        all_amts=all_amts
    )


def detect_frauds_parallel(
    transactions: List[Dict[str, Any]],
    num_workers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Account-Partitioned Multiprocessing Fraud Detection Engine:
    - Safely chunks transaction evaluations across worker processes.
    - Utilizes shared indexed account histories.
    - Deterministically merges outputs in original sorted order.
    """
    if not transactions:
        return []

    if num_workers is None:
        cpu_cnt = os.cpu_count() or 4
        num_workers = min(cpu_cnt, 4)

    # For smaller datasets (<5,000 txns), single-worker avoids IPC process spawn overhead
    if num_workers <= 1 or len(transactions) < 5000:
        return detect_frauds(transactions)

    sorted_txns = sorted(transactions, key=lambda x: x["iso_timestamp"])
    timestamps = [parse_time(t["iso_timestamp"]).timestamp() for t in sorted_txns]

    # Pre-index account event streams
    in_times: Dict[str, List[float]] = {}
    in_amts: Dict[str, List[float]] = {}
    out_times: Dict[str, List[float]] = {}
    out_amts: Dict[str, List[float]] = {}
    all_times: Dict[str, List[float]] = {}
    all_amts: Dict[str, List[float]] = {}

    for idx, t in enumerate(sorted_txns):
        ts = timestamps[idx]
        amt = float(t["amount"])
        sender = t["sender"]
        receiver = t["receiver"]

        if sender not in out_times:
            out_times[sender] = []
            out_amts[sender] = []
        out_times[sender].append(ts)
        out_amts[sender].append(amt)

        if receiver not in in_times:
            in_times[receiver] = []
            in_amts[receiver] = []
        in_times[receiver].append(ts)
        in_amts[receiver].append(amt)

        if sender not in all_times:
            all_times[sender] = []
            all_amts[sender] = []
        all_times[sender].append(ts)
        all_amts[sender].append(amt)

        if receiver not in all_times:
            all_times[receiver] = []
            all_amts[receiver] = []
        all_times[receiver].append(ts)
        all_amts[receiver].append(amt)

    total_len = len(sorted_txns)
    chunk_size = (total_len + num_workers - 1) // num_workers

    chunks: List[List[Dict[str, Any]]] = []
    ts_chunks: List[List[float]] = []

    for i in range(num_workers):
        start_i = i * chunk_size
        end_i = min(total_len, (i + 1) * chunk_size)
        if start_i < total_len:
            chunks.append(sorted_txns[start_i:end_i])
            ts_chunks.append(timestamps[start_i:end_i])

    actual_workers = len(chunks)
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = [
            executor.submit(
                _score_chunk_worker,
                chunks[i],
                ts_chunks[i],
                in_times,
                in_amts,
                out_times,
                out_amts,
                all_times,
                all_amts
            )
            for i in range(actual_workers)
        ]
        results: List[Dict[str, Any]] = []
        for f in futures:
            results.extend(f.result())

    return results


def _score_chunk_worker(chunk, ts_chunk, in_t, in_a, out_t, out_a, all_t, all_a):
    """Top-level picklable entrypoint for multiprocessing workers."""
    return _score_transaction_chunk(chunk, ts_chunk, in_t, in_a, out_t, out_a, all_t, all_a)


def detect_frauds_baseline(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Original Baseline Linear-Scan Fraud Detection Engine (preserved for regression testing).
    """
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
        reasons: List[str] = []

        # R1: FIU-IND Threshold Proximity
        if 750000.0 <= amt <= 999999.0:
            triggered.append("R1")
            reasons.append(f"Amount ?{amt:,.2f} near ?10L threshold (+20)")

        # R2: Cross-Bank Boundary Crossing
        if is_cross:
            triggered.append("R2")
            reasons.append(f"Cross-bank transfer from {t['bank']} (+15)")

        # R3: Fan-In Multi-Source Concentration
        inbounds = account_inbounds.get(receiver, [])
        in_window_inbounds = [
            x for x in inbounds
            if abs((x[0] - t_time).total_seconds()) <= 3600
        ]
        if len(in_window_inbounds) >= 3:
            triggered.append("R3")
            reasons.append(f"Receiver concentration: {len(in_window_inbounds)} inbound transfers in 60m (+20)")

        # R4: Fan-Out Dispersion Velocity
        outbounds = account_outbounds.get(sender, [])
        in_window_outbounds = [
            x for x in outbounds
            if abs((x[0] - t_time).total_seconds()) <= 3600
        ]
        if len(in_window_outbounds) >= 3:
            triggered.append("R4")
            reasons.append(f"Sender dispersion: {len(in_window_outbounds)} outbound transfers in 60m (+20)")

        # R5: Rapid Pass-Through Velocity
        sender_inbounds = account_inbounds.get(sender, [])
        recent_inbounds_to_sender = [
            x for x in sender_inbounds
            if 0 <= (t_time - x[0]).total_seconds() <= 3600
        ]
        pass_through_detected = False
        for in_t, in_amt, _ in recent_inbounds_to_sender:
            if amt >= 0.70 * in_amt or in_amt >= 0.70 * amt:
                pass_through_detected = True
                break

        if not pass_through_detected:
            receiver_outbounds = account_outbounds.get(receiver, [])
            subsequent_outbounds = [
                x for x in receiver_outbounds
                if 0 <= (x[0] - t_time).total_seconds() <= 3600
            ]
            for out_t, out_amt, _ in subsequent_outbounds:
                if out_amt >= 0.70 * amt or amt >= 0.70 * out_amt:
                    pass_through_detected = True
                    break

        if pass_through_detected:
            triggered.append("R5")
            reasons.append("Rapid pass-through: >=70% funds drained within 60m (+20)")

        # R6: Rapid Structuring & Chain Burst
        sender_events = account_all.get(sender, [])
        burst_events = [
            x for x in sender_events
            if abs((x[0] - t_time).total_seconds()) <= 1800
        ]
        total_burst_amt = sum(x[1] for x in burst_events)
        if len(burst_events) >= 2 and total_burst_amt >= 400000.0:
            triggered.append("R6")
            reasons.append(f"Structuring/chain burst: {len(burst_events)} events totaling ?{total_burst_amt:,.2f} in 30m (+15)")

        unique_triggered = []
        for r in triggered:
            if r not in unique_triggered:
                unique_triggered.append(r)

        score = sum(RULE_DEFINITIONS[r]["weight"] for r in unique_triggered)
        score = min(100, score)

        if score >= 65:
            status = "High Risk"
        elif score >= 40:
            status = "Review"
        else:
            status = "Normal"

        rule_details = [
            {
                "rule_id": r,
                "name": RULE_DEFINITIONS[r]["name"],
                "points": RULE_DEFINITIONS[r]["weight"],
                "description": RULE_DEFINITIONS[r]["description"]
            }
            for r in unique_triggered
        ]

        if not reasons:
            explanation = "Standard behavioral profile; no anomalous cross-bank or velocity rules triggered."
        else:
            explanation = ". ".join(reasons) + f". Composite Score: {score}/100 ({status})."

        enriched = dict(t)
        enriched["risk_score"] = score
        enriched["status"] = status
        enriched["triggered_rules"] = unique_triggered
        enriched["rule_details"] = rule_details
        enriched["explanation"] = explanation

        enriched_transactions.append(enriched)

    return enriched_transactions

