"""
G.O.A.T. Shield Stage 2B - GraphSAGE Graph Neural Network Engine
Implements transductive node classification on the multi-bank transaction graph
for privacy-preserving cross-bank mule account detection.
"""

import math
import random
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np


# =========================================================
# 1. Feature Extraction Engine (Zero Ground-Truth Leakage)
# =========================================================

def extract_node_features(
    transactions: List[Dict[str, Any]],
    accounts_dict: Dict[str, List[str]]
) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, Any]]]:
    """
    Extracts 10 observable behavioral features for all 150 accounts.
    Features are derived STRICTLY from transaction timestamps, amounts, senders, receivers, and banks.
    Zero ground-truth labels are accessed.
    """
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
        t_time = datetime.fromisoformat(t["iso_timestamp"])

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

        # 1. In-Degree
        features[i, 0] = tot_in_cnt
        # 2. Out-Degree
        features[i, 1] = tot_out_cnt
        # 3. Log Total Inflow
        features[i, 2] = math.log10(1.0 + tot_in_vol)
        # 4. Log Total Outflow
        features[i, 3] = math.log10(1.0 + tot_out_vol)
        # 5. Average Inflow Amount
        features[i, 4] = (tot_in_vol / max(1.0, tot_in_cnt)) / 100000.0
        # 6. Average Outflow Amount
        features[i, 5] = (tot_out_vol / max(1.0, tot_out_cnt)) / 100000.0
        # 7. Net Flow Balance Ratio
        denom = tot_in_vol + tot_out_vol + 1e-6
        features[i, 6] = (tot_in_vol - tot_out_vol) / denom
        # 8. Cross-Bank Ratio
        features[i, 7] = cross_bank_counts[i] / max(1.0, tot_tx_cnt)
        # 9. Unique Counterparties
        features[i, 8] = float(len(counterparties[i]))
        # 10. Maximum 60-Minute Burst Velocity
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


# =========================================================
# 2. Graph Topology & Normalized Adjacency Construction
# =========================================================

def build_adjacency_matrix(
    transactions: List[Dict[str, Any]],
    acc_idx_map: Dict[str, int]
) -> np.ndarray:
    """
    Constructs a row-normalized bidirectional adjacency matrix with self-loops.
    Implements the GraphSAGE mean neighborhood aggregator:
    A_norm = D^-1 * (A + I)
    """
    num_nodes = len(acc_idx_map)
    adj = np.eye(num_nodes, dtype=np.float32) # self-loops

    for t in transactions:
        u = t["sender"]
        v = t["receiver"]
        if u in acc_idx_map and v in acc_idx_map:
            u_idx = acc_idx_map[u]
            v_idx = acc_idx_map[v]
            adj[u_idx, v_idx] = 1.0
            adj[v_idx, u_idx] = 1.0

    # Row normalize for mean neighborhood aggregation
    row_sum = np.sum(adj, axis=1, keepdims=True)
    adj_norm = adj / np.maximum(1.0, row_sum)
    return adj_norm


# =========================================================
# 3. Ground-Truth Target & Transductive Mask Split
# =========================================================

def get_ground_truth_labels(
    transactions: List[Dict[str, Any]],
    all_accounts: List[str]
) -> np.ndarray:
    """
    Extracts binary node labels y in {0, 1} strictly for supervised training and evaluation.
    """
    mule_accounts = set()
    for t in transactions:
        if t.get("is_mule", False):
            mule_accounts.add(t["sender"])
            mule_accounts.add(t["receiver"])

    y = np.zeros(len(all_accounts), dtype=np.int64)
    for i, acc in enumerate(all_accounts):
        if acc in mule_accounts:
            y[i] = 1

    return y


def create_transductive_splits(
    all_accounts: List[str],
    y: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates stratified 60% Train (90), 20% Val (30), 20% Test (30) boolean node masks.
    """
    rng = random.Random(seed)
    num_nodes = len(all_accounts)

    strata: Dict[Tuple[str, int], List[int]] = {}
    for i, acc in enumerate(all_accounts):
        if "-APX-" in acc:
            b = "APX"
        elif "-BCN-" in acc:
            b = "BCN"
        else:
            b = "CLN"
        key = (b, int(y[i]))
        if key not in strata:
            strata[key] = []
        strata[key].append(i)

    train_indices = []
    val_indices = []
    test_indices = []

    for key, indices in sorted(strata.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(0.60 * n))
        n_val = int(round(0.20 * n))
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))

        train_indices.extend(shuffled[:n_train])
        val_indices.extend(shuffled[n_train: n_train + n_val])
        test_indices.extend(shuffled[n_train + n_val:])

    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    while len(train_set) > 90:
        elem = train_set.pop()
        if len(val_set) < 30:
            val_set.add(elem)
        else:
            test_set.add(elem)
    while len(train_set) < 90 and (val_set or test_set):
        if len(val_set) > 30:
            train_set.add(val_set.pop())
        elif len(test_set) > 30:
            train_set.add(test_set.pop())
        else:
            break

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[list(train_set)] = True
    val_mask[list(val_set)] = True
    test_mask[list(test_set)] = True

    return train_mask, val_mask, test_mask


# =========================================================
# 4. GraphSAGE Mathematical Neural Network Engine (NumPy)
# =========================================================

class GraphSAGEEngine:
    """
    Exact mathematical GraphSAGE 2-Hop architecture with Mean Neighborhood Aggregation.
    h_v^(1) = ReLU( W_self^(1)*x_v + W_neigh^(1)*mean_{u in N(v)}(x_u) + b^(1) )
    h_v^(2) = ReLU( W_self^(2)*h_v^(1) + W_neigh^(2)*mean_{u in N(v)}(h_u^(1)) + b^(2) )
    logits_v = W_cls * h_v^(2) + b_cls
    """
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
        H1 = np.maximum(0.0, Z1) # ReLU activation

        # 2-Hop Message Passing & Aggregation
        H1_neigh = np.dot(A_norm, H1)
        Z2 = np.dot(H1, self.W_self2) + np.dot(H1_neigh, self.W_neigh2) + self.b2
        H2 = np.maximum(0.0, Z2) # ReLU activation

        # Output Logits & Softmax Probabilities
        logits = np.dot(H2, self.W_cls) + self.b_cls
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs, H2, H1, Z1

    def train_step(
        self,
        X: np.ndarray,
        A_norm: np.ndarray,
        y: np.ndarray,
        train_mask: np.ndarray,
        class_weights: np.ndarray,
        lr: float = 0.02
    ) -> float:
        probs, H2, H1, Z1 = self.forward(X, A_norm)
        
        # Weighted Cross-Entropy Loss on train nodes
        train_indices = np.where(train_mask)[0]
        N_train = len(train_indices)
        
        loss = 0.0
        dlogits = np.zeros_like(probs)

        for idx in train_indices:
            label = y[idx]
            p = max(1e-7, probs[idx, label])
            w = class_weights[label]
            loss += -w * math.log(p)
            
            # Gradient of weighted cross entropy w.r.t logits
            for c in range(2):
                if c == label:
                    dlogits[idx, c] = w * (probs[idx, c] - 1.0) / N_train
                else:
                    dlogits[idx, c] = w * probs[idx, c] / N_train

        loss = loss / N_train

        # Backpropagation: Classifier Head
        dW_cls = np.dot(H2.T, dlogits)
        db_cls = np.sum(dlogits[train_mask], axis=0, keepdims=True)
        dH2 = np.dot(dlogits, self.W_cls.T)

        # Backpropagation: Layer 2
        dZ2 = dH2 * (H2 > 0.0).astype(np.float32)
        dW_self2 = np.dot(H1.T, dZ2)
        H1_neigh = np.dot(A_norm, H1)
        dW_neigh2 = np.dot(H1_neigh.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dH1 = np.dot(dZ2, self.W_self2.T) + np.dot(A_norm.T, np.dot(dZ2, self.W_neigh2.T))

        # Backpropagation: Layer 1
        dZ1 = dH1 * (H1 > 0.0).astype(np.float32)
        dW_self1 = np.dot(X.T, dZ1)
        X_neigh = np.dot(A_norm, X)
        dW_neigh1 = np.dot(X_neigh.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Gradient Descent Updates with L2 regularization
        reg = 1e-4
        self.W_cls -= lr * (dW_cls + reg * self.W_cls)
        self.b_cls -= lr * db_cls
        self.W_self2 -= lr * (dW_self2 + reg * self.W_self2)
        self.W_neigh2 -= lr * (dW_neigh2 + reg * self.W_neigh2)
        self.b2 -= lr * db2
        self.W_self1 -= lr * (dW_self1 + reg * self.W_self1)
        self.W_neigh1 -= lr * (dW_neigh1 + reg * self.W_neigh1)
        self.b1 -= lr * db1

        return float(loss)


# =========================================================
# 5. Pipeline Management & Fair Baseline Benchmark
# =========================================================

class GNNPipeline:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.is_trained = False
        self.all_accounts: List[str] = []
        self.observable_stats: Dict[str, Dict[str, Any]] = {}
        self.predictions: List[Dict[str, Any]] = []
        self.test_metrics: Dict[str, Any] = {}
        self.comparison_data: Dict[str, Any] = {}
        self.training_history: List[Dict[str, float]] = []

    def train_and_evaluate(self, transactions: List[Dict[str, Any]], accounts_dict: Dict[str, List[str]]):
        # 1. Extract Features & Adjacency
        all_accounts, X_raw, obs_stats = extract_node_features(transactions, accounts_dict)
        self.all_accounts = all_accounts
        self.observable_stats = obs_stats
        acc_idx_map = {acc: i for i, acc in enumerate(all_accounts)}

        A_norm = build_adjacency_matrix(transactions, acc_idx_map)
        y_np = get_ground_truth_labels(transactions, all_accounts)
        train_mask, val_mask, test_mask = create_transductive_splits(all_accounts, y_np, seed=self.seed)

        # 2. Z-Score Normalization on train_mask features only
        train_mean = np.mean(X_raw[train_mask], axis=0, keepdims=True)
        train_std = np.std(X_raw[train_mask], axis=0, keepdims=True)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        X_norm = (X_raw - train_mean) / train_std

        # 3. Class Weights for Loss Balancing
        train_y = y_np[train_mask]
        num_pos = max(1, int(np.sum(train_y == 1)))
        num_neg = max(1, int(np.sum(train_y == 0)))
        total_train = len(train_y)
        class_weights = np.array([total_train / (2.0 * num_neg), total_train / (2.0 * num_pos)], dtype=np.float32)

        # 4. Initialize GraphSAGE Engine
        model = GraphSAGEEngine(in_features=10, hidden_dim=32, num_classes=2, seed=self.seed)
        
        self.training_history = []
        best_val_loss = float("inf")
        best_weights = None

        # 5. Training Loop (80 Epochs)
        for epoch in range(1, 81):
            lr = 0.03 * (0.98 ** (epoch // 5))
            train_loss = model.train_step(X_norm, A_norm, y_np, train_mask, class_weights, lr=lr)

            # Validation evaluation
            val_probs, _, _, _ = model.forward(X_norm, A_norm)
            val_indices = np.where(val_mask)[0]
            val_loss = 0.0
            for v_idx in val_indices:
                lbl = y_np[v_idx]
                val_loss += -class_weights[lbl] * math.log(max(1e-7, val_probs[v_idx, lbl]))
            val_loss = val_loss / len(val_indices)

            val_preds = (val_probs[val_mask, 1] >= 0.50).astype(int)
            val_acc = float(np.mean(val_preds == y_np[val_mask]))

            if epoch % 5 == 0 or epoch == 80:
                self.training_history.append({
                    "epoch": epoch,
                    "train_loss": round(float(train_loss), 4),
                    "val_loss": round(float(val_loss), 4),
                    "val_accuracy": round(float(val_acc) * 100, 2)
                })

        # 6. Full Graph Inference & Blind Predictions (NO ground truth exposed)
        self.model = model
        self.X_norm = X_norm
        self.A_norm = A_norm
        all_probs, _, _, _ = model.forward(X_norm, A_norm)
        mule_probs = all_probs[:, 1]

        self.predictions = []
        for i, acc in enumerate(all_accounts):
            if "-APX-" in acc:
                bank_name = "Apex Bank"
            elif "-BCN-" in acc:
                bank_name = "Beacon Bank"
            else:
                bank_name = "Crestline Bank"

            prob = float(mule_probs[i])
            score = int(round(prob * 100.0))
            if score >= 65:
                status = "High Risk"
            elif score >= 40:
                status = "Review"
            else:
                status = "Normal"

            self.predictions.append({
                "account_id": acc,
                "bank": bank_name,
                "gnn_mule_probability": round(prob, 4),
                "gnn_risk_score": score,
                "gnn_status": status,
                "observable_stats": obs_stats.get(acc, {})
            })

        # 7. Held-Out Test Set Evaluation strictly on test_mask (30 nodes)
        test_gt = y_np[test_mask]
        test_probs = mule_probs[test_mask]
        test_preds = (test_probs >= 0.50).astype(int)

        tp = int(np.sum((test_preds == 1) & (test_gt == 1)))
        fp = int(np.sum((test_preds == 1) & (test_gt == 0)))
        tn = int(np.sum((test_preds == 0) & (test_gt == 0)))
        fn = int(np.sum((test_preds == 0) & (test_gt == 1)))

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2.0 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        accuracy = round((tp + tn) / len(test_gt), 4)

        self.test_metrics = {
            "evaluation_scope": "Held-out test set strictly masked during training (seed=42)",
            "test_node_count": int(np.sum(test_mask)),
            "test_ground_truth": {
                "mule_nodes": int(np.sum(test_gt == 1)),
                "normal_nodes": int(np.sum(test_gt == 0))
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
            "training_history": self.training_history
        }

        # 8. Fair Apples-to-Apples Comparison on Identical test_mask
        rule_node_scores = np.zeros(len(all_accounts), dtype=np.int64)
        for t in transactions:
            s_acc = t["sender"]
            r_acc = t["receiver"]
            r_score = t.get("risk_score", 0)
            if s_acc in acc_idx_map:
                s_idx = acc_idx_map[s_acc]
                rule_node_scores[s_idx] = max(rule_node_scores[s_idx], r_score)
            if r_acc in acc_idx_map:
                r_idx = acc_idx_map[r_acc]
                rule_node_scores[r_idx] = max(rule_node_scores[r_idx], r_score)

        rule_test_preds = (rule_node_scores[test_mask] >= 65).astype(int)
        r_tp = int(np.sum((rule_test_preds == 1) & (test_gt == 1)))
        r_fp = int(np.sum((rule_test_preds == 1) & (test_gt == 0)))
        r_tn = int(np.sum((rule_test_preds == 0) & (test_gt == 0)))
        r_fn = int(np.sum((rule_test_preds == 0) & (test_gt == 1)))

        r_prec = round(r_tp / (r_tp + r_fp), 4) if (r_tp + r_fp) > 0 else 0.0
        r_rec = round(r_tp / (r_tp + r_fn), 4) if (r_tp + r_fn) > 0 else 0.0
        r_f1 = round(2.0 * (r_prec * r_rec) / (r_prec + r_rec), 4) if (r_prec + r_rec) > 0 else 0.0
        r_acc = round((r_tp + r_tn) / len(test_gt), 4)

        test_accounts = [all_accounts[i] for i in range(len(all_accounts)) if test_mask[i]]
        consensus_count = int(np.sum(test_preds == rule_test_preds))

        divergence_accounts = []
        for idx_in_test, i in enumerate(np.where(test_mask)[0]):
            acc = all_accounts[i]
            r_pred = int(rule_test_preds[idx_in_test])
            g_pred = int(test_preds[idx_in_test])
            gt = int(test_gt[idx_in_test])

            if r_pred != g_pred:
                divergence_accounts.append({
                    "account_id": acc,
                    "rule_prediction": "Mule" if r_pred == 1 else "Normal",
                    "gnn_prediction": "Mule" if g_pred == 1 else "Normal",
                    "ground_truth": "Mule" if gt == 1 else "Normal",
                    "rule_score": int(rule_node_scores[i]),
                    "gnn_score": int(round(mule_probs[i] * 100))
                })

        self.comparison_data = {
            "evaluation_scope": "Fair apples-to-apples evaluation on identical 30 held-out test accounts",
            "test_node_count": len(test_accounts),
            "rule_baseline": {
                "model_name": "Stage 2A Rule-Based Baseline",
                "methodology": "Maximum transaction risk score per account (threshold >= 65)",
                "precision": r_prec,
                "recall": r_rec,
                "f1_score": r_f1,
                "accuracy": r_acc,
                "precision_percent": round(r_prec * 100, 2),
                "recall_percent": round(r_rec * 100, 2),
                "f1_percent": round(r_f1 * 100, 2),
                "accuracy_percent": round(r_acc * 100, 2),
                "confusion_matrix": {"tp": r_tp, "fp": r_fp, "tn": r_tn, "fn": r_fn}
            },
            "graphsage_gnn": {
                "model_name": "Stage 2B GraphSAGE GNN",
                "methodology": "2-Hop neighborhood structural topological embedding",
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "precision_percent": round(precision * 100, 2),
                "recall_percent": round(recall * 100, 2),
                "f1_percent": round(f1 * 100, 2),
                "accuracy_percent": round(accuracy * 100, 2),
                "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
            },
            "comparison_summary": {
                "consensus_percentage": round((consensus_count / len(test_accounts)) * 100, 2),
                "consensus_accounts": consensus_count,
                "divergence_count": len(divergence_accounts),
                "precision_delta": round((precision - r_prec) * 100, 2),
                "recall_delta": round((recall - r_rec) * 100, 2),
                "f1_delta": round((f1 - r_f1) * 100, 2),
                "divergence_sample": divergence_accounts[:5]
            }
        }

        self.is_trained = True


# Global GNN Pipeline Singleton
_gnn_pipeline_instance: Optional[GNNPipeline] = None


def get_or_train_gnn_pipeline(
    transactions: List[Dict[str, Any]],
    accounts_dict: Dict[str, List[str]]
) -> GNNPipeline:
    global _gnn_pipeline_instance
    if _gnn_pipeline_instance is None or not _gnn_pipeline_instance.is_trained:
        pipeline = GNNPipeline(seed=42)
        pipeline.train_and_evaluate(transactions, accounts_dict)
        _gnn_pipeline_instance = pipeline
    return _gnn_pipeline_instance

