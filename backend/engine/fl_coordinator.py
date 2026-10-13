"""
G.O.A.T. Shield Stage 2C-2 - Federated Learning Coordinator
Coordinates privacy-preserving Federated Learning across Apex Bank, Beacon Bank,
and Crestline Bank using Paillier Additive Homomorphic Encryption.

Key-Management & Threat Model:
- The HE Aggregation Server does not possess the Paillier private key and therefore
  cannot decrypt individual ciphertexts under the implemented key-management model.
- Semi-Honest (Honest-but-Curious) threat model.
- Equal-weight FedAvg is used because each participating synthetic bank currently
  contributes an equal number of training accounts (50 accounts each).
- 1024-bit Paillier modulus — prototype/demo configuration.
"""

import math
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from engine.paillier import (
    PaillierPublicKey,
    PaillierPrivateKey,
    generate_paillier_keypair,
    float_to_paillier_integer,
    paillier_integer_to_float,
    encrypt_vector,
    homomorphic_aggregate_vectors,
    decrypt_and_decode_vector,
    get_ciphertext_fingerprint,
    DEFAULT_SCALE_FACTOR
)
from engine.gnn_detector import (
    extract_node_features,
    build_adjacency_matrix,
    get_ground_truth_labels,
    create_transductive_splits
)


# =========================================================
# 1. Compact GraphSAGE for Homomorphic Federated Learning
# =========================================================

class CompactFLGraphSAGE:
    """
    GraphSAGE 1-hop Message Passing Classifier with exact analytical gradients.
    Input Features: 10 observable behavioral features
    Hidden Dimension: 8
    Classes: 2 (Normal=0, Mule=1)
    
    Trainable Parameters (186 Total):
    - W_self:  (10, 8) = 80
    - W_neigh: (10, 8) = 80
    - b1:      (1, 8)  = 8
    - W_cls:   (8, 2)  = 16
    - b_cls:   (1, 2)  = 2
    """
    def __init__(self, in_features: int = 10, hidden_dim: int = 8, num_classes: int = 2, seed: int = 42):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        rng = np.random.RandomState(seed)

        scale1 = math.sqrt(2.0 / (in_features + hidden_dim))
        self.W_self = (rng.randn(in_features, hidden_dim) * scale1).astype(np.float32)
        self.W_neigh = (rng.randn(in_features, hidden_dim) * scale1).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)

        scale_cls = math.sqrt(2.0 / (hidden_dim + num_classes))
        self.W_cls = (rng.randn(hidden_dim, num_classes) * scale_cls).astype(np.float32)
        self.b_cls = np.zeros((1, num_classes), dtype=np.float32)

    def get_flat_weights(self) -> List[float]:
        """Flattens all trainable parameters into a 1D Python float list."""
        vec = np.concatenate([
            self.W_self.ravel(),
            self.W_neigh.ravel(),
            self.b1.ravel(),
            self.W_cls.ravel(),
            self.b_cls.ravel()
        ])
        return [float(x) for x in vec]

    def set_flat_weights(self, flat_weights: List[float]):
        """Restores model weights from a 1D float list."""
        arr = np.array(flat_weights, dtype=np.float32)
        idx = 0

        w_self_len = self.in_features * self.hidden_dim
        self.W_self = arr[idx: idx + w_self_len].reshape(self.in_features, self.hidden_dim)
        idx += w_self_len

        w_neigh_len = self.in_features * self.hidden_dim
        self.W_neigh = arr[idx: idx + w_neigh_len].reshape(self.in_features, self.hidden_dim)
        idx += w_neigh_len

        b1_len = self.hidden_dim
        self.b1 = arr[idx: idx + b1_len].reshape(1, self.hidden_dim)
        idx += b1_len

        w_cls_len = self.hidden_dim * self.num_classes
        self.W_cls = arr[idx: idx + w_cls_len].reshape(self.hidden_dim, self.num_classes)
        idx += w_cls_len

        b_cls_len = self.num_classes
        self.b_cls = arr[idx: idx + b_cls_len].reshape(1, self.num_classes)

    def forward(self, X: np.ndarray, A_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes softmax probabilities and hidden representations."""
        X_neigh = np.dot(A_norm, X)
        Z1 = np.dot(X, self.W_self) + np.dot(X_neigh, self.W_neigh) + self.b1
        H1 = np.maximum(0.0, Z1) # ReLU

        logits = np.dot(H1, self.W_cls) + self.b_cls
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs, H1

    def train_step(
        self,
        X: np.ndarray,
        A_norm: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        class_weights: np.ndarray,
        lr: float = 0.02
    ) -> float:
        """Executes one local SGD step on masked nodes."""
        probs, H1 = self.forward(X, A_norm)
        indices = np.where(mask)[0]
        n_samples = len(indices)
        if n_samples == 0:
            return 0.0

        loss = 0.0
        dlogits = np.zeros_like(probs)

        for idx in indices:
            label = y[idx]
            p = max(1e-7, probs[idx, label])
            w = class_weights[label]
            loss += -w * math.log(p)

            for c in range(self.num_classes):
                if c == label:
                    dlogits[idx, c] = w * (probs[idx, c] - 1.0) / n_samples
                else:
                    dlogits[idx, c] = w * probs[idx, c] / n_samples

        loss = loss / n_samples

        # Backpropagation
        dW_cls = np.dot(H1.T, dlogits)
        db_cls = np.sum(dlogits[mask], axis=0, keepdims=True)

        dH1 = np.dot(dlogits, self.W_cls.T)
        dZ1 = dH1 * (H1 > 0.0).astype(np.float32)

        X_neigh = np.dot(A_norm, X)
        dW_self = np.dot(X.T, dZ1)
        dW_neigh = np.dot(X_neigh.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        reg = 1e-4
        self.W_cls -= lr * (dW_cls + reg * self.W_cls)
        self.b_cls -= lr * db_cls
        self.W_self -= lr * (dW_self + reg * self.W_self)
        self.W_neigh -= lr * (dW_neigh + reg * self.W_neigh)
        self.b1 -= lr * db1

        return float(loss)


# =========================================================
# 2. Local Bank Silo Partition
# =========================================================

class BankSilo:
    """
    Represents an isolated financial institution (Apex, Beacon, or Crestline).
    Performs local training strictly on its internal account partition and transaction subgraph.
    """
    def __init__(
        self,
        bank_id: str,
        bank_name: str,
        account_ids: List[str],
        X_all: np.ndarray,
        A_norm_all: np.ndarray,
        y_all: np.ndarray,
        train_mask_all: np.ndarray,
        acc_idx_map: Dict[str, int],
        mule_weight_multiplier: float = 2.0
    ):
        self.bank_id = bank_id
        self.bank_name = bank_name
        self.account_ids = account_ids

        # Local node indices
        self.local_indices = np.array([acc_idx_map[acc] for acc in account_ids], dtype=np.int64)
        
        # Local training mask (only nodes belonging to this bank and marked in train_mask)
        self.local_train_mask = np.zeros(len(acc_idx_map), dtype=bool)
        for idx in self.local_indices:
            if train_mask_all[idx]:
                self.local_train_mask[idx] = True

        self.X = X_all
        self.A_norm = A_norm_all
        self.y = y_all

        # Compute balanced class weights with validated mule weight multiplier
        train_y = self.y[self.local_train_mask]
        n_pos = max(1, int(np.sum(train_y == 1)))
        n_neg = max(1, int(np.sum(train_y == 0)))
        tot = len(train_y)
        self.class_weights = np.array([tot / (2.0 * n_neg), mule_weight_multiplier * tot / (2.0 * n_pos)], dtype=np.float32)

        # Local model instance
        self.local_model = CompactFLGraphSAGE(seed=42)

    def train_local(
        self,
        global_weights: List[float],
        epochs: int = 3,
        lr: float = 0.02
    ) -> Tuple[List[float], float]:
        """
        Loads global model weights, trains for E local epochs on bank data,
        and returns local weight delta: delta_W = W_local - W_global.
        """
        self.local_model.set_flat_weights(global_weights)
        final_loss = 0.0
        for _ in range(epochs):
            final_loss = self.local_model.train_step(
                self.X,
                self.A_norm,
                self.y,
                self.local_train_mask,
                self.class_weights,
                lr=lr
            )

        updated_weights = self.local_model.get_flat_weights()
        deltas = [u - g for u, g in zip(updated_weights, global_weights)]
        return deltas, final_loss

    def train_isolated_silo(self, epochs: int = 30, lr: float = 0.02) -> CompactFLGraphSAGE:
        """Trains an isolated local model without any federated collaboration (for fair benchmark)."""
        isolated_model = CompactFLGraphSAGE(seed=42)
        for _ in range(epochs):
            isolated_model.train_step(
                self.X,
                self.A_norm,
                self.y,
                self.local_train_mask,
                self.class_weights,
                lr=lr
            )
        return isolated_model


# =========================================================
# 3. Federated Learning Coordinator & HE Aggregator
# =========================================================

class FLCoordinator:
    """
    Coordinates Paillier-encrypted Federated Learning across Apex, Beacon, and Crestline banks.
    """
    def __init__(self, seed: int = 42, eval_threshold: float = 0.55, mule_weight_multiplier: float = 2.0):
        self.seed = seed
        self.eval_threshold = eval_threshold
        self.mule_weight_multiplier = mule_weight_multiplier
        self.current_round = 0
        self.is_initialized = False

        # Cryptographic Keys (1024-bit prototype modulus)
        self.pubkey: Optional[PaillierPublicKey] = None
        self.privkey: Optional[PaillierPrivateKey] = None # Held by Decryption Authority

        # Global Model
        self.global_model: Optional[CompactFLGraphSAGE] = None

        # Bank Silos
        self.silos: Dict[str, BankSilo] = {}

        # Shared Evaluation Dataset (Exact Stage 2B Held-Out Test Set)
        self.all_accounts: List[str] = []
        self.X_all: Optional[np.ndarray] = None
        self.A_norm_all: Optional[np.ndarray] = None
        self.y_all: Optional[np.ndarray] = None
        self.test_mask: Optional[np.ndarray] = None

        # Telemetry & History
        self.history: List[Dict[str, Any]] = []
        self.latest_round_telemetry: Dict[str, Any] = {}
        self.benchmark_comparison: Dict[str, Any] = {}

    def initialize(
        self,
        transactions: List[Dict[str, Any]],
        accounts_dict: Dict[str, List[str]]
    ):
        """Prepares datasets, transductive masks, Paillier keypair, and initializes global model."""
        # 1. Generate 1024-bit Paillier Keypair
        self.pubkey, self.privkey = generate_paillier_keypair(prime_bits=512)

        # 2. Extract Features and Normalized Adjacency
        self.all_accounts, X_raw, obs_stats = extract_node_features(transactions, accounts_dict)
        acc_idx_map = {acc: i for i, acc in enumerate(self.all_accounts)}

        # Feature normalization
        mean = np.mean(X_raw, axis=0, keepdims=True)
        std = np.std(X_raw, axis=0, keepdims=True) + 1e-6
        self.X_all = (X_raw - mean) / std
        self.A_norm_all = build_adjacency_matrix(transactions, acc_idx_map)
        self.y_all = get_ground_truth_labels(transactions, self.all_accounts)

        # 3. Transductive 60/20/20 Split (90 train, 30 val, 30 test)
        train_mask, val_mask, test_mask = create_transductive_splits(self.all_accounts, self.y_all, seed=self.seed)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        # 4. Instantiate 3 Bank Silos with validated mule class weight multiplier
        self.silos = {
            "APEX": BankSilo("APEX", "Apex Bank", accounts_dict.get("Apex Bank", []), self.X_all, self.A_norm_all, self.y_all, train_mask, acc_idx_map, mule_weight_multiplier=self.mule_weight_multiplier),
            "BEACON": BankSilo("BEACON", "Beacon Bank", accounts_dict.get("Beacon Bank", []), self.X_all, self.A_norm_all, self.y_all, train_mask, acc_idx_map, mule_weight_multiplier=self.mule_weight_multiplier),
            "CRESTLINE": BankSilo("CRESTLINE", "Crestline Bank", accounts_dict.get("Crestline Bank", []), self.X_all, self.A_norm_all, self.y_all, train_mask, acc_idx_map, mule_weight_multiplier=self.mule_weight_multiplier)
        }

        # 5. Initialize Global Model & Round State
        self.global_model = CompactFLGraphSAGE(seed=self.seed)
        self.current_round = 0
        self.history = []

        # Evaluate Round 0 (Baseline before FL)
        init_metrics = self._evaluate_model(self.global_model)
        self.history.append({
            "round": 0,
            "accuracy": init_metrics["accuracy"],
            "precision": init_metrics["precision"],
            "recall": init_metrics["recall"],
            "f1": init_metrics["f1"],
            "loss": init_metrics["loss"],
            "execution_time_ms": 0.0
        })

        self._compute_benchmarks()
        self.is_initialized = True

    def _evaluate_model(
        self,
        model: CompactFLGraphSAGE,
        mask: Optional[np.ndarray] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """Evaluates a model on specified mask (defaults to held-out 30 test accounts) and decision threshold."""
        eval_mask = mask if mask is not None else self.test_mask
        eval_threshold = threshold if threshold is not None else self.eval_threshold
        probs, _ = model.forward(self.X_all, self.A_norm_all)
        eval_gt = self.y_all[eval_mask]
        eval_probs = probs[eval_mask, 1]
        eval_preds = (eval_probs >= eval_threshold).astype(int)

        tp = int(np.sum((eval_preds == 1) & (eval_gt == 1)))
        fp = int(np.sum((eval_preds == 1) & (eval_gt == 0)))
        tn = int(np.sum((eval_preds == 0) & (eval_gt == 0)))
        fn = int(np.sum((eval_preds == 0) & (eval_gt == 1)))

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2.0 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        accuracy = round((tp + tn) / len(eval_gt), 4)

        # Cross entropy loss on evaluation set
        eval_loss = 0.0
        for i, idx in enumerate(np.where(eval_mask)[0]):
            lbl = self.y_all[idx]
            p = max(1e-7, probs[idx, lbl])
            eval_loss += -math.log(p)
        eval_loss = round(eval_loss / len(eval_gt), 4)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": eval_loss,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }

    def _compute_benchmarks(self):
        """Computes true isolated local-silo performance and centralized baseline on held-out test nodes."""
        silo_metrics = {}
        for silo_id, silo in self.silos.items():
            silo_model = silo.train_isolated_silo(epochs=30)
            m = self._evaluate_model(silo_model)
            silo_metrics[silo_id] = {
                "bank_name": silo.bank_name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "loss": m["loss"]
            }

        self.benchmark_comparison = {
            "evaluation_node_count": int(np.sum(self.test_mask)),
            "silo_isolated": silo_metrics,
            "equal_weight_fedavg_rationale": "Equal-weight FedAvg is used because each participating synthetic bank currently contributes an equal number of training accounts (50 accounts each)."
        }

    def execute_round(self, local_epochs: int = 5, lr: float = 0.02) -> Dict[str, Any]:
        """
        Executes one full privacy-preserving federated learning round:
        1. Broadcast current global weights to Apex, Beacon, and Crestline.
        2. Local SGD training at each bank -> local weight deltas delta_W_i.
        3. Signed fixed-point quantization & Paillier encryption -> C_i.
        4. HE Aggregation Server computes C_agg = prod_i(C_i) mod n^2 (Ciphertexts only).
        5. Authorized Decryption Authority decrypts C_agg -> sum(delta_W_i).
        6. Decode fixed-point integer & divide by 3 -> delta_W_global.
        7. Global model updated: W^(t+1) = W^(t) + delta_W_global.
        8. Evaluates updated global model on held-out test set.
        """
        if not self.is_initialized:
            raise RuntimeError("FL Coordinator not initialized. Call initialize() first.")

        t_start = time.perf_counter()
        global_weights = self.global_model.get_flat_weights()
        num_params = len(global_weights)

        # ---------------------------------------------------------
        # Step 1 & 2: Local Silo Training
        # ---------------------------------------------------------
        t_train_start = time.perf_counter()
        bank_deltas = {}
        bank_losses = {}
        for silo_id, silo in self.silos.items():
            deltas, loss = silo.train_local(global_weights, epochs=local_epochs, lr=lr)
            bank_deltas[silo_id] = deltas
            bank_losses[silo_id] = round(loss, 4)
        t_train_ms = round((time.perf_counter() - t_train_start) * 1000, 2)

        # ---------------------------------------------------------
        # Step 3: Fixed-Point Quantization & Paillier Encryption
        # ---------------------------------------------------------
        t_enc_start = time.perf_counter()
        bank_ciphertexts = {}
        ciphertext_fingerprints = {}
        for silo_id, deltas in bank_deltas.items():
            encrypted_vec = encrypt_vector(self.pubkey, deltas, scale=DEFAULT_SCALE_FACTOR)
            bank_ciphertexts[silo_id] = encrypted_vec
            ciphertext_fingerprints[silo_id] = [
                get_ciphertext_fingerprint(c) for c in encrypted_vec[:3]
            ]
        t_enc_ms = round((time.perf_counter() - t_enc_start) * 1000, 2)

        # ---------------------------------------------------------
        # Step 4: HE Aggregation Server (Ciphertexts only)
        # ---------------------------------------------------------
        t_agg_start = time.perf_counter()
        all_ciphertext_vectors = [bank_ciphertexts["APEX"], bank_ciphertexts["BEACON"], bank_ciphertexts["CRESTLINE"]]
        aggregated_ciphertexts = homomorphic_aggregate_vectors(self.pubkey, all_ciphertext_vectors)
        t_agg_ms = round((time.perf_counter() - t_agg_start) * 1000, 2)

        # ---------------------------------------------------------
        # Step 5 & 6: Authorized Decryption Authority & FedAvg Update
        # ---------------------------------------------------------
        t_dec_start = time.perf_counter()
        # Equal-weight FedAvg with divisor = 3.0
        global_delta_avg = decrypt_and_decode_vector(
            self.privkey,
            aggregated_ciphertexts,
            scale=DEFAULT_SCALE_FACTOR,
            divisor=3.0
        )
        t_dec_ms = round((time.perf_counter() - t_dec_start) * 1000, 2)

        # Step 7: Apply FedAvg update to Global Model
        new_global_weights = [g + d for g, d in zip(global_weights, global_delta_avg)]
        self.global_model.set_flat_weights(new_global_weights)
        self.current_round += 1

        # Step 8: Evaluate on Held-Out Test Set
        eval_metrics = self._evaluate_model(self.global_model)
        total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

        round_record = {
            "round": self.current_round,
            "accuracy": eval_metrics["accuracy"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "f1": eval_metrics["f1"],
            "loss": eval_metrics["loss"],
            "execution_time_ms": total_time_ms
        }
        self.history.append(round_record)

        # Safe Non-Confidential Telemetry (Zero Plaintext Gradients Exposed)
        self.latest_round_telemetry = {
            "round": self.current_round,
            "parameter_count": num_params,
            "participating_banks": ["Apex Bank", "Beacon Bank", "Crestline Bank"],
            "local_losses": bank_losses,
            "timing_breakdown_ms": {
                "local_training": t_train_ms,
                "paillier_encryption": t_enc_ms,
                "homomorphic_aggregation": t_agg_ms,
                "decryption_and_fedavg": t_dec_ms,
                "total_round_latency": total_time_ms
            },
            "ciphertext_sample_fingerprints": ciphertext_fingerprints,
            "aggregate_sample_fingerprints": [
                get_ciphertext_fingerprint(c) for c in aggregated_ciphertexts[:3]
            ],
            "security_status": {
                "paillier_modulus": "1024-bit Paillier modulus — prototype/demo configuration",
                "aggregator_role": "HE Aggregation Server (Possesses PK only, cannot decrypt individual ciphertexts)",
                "aggregation_law": "∏ Enc(ΔW_i) mod n² = Enc(∑ ΔW_i mod n)"
            }
        }

        return {
            "round": self.current_round,
            "metrics": eval_metrics,
            "telemetry": self.latest_round_telemetry
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns overall FL system status."""
        return {
            "is_initialized": self.is_initialized,
            "current_round": self.current_round,
            "participating_banks": ["Apex Bank (50 Accounts)", "Beacon Bank (50 Accounts)", "Crestline Bank (50 Accounts)"],
            "model_architecture": {
                "type": "GraphSAGE 1-Hop Mean Neighborhood Aggregator",
                "input_features": 10,
                "hidden_dim": 8,
                "output_classes": 2,
                "trainable_parameters": 186
            },
            "public_key": self.pubkey.to_dict() if self.pubkey else None,
            "benchmark_comparison": self.benchmark_comparison,
            "rounds_completed": len(self.history) - 1 if self.history else 0
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Returns full training history across rounds."""
        return {
            "current_round": self.current_round,
            "history": self.history,
            "benchmark_comparison": self.benchmark_comparison
        }

    def get_security_disclosures(self) -> Dict[str, Any]:
        """Returns transparent threat model disclosures and non-claims."""
        return {
            "threat_model": "Semi-Honest (Honest-but-Curious)",
            "key_management": "Paillier private key is held exclusively by the Authorized Decryption Authority. The HE Aggregation Server possesses only the public key (n, g) and cannot decrypt individual ciphertexts under the implemented key-management model.",
            "modulus_configuration": "1024-bit Paillier modulus — prototype/demo configuration",
            "equal_weighting_note": "Equal-weight FedAvg is used because each participating synthetic bank currently contributes an equal number of training accounts (50 accounts each).",
            "explicit_non_claims": [
                "No Malicious Security: Does not verify zero-knowledge range proofs for participant gradient validity.",
                "No Byzantine Resistance: Does not filter adversarial poison updates over encrypted ciphertexts.",
                "No Differential Privacy: Does not inject Gaussian/Laplace DP-SGD noise into aggregated gradients.",
                "No Multi-Party Decryption: Single decryption keyholder rather than threshold MPC key shares.",
                "No Protection against all Model Inversion: Global decrypted weights are shared among participating banks."
            ]
        }


# Singleton Coordinator Instance
_fl_coordinator_instance: Optional[FLCoordinator] = None

def get_fl_coordinator() -> FLCoordinator:
    global _fl_coordinator_instance
    if _fl_coordinator_instance is None:
        _fl_coordinator_instance = FLCoordinator(seed=42)
    return _fl_coordinator_instance
