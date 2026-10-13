"""
Unit Test Suite for G.O.A.T. Shield Stage 2C-2:
Federated Learning with Paillier Homomorphic Encryption.
"""

import math
import secrets
import numpy as np
import pytest

from engine.paillier import (
    generate_paillier_keypair,
    float_to_paillier_integer,
    paillier_integer_to_float,
    encrypt_vector,
    homomorphic_aggregate_vectors,
    decrypt_and_decode_vector,
    get_ciphertext_fingerprint,
    DEFAULT_SCALE_FACTOR
)
from engine.fl_coordinator import FLCoordinator, CompactFLGraphSAGE
from engine.simulator import generate_accounts, generate_synthetic_dataset


# =========================================================
# 1. Cryptographic Invariant Tests
# =========================================================

def test_paillier_keygen():
    """Verify 1024-bit keypair generation and mathematical properties."""
    pubkey, privkey = generate_paillier_keypair(prime_bits=512)
    assert pubkey.bits == 1024
    assert pubkey.n.bit_length() >= 1020
    assert pubkey.g == pubkey.n + 1
    assert pubkey.n_sq == pubkey.n * pubkey.n
    assert privkey.pubkey == pubkey
    assert privkey.lmbda > 0
    assert privkey.mu > 0


def test_paillier_encrypt_decrypt_integer():
    """Verify Paillier encryption and decryption of random integers in Z_n."""
    pubkey, privkey = generate_paillier_keypair(prime_bits=512)
    for _ in range(5):
        m = secrets.randbelow(pubkey.n - 1000) + 100
        c = pubkey.encrypt(m)
        decrypted_m = privkey.decrypt(c)
        assert decrypted_m == m, f"Expected {m}, got {decrypted_m}"


def test_paillier_homomorphic_addition():
    """Verify E(a) * E(b) mod n^2 = E(a + b mod n)."""
    pubkey, privkey = generate_paillier_keypair(prime_bits=512)
    a = 123456789
    b = 987654321
    ca = pubkey.encrypt(a)
    cb = pubkey.encrypt(b)

    c_sum = pubkey.homomorphic_add(ca, cb)
    decrypted_sum = privkey.decrypt(c_sum)
    assert decrypted_sum == (a + b), f"Expected {a + b}, got {decrypted_sum}"


# =========================================================
# 2. Signed Fixed-Point Quantization Tests
# =========================================================

def test_fixed_point_positive_and_negative_floats():
    """Verify lossless signed encoding and decoding for floats in [-1.0, 1.0]."""
    pubkey, privkey = generate_paillier_keypair(prime_bits=512)
    test_values = [0.12345, -0.67890, 0.0, -0.00001, 0.99999, -0.99999]

    for val in test_values:
        m = float_to_paillier_integer(val, pubkey.n, scale=DEFAULT_SCALE_FACTOR)
        c = pubkey.encrypt(m)
        decrypted_m = privkey.decrypt(c)
        recovered_val = paillier_integer_to_float(decrypted_m, pubkey.n, scale=DEFAULT_SCALE_FACTOR, divisor=1.0)
        assert abs(recovered_val - val) < 1e-5, f"Original: {val}, Recovered: {recovered_val}"


def test_three_bank_homomorphic_aggregation():
    """
    Test decode(decrypt(E(ΔW_A) * E(ΔW_B) * E(ΔW_C))) == ΔW_A + ΔW_B + ΔW_C
    within fixed-point tolerance.
    """
    pubkey, privkey = generate_paillier_keypair(prime_bits=512)
    delta_A = [0.03456, -0.01234, 0.00500]
    delta_B = [-0.02100, -0.04500, 0.08900]
    delta_C = [0.01500, 0.03200, -0.01100]

    enc_A = encrypt_vector(pubkey, delta_A, scale=DEFAULT_SCALE_FACTOR)
    enc_B = encrypt_vector(pubkey, delta_B, scale=DEFAULT_SCALE_FACTOR)
    enc_C = encrypt_vector(pubkey, delta_C, scale=DEFAULT_SCALE_FACTOR)

    # Aggregation on HE Server (Ciphertexts only)
    c_agg = homomorphic_aggregate_vectors(pubkey, [enc_A, enc_B, enc_C])

    # Decryption & Averaging (divisor = 3.0)
    avg_recovered = decrypt_and_decode_vector(privkey, c_agg, scale=DEFAULT_SCALE_FACTOR, divisor=3.0)
    
    for i in range(len(delta_A)):
        expected_avg = (delta_A[i] + delta_B[i] + delta_C[i]) / 3.0
        assert abs(avg_recovered[i] - expected_avg) < 1e-5, f"Expected {expected_avg}, got {avg_recovered[i]}"


# =========================================================
# 3. Model Architecture & Coordinator Integration Tests
# =========================================================

def test_compact_graphsage_forward_and_params():
    """Verify GraphSAGE parameter packing and unpacking."""
    model = CompactFLGraphSAGE(in_features=10, hidden_dim=8, num_classes=2, seed=42)
    flat_weights = model.get_flat_weights()
    assert len(flat_weights) == 186

    X = np.random.randn(5, 10).astype(np.float32)
    A = np.eye(5, dtype=np.float32)
    probs, H1 = model.forward(X, A)
    assert probs.shape == (5, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    # Modify and restore
    new_weights = [w + 0.1 for w in flat_weights]
    model.set_flat_weights(new_weights)
    restored = model.get_flat_weights()
    assert np.allclose(restored, new_weights, atol=1e-5)


def test_fl_coordinator_full_flow():
    """Verify full multi-bank federated training loop with Paillier HE."""
    accounts_dict = generate_accounts()
    transactions = generate_synthetic_dataset(seed=42)

    coordinator = FLCoordinator(seed=42)
    coordinator.initialize(transactions, accounts_dict)

    status = coordinator.get_status()
    assert status["is_initialized"] is True
    assert status["current_round"] == 0
    assert status["public_key"]["modulus_bits"] == 1024

    # Execute Round 1
    r1 = coordinator.execute_round(local_epochs=2, lr=0.02)
    assert r1["round"] == 1
    assert "accuracy" in r1["metrics"]
    assert "telemetry" in r1
    assert len(r1["telemetry"]["ciphertext_sample_fingerprints"]["APEX"]) == 3

    # Execute Round 2
    r2 = coordinator.execute_round(local_epochs=2, lr=0.02)
    assert r2["round"] == 2

    metrics = coordinator.get_metrics()
    assert len(metrics["history"]) == 3 # Round 0, 1, 2
    assert metrics["current_round"] == 2

    disclosures = coordinator.get_security_disclosures()
    assert "1024-bit" in disclosures["modulus_configuration"]
    assert len(disclosures["explicit_non_claims"]) == 5
