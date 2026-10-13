"""
G.O.A.T. Shield Stage 2C-1 - Private Set Intersection (PSI) Test Suite
Tests:
1. Group and Hash-to-Group validity (h in G_q, h^q = 1 mod p, invalid element rejection)
2. Commutative exponentiation invariance ((H(x)^k_A)^k_B == (H(x)^k_B)^k_A mod p)
3. Exact intersection correctness (controlled synthetic overlaps discovered)
4. Non-intersection isolation (benign accounts never falsely matched)
5. Zero raw identifier or secret exponent exposure in safe vault summaries / API outputs
6. Pairwise PSI execution across all 3 bank pairs (Apex-Beacon, Apex-Crestline, Beacon-Crestline)
7. Tripartite multi-party intersection (Apex ∩ Beacon ∩ Crestline)
8. Ephemeral key rotation (secret exponent changes on reset)
9. Deterministic entity mapping consistency
10. Timing and execution performance
"""

import unittest
import sys
from pathlib import Path

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine.psi_engine import (
    psi_coordinator,
    hash_to_group,
    generate_bank_key,
    is_valid_subgroup_element,
    MODP_PRIME,
    GROUP_ORDER_Q,
    BankParty,
)


class TestStage2CPSIEngine(unittest.TestCase):

    def test_01_group_parameters_and_subgroup_membership(self):
        """Verify RFC 3526 2048-bit MODP Group 14 parameters and subgroup membership validation"""
        self.assertEqual(MODP_PRIME.bit_length(), 2048)
        self.assertEqual(GROUP_ORDER_Q, (MODP_PRIME - 1) // 2)

        # Hash-to-group must produce a valid element in G_q (order q)
        test_entity = "TEST_ENTITY_KYC_99182"
        h = hash_to_group(test_entity)
        
        self.assertTrue(is_valid_subgroup_element(h), "Hash-to-group must produce valid subgroup element")
        self.assertEqual(pow(h, GROUP_ORDER_Q, MODP_PRIME), 1, "Element in G_q must satisfy h^q = 1 mod p")
        
        # Test rejection of invalid elements (e.g. 0, 1, p, negative)
        self.assertFalse(is_valid_subgroup_element(0))
        self.assertFalse(is_valid_subgroup_element(1))
        self.assertFalse(is_valid_subgroup_element(MODP_PRIME))
        self.assertFalse(is_valid_subgroup_element(-5))

    def test_02_commutative_exponentiation_property(self):
        """Verify (H(x)^k_A)^k_B == (H(x)^k_B)^k_A mod p"""
        test_id = "TEST-DEVICE-FINGERPRINT-X"
        h = hash_to_group(test_id)
        k_a = generate_bank_key()
        k_b = generate_bank_key()

        self.assertTrue(1 < k_a < GROUP_ORDER_Q)
        self.assertTrue(1 < k_b < GROUP_ORDER_Q)

        # Route 1: A blinds first, then B blinds
        blind_a = pow(h, k_a, MODP_PRIME)
        self.assertTrue(is_valid_subgroup_element(blind_a))
        double_ab = pow(blind_a, k_b, MODP_PRIME)

        # Route 2: B blinds first, then A blinds
        blind_b = pow(h, k_b, MODP_PRIME)
        self.assertTrue(is_valid_subgroup_element(blind_b))
        double_ba = pow(blind_b, k_a, MODP_PRIME)

        self.assertEqual(double_ab, double_ba, "Commutative group exponentiation must be strictly identical")

    def test_03_deterministic_entity_mapping(self):
        """Verify the same entity identifier always maps deterministically to the same subgroup element"""
        ent_id = "CONSISTENT_DEVICE_FINGERPRINT_123"
        h1 = hash_to_group(ent_id)
        h2 = hash_to_group(ent_id)
        self.assertEqual(h1, h2, "Hash-to-group must be deterministic")

    def test_04_no_raw_identifier_or_secret_key_exposure(self):
        """Verify get_vault_summary returns only safe metadata, zero secret exponents or raw IDs"""
        summary = psi_coordinator.get_vault_summary()
        self.assertEqual(summary["status"], "ready")
        self.assertEqual(len(summary["banks"]), 3)

        for b in summary["banks"]:
            self.assertEqual(b["local_entity_count"], 50)
            self.assertTrue(b["key_fingerprint"].startswith(f"k_{b['bank_code']}_"))
            # Secret key must not appear
            self.assertNotIn(str(psi_coordinator.banks[b["bank_name"]].private_key), str(summary))
            # Masked sample checks
            for sample in b["sample_masked_records"]:
                self.assertTrue(sample["masked_pan"].startswith("PAN-***-"))
                self.assertNotIn("DEV-SYND", sample["masked_pan"])

    def test_05_pairwise_psi_execution_all_three_pairs(self):
        """Verify pairwise PSI across Apex-Beacon, Apex-Crestline, and Beacon-Crestline"""
        apx = psi_coordinator.banks["Apex Bank"]
        bcn = psi_coordinator.banks["Beacon Bank"]
        cln = psi_coordinator.banks["Crestline Bank"]

        # 1. Apex <-> Beacon
        res_ab = psi_coordinator.execute_pairwise_psi(apx, bcn)
        self.assertGreater(res_ab["intersection_count"], 0)
        self.assertLess(res_ab["elapsed_ms"], 10000)
        for m in res_ab["matches"]:
            self.assertTrue(m["is_mule"])
            self.assertIn("verification_token", m)

        # 2. Apex <-> Crestline
        res_ac = psi_coordinator.execute_pairwise_psi(apx, cln)
        self.assertGreater(res_ac["intersection_count"], 0)

        # 3. Beacon <-> Crestline
        res_bc = psi_coordinator.execute_pairwise_psi(bcn, cln)
        self.assertGreater(res_bc["intersection_count"], 0)

    def test_06_tripartite_multi_party_intersection(self):
        """Verify tripartite multi-party intersection (Apex ∩ Beacon ∩ Crestline)"""
        results = psi_coordinator.execute_full_network_psi()
        self.assertIn("session_id", results)
        self.assertIn("tripartite_results", results)
        
        tripartite = results["tripartite_results"]["common_entities"]
        self.assertEqual(len(tripartite), 4, "Must identify 4 common cross-bank syndicates")
        for entity in tripartite:
            self.assertIn("verification_token", entity)
            self.assertIsNotNone(entity["apex_account"])
            self.assertIsNotNone(entity["beacon_account"])
            self.assertIsNotNone(entity["crestline_account"])

    def test_07_non_intersection_benign_isolation(self):
        """Verify benign single-bank accounts are never leaked or included in the intersection"""
        results = psi_coordinator.execute_full_network_psi()
        for match in results["all_pairwise_matches"]:
            self.assertTrue(match["is_mule"], f"Benign account falsely included in intersection: {match}")

    def test_08_ephemeral_key_rotation(self):
        """Verify that running PSI or resetting session rotates secret exponents"""
        key_apx_1 = psi_coordinator.banks["Apex Bank"].private_key
        key_bcn_1 = psi_coordinator.banks["Beacon Bank"].private_key

        psi_coordinator.execute_full_network_psi()

        key_apx_2 = psi_coordinator.banks["Apex Bank"].private_key
        key_bcn_2 = psi_coordinator.banks["Beacon Bank"].private_key

        self.assertNotEqual(key_apx_1, key_apx_2, "Apex key must rotate on new session")
        self.assertNotEqual(key_bcn_1, key_bcn_2, "Beacon key must rotate on new session")


if __name__ == "__main__":
    unittest.main()
