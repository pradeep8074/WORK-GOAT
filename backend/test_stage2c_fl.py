"""
Standalone runner for Stage 2C-2: Federated Learning with Paillier Homomorphic Encryption.
"""
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(__file__))

import pytest

if __name__ == "__main__":
    test_path = os.path.join(os.path.dirname(__file__), "tests", "test_stage2c_fl.py")
    print(f"Running Stage 2C-2 Paillier & FL Tests: {test_path}")
    exit_code = pytest.main(["-v", test_path])
    sys.exit(exit_code)
