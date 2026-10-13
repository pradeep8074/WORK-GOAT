"""
G.O.A.T. Shield Stage 2C-2 - Paillier Homomorphic Encryption Engine
Implements the Paillier Additive Homomorphic Cryptosystem over Z_{n^2}*
with signed fixed-point parameter quantization for privacy-preserving
Federated Learning across Apex, Beacon, and Crestline banks.

Configuration: 1024-bit Paillier modulus — prototype/demo configuration.
"""

import math
import secrets
from typing import List, Tuple, Dict, Any, Optional


# =========================================================
# 1. Miller-Rabin Primality & Key Generation
# =========================================================

# Precomputed small primes for fast sieving before Miller-Rabin
_SMALL_PRIMES = [
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
    211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331,
    337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457
]

def _is_probable_prime(n: int, k: int = 32) -> bool:
    """Miller-Rabin probabilistic primality test with small prime sieving."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    # Small prime sieve eliminates >88% of non-primes instantly
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False

    # Write n - 1 as 2^s * d
    s = 0
    d = n - 1
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_prime(bits: int = 512) -> int:
    """Generates a cryptographically secure prime of the specified bit length."""
    while True:
        # Ensure highest and lowest bits are set
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(candidate):
            return candidate


def _lcm(a: int, b: int) -> int:
    """Least common multiple of two integers."""
    return abs(a * b) // math.gcd(a, b)


# =========================================================
# 2. Paillier Public & Private Key Classes
# =========================================================

class PaillierPublicKey:
    """
    Public Key (n, g) for Paillier Additive Homomorphic Encryption.
    Modulus: n = p * q
    Modulus squared: n^2
    Generator: g = n + 1
    """
    def __init__(self, n: int, bits: int = 1024):
        self.n = n
        self.n_sq = n * n
        self.g = n + 1
        self.bits = bits

    def encrypt(self, m: int, r: Optional[int] = None) -> int:
        """
        Encrypts an integer message m in Z_n into ciphertext c in Z_{n^2}*.
        c = ( (1 + m * n) * r^n ) mod n^2
        """
        m = m % self.n
        if r is None:
            while True:
                r = secrets.randbelow(self.n - 1) + 1
                if math.gcd(r, self.n) == 1:
                    break
        else:
            r = r % self.n

        # Optimization: (1 + m*n) mod n^2 since g = n + 1
        g_m = (1 + m * self.n) % self.n_sq
        r_n = pow(r, self.n, self.n_sq)
        c = (g_m * r_n) % self.n_sq
        return c

    def homomorphic_add(self, c1: int, c2: int) -> int:
        """
        Homomorphic addition: E(m1) * E(m2) mod n^2 = E(m1 + m2 mod n)
        """
        return (c1 * c2) % self.n_sq

    def homomorphic_aggregate(self, ciphertexts: List[int]) -> int:
        """
        Aggregates a list of ciphertexts: prod(c_i) mod n^2 = E(sum(m_i) mod n)
        """
        if not ciphertexts:
            raise ValueError("Cannot aggregate empty list of ciphertexts")
        res = 1
        for c in ciphertexts:
            res = (res * c) % self.n_sq
        return res

    def to_dict(self) -> Dict[str, Any]:
        """Serializes public key metadata for telemetry."""
        return {
            "modulus_bits": self.bits,
            "modulus_hex_preview": hex(self.n)[:18] + "..." + hex(self.n)[-8:],
            "generator": "n + 1 (standard Paillier)",
            "key_type": "Paillier Asymmetric Public Key (1024-bit prototype/demo configuration)"
        }


class PaillierPrivateKey:
    """
    Private Key (lambda, mu) held by the Authorized Decryption Authority.
    lambda = lcm(p-1, q-1)
    mu = lambda^(-1) mod n
    """
    def __init__(self, lmbda: int, mu: int, pubkey: PaillierPublicKey):
        self.lmbda = lmbda
        self.mu = mu
        self.pubkey = pubkey

    def decrypt(self, c: int) -> int:
        """
        Decrypts ciphertext c in Z_{n^2}* back to integer message m in Z_n.
        m = L(c^lambda mod n^2) * mu mod n, where L(u) = (u - 1) // n
        """
        c = c % self.pubkey.n_sq
        u = pow(c, self.lmbda, self.pubkey.n_sq)
        l_val = (u - 1) // self.pubkey.n
        m = (l_val * self.mu) % self.pubkey.n
        return m


def generate_paillier_keypair(prime_bits: int = 512) -> Tuple[PaillierPublicKey, PaillierPrivateKey]:
    """
    Generates a Paillier keypair with two distinct primes of prime_bits each (1024-bit modulus).
    """
    p = generate_prime(prime_bits)
    while True:
        q = generate_prime(prime_bits)
        if q != p:
            break

    n = p * q
    lmbda = _lcm(p - 1, q - 1)
    pubkey = PaillierPublicKey(n=n, bits=prime_bits * 2)

    # mu = lambda^(-1) mod n
    mu = pow(lmbda, -1, n)
    privkey = PaillierPrivateKey(lmbda=lmbda, mu=mu, pubkey=pubkey)
    return pubkey, privkey


# =========================================================
# 3. Signed Fixed-Point Parameter Quantization
# =========================================================

DEFAULT_SCALE_FACTOR = 100000  # S = 10^5 (5 decimal digits of float precision)

def float_to_paillier_integer(val: float, n: int, scale: int = DEFAULT_SCALE_FACTOR) -> int:
    """
    Encodes a floating-point model weight delta into a non-negative integer m in Z_n.
    m_tilde = round(val * scale)
    If m_tilde >= 0: m = m_tilde
    If m_tilde < 0:  m = n + m_tilde (modular negative representation)
    """
    m_tilde = int(round(val * scale))
    if m_tilde >= 0:
        return m_tilde % n
    else:
        return (n + m_tilde) % n


def paillier_integer_to_float(
    m: int,
    n: int,
    scale: int = DEFAULT_SCALE_FACTOR,
    divisor: float = 1.0
) -> float:
    """
    Decodes an integer m in Z_n back to signed floating point and applies FedAvg divisor.
    If m > n // 2: signed_val = m - n
    Else:          signed_val = m
    float_val = (signed_val / scale) / divisor
    """
    half_n = n // 2
    if m > half_n:
        signed_val = m - n
    else:
        signed_val = m
    return (float(signed_val) / float(scale)) / float(divisor)


# =========================================================
# 4. Vector Operations for Graph Neural Network Parameters
# =========================================================

def encrypt_vector(
    pubkey: PaillierPublicKey,
    values: List[float],
    scale: int = DEFAULT_SCALE_FACTOR
) -> List[int]:
    """Encrypts a list of float parameters into Paillier ciphertexts."""
    ciphertexts = []
    for v in values:
        m = float_to_paillier_integer(v, pubkey.n, scale=scale)
        c = pubkey.encrypt(m)
        ciphertexts.append(c)
    return ciphertexts


def homomorphic_aggregate_vectors(
    pubkey: PaillierPublicKey,
    ciphertext_vectors: List[List[int]]
) -> List[int]:
    """
    Homomorphically adds corresponding ciphertexts across multiple banks:
    C_agg[j] = prod_i(C_i[j]) mod n^2
    """
    if not ciphertext_vectors:
        raise ValueError("No ciphertext vectors provided")
    num_params = len(ciphertext_vectors[0])
    for vec in ciphertext_vectors:
        if len(vec) != num_params:
            raise ValueError(f"Mismatched vector lengths: expected {num_params}, got {len(vec)}")

    aggregated = []
    for j in range(num_params):
        prod = 1
        for vec in ciphertext_vectors:
            prod = (prod * vec[j]) % pubkey.n_sq
        aggregated.append(prod)
    return aggregated


def decrypt_and_decode_vector(
    privkey: PaillierPrivateKey,
    ciphertexts: List[int],
    scale: int = DEFAULT_SCALE_FACTOR,
    divisor: float = 3.0
) -> List[float]:
    """
    Decrypts aggregate ciphertexts and decodes them into averaged floating point parameters.
    """
    n = privkey.pubkey.n
    floats = []
    for c in ciphertexts:
        m = privkey.decrypt(c)
        f = paillier_integer_to_float(m, n, scale=scale, divisor=divisor)
        floats.append(f)
    return floats


def get_ciphertext_fingerprint(c: int) -> str:
    """Returns a safe non-confidential telemetry hex fingerprint for a ciphertext."""
    hex_str = hex(c)[2:]
    if len(hex_str) <= 16:
        return f"0x{hex_str}"
    return f"0x{hex_str[:8]}...{hex_str[-8:]}"
