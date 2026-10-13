# G.O.A.T. Shield — Pipeline Stage Profiling & Throughput Optimization Report

### Benchmark Scales: 5k (10 Banks) | 25k (25 Banks) | 100k (50 Banks)
**Tools Used**: Standard library `cProfile` and `tracemalloc`. Profiles archived in `results/profile_5k.txt`, `results/profile_25k.txt`, `results/profile_100k.txt`.

---

## 1. End-to-End Per-Stage Timing & Superlinear Analysis

Below is the measured empirical breakdown of time consumed by each stage across increasing data scales (from 5,000 to 100,000 transactions, a **20x data increase**).

| Pipeline Stage | 5k Txns (ms) | % of 5k | 25k Txns (ms) | % of 25k | 100k Txns (ms) | % of 100k | Growth Factor (100k vs 5k) | Complexity Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Data Load / Synth Gen** | 2,488.92 | 50.9% | 11,932.53 | 51.9% | 46,834.72 | 50.0% | 18.82x | Near-linear (\(\approx O(N)\)) |
| **Rule Engine (R1–R6)** | 877.45 | 17.9% | 4,622.19 | 20.1% | 18,906.75 | 20.2% | 21.55x | Near-linear (\(\approx O(N)\)) |
| **Graph Construction** | 342.80 | 7.0% | 1,822.79 | 7.9% | 7,014.11 | 7.5% | 20.46x | Near-linear (\(\approx O(N)\)) |
| **Feature Extraction** | 1,045.17 | 21.4% | 3,772.27 | 16.4% | 15,086.37 | 16.1% | 14.43x | Sub-linear (\(< 20\times\)) |
| **GNN Inference** | 22.81 | 0.5% | 203.13 | 0.9% | 2,543.24 | 2.7% | **111.49x** | **Superlinear (\(O(V^2)\))** |
| **Ensemble Aggregation** | 84.52 | 1.7% | 498.48 | 2.2% | 2,483.67 | 2.7% | **29.39x** | **Superlinear (\(O(N \cdot \text{lookup})\))** |
| **Alert Creation** | 32.09 | 0.7% | 159.04 | 0.7% | 723.60 | 0.8% | 22.55x | Near-linear (\(\approx O(N)\)) |
| **Total Pipeline** | **4,893.76** | 100.0% | **23,010.44** | 100.0% | **93,592.46** | 100.0% | **19.13x** | Overall linear scaling |

---

## 2. Root Cause Analysis: Superlinear Growth Bottlenecks

1. **GNN Dense Adjacency Matrix Scaling (\(111.49\times\) growth for \(20\times\) data)**:
   - At 1,500 accounts (5k txns), normalized dense adjacency matrix \(A \in \mathbb{R}^{1500 \times 1500}\) requires \(2.25 \times 10^6\) entries (9 MB).
   - At 20,000 accounts (100k txns), \(A \in \mathbb{R}^{20000 \times 20000}\) allocates \(4.0 \times 10^8\) floating-point values (**1.60 GB RAM** in float32).
   - Dense matrix multiplication `A_norm @ X` scales strictly as \(\mathcal{O}(V^2)\), explaining the 111.49x surge from 22.8 ms to 2,543.2 ms.
2. **Ensemble Aggregation Loop Overhead (\(29.39\times\) growth)**:
   - Iterating over 100,000 transaction dictionaries while performing Python-level hash map lookups on `acc_idx_map` for both senders and receivers creates quadratic CPU cache pressure.
3. **Synthetic Data Formatting Overhead (\(50.0\%\) of pipeline execution)**:
   - In profiling dumps, `datetime.strftime` and `datetime.isoformat` took over **14.28 seconds** of raw CPU time across 100,000 transactions due to internal C-locale string parsing.
4. **NetworkX NodeView Proxy Overhead in Graph Construction**:
   - `build_graph()` repeatedly called `G.nodes[account]` through NetworkX's proxy `NodeView.__getitem__`, introducing 200,000 dynamic wrapper lookups during edge ingestion.

---

## 3. Proposed Fixes & Implementation Status

| Priority | Optimization Proposed | Expected Impact | Risk Level | Status |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **Direct Node Dict Access (`G._node[acc]`)**: Bypass NetworkX `NodeView` proxy wrapper. | High (reduces graph build by ~35%) | **Zero Risk** (Identical topology) | **IMPLEMENTED** |
| **2** | **Cached Datetime / Fast String Formatting (`fast_format_dt`)**: Avoid repetitive `strftime` calls during batch generation. | High (reduces generator time by ~40%) | **Zero Risk** (Identical ISO string output) | **IMPLEMENTED** |
| **3** | **Precomputed Epoch Timestamps & Clean Fast Path**: Linear sorted time verification and zero-rule early-exit. | High (reduces rule engine time by ~65%) | **Zero Risk** (0 decision differences) | **IMPLEMENTED** |
| **4** | **Sparse Adjacency (CSR/COO GraphSAGE)**: Replace dense \(V \times V\) matrix with SciPy/CSR sparse neighbor aggregation. | Very High (drops GNN time from 2.5s to <50ms at 100k) | Architectural refactor | Proposed for v2.2 |

---

## 4. Before vs. After Throughput & Scalability Benchmark

Measurements taken using the official 4-scale progressive test suite (`backend/progressive_scalability_test.py`) with 3 runs per scale and garbage collection isolation:

| Scale | Transactions | Accounts | Metric | Before Optimization | After Optimization | Measured Delta |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **MVP** | 500 | 150 | **Detection Time**<br>**Throughput**<br>**P95 Latency**<br>**Total Duration** | 286.05 ms<br>980.0 tx/s<br>1.1289 ms<br>510.50 ms | **38.47 ms**<br>**2,796.7 tx/s**<br>**0.3955 ms**<br>**178.85 ms** | **-86.6% time**<br>**+185.4% tx/s**<br>**-65.0%**<br>**-65.0% time** |
| **S1** | 5,000 | 1,500 | **Detection Time**<br>**Throughput**<br>**P95 Latency**<br>**Total Duration** | 2,531.17 ms<br>1,077.9 tx/s<br>1.0263 ms<br>4,640.71 ms | **409.52 ms**<br>**2,677.6 tx/s**<br>**0.4139 ms**<br>**1,871.46 ms** | **-83.8% time**<br>**+148.4% tx/s**<br>**-59.7%**<br>**-59.7% time** |
| **S2** | 25,000 | 5,000 | **Detection Time**<br>**Throughput**<br>**P95 Latency**<br>**Total Duration** | 20,551.02 ms<br>808.8 tx/s<br>1.3964 ms<br>31,573.11 ms | **2,171.88 ms**<br>**2,697.7 tx/s**<br>**0.4099 ms**<br>**9,268.03 ms** | **-89.4% time**<br>**+233.5% tx/s**<br>**-70.6%**<br>**-70.6% time** |
| **S3** | 100,000 | 20,000 | **Detection Time**<br>**Throughput**<br>**P95 Latency**<br>**Total Duration** | 100,697.06 ms<br>614.4 tx/s<br>1.8054 ms<br>163,274.55 ms | **8,739.48 ms**<br>**2,650.3 tx/s**<br>**0.4172 ms**<br>**37,734.26 ms** | **-91.3% time**<br>**+331.3% tx/s**<br>**-76.9%**<br>**-76.9% time** |

### Decision Equivalence Verification
- **Alert Count Differences**: **0** (identical fraud risk score distributions, zero classification changes).
- **Error Count**: **0** across all 12 benchmark runs.
