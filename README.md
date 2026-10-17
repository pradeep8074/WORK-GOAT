# G.O.A.T. Shield
### **Privacy-Preserving Federated Fraud Intelligence for Cross-Bank Mule Detection**


---

## 1. Project Overview & Problem Statement

Financial crime syndicates increasingly exploit **institutional data silos** between banks. By utilizing techniques such as **smurfing (micro-structuring)**, **rapid pass-throughs**, **multi-hop layering chains**, and **circular loops**, illicit funds are dispersed across multiple independent banking institutions. 

Under regulatory frameworks (such as PMLA and FIU-IND guidelines), banks cannot share raw, unredacted customer transaction ledgers with peer institutions. Because each bank observes only its local slice of the transaction network, cross-bank syndicates operate with near impunity.

**G.O.A.T. Shield** resolves this regulatory-technical conflict by providing a dual-layer cryptographic and graph-powered fraud detection architecture:
1. **Private Set Intersection (DH-PSI)**: Resolves suspicious cross-bank counterparty identities using commutative 2048-bit Diffie-Hellman exponentiation—**zero cleartext customer records or identifiers are disclosed**.
2. **Homomorphic Federated Learning**: Trains a Graph Neural Network (**GraphSAGE**) across isolated bank silos, aggregating gradient updates under **1024-bit Paillier Additive Homomorphic Encryption**—**zero plaintext weights or model deltas are exposed to the central server**.

---

## 2. Key Features

- **Deterministic Rule Engine (R1–R6)**: $O(\log K)$ temporal sliding-window analysis detecting FIU-IND threshold proximity (₹7.5L–₹10L), cross-bank boundaries, fan-in concentration, fan-out velocity, rapid pass-throughs, and structuring bursts.
- **Cross-Bank Transaction Graph (NetworkX)**: In-memory directed multi-graph mapping 150 accounts and 500 transactions across 3 banks (Apex, Beacon, Crestline).
- **GraphSAGE Neural Network**: 2-hop mean neighborhood aggregation model combining graph topology with 10 observable node behavioral features.
- **Privacy-Preserving Entity Matching**: Commutative Diffie-Hellman PSI over RFC 3526 2048-bit MODP Group 14 safe-prime group.
- **Homomorphic Federated Learning**: Secure FedAvg aggregation over 186 trainable parameters using 1024-bit Paillier encryption.
- **Real-Time Live Feed**: Live streaming transaction scoring and alert feed with sub-5ms latency.
- **Comprehensive Audit Dashboard**: Full React 18 SPA featuring graph visualizers, alert management, case investigation, and empirical performance metrics.

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        React 18 SPA Dashboard                          │
│   (Overview | Transactions | Alerts | Network Graph | FL | PSI | Perf) │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ HTTP / REST (Axios)
┌───────────────────────────────────▼────────────────────────────────────┐
│                       FastAPI Backend Service                          │
│                                                                        │
│  ┌──────────────────────────────┐   ┌───────────────────────────────┐  │
│  │    Rule Engine (R1-R6)       │   │    NetworkX Graph Engine      │  │
│  │   O(log K) Bisect Windows    │   │  MultiDiGraph Topology & Hubs │  │
│  └──────────────────────────────┘   └───────────────────────────────┘  │
│  ┌──────────────────────────────┐   ┌───────────────────────────────┐  │
│  │    GraphSAGE GNN Engine      │   │    SQLite WAL Persistence     │  │
│  │  2-Hop Message Passing (ML)  │   │  ACID Document-Relational DB  │  │
│  └──────────────────────────────┘   └───────────────────────────────┘  │
│  ┌──────────────────────────────┐   ┌───────────────────────────────┐  │
│  │     Paillier HE Engine       │   │       DH-PSI Engine           │  │
│  │ 1024-bit Homomorphic FedAvg  │   │   RFC 3526 MODP-2048 Group    │  │
│  └──────────────────────────────┘   └───────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Technology Stack (Actual Implementation)

| Layer | Technology | Version | Purpose in Prototype |
| :--- | :--- | :--- | :--- |
| **Frontend** | React | 18.3.1 | Core component UI framework |
| **Frontend** | Vite | 6.1.0 | High-performance build tool and dev server |
| **Frontend** | Tailwind CSS | 3.4.17 | Dark-mode design system & utility styling |
| **Frontend** | Lucide React | 0.475.0 | Financial & security iconography |
| **Frontend** | Axios | 1.7.9 | Centralized HTTP client for backend REST APIs |
| **Backend** | Python | 3.11+ | Computational runtime |
| **Backend** | FastAPI | 0.115.0+ | Asynchronous ASGI REST API framework |
| **Backend** | Uvicorn | 0.30.0+ | High-throughput ASGI application server |
| **Backend** | Pydantic | 2.0.0+ | Data validation and request/response serialization |
| **Database** | SQLite (WAL Mode)| 3.45+ | ACID document-relational persistence layer |
| **Graph Engine** | NetworkX | 3.3+ | Directed transaction network representation |
| **Math / ML** | NumPy | 1.26.0+ | GraphSAGE tensors, analytical backprop, FL weights |
| **Cryptography** | RFC 3526 MODP-2048 | Group 14 | Commutative Diffie-Hellman Private Set Intersection |
| **Cryptography** | Paillier Cryptosystem | 1024-bit | Additive Homomorphic Encryption for Federated Learning |

---

## 5. Project Structure

```
hack/
├── backend/
│   ├── engine/                     # Core computational, ML, graph, and cryptographic engines
│   │   ├── detector.py             # Heuristic fraud detection (R1-R6 rules, windowed bisect algorithms)
│   │   ├── fl_coordinator.py       # Federated Learning coordinator (CompactFLGraphSAGE + FedAvg)
│   │   ├── gnn_detector.py         # 2-hop GraphSAGE GNN engine, feature extraction, train/val/test splits
│   │   ├── graph_builder.py        # NetworkX multi-directed transaction graph constructor & topological metrics
│   │   ├── metrics.py              # Mathematical evaluation engine (Precision, Recall, F1, Confusion Matrix)
│   │   ├── paillier.py             # 1024-bit Paillier Additive Homomorphic Encryption cryptosystem
│   │   ├── psi_engine.py           # Multi-bank Diffie-Hellman Private Set Intersection (RFC 3526 MODP-2048)
│   │   └── simulator.py            # Synthetic dataset generator with 4 planted mule topologies
│   ├── data/                       # Initial bank metadata and seed records
│   │   ├── banks.json              # Bank metadata for Apex, Beacon, Crestline
│   │   ├── transactions.json       # Stage 1 baseline transactions
│   │   └── alerts.json             # Stage 1 baseline alerts
│   ├── tests/                      # Verification and regression test suites
│   │   ├── test_stage2c_fl.py      # Paillier HE & Federated Learning round verification tests
│   │   ├── test_stage2c_psi.py     # Diffie-Hellman commutative PSI correctness tests
│   │   └── test_storage.py         # SQLite WAL persistence layer unit tests
│   ├── main.py                     # Primary FastAPI application entry point, routing, endpoints
│   ├── storage.py                  # Thread-safe SQLite persistence layer (WAL mode)
│   ├── goat.db                     # Active SQLite database file
│   ├── requirements.txt            # Python dependencies
│   ├── test_e2e.py                 # End-to-end API integration verification script
│   ├── test_stage2a.py             # Stage 2A verification test script (42 unit checks)
│   ├── test_stage2b.py             # Stage 2B GraphSAGE evaluation test script
│   └── run_progressive_scalability_benchmark.py # Multi-scale empirical benchmark script
├── frontend/
│   ├── src/
│   │   ├── components/             # Reusable UI component library (BankCard, StatCard, TransactionDrawer, etc.)
│   │   ├── pages/                  # Route views (Overview, Transactions, Alerts, Network, Federated, Privacy, etc.)
│   │   ├── services/api.js         # Centralized Axios API client
│   │   ├── utils/currency.js       # INR currency formatting utilities (Lakhs / Crores)
│   │   ├── data/fallbackData.js    # Graceful fallback state for demo resilience
│   │   ├── App.jsx                 # Client-side router and navigation frame
│   │   └── index.css               # Tailwind CSS theme declarations
│   ├── package.json                # Frontend package dependencies
│   ├── vite.config.js              # Vite bundler configuration
│   └── tailwind.config.js          # Tailwind CSS configuration
├── results/                        # Persisted scientific experiment logs & benchmark outputs
│   ├── ablation_results.json       # 5-seed empirical ablation test outputs
│   ├── fresh_seeds_results.json    # Confirmation run on unpeeked seeds 5–9
│   ├── raw_advanced_benchmark_results.json # Latency and evasive pattern dataset
│   └── ablation_report.md          # Scientific methodology breakdown
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git exclusion rules
└── README.md                       # Comprehensive judge documentation
```

---

## 6. Quick Start & Setup Instructions

### Prerequisites
- **Python 3.10+** (Tested on Python 3.11)
- **Node.js 18+** (Tested on Node.js 20+)
- **npm** or **pnpm**

---

### Step 1: Run the Backend (FastAPI)

1. Open a terminal and navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. (Optional but recommended) Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
   *The backend will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)*  
   *Interactive OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)*

---

### Step 2: Run the Frontend (React + Vite)

1. Open a second terminal and navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install frontend dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev -- --host 127.0.0.1 --port 5173
   ```
   *Open your browser and navigate to: [http://127.0.0.1:5173](http://127.0.0.1:5173)*

---

## 7. Verification & Automated Testing

You can verify the entire backend logic and unit tests using the standard test runners:

```bash
cd backend

# Run the 42-assertion Stage 2A verification suite
python test_stage2a.py

# Run all unit tests (PSI, Federated Learning, Storage layer)
python -m unittest discover -s . -p "test_*.py"
```
**Expected Outcome**: All tests pass with **100% success rate**.

---

## 8. Primary REST API Reference

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/api/health` | `GET` | System health check (`"healthy"`). |
| `/api/banks` | `GET` | Metadata and status of the 3 participating bank nodes. |
| `/api/v2/simulate` | `GET` | Paginated 500 synthetic transactions scored by rule engine. |
| `/api/v2/transactions/live` | `GET` | Real-time live transaction feed with instantaneous scoring. |
| `/api/v2/alerts/live` | `GET` | Live security fraud alerts emitted by the detection engine. |
| `/api/v2/graph` | `GET` | NetworkX directed cross-bank transaction graph (150 nodes, 500 edges). |
| `/api/v2/privacy/psi/run` | `POST` | Executes 3-bank Diffie-Hellman Private Set Intersection. |
| `/api/v2/federated/round` | `POST` | Executes 1 round of Paillier-encrypted Federated Learning. |
| `/api/v2/metrics/performance` | `GET` | Empirical ablation metrics and benchmark results from disk. |

---

## 9. Performance & Empirical Evaluation

- **Detection Baseline (500 Transactions)**:
  - Accuracy: **96.2%**
  - Precision: **88.2%**
  - Recall: **87.2%**
  - F1 Score: **87.7%**
  - Detection rate on Fan-In and Fan-Out mule rings: **100.0%**
- **Ensemble Robustness (35,000 Transactions)**:
  - Validation-Tuned Hybrid Ensemble F1: **$96.0 \pm 0.7\%$** across 5 seeds.
  - Confirmation run on fresh, unpeeked seeds 5–9: **$95.7 \pm 0.8\%$** F1.
- **Inference Efficiency**:
  - Single-transaction scoring latency: **$4.42 \pm 0.81\text{ ms}$** (P95: 5.75 ms).
  - Graph memory reduction: Sparse CSR consumes **0.88 MB** vs **186.9 MB** for dense matrices (**212.4× reduction**).

---

## 10. Known Prototype Limitations & Future Scope

### Current Prototype Limitations
1. **Synthetic Data**: Workload datasets are parametrically generated with modeled mule typologies; real-world banking ledgers exhibit non-stationary multi-modal distributions.
2. **Semi-Honest Security Model**: The DH-PSI and Paillier HE implementations operate under the semi-honest (honest-but-curious) adversary model.
3. **Single-Node Simulation**: Multi-bank silos and coordinator run in separate memory contexts on a single machine rather than physically separated WAN cloud instances.

### Future Scope
1. **Malicious-Secure Zero-Knowledge Proofs**: Integrate ZK proofs of discrete-log exponentiation to prevent adversarial input tampering in PSI.
2. **Distributed Node Deployment**: Deploy bank nodes on independent cloud clusters (AWS/GCP) communicating via mutual TLS (mTLS).
3. **Differential Privacy ($\epsilon, \delta$-DP)**: Introduce bounded Laplacian or Gaussian noise to encrypted gradients before Paillier aggregation.
4. **Direct Payment Rail Hook**: Connect to IMPS / UPI real-time payment switches via ISO 20022 message adapters.

---

## 11. Team & Project Credits
- **Project**: G.O.A.T. Shield
  
- **Track**: AI/ML, Privacy-Preserving Computing, FinTech Security
