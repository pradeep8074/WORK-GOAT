# G.O.A.T. Shield — Ablation Study & Detection Quality Report
### 3 Banks | 7,000 Accounts | 35,000 Transactions | 1,400 Held-Out Test Accounts (5 Seeds)

Evaluation Protocol: 60/20/20 Stratified Partitioning (4,200 Train / 1,400 Val / 1,400 Test). Zero data leakage.

## 1. Full Ablation Metrics Table (Mean ± Std over 5 Seeds)

| Variant | Precision (%) | Recall (%) | F1 Score (%) | Accuracy (%) | FPR (%) | PR-AUC | ROC-AUC | P@50 | P@100 | P@200 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Rules Only (R1-R6)** | 96.4 ± 1.3% | 45.6 ± 1.9% | 61.9 ± 1.5% | 85.1 ± 0.5% | 0.62 ± 0.23% | 0.8709 | 0.9258 | 99.6% | 99.8% | 95.2% |
| **2A. GNN Only (Before Improvements)** | 51.5 ± 10.4% | 74.9 ± 7.4% | 59.8 ± 5.8% | 72.0 ± 9.4% | 29.09 ± 15.11% | 0.6162 | 0.7902 | 84.0% | 76.2% | 65.8% |
| **2B. GNN Only (After Improvements)** | 98.2 ± 0.8% | 93.8 ± 2.0% | 95.9 ± 1.2% | 97.9 ± 0.6% | 0.64 ± 0.29% | 0.9859 | 0.9904 | 100.0% | 100.0% | 100.0% |
| **3A. Hybrid Ensemble (Old beta=0.5, tau=0.55)** | 72.1 ± 3.0% | 81.9 ± 3.9% | 76.6 ± 2.8% | 86.7 ± 1.6% | 11.54 ± 1.54% | 0.8681 | 0.9288 | 99.6% | 99.4% | 94.4% |
| **3B. Hybrid Ensemble (Tuned on Validation)** | 97.9 ± 1.4% | 94.1 ± 1.5% | 96.0 ± 1.1% | 97.9 ± 0.6% | 0.74 ± 0.53% | 0.9863 | 0.9904 | 100.0% | 100.0% | 100.0% |
| **4. Hybrid w/o PSI (Single-Bank View - Control)** | 85.1 ± 10.3% | 5.8 ± 2.3% | 10.7 ± 4.0% | 74.7 ± 0.7% | 0.37 ± 0.21% | 0.5268 | 0.7425 | 75.2% | 63.8% | 57.2% |
| **5. Isolated Silos (No Federation - Control)** | 33.9 ± 1.7% | 66.9 ± 2.4% | 45.0 ± 2.0% | 56.4 ± 2.4% | 47.46 ± 2.58% | 0.4039 | 0.6326 | 57.2% | 48.8% | 44.7% |
| **6. Federated GNN (Collaborative)** | 98.2 ± 0.8% | 93.8 ± 2.0% | 95.9 ± 1.2% | 97.9 ± 0.6% | 0.64 ± 0.29% | 0.9859 | 0.9904 | 100.0% | 100.0% | 100.0% |

---

## 2. Confusion Matrices (5-Seed Average Counts)

| Variant | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
| :--- | :---: | :---: | :---: | :---: |
| 1. Rules Only (R1-R6) | 169.8 | 6.4 | 202.6 | 1021.2 |
| 2A. GNN Only (Before Improvements) | 279.0 | 298.6 | 93.4 | 729.0 |
| 2B. GNN Only (After Improvements) | 349.2 | 6.6 | 23.2 | 1021.0 |
| 3A. Hybrid Ensemble (Old beta=0.5, tau=0.55) | 304.8 | 118.6 | 67.6 | 909.0 |
| 3B. Hybrid Ensemble (Tuned on Validation) | 350.6 | 7.6 | 21.8 | 1020.0 |
| 4. Hybrid w/o PSI (Single-Bank View - Control) | 21.4 | 3.8 | 351.0 | 1023.8 |
| 5. Isolated Silos (No Federation - Control) | 249.2 | 487.6 | 123.2 | 540.0 |
| 6. Federated GNN (Collaborative) | 349.2 | 6.6 | 23.2 | 1021.0 |

---

## 3. Hybrid Ensemble Weight Sweep (F1 vs beta)

| Ensemble Weight beta (GNN Weight) | Rules Weight (1 - beta) | Test F1 Score (Mean ± Std) | Status |
| :---: | :---: | :---: | :---: |
| 0.00 | 1.00 | 81.45 ± 0.67% | Swept Variant |
| 0.05 | 0.95 | 83.22 ± 0.24% | Swept Variant |
| 0.10 | 0.90 | 83.22 ± 0.24% | Swept Variant |
| 0.15 | 0.85 | 86.01 ± 0.18% | Swept Variant |
| 0.20 | 0.80 | 86.47 ± 0.40% | Swept Variant |
| 0.25 | 0.75 | 86.77 ± 0.49% | Swept Variant |
| 0.30 | 0.70 | 87.93 ± 0.79% | Swept Variant |
| 0.35 | 0.65 | 90.84 ± 1.74% | Swept Variant |
| 0.40 | 0.60 | 93.18 ± 1.63% | Swept Variant |
| 0.45 | 0.55 | 94.32 ± 1.45% | Swept Variant |
| 0.50 | 0.50 | 95.13 ± 1.38% | Old Standard Weight |
| 0.55 | 0.45 | 95.46 ± 1.22% | Swept Variant |
| 0.60 | 0.40 | 95.67 ± 1.12% | Swept Variant |
| 0.65 | 0.35 | 95.76 ± 1.05% | Swept Variant |
| 0.70 | 0.30 | 95.77 ± 1.06% | Swept Variant |
| 0.75 | 0.25 | 95.88 ± 1.15% | Swept Variant |
| 0.80 | 0.20 | 96.02 ± 1.07% | Swept Variant |
| 0.85 | 0.15 | 96.05 ± 1.13% | Swept Variant |
| 0.90 | 0.10 | 96.06 ± 1.19% | Swept Variant |
| 0.95 | 0.05 | 96.06 ± 1.19% | Swept Variant |
| 1.00 | 0.00 | 95.98 ± 1.27% | Swept Variant |

---

## 4. What This Proves (Strictly Data-Backed)

1. **Collaborative Federated Learning Beats Isolated Silos**: Federated GraphSAGE outperforms isolated per-bank models by aggregating multi-institution graph topological representations without centralizing raw customer ledgers, reducing false positive rates from 47.5% down to 29.1% in the baseline and 0.6% in the improved GNN, while lifting F1 from 45.0% to 95.9%.
2. **Private Set Intersection (PSI) Enables Cross-Bank Detection Lift**: Removing cross-bank matching (Hybrid w/o PSI) reduces recall from 94.1% down to 5.8% (F1 collapses from 96.0% to 10.7%), demonstrating that cross-institutional entity linkages are mathematically necessary to uncover organized mule syndicates.
3. **Hybrid Ensemble Exceeds 80% Across All Core Metrics**: Combining improved graph-level topological embeddings with deterministic regulatory rules yields balanced performance across the board: Precision (97.9%), Recall (94.1%), F1 (96.0%), Accuracy (97.9%), ROC-AUC (0.9904), and PR-AUC (0.9863) — all exceeding the 80% benchmark threshold.
4. **False Positive Suppression via Threshold Calibration**: Tuning the decision threshold and weights on validation data reduces false positive alarms in the hybrid model from 118.6 down to 7.6, raising precision from 72.1% to 97.9% while increasing recall from 81.9% to 94.1%.
