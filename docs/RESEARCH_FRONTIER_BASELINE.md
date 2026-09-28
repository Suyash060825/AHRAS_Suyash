# AHRAS Research Frontier Baseline & Repository Integrity Audit

> **Platform**: Adaptive Hybrid Risk-Aware Security (AHRAS)  
> **Audit Timestamp**: 2026-09-28T10:05:00Z  
> **Lead Architect & Reviewer**: Antigravity Technical Audit Team  
> **Git Commit**: `4a0214e596aa39f52ca4b0cdd221f34d15fab322` (`new-features`)

---

## 1. Codebase Identification & Version Control

* **Active Git Branch**: `new-features`
* **Current Git Commit**: `4a0214e596aa39f52ca4b0cdd221f34d15fab322`
* **Repository Architecture**: Research-grade adaptive multimodal cyber defense platform implementing the full closed-loop architecture:
  $$\text{OBSERVE} \rightarrow \text{DETECT} \rightarrow \text{CORRELATE} \rightarrow \text{ASSESS} \rightarrow \text{EXPLAIN} \rightarrow \text{PREDICT} \rightarrow \text{RESPOND} \rightarrow \text{VERIFY} \rightarrow \text{LEARN} \rightarrow \text{ADAPT}$$
* **Core Philosophy**: Zero fabricated metrics; strictly reproducible benchmarks; deterministic safety boundaries; DRY_RUN response defaults.

---

## 2. Runtime & Execution Environment

Cataloged in [`evaluation/environment_manifest.json`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/evaluation/environment_manifest.json).

* **Hardware Host**: 11th Gen Intel(R) Core(TM) i5-11260H @ 2.60GHz (12 vCPUs), 31.0 GB RAM, 8.0 GB Swap
* **Operating System**: Linux 7.1.8-100.fc43.x86_64 (x86_64 architecture with glibc 2.42)
* **Compiler**: GCC 15.2.1 20260123 (Red Hat 15.2.1-7)
* **Python Runtime**: `3.14.6` (main, Jun 11 2026, 00:00:00)
* **Core Dependency Versions**:
  - `numpy`: `2.5.2`
  - `scikit-learn`: `1.9.0`
  - `scipy`: `1.18.0`
  - `pandas`: `3.0.5`
  - `pytest`: `9.1.1`
  - `pydantic`: `2.13.4`
  - `fastapi`: `0.141.1`
  - `uvicorn`: `0.52.1`

---

## 3. Dataset Registry & Ingestion Status

All datasets are cataloged in [`evaluation/data/dataset_registry.json`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/evaluation/data/dataset_registry.json).

| Dataset Name | Relative Path | File Size | SHA-256 Digest | Records | Status |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **CIC-IDS2017 (Wednesday)** | `data/cicids2017/Wednesday-workingHours.pcap_ISCX.csv` | 214.74 MB | `893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a` | 692,703 | **AUTHENTICATED / LOCAL** |
| **UNSW-NB15 (Part 1)** | `data/unsw_nb15/UNSW-NB15_1.csv` | 1.07 MB | `a71a9dc641b2878d63133d46a2a5bc8488e82010208cf237c89f3fa710b8c628` | 5,000 | **AUTHENTICATED / LOCAL** |
| **CIC-IDS2017 (Friday)** | `data/cicids2017/Friday-*.pcap_ISCX.csv` | — | Pending External Download | — | **QUEUED FOR INGESTION** |
| **CSE-CIC-IDS2018** | `data/cse_cic_ids2018/*.csv` | — | Pending External Download | — | **QUEUED FOR INGESTION** |

### Policy on Missing Datasets
When an external raw dataset is unavailable, the evaluation harness fails with an explicit `REAL_DATA_NOT_AVAILABLE` status. Synthetic data is NEVER silently substituted for authentic benchmarks.

---

## 4. Test Suite Execution & Integrity Check

* **Execution Command**: `pytest -q`
* **Test Inventory**: 61 test files under `tests/`
* **Pass Rate**: **100% (549 passed, 43 subtests passed = 592 test execution units in 69.95s)**
* **Regressions**: 0
* **Recent Verified Hardening Measures**:
  1. Safe DOM construction in `web/index.html` preventing XSS via malicious event entity strings.
  2. Jittered exponential WebSocket backoff (1s to 30s) preventing SOC dashboard thundering herd reconnects.
  3. Strict Content Security Policy (`script-src 'self'`, `object-src 'none'`) in `api/server.py`.
  4. Development mode warning for default credentials in `auth/manager.py`.
  5. Cryptographic Evidence Ledger TTL pruning (`prune_older_than_days()`) preventing unbounded memory growth.
  6. Conformal selective gate diagnostic property `calibration_status` detecting degenerate $\tau^* \ge 0.99$.
  7. Holm-Bonferroni correction applied across all 24 ablation components in `paper/main.tex`.

---

## 5. Empirical Benchmark Results (Authentic Real Datasets)

Recorded in [`evaluation/results/real_world_benchmarks_report.json`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/evaluation/results/real_world_benchmarks_report.json).

### 5.1 CICIDS2017 (Wednesday Working Hours — Botnet & DoS Traffic)
* **Sample Evaluated**: 10,000 authentic flow records (Stratified temporal sampling across 692,703 records).
* **Partitioning**: Entity-disjoint split (Train: 7,000, Val: 1,480, Test: 1,520).
* **Leakage Verification**: Passed (Zero train/test IP contamination).
* **Locked Hyperparameters**: $\tau^* = 0.22$, Platt calibration slope = 0.0881, intercept = -0.5596.
* **Test Performance**:
  - **F1 Score**: **0.6133** (95% CI: [0.5804, 0.6442]) (improved from previous 0.2408)
  - **Precision**: **0.5497**
  - **Recall**: **0.6935**
  - **Balanced Accuracy**: **68.20%**
  - **Conformal Gate $\tau^*$**: **0.6356** (VALID, non-degenerate; selective gate actively routes uncertain flows to Tier 2 inspection)
  - **Mean Latency**: **20.08 ms** (P95: 21.09 ms)
  - **External Baselines**: Random Forest F1=0.9849, Gradient Boosting F1=0.9741, Isolation Forest F1=0.0778.

### 5.2 UNSW-NB15 (Part 1 — 5,000 Authentic Records)
* **Sample Evaluated**: 5,000 authentic flow records.
* **Partitioning**: Chronological split (Train: 3,500, Val: 750, Test: 750).
* **Leakage Verification**: Passed.
* **Locked Hyperparameters**: $\tau^* = 0.34$, Platt calibration slope = 9.1342, intercept = -3.8689.
* **Test Performance**:
  - **F1 Score**: **0.9764** (95% CI: [0.9587, 0.9897])
  - **Precision**: **0.9538**
  - **Recall**: **1.0000** (100% attack capture)
  - **PR-AUC**: **0.9996**
  - **ROC-AUC**: **0.9998**
  - **False Positive Rate**: **1.60%**
  - **Balanced Accuracy**: **99.20%**
  - **Brier Score**: **0.0340**
  - **Conformal Gate $\tau^*$**: **0.2002** (VALID, confident autonomous gating)
  - **Mean Latency**: **21.32 ms** (P95: 21.57 ms)
  - **External Baselines**: **AHRAS (0.9764)** outperforms Random Forest (0.9526), Gradient Boosting (0.9556), and Isolation Forest (0.5179).

---

## 6. Throughput & Latency Hierarchy

AHRAS resolves the latency dilemma through a documented 3-tier processing pipeline:

```
+-------------------------------------------------------------------------+
| Tier 0: Fast Screening (Streaming Sketch + Early-Exit Router)           |
| Mean Latency: 0.04 ms (40 µs) | Throughput: >25,000 EPS                 |
| Screens benign background traffic with zero stateful ML overhead        |
+-------------------------------------------------------------------------+
                                    | (Ambiguous / Elevated Risk)
                                    v
+-------------------------------------------------------------------------+
| Tier 1: In-Memory Triage (Hybrid Tri-Engine + Conformal Gate)           |
| Mean Latency: 2.85 ms         | Throughput: >350 EPS                    |
| Signature + ML Anomaly + Statistical drift fusion + Split conformal     |
+-------------------------------------------------------------------------+
                                    | (High Risk / Multi-Hop / Novel Attack)
                                    v
+-------------------------------------------------------------------------+
| Tier 2: Full Analytical (GNN / TGNN + Security Twin + Causal XAI)       |
| Mean Latency: 19.61 - 21.32 ms| Throughput: ~50 EPS                     |
| Deep graph relational inference, Monte Carlo simulation, Causal DAG    |
+-------------------------------------------------------------------------+
```

---

## 7. Known Scientific Limitations & Target Research Frontiers

1. **Rule Diversity**: Current signature rules cover 23 MITRE ATT&CK techniques; production SOC operations require systematic coverage of at least 50+ ATT&CK techniques.
2. **Telemetry Incompleteness**: While two-tier host sensors handle syscalls and entropy, quantitative observability scores under network packet drops or missing host telemetry need explicit metric tracking (Telemetry Adequacy).
3. **Adversarial Perturbation Defense**: Evasion mutators (e.g. quote insertion, packet padding) currently evade signature matching at rates up to 26.67%, requiring robust feature normalization and representation regularization.
