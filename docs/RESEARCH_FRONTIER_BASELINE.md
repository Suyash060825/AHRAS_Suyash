# AHRAS Research Frontier Baseline & Repository Integrity Audit

> **Platform**: Adaptive, Hybrid, Risk-Aware Security Intelligence and Defense Platform  
> **Audit Timestamp**: 2026-09-26T16:40:00Z  
> **Lead Architect & Reviewer**: Antigravity Technical Audit Team

---

## 1. Codebase Identification & Version Control

* **Active Git Branch**: `new-features`
* **Current Git Commit**: `7ba27f516f43704833f3893be684342203fd3a1f`
* **Repository Architecture**: Multi-module cyber defense platform with OCSF standard telemetry, hybrid detection engines (signature, ML anomaly, statistical Welford drift), dynamic trust scoring, causal XAI, conformal risk gating, and closed-loop SOAR orchestration.

---

## 2. Runtime & Execution Environment

* **Python Version**: `3.14.6` (GCC 15.0.1 20250116, 64-bit Linux)
* **Operating System**: Linux x86_64
* **Core Dependency Versions**:
  - `numpy`: `2.3.0`
  - `scikit-learn`: `1.9.0`
  - `scipy`: `1.18.0`
  - `pandas`: `3.0.5`
  - `pytest`: `9.1.1`
  - `pydantic`: `2.13.4`
  - `fastapi`: `0.135.3`
  - `uvicorn`: `0.52.1`

---

## 3. Dataset Registry & Ingestion Status

All datasets are cataloged in [`evaluation/data/dataset_registry.json`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/evaluation/data/dataset_registry.json).

| Dataset Name | Expected Relative Path | File Size | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- |
| **CIC-IDS2017 (Wednesday)** | `data/cicids2017/Wednesday-workingHours.pcap_ISCX.csv` | 214.74 MB | `893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a` | **AUTHENTICATED / LOCAL** |
| **UNSW-NB15 (Part 1)** | `data/unsw_nb15/UNSW-NB15_1.csv` | 1.07 MB | `a71a9dc641b2878d63133d46a2a5bc8488e82010208cf237c89f3fa710b8c628` | **AUTHENTICATED / LOCAL** |
| **CIC-IDS2017 (Friday/Thursday)** | `data/cicids2017/Friday-*.pcap_ISCX.csv` | Absent | Pending | **PENDING EXTERNAL DOWNLOAD** |
| **CSE-CIC-IDS2018** | `data/cse_cic_ids2018/*.csv` | Absent | Pending | **PENDING EXTERNAL DOWNLOAD** |

### Policy on Missing Datasets
When an external raw dataset is unavailable, the evaluation harness fails with an explicit `REAL_DATA_NOT_AVAILABLE` status. Synthetic data is NEVER silently substituted for authentic benchmarks.

---

## 4. Test Suite Execution & Integrity Check

* **Execution Command**: `pytest -q`
* **Total Collected Tests**: 472 test cases across 48 test suites.
* **Pass Rate**: **100% (472 passed in 73.93s)**.
* **Regressions**: 0.
* **Recent Critical Fixes Verified by Tests**:
  1. `TemporalGNN` deterministic hash-based spectral projection verified in `tests/test_graph_correlation_evaluation.py`.
  2. `SecurityGNN` message-passing supervised training verified.
  3. Slow HTTP / DoS connection-holding signatures verified against authentic Wednesday test split.
  4. SQLite-backed persistent token revocation store verified in `tests/test_rbac.py` (`test_07_persistent_token_revocation`).
  5. Default credential security gating behind `DEV_MODE` verified.

---

## 5. Empirical Benchmark Results

### 5.1 Authentic Real Benchmark Evaluation (Wednesday Working Hours)
* **Records Evaluated**: 10,000 authentic flow records (Stratified temporal sampling, Stride 69 across 692,703 records).
* **Partitioning**: Chronological split (70% Train = 7,000, 15% Val = 1,500, 15% Test = 1,500).
* **Leakage Verification**: Passed (Zero train/test entity contamination).
* **Locked Hyperparameters on Validation**: $\tau^* = 0.300$, Platt calibration slope = 5.0.
* **Test Performance**:
  - **Attack Recall**: **98.57%** (69/70 attacks caught)
  - **Balanced Accuracy**: **84.11%**
  - **ROC-AUC**: **0.7358**
  - **PR-AUC**: **0.0998**
  - **Test F1 Score**: **0.2408** (95% CI: [0.1948, 0.2845])
  - **Mean Decision Latency**: **19.61 ms** (P95: 19.99 ms)
  - **External Baselines**: Random Forest F1=0.5385, Gradient Boosting F1=0.5098, Isolation Forest F1=0.0000.

### 5.2 Cross-Dataset & Temporal Shift Evaluation (EXP-02)
* **In-Domain F1**: 0.7209
* **Temporal Shift F1**: 0.6667 (degradation bounded to 7.52% $\le 15\%$)
* **Cross-Dataset F1 (UNSW-NB15)**: 0.8968 (zero degradation vs baseline models which collapsed to 0.0)
* **Paired Permutation Significance**: $p = 0.0001$.

---

## 6. Path Portability & Hardcoding Audit

* **Repository Root Handling**: Standardized to `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`.
* **Hard-coded Paths**: No `/home/...` or machine-specific absolute paths exist in execution source code. Paths default to environment variables (`CICIDS2017_PATH`, `UNSW_PATH`, `AHRAS_TOKEN_BLACKLIST_DB`) with clean fallback to relative `./data/` paths.
* **Artifact Output Paths**: Re-routed to `evaluation/results/` and `docs/`.

---

## 7. Known Limitations & Scientific Boundaries

1. **UNSW-NB15 Scale**: Only `UNSW-NB15_1.csv` (1.07 MB sample) is locally present; full multi-file evaluation requires downloading parts 2–4.
2. **Technique-Level vs Implementation-Level Coverage**: Current rules tag MITRE ATT&CK techniques at the aggregate level (e.g. `T1071`, `T1499`), but do not yet track behavioral implementation variants (e.g. PowerShell vs schtasks vs WMI task execution). This motivates **Research Frontier A (Threat-Informed Detection Coverage)**.
3. **Telemetry Completeness**: OCSF event normalizers handle available fields, but do not yet explicitly quantify observability degradation when critical fields are dropped by upstream sensors. This motivates **Research Frontier B (Telemetry Adequacy)**.
4. **Adversarial Robustness**: Implicit robustness is provided via conformal abstention, but quantitative robustness envelopes under bounded traffic perturbation have not yet been evaluated systematically. This motivates **Research Frontier D (Adversarial Mutation Lab)**.
