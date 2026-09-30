# AHRAS Deep Baseline & System Specifications
**Date:** September 30, 2026  
**Auditor:** Comprehensive System Verification Team  
**Git Commit:** `01fbb82f900edcb8e5a87ce201da9dcf73167dc0`  
**Git Branch:** `home2`  

---

## 1. Execution Environment & Dependencies

- **Python Version:** `3.14.6`
- **Platform:** `Linux-7.1.8-100.fc43.x86_64` (glibc 2.42)
- **Pytest Version:** `9.1.1`
- **Key Python Libraries:**
  - `numpy`: `2.5.2`
  - `scikit-learn`: `1.9.0`
  - `fastapi`: `0.115.0+`
  - `pyyaml`: `6.0.3`
  - `pydantic`: `2.x`

---

## 2. Test Suite & Benchmark Baseline

- **Collected Tests:** `650` tests (`pytest --collect-only -q`)
- **Execution Result:** `650 passed, 43 subtests passed in 75.65s (100% green)`
- **Evaluation Engine Execution:**
  - `evaluation/master_runner.py`: Successful zero-error run.
  - Verified Pairwise Scenarios: 90
  - Verified 3-Way Scenarios: 30
  - Metamorphic 7-Relation Pass Rate: 100%
  - Telemetry Fault Tolerancing: Verified (loss, disorder, corruption).

---

## 3. Dataset Registry & Provenance Baseline

- **Locally Staged & Cryptographically Verified Datasets:**
  1. `cicids2017`: `Wednesday-workingHours.pcap_ISCX.csv` (SHA-256: `893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a`, 225 MB)
  2. `unsw_nb15`: `UNSW-NB15_1.csv` (SHA-256: `a71a9dc641b2878d63133d46a2a5bc8488e82010208cf237c89f3fa710b8c628`, 1.1 MB)
- **Remote / Staged Missing Datasets:**
  - `cse_cic_ids2018`, `ugr16`, `ctu13`, `iot23`, `lanl_cyber1`: Explicitly **`BLOCKED`** in `DatasetRegistry`. Silent synthetic fallback in strict real benchmark mode is prohibited.
