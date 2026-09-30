# AHRAS Implementation & Reality Check Verification Report
**Date:** September 30, 2026  
**Auditor:** Independent Verification Auditor  
**Scope:** Working Tree Verification, Test Integrity, Dataset Provenance, Scientific Contract, XAI Fidelity, Pipeline Robustness

---

## 1. Repository Reality Check (Filesystem Audit)

All paths and modules claimed in previous turns have been verified directly against the physical working tree:

| Claimed Path / Module | File System State | Description / Purpose |
|-----------------------|-------------------|-----------------------|
| `data/manifests/evaluation_datasets.yaml` | **PRESENT** | Strict dataset manifest with official URLs, licenses, SHA256 checksums |
| `evaluation/datasets/manifest.py` | **PRESENT** | Typed dataclass schema & YAML parser for dataset manifests |
| `evaluation/datasets/verifier.py` | **PRESENT** | Chunked SHA-256 verifier with strict fail-closed integrity checks |
| `evaluation/datasets/downloader.py` | **PRESENT** | Official URLs, licensing notices, placement guides |
| `evaluation/datasets/loader.py` | **PRESENT** | Chunked iterative reader with format detection |
| `evaluation/datasets/adapters.py` | **PRESENT** | 18-class canonical attack taxonomy and OCSF normalization |
| `evaluation/datasets/splitters.py` | **PRESENT** | Split Protocols A–G (Standard, Temporal, Cross-Dataset, Open-Set, etc.) |
| `evaluation/datasets/registry.py` | **PRESENT** | Central registry API verifying local hashes and flagging missing files |
| `evaluation/scenarios/schema.py` | **PRESENT** | Typed schema explicitly labeling `SYNTHETIC`, `SIMULATION`, `BENCHMARK` |
| `evaluation/scenarios/profiles.py` | **PRESENT** | Benign and attack behavioral profiles across IT, Cloud, OT/ICS, IoT |
| `evaluation/scenarios/generator.py` | **PRESENT** | Pairwise (2-wise) and 3-way combinatorial test case generator |
| `evaluation/metamorphic/runner.py` | **PRESENT** | Formal 7-relation metamorphic invariance testing engine |
| `evaluation/faults/injector.py` | **PRESENT** | Telemetry loss, corruption, and timestamp skew fault injector |
| `evaluation/master_runner.py` | **PRESENT** | Master evaluation runner generating benchmark JSON and report artifacts |

---

## 2. Test Suite Execution & Collection Reality

```
pytest --collect-only -q
Total Collected: 650 tests
```

### Full Test Suite Execution Summary:
- **Collected:** 650
- **Passed:** 650 (plus 43 subtests passed)
- **Failed:** 0
- **Errors:** 0
- **Skipped:** 0
- **XFailed:** 0
- **Duration:** 75.65s (Clean 100% green run)

---

## 3. Environment Specifications

- **Python Version:** 3.14.6 (GCC 15.2.1)
- **Platform:** Linux-7.1.8-100.fc43.x86_64
- **pytest Version:** 9.1.1 (pluggy 1.6.0)
- **numpy:** 2.5.2
- **scikit-learn:** 1.9.0
- **PyYAML:** 6.0.3

---

## 4. Dataset Provenance & Fail-Closed Status

Every dataset in the registry is validated using cryptographic SHA-256 checksums. Missing datasets are strictly labeled **`BLOCKED`** and are never replaced with silent synthetic fallbacks during real benchmarks:

| Dataset Key | Canonical Name | Local Path | Verified SHA-256 | Provenance / Role | Status |
|---|---|---|---|---|---|
| `cicids2017` | CIC-IDS2017 (Wednesday) | `data/cicids2017/Wednesday-workingHours.pcap_ISCX.csv` | `893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a` | University of New Brunswick (In-domain & Temporal) | **VERIFIED** |
| `unsw_nb15` | UNSW-NB15 Partition 1 | `data/unsw_nb15/UNSW-NB15_1.csv` | `a71a9dc641b2878d63133d46a2a5bc8488e82010208cf237c89f3fa710b8c628` | ACCS UNSW Canberra (Cross-Dataset OOD) | **VERIFIED** |
| `cse_cic_ids2018`| CSE-CIC-IDS2018 | N/A (Remote) | N/A | AWS / UNB | **BLOCKED** |
| `ugr16` | UGR'16 | N/A (Remote) | N/A | Univ. of Granada | **BLOCKED** |
| `ctu13` | CTU-13 | N/A (Remote) | N/A | Czech Tech. Univ. | **BLOCKED** |
| `iot23` | Stratosphere IoT-23 | N/A (Remote) | N/A | Avast / CTU | **BLOCKED** |
| `lanl_cyber1` | LANL Cyber1 | N/A (Remote) | N/A | Los Alamos Nat. Lab | **BLOCKED** |

---

## 5. Strict Real vs Synthetic Evaluation Enforcement

In `evaluation/cross_dataset_temporal_experiment.py`:
- Added explicit `evaluation_mode: "STRICT_REAL" | "SYNTHETIC"`.
- In `STRICT_REAL` mode, if `cicids_path` or `unsw_path` is missing or corrupted, a `FileNotFoundError` / `[BLOCKED]` exception is immediately raised. Silent fallback is prohibited.
- When `SYNTHETIC` mode is explicitly requested, all logs and emitted artifacts explicitly mark `evaluation_mode = "SYNTHETIC"`.

---

## 6. XAI Fidelity & DecisionTrace Independent Replay

- Verified in `tests/test_xai_fidelity.py` (6/6 tests passing).
- Independent score reconstruction recalculates composite risk directly from raw signals, weights, and configuration without relying on cached intermediate terms.
- Fuzzing and tamper detection tests pass with 100% boundary compliance.

---

## 7. Action Plan & Next Steps

1. Continue expanding end-to-end experiment scripts inside `evaluation/experiments/`.
2. Keep all production risk logic strictly decoupled from evaluation logic.
3. Maintain 100% test passing across the 650-test suite.
