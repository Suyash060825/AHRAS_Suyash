# AHRAS Evaluation System — Current State Audit
**Audit Date:** September 30, 2026  
**Auditor:** AHRAS Principal Research Systems Engineer & Evaluation Architect  
**Scope:** Exhaustive audit of all evaluation scripts, real dataset subsets, synthetic generators, metrics calculations, manifests, and test suites.

---

## 1. Executive Summary

The AHRAS repository currently possesses an extensive collection of specialized evaluation scripts and benchmark modules across 30+ files in `evaluation/`, backed by unit and integration tests in `tests/`. The system has made massive progress in replacing mock metrics with live algorithmic computations.

However, the current evaluation subsystem is organized primarily as a flat collection of one-off evaluation runners rather than a unified, modular, protocol-driven evaluation architecture. To support defensible research claims, a structured dataset registry, rigorous split protocols (temporal, cross-dataset, open-set), combinatorial scenario generation (2-wise/3-wise), metamorphic testing, and explicit lineage tracking are required.

---

## 2. Inventory of Current Evaluation Artifacts

### 2.1 Real Datasets Present
| Dataset | Location | File Size | Description & Status |
| :--- | :--- | :--- | :--- |
| **CIC-IDS2017** | `data/cicids2017/Wednesday-workingHours.pcap_ISCX.csv` | 225.1 MB | Full Wednesday subset containing DoS, Heartbleed, and Benign network flows. Raw CSV format. |
| **UNSW-NB15** | `data/unsw_nb15/UNSW-NB15_1.csv` | 1.1 MB | Extracted subset containing Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance. |

### 2.2 Synthetic Data Generators
* `evaluation/generate_synthetic_dataset.py`: Generates structured multi-attack network/host records for unit test baselines and deterministic reproducibility.
* `evaluation/run_controlled_synthetic.py`: Controlled perturbation tests for parameter sensitivity.

### 2.3 Existing Evaluation Scripts & Test Suites
1. **Core Pipeline & Multi-Objective:**
   * `evaluation/run_comprehensive_research.py`: Master live research pipeline producing `RESULTS.json`.
   * `evaluation/run_strategic_pareto_scorecard.py`: Multi-objective Pareto frontier scorecard ($RASE$, Brier, F1, Latency).
2. **Detection & Graph Reasoning:**
   * `evaluation/run_attack_path_reasoning.py`: Temporal GNN kill-chain stage progression.
   * `evaluation/run_graph_correlation_evaluation.py`: Entity subgraph clustering and cross-hop correlation.
   * `evaluation/run_detection_coverage.py`: MITRE ATT&CK technique coverage audit.
3. **Robustness & Temporal Adaptation:**
   * `evaluation/run_evasion_robustness.py`: Feature perturbation, dead-zone masking, jitter attacks.
   * `evaluation/run_prequential_drift.py`: Sequential streaming evaluation under concept drift.
   * `evaluation/run_continual_learning_eval.py`: Replay buffer and catastrophic forgetting prevention.
4. **Active Defense & Security Twin:**
   * `evaluation/run_security_twin_evaluation.py`: Digital twin state synchronization and lookahead simulation.
   * `evaluation/run_safe_response.py`: Conformal gating ($\tau^*$) and blast-radius bounding.
   * `evaluation/run_resilience_recovery_eval.py`: 6-stage MTTC/MTTR recovery tracking.
5. **XAI & Explainability:**
   * `evaluation/xai_fidelity_experiment.py`: Analytical Shapley sum-check verification.
   * `evaluation/xai_reliability_audit.py`: Explanation stability and permutation robustness.

---

## 3. Current Preprocessing & Split Methodology

* **OCSF Normalization:** Handled via `normalizer/ocsf_normalizer.py` and `evaluation/dataset_loader.py`.
* **Splits:** Currently implemented mostly as train/test splits within individual scripts or 5-fold cross validation. Protocol-specific withholdings (e.g. cross-dataset, temporal cutoff, open-set family holdout) are implemented script-by-script rather than via a standardized splitting engine.

---

## 4. Current Manifests and Metadata
* `evaluation/results/RESULTS.json`: Live consolidated benchmark metrics.
* `evaluation/results/experiment_manifest.json`: Checksum records for table outputs.
* `CLAIMS_MANIFEST_FINAL.json`: Central claims verified by executable evidence.
