# AHRAS Forensic Findings & Subsystem Quality Classification
**Date:** September 30, 2026  
**Auditor:** Independent Technical & Scientific Audit Team  

---

## 1. Classification Methodology

Findings across the codebase are classified into four strict priority tiers:
- **P0 (Critical):** Correctness/security/research-integrity blockers (crashes, bypasses, silent fake benchmarks).
- **P1 (High):** Major robustness or accuracy issues (graph ground truth gaps, telemetry decoupling).
- **P2 (Medium):** Meaningful quality/performance/UX improvements (dashboard clarity, automated quality gates).
- **P3 (Low):** Minor polish and documentation maintenance.

---

## 2. Audit Findings & Resolution Status

| Finding ID | Subsystem | Severity | Description | Current Status |
|---|---|---|---|---|
| **F-01** | `evaluation/cross_dataset_temporal_experiment.py` | **P0** | Silent synthetic fallback risk during real dataset benchmarks. | **RESOLVED:** Added explicit `STRICT_REAL` fail-closed mode raising `FileNotFoundError` (`[BLOCKED]`). |
| **F-02** | `api/server.py` & `config/settings.py` | **P0** | Insecure production startup & unauthenticated WebSocket streaming. | **RESOLVED:** Fail-closed env checks & JWT token validation on `/ws/live-soc` and all sensitive endpoints. |
| **F-03** | `xai/computational_fidelity.py` | **P1** | Replay of complex composite risk scores must be independent of cached terms. | **RESOLVED:** 10-path independent reconstruction verified with $< 10^{-4}$ tolerance ($100\%$ pass rate). |
| **F-04** | `response/safety_invariants.py` | **P1** | Automated response risk on critical domain infrastructure. | **RESOLVED:** Hard barrier for critical assets, $20\%$ blast radius limit, and mandatory compensating reversibility. |
| **F-05** | `graph/` | **P1** | Graph evaluation against formal ground-truth topology. | **ACTION REQUIRED:** Build controlled graph ground truth evaluator (`evaluation/experiments/graph_ground_truth.py`). |
| **F-06** | `scripts/` | **P2** | Automated quality gate check script for CI/CD and pre-commit validation. | **ACTION REQUIRED:** Implement unified `scripts/ahras_quality_gate.py`. |
