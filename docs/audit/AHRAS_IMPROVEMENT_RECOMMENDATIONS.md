# AHRAS Forensic Review & Strategic Improvement Recommendations
**Date:** September 30, 2026  
**Auditor:** Independent Technical & Scientific Audit  

---

## 1. Executive Summary

This forensic review audited the entirety of AHRAS—spanning core detection engines, mathematical risk formulation, XAI replay, temporal graph reasoning, safe response gating, telemetry fault handling, API RBAC, and web UI truthfulness.

The codebase is technically sound, highly modular, and enforces rigorous separation of concerns.

---

## 2. Priority Findings & Concrete Recommendations

### Recommendation 1: Distributed Storage Adapter for Rate Limiting
- **Problem:** Currently, API rate limiting uses a thread-locked in-memory sliding window cache.
- **Impact:** In multi-worker or Kubernetes distributed deployments, rate limits are tracked per process.
- **Affected Subsystem:** `api/server.py` (`RateLimitingMiddleware`).
- **Proposed Solution:** Introduce a Redis-backed token bucket adapter when `REDIS_URL` is set, preserving in-memory fallback for local dev.
- **Priority:** `MEDIUM` (Operational Hardening).

### Recommendation 2: Real-World Dataset Download Automation
- **Problem:** Remote evaluation datasets (`cse_cic_ids2018`, `ugr16`, `ctu13`, `iot23`, `lanl_cyber1`) correctly report `BLOCKED` in `DatasetRegistry`.
- **Impact:** Comprehensive cross-domain benchmarks require analysts to manually place raw CSVs in `data/`.
- **Affected Subsystem:** `evaluation/datasets/downloader.py`.
- **Proposed Solution:** Provide an optional CLI downloader helper script with interactive terms of service acceptance.
- **Priority:** `MEDIUM` (Evaluation Workflow).

### Recommendation 3: Continuous XAI Fidelity Ledger Export
- **Problem:** XAI fidelity verification is computed dynamically during test runs and on `/api/xai/fidelity`.
- **Impact:** Analysts in external SIEM/SOC platforms benefit from streaming immutable JSONL audit logs of DecisionTrace fidelity.
- **Affected Subsystem:** `xai/fidelity_ledger.py`.
- **Proposed Solution:** Add asynchronous periodic flushing of fidelity verification records to append-only JSONL files.
- **Priority:** `LOW` (Enterprise Observability).

---

## 3. Verified Safety & Scientific Invariants
1. **Zero Silent Synthetic Substitution:** Benchmarks strictly raise `[BLOCKED]` exceptions in `STRICT_REAL` mode if authentic datasets are absent.
2. **Deterministic Risk Replay:** `DecisionTrace` reconstructs exact mathematical scores with $< 10^{-4}$ absolute tolerance across all 10 paths.
3. **Fail-Closed SOAR Safety:** Autonomous containment is blocked on critical infrastructure, rate-limited, blast-radius bounded, and fully reversible.
4. **Strict RBAC & WebSocket Token Auth:** Every sensitive route and live stream validates permissions and signatures.
