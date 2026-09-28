# AHRAS Implementation Priority Matrix

**Platform**: Adaptive Hybrid Risk-Aware Security (AHRAS)  
**Standard**: Modular, Non-Monolithic, Auditable, Uncertainty-Bounded Defense Platform  
**Audit Baseline**: Commit `4a0214e`, 549 unit tests passed, 43 subtests passed (100% pass rate)

---

## 1. Overview & Gating Principles

All extensions and upgrades in AHRAS strictly adhere to the six permanent research pillars:
1. **Generalization** (cross-dataset resilience, non-IID federation)
2. **Adaptation** (5-bank replay, online weight learning, response efficacy learning)
3. **Unknown-Attack Detection** (OpenMax, Energy-based OOD representation)
4. **Relational Reasoning** (RGCN message passing, attack-path provenance reconstruction)
5. **Trustworthy Explanation** (computational fidelity, causal DAG, stability auditing)
6. **Safe Response** (split conformal prediction gating, DRY_RUN default, safety invariants)

### Absolute Engineering Gates:
- **Gate 0 (Audit & Reproducibility)**: Machine-readable environment and dataset manifests locked before code changes.
- **Gate 1 (Zero Regressions)**: Complete test suite must maintain 100% pass rate after every single commit.
- **Gate 2 (Deterministic Semantics)**: Never silently alter existing risk engine formulas or thresholds.
- **Gate 3 (Empirical Provenance)**: Never fabricate or transfer numerical benchmark results; write real JSON artifacts.

---

## 2. Phase-by-Phase Implementation Matrix

| Phase | Subsystem / Focus | Modules / Packages | Priority | Core Status | Extension Scope | Prerequisites |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **Phase 0** | **Repository Audit & Baseline Locking** | `evaluation/`, `docs/`, `tests/` | **P0** | **COMPLETED** | Dataset SHA-256 digests, environment manifest, capability matrices, test reproducibility verification. | None |
| **Phase 1** | **Alert Intelligence & Threat-Informed Coverage** | `alert_intelligence/`, `coverage/`, `detection/` | **P0** | **PARTIAL** | Unify raw alerts into high-context incident clusters; expand MITRE technique coverage from 23 to 50+; compute concrete implementation depth. | Phase 0 |
| **Phase 2** | **Adversarial Mutation & Robustness Hardening** | `adversarial/`, `detection/anomaly_engine/` | **P1** | **IMPLEMENTED** | Defend against quote insertion, padding, and jitter; harden Platt calibration under adversarial perturbation. | Phase 1 |
| **Phase 3** | **Real-Time Provenance & Graph Reasoning** | `provenance/`, `detection/gnn_engine.py` | **P1** | **IMPLEMENTED** | Scale heterogeneous DAG reconstruction to multi-hop process lineages; integrate Linux auditd/eBPF provenance. | Phase 1 |
| **Phase 4** | **Streaming Sketch & Fast-Path Filtering** | `detection/streaming_sketch.py`, `detection/model_router.py` | **P1** | **IMPLEMENTED** | Space-Saving top-k heavy hitters; sub-100µs Tier 0 screening to guarantee >25k EPS line rate. | Phase 0 |
| **Phase 5** | **Dynamic Resource Controller & Adaptive Scheduling** | `controller/`, `detection/` | **P2** | **IMPLEMENTED** | Latency budget enforcement (P95 < 25ms); adaptive tier skipping under high event velocity bursts. | Phase 4 |
| **Phase 6** | **XAI Reliability 2.0 & Counterfactual Validation** | `xai/reliability_audit.py`, `xai/counterfactual.py` | **P1** | **IMPLEMENTED** | Multi-dimensional audit (Sufficiency, Comprehensiveness, Rank Stability); real-time explanation confidence. | Phase 0 |
| **Phase 7** | **Security Twin & Closed-Loop Simulation** | `security_twin/`, `response/` | **P1** | **IMPLEMENTED** | Pre-execution Monte Carlo simulation of blast radius and collateral cost before autonomous response. | Phase 3, Phase 6 |
| **Phase 8** | **Active Deception & Information-Gain Honeypots** | `deception/`, `threat_intel/` | **P2** | **IMPLEMENTED** | Dynamic breadcrumb and lure deployment to confirm lateral movement; accelerate time-to-confirm. | Phase 7 |
| **Phase 9** | **Continual Learning & Gated Pseudo-Labeling** | `adaptive_learning/pseudo_labeler.py`, `weight_learner.py` | **P1** | **IMPLEMENTED** | Epistemic uncertainty-gated pseudo-labeling; 5-bank replay memory preventing catastrophic forgetting. | Phase 1, Phase 6 |
| **Phase 10**| **Strategic Pareto Scorecard & Production Readiness** | `evaluation/multi_objective_scorecard.py`, `api/`, `web/` | **P0** | **IMPLEMENTED** | Multi-objective optimization (12/12 targets); RBAC security; D3.js real-time topology; production Docker/K8s. | Phases 1–9 |

---

## 3. Detailed Work Breakdown & Deliverables for Immediate Phase 1

### Phase 1: Alert Intelligence Layer (`alert_intelligence/`)
- **Objective**: Eliminate alert fatigue by clustering raw low-level detector outputs into unified, high-context incident entities mapped to MITRE ATT&CK tactics.
- **Key Deliverables**:
  1. `alert_intelligence/clustering.py`: Temporal-spatial alert correlator using graph connectivity and Jaccard similarity.
  2. `alert_intelligence/prioritizer.py`: Exposure-aware and asset-criticality incident scoring.
  3. `alert_intelligence/deduplication.py`: Sliding-window deduplication with exponential suppression for repetitive noise.
  4. `tests/test_alert_intelligence.py`: Comprehensive test suite verifying zero alert loss, correct clustering, and sub-millisecond overhead.
  5. `evaluation/run_alert_intelligence_eval.py`: Experiment script producing `evaluation/results/ALERT_INTELLIGENCE_REPORT.json`.
- **Safety Invariant**: All alerts must preserve raw evidence references (`evidence_id`) linking back to the tamper-evident ledger.
