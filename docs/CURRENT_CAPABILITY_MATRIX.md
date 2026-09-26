# AHRAS Current Capability Matrix & Subsystem Audit

**Platform**: Adaptive, Hybrid, Risk-Aware Security (AHRAS)  
**Audit Date**: 2026-09-26  
**Status Categories**:
- **A. Already implemented**: Full production/research-grade code operational with unit & integration tests.
- **B. Partially implemented**: Core algorithm functional, but requires modular hardening, scaling, or interface completeness.
- **C. Planned but not implemented**: Architecture roadmap artifact defined; code stub or placeholder only.
- **D. Implemented but not properly evaluated**: Implementation exists but relies on synthetic data or lacks real-world benchmarking.
- **E. Completely missing**: Not present in the repository.

---

## 1. Capability Matrix

| Capability | Existing Module | Status | Tests | Evaluation | Keep/Extend/Replace |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Detection (Signatures)** | `detection/signature_engine/rules.py` | **A** | `tests/test_module1.py`, `tests/test_full_system.py` | Baseline $B_0$, 23 curated MITRE rules | **Keep & Extend** (Add encrypted metadata & periodicity signatures) |
| **Detection (ML Anomaly)** | `detection/anomaly_engine/ml_engine.py` | **A** | `tests/test_module1.py`, `tests/test_ablation_suite.py` | Tri-model ensemble (IF, AE, OC-SVM) | **Keep** (Maintain standardized 14-dim contract) |
| **Detection (Statistical Drift)** | `detection/statistical_engine/stat_engine.py` | **A** | `tests/test_module1.py` | Welford streaming running mean, variance, EWMA | **Keep** (Ultra-low latency streaming path) |
| **Detection (Open-Set / OOD)** | `detection/representation_engine.py` | **A** | `tests/test_open_set_detection_evaluation.py` | Phase 8 EXP-03 (`OPEN_SET_DETECTION_REPORT.json`): 96.54% Zero-Day Recall, 4.61% FUR | **Keep** (Crucial for unknown attack pillar) |
| **Detection (Hybrid Combiner)**| `detection/hybrid_engine.py` | **A** | `tests/test_module1.py` | Multi-engine fusion (0.50 Sig / 0.30 ML / 0.20 Stat) | **Keep & Extend** (Add early-exit routing stage) |
| **Risk Engine** | `detection/risk_engine.py` | **A** | `tests/test_module4.py`, `tests/test_computational_fidelity.py` | 8-term weighted formula, explicit ordering, $\Delta \le 10^{-6}$ replay | **Keep** (Do NOT alter arithmetic semantics) |
| **Evidence Ledger** | `ahras/evidence/ledger.py`, `ahras/evidence/models.py` | **B** | `tests/test_evidence_ledger.py` | In-memory cryptographic hash chain; lacks TTL & disk checkpointing | **Extend** (Add epoch Merkle block flushing & TTL retention) |
| **Graph (RGCN Message Passing)**| `detection/gnn_engine.py` | **B** | `tests/test_graph_correlation_evaluation.py` | Numpy RGCN layer with truncated backprop; needs proper multi-hop learning | **Extend** (Keep numpy contract; formalize inductive relational projection) |
| **Graph (Temporal TGNN)** | `graph/tgnn.py` | **B** | `tests/test_module1.py` | Heuristic exponential time-decay vector drift | **Extend** (Formalize temporal state transition matrices) |
| **Graph (Path Reasoning)** | `detection/attack_path.py` | **A** | `tests/test_attack_path.py` | Noisy-OR Bayesian path accumulation, campaign attribution | **Keep & Extend** (Form foundation for provenance attack reconstruction) |
| **Historical Context** | `historical_risk/engine.py` | **A** | `tests/test_historical_context_evaluation.py`, `tests/test_historical_risk.py` | Phase 6 EXP-06-HIST: Recidivism boost (7d, 30d, >30d decay) | **Keep** (Stable recidivism memory) |
| **Threat Intelligence** | `threat_intel/intel.py`, `threat_intel/stix_ingestor.py` | **A** | `tests/test_threat_intel.py` | STIX 2.1 parser, TAXII 2.1 poll, memory cache | **Keep & Extend** (Integrate with adaptive deception trigger) |
| **XAI (Computational Fidelity)**| `xai/computational_fidelity.py`, `xai/fidelity_ledger.py` | **A** | `tests/test_xai_fidelity.py`, `tests/test_computational_fidelity.py` | 10-path sum-check, 10,000 trace replay pass rate 100% | **Keep** (Foundation of explainability contract) |
| **XAI (Causal Explainer DAG)** | `xai/causal_explainer.py` | **A** | `tests/test_causal_explainer.py` | Mechanistic DAG, finite-difference sensitivity gradients | **Keep & Extend** (Bridge to Explanation Reliability Audit 2.0) |
| **XAI (Faithfulness / Monotonicity)**| `xai/faithfulness.py` | **B** | `tests/test_module1.py` | Feature deletion & insertion monotonicity, Gaussian noise stability | **Extend** (Upgrade to multidimensional XAI Reliability Audit 2.0) |
| **Counterfactuals** | `xai/counterfactual.py` | **A** | `tests/test_response_policy_gating.py` | Minimal feature intervention delta computation on DecisionTrace | **Keep & Extend** (Integrate with Security Twin response counterfactuals) |
| **Active Learning** | `adaptive_learning/active_learner.py`| **A** | `tests/test_active_learner.py`, `tests/test_adaptive_learning.py` | Acquisition function $a(x) = u \cdot h \cdot (1 + \text{ood})$, budget window | **Keep & Extend** (Add confidence-gated pseudo-labeling) |
| **Continual Learning** | `adaptive_learning/weight_learner.py`| **A** | `tests/test_continual_learning_evaluation.py` | 5-bank memory replay (Recent, Attack, Hard-Neg, Drift, Prototype) | **Keep & Extend** (Add pseudo-label provenance tracking) |
| **Federated Learning** | `federated/fed_learning.py` | **B** | `tests/test_federated_evaluation.py`, `tests/test_federated_reputation.py` | Coordinate-median, FedKD, client reputation; lacks mTLS/token enforcement | **Extend** (Add cryptographic token/mTLS enforcement) |
| **Multimodal Encoder** | `detection/multimodal_encoder.py` | **A** | `tests/test_multimodal_fusion_evaluation.py`, `tests/test_multimodal_encoder.py` | Phase 14 EXP-10: 4-modality attention fusion, 50% missingness resilience | **Keep** (Core representation pillar) |
| **Host Telemetry** | `sensors/two_tier_telemetry.py`, `sensors/host_agent.py` | **A** | `tests/test_host_telemetry_evaluation.py` | Phase 13 EXP-09: Tier 1 syscalls, Tier 2 Shannon entropy (7.20 threshold), process lineage | **Keep** (High-throughput endpoint collector) |
| **Forecasting** | `forecast/predictor.py` | **A** | `tests/test_proactive_forecasting_evaluation.py`, `tests/test_forecast.py` | Phase 12 EXP-08: Holt linear causal smoothing, hazard thresholding ($\ge 3$ lead time) | **Keep** (Zero lookahead validated) |
| **Conformal Selective Gate** | `detection/selective_gate.py` | **A** | `tests/test_conformal_autonomy_evaluation.py`, `tests/test_selective_gate.py` | Phase 10 EXP-06: Split conformal quantile calibration, 7 action tiers | **Keep** (Safe autonomy foundation) |
| **Response Orchestrator** | `response/orchestrator.py` | **B** | `tests/test_response_policy_gating.py` | RASE expected utility policy, staged queue; lacks learned observed feedback | **Extend** (Add Response Efficacy Learning loop) |
| **Deception / Honeypot** | `deception/honeypot_manager.py` | **A** | `tests/test_adaptive_deception.py` | Phase 8 EXP-18: DeceptionValue optimization, 4 dynamic lure types, Time-to-Confirm 4.0 -> 1.3 steps, 85% FP cut, Delta U=0.49 | **Implemented (Phase 8)** |
| **RAG / LLM Threat Narrator** | `xai/llm_narrator.py` | **B** | `tests/test_adversarial_redteam.py` | Template fallback + local Ollama; prompt injection sanitized; lacks provenance tag | **Extend** (Surface provenance flag in UI/API) |
| **Dashboard & API** | `api/server.py`, `web/index.html` | **B** | `tests/test_module4.py` | FastAPI 6.1.0, WebSockets; needs RBAC dependencies attached to routes | **Extend** (Enforce RBAC dependencies, add new operational views) |
| **Explanation Reliability 2.0**| `xai/reliability_audit.py` | **A** | `tests/test_xai_reliability_audit.py` | Phase 1 EXP-11: Multi-dimensional XAI audit, Sufficiency k in {3,5,10}, Comprehensiveness, Rank Stability J=0.88, Monotonicity | **Implemented (Phase 1)** |
| **Security Twin Simulation** | `security_twin/` | **A** | `tests/test_security_twin.py` | Phase 2 EXP-12: Mean Optimal Containment 73.6%, Risk Reduction 85.0%, Path Breakage 100%, Blast-radius evaluated, N=500 MC sampling | **Implemented (Phase 2)** |
| **Provenance Attack Reconstruction**| `provenance/` | **A** | `tests/test_provenance_reconstruction.py` | Phase 3 EXP-13: 12-node/12-edge heterogeneous DAG, Clean Graph F1 1.000, 50% Missingness F1 0.612, Path Completeness 100%, Gap Reasoning | **Implemented (Phase 3)** |
| **Temporal Epistemic Instability**| `instability/` | **A** | `tests/test_temporal_instability.py` | Phase 4 EXP-14: 5-component volatility metric, 100% FAIR reduction (17->0), 100% oscillating abstention recall, ECE 0.207->0.184 | **Implemented (Phase 4)** |
| **Early-Exit Model Router** | `detection/model_router.py` | **A** | `tests/test_model_router.py` | Phase 5 EXP-15: 4-stage adaptive cascade, 2.86x throughput speedup (54.8 -> 157.0 EPS), 65.1% latency reduction, P50 17.85ms -> 0.05ms, Zero F1 loss (0.2869) | **Implemented (Phase 5)** |
| **Streaming Sketch Fast Path** | `detection/streaming_sketch.py` | **A** | `tests/test_streaming_sketch.py` | Phase 6 EXP-16: Count-Min + HLL fan-out, O(1) space (0.83MB vs 2.94MB exact), 10.3k EPS, P50 91.6us, HH F1 0.8889, 21.6% workload screened | **Implemented (Phase 6)** |
| **Encrypted Session Intelligence**| `detection/encrypted_session.py` | **A** | `tests/test_encrypted_session.py` | Phase 7 EXP-17: Payload-blind sequence dynamics + IAT autocorrelation, F1 0.9362 vs 0.0000 Flow-Only, 90.5% Unknown Attack Recall, 3.9k SPS | **Implemented (Phase 7)** |
| **Adaptive Deception Sensor** | `deception/honeypot_manager.py` | **A** | `tests/test_adaptive_deception.py` | Phase 8 EXP-18: Bayesian Info-Gain lure optimization, Time-to-confirmation 4.0 -> 1.3 steps (-2.7 steps), 100% path completeness, 85% FP reduction | **Implemented (Phase 8)** |
| **Response Efficacy Learning**| `response/efficacy_learner.py` | **A** | `tests/test_response_efficacy.py` | Phase 9 EXP-19: Online Bayesian Beta beliefs, Twin counterfactual simulation, Zero safety violations (0.0%), +545.7% security utility vs static SOAR | **Implemented (Phase 9)** |
| **Confidence-Gated Pseudo-Labeling**| `adaptive_learning/pseudo_labeler.py`| **A** | `tests/test_pseudo_labeler.py` | Phase 10 EXP-20: Multi-condition epistemic gating, 1.0000 Hold-out Macro F1, 100.0% pseudo purity, 0 OOD pollution, continual replay isolation | **Implemented (Phase 10)** |

---

## 2. Subsystem Architectural Review

### 2.1 Strengths
1. **Mathematical Cleanliness of DecisionTrace**: The risk engine (`detection/risk_engine.py`) explicitly records all intermediate terms, allowing exact computational reconstruction within $\Delta \le 10^{-6}$.
2. **Empirical Rigor in Existing Extensions**: Multi-modal fusion (`EXP-10`), Host telemetry (`EXP-09`), Holt forecasting (`EXP-08`), and Conformal gating (`EXP-06`) possess rigorous paired permutation testing, Holm-Bonferroni correction, and well-structured report schemas.
3. **Multi-Memory Continual Learning**: The 5-compartment memory replay (`adaptive_learning/weight_learner.py`) is well-conceived, segregating prototypes, hard negatives, and drift instances.

### 2.2 Critical Implementation Gaps Addressed in Next-Gen Roadmap
1. **XAI Faithfulness Gap**: Existing faithfulness checks (`xai/faithfulness.py`) only compute basic deletion step counts and Gaussian Jaccard stability; it lacks Sufficiency, Comprehensiveness, Spurious Feature Robustness, and Cross-Run Consensus.
2. **Simulation / Pre-Execution Verification Gap**: No safe digital twin exists to test "What happens if action $A$ is taken?" before executing mitigation.
3. **Graph Scope Gap**: Graph analysis operates on 2-hop entity neighborhoods without reconstructing full attack provenance chains across processes, sockets, files, and techniques.
4. **Latency / Throughput Scalability**: Every event currently triggers the full detection pipeline without early-exit filtering or streaming sketch summarization.
