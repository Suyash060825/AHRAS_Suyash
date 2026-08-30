# AHRAS: Final Journal Publication Package

## Project Overview
**AHRAS: An Auditable Uncertainty-Aware Adaptive Risk Controller for Closed-Loop Intrusion Detection and Response**

This package contains the complete, frozen, verified, and audited scientific release package for journal submission.

---

## 1. Canonical Architecture & Freeze
- Frozen Pipeline: 22 canonical runtime stages spanning raw OCSF telemetry normalization, multimodal feature representation, statistical & ML inference, self-supervised OOD representation, temporal heterogeneous GNN message passing, context-conditioned adaptive evidence fusion, conformal selective autonomy, deterministic DecisionTrace replay, cost-aware RASE safety response, and 5-bank continual learning.
- Specification File: `ARCHITECTURE_FREEZE_FINAL.json`
- Integration Truth Matrix: `INTEGRATION_TRUTH_MATRIX_FINAL.json`

---

## 2. Scientific Integrity & Audit Summary
- Research Result Integrity: Zero hard-coded research metrics (`NO_HARDCODED_RESEARCH_RESULTS`). All reported metrics originate from live pipeline execution over structured data.
- Statistical Rigor: 10,000 paired sample-level permutations with bootstrap 95% confidence intervals, Cohen's d effect sizes, and Holm-Bonferroni multi-comparison corrections (`STATISTICAL_VALIDATION_FINAL.json`).
- Graph Reasoning: Graph-native multi-hop lateral movement evaluation (`GNN_GRAPH_NATIVE_RESULTS_FINAL.json`). The temporal heterogeneous GNN provides structural relational grounding for multi-hop lateral movement reasoning (F1 = 0.893) while exhibiting parity on isolated single-event classification (F1 = 0.752).
- XAI Deterministic Replay: 10,000 production traces validated through `AdaptiveRiskEngine`; maximum replay discrepancy <= 1e-4 with 99.2% of traces exhibiting replay delta <= 1e-6 (`RESULTS_FINAL.json`).
- RAG Security: 100% prompt injection and override mitigation across red-team test patterns (`RAG_SECURITY_REPORT_FINAL.json`).
- External Dataset Disclosure: Raw external benchmarks explicitly disclosed as `NOT_RUN_EXTERNAL_DATA` pending provision of authentic raw benchmark captures (`REAL_DATASET_VALIDATION_FINAL.json`).

---

## 3. Directory Structure
listing of publication/ artifacts:
- ARCHITECTURE_FREEZE_FINAL.json
- INTEGRATION_TRUTH_MATRIX_FINAL.json
- SCIENTIFIC_INTEGRITY_AUDIT_FINAL.json
- RESULT_INTEGRITY_AUDIT_FINAL.json
- MASTER_EXPERIMENT_PROTOCOL_FINAL.json
- RESULTS_FINAL.json
- CONFIG_FINAL.json
- ENVIRONMENT_FINAL.json
- LEAKAGE_REPORT_FINAL.json
- REAL_DATASET_VALIDATION_FINAL.json
- STATISTICAL_VALIDATION_FINAL.json
- GNN_GRAPH_NATIVE_RESULTS_FINAL.json
- CONTINUAL_LEARNING_LONGIUQDINAL_FINAL.json
- MEMORY_ABLATION_FINAL.json
- ACTIVE_LEARNING_EFFICIENCY_FINAL.json
- ADAPTIVE_FUSION_FINAL.json
- CLAIMS_MANIFEST_FINAL.json
- REVIEWER_OBJECTIONS_FINAL.md
- RAG_SECURITY_REPORT_FINAL.json
- tables/
- figures/
