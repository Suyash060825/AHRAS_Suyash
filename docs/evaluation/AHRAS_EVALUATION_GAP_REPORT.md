# AHRAS Evaluation Gap Report & Architectural Roadmap
**Audit Date:** September 30, 2026  
**Document Status:** Complete & Actionable

---

## 1. Identified Architecture & Methodology Gaps

| Area | Current Implementation | Target Research Standard (Master Prompt) | Gap Severity | Action Required |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset Registry** | Ad-hoc file loading in `dataset_loader.py` | Unified registry with YAML manifest, explicit hashes, license tracking, and official URLs (`evaluation/datasets/`) | **HIGH** | Build `evaluation/datasets/` package and `data/manifests/evaluation_datasets.yaml`. |
| **Split Protocols** | Script-specific splits | 7 Formal Protocols (Standard, Temporal, Cross-Dataset, Cross-Environment, Open-Set, Cross-Attack, Benign Shift) | **HIGH** | Implement `evaluation/datasets/splitters.py` with reproducible seed and zero future leakage. |
| **Attack Family Taxonomy** | Raw dataset labels or heuristic names | Standardized 18-class canonical AHRAS attack taxonomy with confidence and MITRE mappings | **MEDIUM** | Implement `evaluation/datasets/adapters.py` with explicit canonical label normalization. |
| **Combinatorial Scenarios** | Static/random scenario generator | Pairwise (2-wise) and 3-wise IPO/AETG combinatorial coverage + Latin Hypercube continuous sampling | **HIGH** | Build `evaluation/scenarios/` modular generator with coverage metric accounting. |
| **Metamorphic Testing** | Limited permutation tests | 7 Formal Metamorphic Relations (duplication, irrelevant field scaling, monotonic evidence, order invariance, etc.) | **HIGH** | Create `evaluation/metamorphic/` suite with explicit violation checks. |
| **Fault Injection** | Noise tests in evasion script | Multi-tier observation failure engine (0%, 5%, 10%, 20%, 30%, 50% loss, delays, clock skew, out-of-order) | **HIGH** | Build `evaluation/faults/` injection harness tracking threat vs observation confidence. |
| **Lineage & Provenance** | Central `RESULTS.json` | Full end-to-end lineage graph (`RESULT_LINEAGE.json`) linking result_id → experiment_id → dataset_hash → config | **MEDIUM** | Implement `evaluation/reproducibility/lineage.py`. |
| **Execution Modes** | Single test/run invocations | Multi-mode runner (`--strict-real`, `--synthetic`, `--simulation`, `--smoke`, `--full`, `--replay`) | **HIGH** | Build unified modular runner CLI in `evaluation/runner.py`. |

---

## 2. Implementation Roadmap

1. **Phase 1-6:** Dataset Registry, Manifest, Adapters, Preprocessing, Protocols A-G, and Canonical Taxonomy.
2. **Phase 7-11:** Multimodal Realism, Controlled Scenario Generation, and 2-wise/3-wise Combinatorial Coverage Engine.
3. **Phase 12-13:** Metamorphic Testing Suite and Telemetry Fault Injection Harness.
4. **Phase 14-24:** Specialized Evaluation Engines (Open-Set, Temporal Drift, Cross-Dataset Matrix, Attack Chains, Detector Disagreement, Context Sensitivity, Scale, Security Twin).
5. **Phase 25-35:** Coverage Matrix, Lineage Tracking, Data Quality Profiler, and Modular CLI Runner.
6. **Phase 36-40:** 4-Tier Validation Reports, Publication Artifacts (JSON, CSV, LaTeX, Figures), and Final Verification Gates.
