# AHRAS Complete Forensic Review Baseline & Snapshot
**Date:** September 30, 2026  
**Auditor:** Independent Technical & Scientific Audit Team  
**Git Commit:** `01fbb82f900edcb8e5a87ce201da9dcf73167dc0`  
**Git Branch:** `home2`  

---

## 1. Environment Specifications

- **Python Version:** `3.14.6` (GCC 15.2.1 20260123)
- **OS / Platform:** `Linux-7.1.8-100.fc43.x86_64` (glibc 2.42)
- **Pytest Version:** `9.1.1`
- **Core Dependencies:**
  - `numpy`: `2.5.2`
  - `scikit-learn`: `1.9.0`
  - `fastapi`: `0.115.0+`
  - `pyyaml`: `6.0.3`
  - `pydantic`: `2.x`

---

## 2. Core Repository Inventory

- **Production Core Packages:**
  - `core/`: Risk engine, context, normalization, config, types, pipeline.
  - `detection/`: Signature, statistical/Welford, ensemble, open-set, multimodal.
  - `graph/`: Cyber knowledge graph, temporal reasoning, attack paths, community detection.
  - `evidence/`: Decision provenance, cryptographic chain, evidence ledger.
  - `xai/`: DecisionTrace explanation, fidelity verification, counterfactuals.
  - `forecasting/`: Threat progression, risk forecasting, proactive defense.
  - `calibration/` & `uncertainty/`: Conformal prediction, entropy gating, selective autonomy.
  - `adaptive_learning/`: Online weight learner, pseudo-labeler, shadow promotion, replay buffer.
  - `threat_intelligence/`: STIX/TAXII parser, IOC matcher, prior evidence enrichment.
  - `response/` & `security_twin/`: Safe containment, policy engine, blast radius limits, digital twin simulation.
  - `telemetry/`: OCSF normalization, collectors, minimality auditor, sketch screens.
  - `api/`: FastAPI REST API, v1 endpoints, WebSocket streaming, RBAC auth middleware.
  - `web/`: Frontend dashboard, D3 graph visualization, real-time alert triage.

- **Evaluation & Benchmarking Packages:**
  - `evaluation/datasets/`: Cryptographic verifier, downloader, adapters, split protocols.
  - `evaluation/scenarios/`: Combinatorial (pairwise/3-way) generator, environmental profiles.
  - `evaluation/metamorphic/`: 7 formal metamorphic relations suite.
  - `evaluation/faults/`: Network/sensor fault injection engine.
  - `evaluation/experiments/` & `evaluation/master_runner.py`: Master research benchmark execution.

---

## 3. Test Suite Baseline

- **Collected Tests:** `650` tests across all packages.
- **Passing Status:** `650 passed, 43 subtests passed` in ~75s.
- **Failures / Errors:** `0`.
