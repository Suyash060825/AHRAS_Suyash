# Phase 4 Completion Report: Security Knowledge Graph, Attack Flow, Campaign Similarity & Memory

**Platform**: Adaptive Hybrid Risk-Aware Security (AHRAS)  
**Standard**: Non-Monolithic, Auditable, Uncertainty-Bounded Defense Platform  
**Target Specifications**: Sections 21, 22, 23, 47, 48 of the Ultimate Implementation Architecture  
**Benchmark ID**: `EXP-33` (Security Knowledge Graph & Campaign Reasoning Benchmark)  
**Target Package**: [`knowledge_graph/`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/knowledge_graph), [`api/`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/api)  
**Primary Artifacts**:
- [`evaluation/results/KNOWLEDGE_GRAPH_REPORT.json`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/evaluation/results/KNOWLEDGE_GRAPH_REPORT.json)
- [`publication/tables/knowledge_graph.tex`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/publication/tables/knowledge_graph.tex)
- [`tests/test_knowledge_graph.py`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/tests/test_knowledge_graph.py) (10/10 Passed)  
- [`tests/test_api_endpoints.py`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/tests/test_api_endpoints.py) (7/7 Passed)  
**Status**: **COMPLETED & EMPIRICALLY VALIDATED (Zero Regressions, 100% Pass Rate)**

---

## 1. Executive Summary & Capabilities Delivered

Phase 4 equips AHRAS with high-level relational reasoning, semantic capability dependencies, standard-compliant attack flow serialization, and case-based memory with strict temporal isolation:

1. **Security & Detection Knowledge Graph ([`knowledge_graph/security_kg.py`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/knowledge_graph/security_kg.py) / Section 21)**:
   - Heterogeneous semantic graph connecting 11 node types (`TECHNIQUE`, `IMPLEMENTATION`, `TELEMETRY`, `SENSOR`, `DETECTION_RULE`, `ML_MODEL`, `EVIDENCE`, `ASSET`, `VULNERABILITY`, `THREAT_INTEL`, `RESPONSE_ACTION`) across 9 relational edge types (`requires`, `observed_by`, `detected_by`, `affects`, `mitigated_by`, `depends_on`, `validated_by`, `blocked_by`, `exposed_by`).
   - Answers core SOC operational questions:
     - *"What currently enables detection of this behavior?"* (`what_enables_detection()`)
     - *"What sensor is missing to observe this technique?"* (`find_missing_sensors()`)
     - *"Which detections depend on an unhealthy sensor?"* (`detections_affected_by_sensor()`)
     - *"Which response actions mitigate this multi-stage attack path?"* (`mitigating_responses_for_path()`)

2. **Attack Flow Interoperability ([`knowledge_graph/attack_flow.py`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/knowledge_graph/attack_flow.py) / Section 22)**:
   - Bridges low-level causal provenance graphs into standardized MITRE Center for Threat-Informed Defense Attack Flow representations.
   - Preserves timestamps, epistemic uncertainty, confidence, and hypothetical edges.
   - Computes formal structural and temporal validity metrics (`stage_completeness`, `edge_completeness`, `entity_completeness`, `temporal_ordering_validity`).

3. **Attack Campaign Similarity Engine ([`knowledge_graph/campaign_similarity.py`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/knowledge_graph/campaign_similarity.py) / Section 23)**:
   - Multi-attribute behavioral indexing combining Jaccard technique overlap, Longest Common Subsequence (LCS) sequence alignment, entity graph structural cosine similarity, and evidence hash verification.
   - **Safety Invariant Enforced**: Strictly adheres to *"Never assert same attacker/actor without supporting evidence"*. Incidents with matching behavioral patterns but differing/absent cryptographic IOCs are strictly designated `UNATTRIBUTED`.

4. **Vulnerability & Exposure Intelligence ([`knowledge_graph/vulnerability_intelligence.py`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/knowledge_graph/vulnerability_intelligence.py) / Section 47)**:
   - Evaluates dynamic risk based on real-time attack-path reachability, network exposure zone (DMZ vs Isolated), EPSS exploitability, CISA KEV status, and asset criticality:
     $$\text{DynamicRisk} = \text{BaseCVSS} \times (1.0 + 2.0 \times \text{EPSS}) \times \text{ZoneMult} \times \text{PathMult} \times \text{CriticalityMult}$$
   - Flips static ordering: prioritizes actively exploited flaws sitting directly on an adversary's lateral movement path over isolated high-CVSS vulnerabilities.

5. **Case-Based Security Memory ([`knowledge_graph/case_memory.py`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/knowledge_graph/case_memory.py) / Section 48)**:
   - Indexes past resolved incidents, attack graphs, applied playbooks, and verified operational containment outcomes.
   - **Strict Temporal Invariant**: *"Never use historical cases containing future information relative to an evaluation event"*. Any case where `closed_at > query_timestamp` is strictly filtered out, guaranteeing zero future data leakage.

6. **REST API Extensions ([`api/server.py`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/api/server.py))**:
   - `GET /api/knowledge-graph/enables/{technique_id}`
   - `GET /api/knowledge-graph/missing-sensors/{technique_id}`
   - `GET /api/knowledge-graph/sensor-impact/{sensor_id}`
   - `POST /api/campaign/match`
   - `POST /api/vulnerabilities/prioritize`

---

## 2. Empirical Benchmark Verification (EXP-33)

*Machine-readable artifact*: [`evaluation/results/KNOWLEDGE_GRAPH_REPORT.json`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/evaluation/results/KNOWLEDGE_GRAPH_REPORT.json)  
*LaTeX table*: [`publication/tables/knowledge_graph.tex`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/publication/tables/knowledge_graph.tex)

### 2.1 Attack Flow Structural Completeness & Temporal Validity

| Attack Scenario | Stage Completeness (%) | Edge Completeness (%) | Entity Completeness (%) | Temporal Monotonicity |
| :--- | :---: | :---: | :---: | :---: |
| **Ransomware Multi-Stage Deployment** | 100.0% | 100.0% | 100.0% | **Valid (0 violations)** |
| **Data Exfiltration Campaign** | 100.0% | 100.0% | 100.0% | **Valid (0 violations)** |
| **Cloud Privilege Escalation** | 100.0% | 100.0% | 100.0% | **Valid (0 violations)** |

### 2.2 Attribution Safety & Campaign Similarity
- **Top Match**: `camp-02` (Conti Ransomware Pipeline, Composite Similarity = 0.8875).
- **Matching Techniques**: `T1021`, `T1059`, `T1190`, `T1486`.
- **Attribution Invariant**: `UNATTRIBUTED` (Passed). In the absence of matching cryptographic evidence hashes, attribution was safely withheld with the formal justification: *"High behavioral & technique similarity; actor attribution unconfirmed per AHRAS safety invariants"*.

### 2.3 Context-Aware Vulnerability Prioritization
- **Observed Inversion**: `CVE-ACTIVE-PATH` (Base CVSS 6.8, on DMZ Web Server sitting on active lateral movement bridge) achieved Dynamic Risk **143.9**, outranking `CVE-STATIC-CRITICAL` (Base CVSS 9.8, isolated database) with Dynamic Risk **12.9** ($11.1\times$ higher priority).
- Confirms that operational exposure and path reachability supersede static severity.

### 2.4 Case Memory Temporal Isolation
- **Evaluation**: 100 historical queries across overlapping time horizons (-500s to +490s relative to evaluation timestamp).
- **Future Leakage Violations**: **0 / 100** (**0.0% leakage rate**).

---

## 3. Test & Regression Verification

- **Phase 4 Unit Tests**: 10 passed in `tests/test_knowledge_graph.py`.
- **API Endpoint Tests**: 7 passed in `tests/test_api_endpoints.py`.
- **Full Repository Regression Suite**: **596 passed, 43 subtests passed** (639 total test units) in 69.84s with **100% pass rate** and **zero regressions**.
