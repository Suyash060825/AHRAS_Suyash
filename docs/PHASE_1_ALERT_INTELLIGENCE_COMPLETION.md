# Phase 1 Completion Report: Alert Intelligence Layer

**Platform**: Adaptive Hybrid Risk-Aware Security (AHRAS)  
**Standard**: Non-Monolithic, Auditable, Uncertainty-Bounded Defense Platform  
**Benchmark ID**: `EXP-ALERT-INTEL-01`  
**Package**: [`alert_intelligence/`](alert_intelligence)  
**Primary Artifact**: [`evaluation/results/ALERT_INTELLIGENCE_REPORT.json`](evaluation/results/ALERT_INTELLIGENCE_REPORT.json)  
**Test Suite**: [`tests/test_alert_intelligence.py`](tests/test_alert_intelligence.py) (9/9 Passed)  
**Status**: **COMPLETED & EMPIRICALLY VALIDATED**

---

## 1. Executive Summary & Problem Formulation

In modern enterprise Security Operations Centers (SOCs), raw detector alert fatigue represents a fatal vulnerability: high-volume repetitive alerts (e.g. DoS floods, automated port scans, brute-force bursts) flood event queues at thousands of events per second, obscuring low-and-slow, multi-stage Advanced Persistent Threats (APTs).

Phase 1 introduces the **AHRAS Alert Intelligence Layer (`alert_intelligence/`)**, an online stream processing layer that:
1. **Suppresses Repetitive Noise**: Folds identical burst alerts into canonical representations using geometric emission thresholds while maintaining an append-only cryptographic ledger of all associated evidence identifiers.
2. **Synthesizes Multi-Stage Incidents**: Correlates alerts across entities and time into unified `IncidentCluster` entities tracking MITRE ATT&CK kill-chain progression ($S_{\text{prog}}$).
3. **Calibrates Triage Priority**: Weighs detector severity against enterprise asset criticality (Tier 1 Crown Jewels vs Tier 3 Workstations), network zone exposure (Internet/DMZ vs Isolated), known CVE exploitability, and dampens priority under epistemic model uncertainty.
4. **Preserves 100% Cryptographic Provenance**: Enforces a strict invariant: **Zero Evidence Loss** ($100.00\%$ retention completeness across all deduplicated and clustered alerts).

---

## 2. Architectural Design & Core Components

```mermaid
flowchart TD
    subgraph StreamInput [Raw Ingestion Stream]
        RA["RawAlerts from Detectors\n(Sig, ML Anomaly, Stat, TGNN)"]
    end

    subgraph Deduplication [Adaptive Deduplicator]
        DEDUP["AdaptiveAlertDeduplicator\n(Sliding Window: 60s, Geometric: 2x)"]
        NOISE["Suppressed Duplicates\n(Evidence Folded into Canonical Alert)"]
    end

    subgraph Clustering [Spatio-Temporal Clustering]
        CLUSTER["AlertClusteringEngine\n(Spatial Proximity + Lateral Pivot Merge)"]
        KILLCHAIN["Kill-Chain Progression Tracker\n(MITRE Tactic Advancement TA0001 -> TA0040)"]
    end

    subgraph Triage [Exposure-Aware Prioritizer]
        PRIO["ExposureAwarePrioritizer\n(Asset Tier + Zone + CVSS + Uncertainty Dampening)"]
        DEC["TriageDecision\n(CRITICAL | HIGH | MEDIUM | LOW | INFO)"]
    end

    subgraph API [SOC Operations & Dispatch]
        REST["FastAPI Endpoints\nGET /api/incidents\nPOST /api/alerts/ingest"]
        WS["Live SOC WebSocket Stream"]
    end

    RA --> DEDUP
    DEDUP -->|Duplicate| NOISE
    DEDUP -->|New / Milestone / Evicted| CLUSTER
    CLUSTER --> KILLCHAIN
    KILLCHAIN --> CLUSTER
    CLUSTER --> PRIO
    PRIO --> DEC
    DEC --> REST
    DEC --> WS
```

### 2.1 Component Specifications

1. **`AdaptiveAlertDeduplicator`** ([`alert_intelligence/deduplication.py`](alert_intelligence/deduplication.py)):
   - Generates composite deduplication key: $K = (\text{entity\_key}, \text{source\_engine}, \text{detector\_name}, \text{mitre\_technique})$.
   - Implements exponential geometric emission ($C \ge 5, 10, 20, 40, \dots$) to update downstream systems during sustained volume floods without flooding queues.
   - Enforces eviction callback (`on_evict`) on window expiry to forward all accumulated cryptographic evidence pointers into active clusters.

2. **`AlertClusteringEngine`** ([`alert_intelligence/clustering.py`](alert_intelligence/clustering.py)):
   - Tracks active open clusters within correlation window $\Delta t = 300\text{s}$.
   - Dynamically merges separate clusters when a lateral movement or pivot alert bridges multiple previously independent entity keys.
   - Calculates kill-chain progression score:
     $$S_{\text{prog}} = \min\left(1.0, 0.20 \cdot |\mathcal{T}_{\text{present}}| + 0.50 \cdot \frac{\text{span}(\mathcal{T}_{\text{order}})}{|\mathcal{T}_{\text{total}}|}\right)$$

3. **`ExposureAwarePrioritizer`** ([`alert_intelligence/prioritizer.py`](alert_intelligence/prioritizer.py)):
   - Computes calibrated triage priority:
     $$P = \text{clip}\left( \left(0.35 \cdot S_{\text{det}} + 0.25 \cdot S_{\text{prog}} + 0.20 \cdot S_{\text{crit}} + 0.20 \cdot S_{\text{expo}}\right) \cdot (1 - 0.30 \cdot \bar{U}), 0.0, 1.0 \right)$$
   - Maps $P$ to 5 triage tiers:
     - $P \ge 0.80 \implies \text{CRITICAL}$ (Immediate containment dispatch)
     - $0.60 \le P < 0.80 \implies \text{HIGH}$ (Stage automation & engage honeypot lure)
     - $0.35 \le P < 0.60 \implies \text{MEDIUM}$ (Active provenance tracking)
     - $0.15 \le P < 0.35 \implies \text{LOW}$ (Continuous monitoring)
     - $P < 0.15 \implies \text{INFO}$ (Telemetry record)

4. **`AlertIntelligencePipeline`** ([`alert_intelligence/pipeline.py`](alert_intelligence/pipeline.py)):
   - Unified facade orchestrating deduplication, clustering, and prioritization with flush and query APIs.

---

## 3. Empirical Benchmark Results (EXP-ALERT-INTEL-01)

Evaluated across a benchmark stream of **5,000 alerts** including volumetric DoS bursts, multi-stage APT campaigns, and sporadic background anomalies across 6 monitored enterprise entities.

Recorded in [`evaluation/results/ALERT_INTELLIGENCE_REPORT.json`](evaluation/results/ALERT_INTELLIGENCE_REPORT.json):

| Metric | Result | Target / Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Ingestion Throughput** | **45,925.6 EPS** | $\ge 25,000$ EPS (Line-Rate) | **EXCEEDED** |
| **Mean Ingestion Latency** | **21.64 µs** | $\le 100$ µs | **EXCEEDED** |
| **P50 (Median) Latency** | **3.28 µs** | $\le 10$ µs | **EXCEEDED** |
| **P95 Latency** | **51.36 µs** | $\le 200$ µs | **EXCEEDED** |
| **Duplicate Alert Suppression** | **4,876 / 5,000 (97.52%)** | $\ge 90.0\%$ | **EXCEEDED** |
| **Alert Compression Factor** | **1,250 : 1** | $\ge 100 : 1$ | **EXCEEDED** |
| **Evidence Retention Completeness** | **5,007 / 5,007 (100.00%)** | $100.00\%$ Strict Invariant | **VERIFIED** |
| **Multi-Stage Attack Priority** | **$P = 0.9314$, CRITICAL** | Top Queue Rank | **VERIFIED** |

### 3.1 Top Incident Case Study
The engine autonomously isolated and synthesized the multi-stage APT:
- **Title**: *Multi-Stage Attack: Initial Access -> Execution -> Persistence on host:web-public-01 (+2 entities)*
- **Entities**: `host:web-public-01` $\to$ `host:workstation-101` $\to$ `host:dc-prod-01`
- **Kill-Chain Span**: 10 MITRE tactics (`TA0001` through `TA0040`)
- **Progression Score**: $1.0000$ (Complete kill chain)
- **Triage Level**: **CRITICAL** ($P = 0.9314$)
- **Recommended Action**: `IMMEDIATE_CONTAINMENT_DISPATCH`

---

## 4. REST API Endpoints Implemented

Integrated into [`api/server.py`](api/server.py):

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/incidents` | `GET` | Lists all active incident clusters sorted by priority score (descending). |
| `/api/incidents/{incident_id}` | `GET` | Retrieves full contextual details, member alerts, and rationale for a cluster. |
| `/api/alerts/ingest` | `POST` | Ingests a `RawAlert` into the online deduplication and clustering pipeline. |
| `/api/alert-intelligence/metrics` | `GET` | Returns real-time operational telemetry (dedup ratio, triage distribution). |

---

## 5. Verification & Test Suite Summary

- **Unit Test File**: [`tests/test_alert_intelligence.py`](tests/test_alert_intelligence.py)
- **Test Results**: **9 passed in 1.26s (100% pass rate)**.
- **Invariants Verified**:
  - `test_models_and_evidence_preservation`: Atomic evidence pointers retained through clustering.
  - `test_deduplication_basic_and_evidence_accumulation`: Suppression of identical alerts within sliding window.
  - `test_deduplication_geometric_emission`: Exponential milestone emission.
  - `test_clustering_spatio_temporal_and_progression`: Progressive kill chain tracking.
  - `test_clustering_dynamic_merge`: Multi-cluster merging on lateral movement bridges.
  - `test_prioritizer_asset_criticality_and_exposure`: Tier 1 Crown Jewel priority dominance.
  - `test_prioritizer_uncertainty_dampening`: Epistemic uncertainty dampening.
  - `test_pipeline_end_to_end`: Stream processing under heterogeneous workloads.
  - `test_api_alert_intelligence_endpoints`: REST API integration via `TestClient`.
