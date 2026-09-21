# AHRAS Architecture Roadmap (v6 – v14)

> **Platform**: Adaptive, Hybrid, Risk-Aware Security Intelligence and Defense Platform  
> **Core Operational Loop**: $\text{OBSERVE} \to \text{DETECT} \to \text{UNDERSTAND} \to \text{PREDICT} \to \text{PROTECT} \to \text{LEARN} \to \text{repeat}$  
> **Governing Architecture Principle**: Non-monolithic, multi-signal detection fabric feeding a unified, auditable risk controller.

---

## 1. High-Level Target Architecture

The target architecture enforces strict separation between **evidence generation** (specialized, swappable detectors) and **evidence fusion** (the uncertainty-aware risk engine). No single deep learning model owns the final security verdict.

```
                    ┌──────────────────────────────────────────────┐
                    │                    AHRAS                     │
                    └──────────────────────┬───────────────────────┘
                                           │
             ┌─────────────────────────────┴─────────────────────────────┐
             ▼                                                           ▼
┌──────────────────────────┐                               ┌──────────────────────────┐
│  Network Sensor (Kafka)  │                               │ Host / Endpoint Agent    │
│  (PCAP / eBPF / NetFlow) │                               │ (Process, Entropy, Tree) │
└────────────┬─────────────┘                               └─────────────┬────────────┘
             │                                                           │
             └─────────────────────────────┬─────────────────────────────┘
                                           │ (OCSF Normalization)
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │          Unified Detection Fabric            │
                    │  ┌────────────┐ ┌────────────┐ ┌──────────┐  │
                    │  │ Signatures │ │ Anomaly ML │ │ Stat /   │  │
                    │  │ (23 Rules) │ │ (IF/AE/SVM)│ │ Drift    │  │
                    │  └────────────┘ └────────────┘ └──────────┘  │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │    Open-Set Layer (Latent Mahalanobis OOD)   │
                    │    [Benign | Known | Unknown/Novel | Uncert] │
                    └──────────────────────┬───────────────────────┘
                                           │ Structured Evidence Records
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │     Multi-Signal Adaptive Risk Engine        │
                    │   R_t = Clip[(Σ w_i E_i)·A_crit·(1-U) - w_T] │
                    └──────────┬───────────────────┬───────────────┘
                               │                   │
             ┌─────────────────┴───────┐   ┌───────┴─────────────────┐
             ▼                         ▼   ▼                         ▼
┌──────────────────────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────────┐
│ Dynamic Recidivism Trust │ │ Temp GNN  │ │ MITRE CTI │ │ Holt Forecasting  │
│ (Historical Decay/Recov) │ │ (Noisy-OR)│ │ (STIX 2.1)│ │ (h1, h3, h5 Horiz)│
└──────────────────────────┘ └───────────┘ └───────────┘ └───────────────────┘
             │                         │   │                         │
             └─────────────────┬───────┴───┴───────┬─────────────────┘
                               │                   │
                               ▼                   ▼
                    ┌──────────────────────┐ ┌───────────────────────┐
                    │ DecisionTrace Ledger │ │ Split Conformal Gate  │
                    │ (Fidelity Δ <= 1e-4) │ │ (Autonomous / Abstain)│
                    └──────────┬───────────┘ └───────────┬───────────┘
                               │                         │
                               ▼                         ▼
                    ┌──────────────────────┐ ┌───────────────────────┐
                    │ Causal XAI & Counter-│ │ Safety-Gated SOAR     │
                    │ factual Engine       │ │ (RASE Utility Policy) │
                    └──────────────────────┘ └───────────┬───────────┘
                                                         │
                                                         ▼
                    ┌──────────────────────────────────────────────┐
                    │ Active Analyst Loop & 5-Bank Continual Replay│
                    │ [Recent | Attack | Hard-Neg | Drift | Proto] │
                    └──────────────────────────────────────────────┘
```

---

## 2. Target Release Evolution Matrix (v6 to v14)

Each version is self-contained, rigorously tested, and backwards-compatible with existing evidence ledgers and OCSF schemas.

| Release | Codename | Core Focus | Key Subsystems & Deliverables | Primary Verification Metric |
| :--- | :--- | :--- | :--- | :--- |
| **AHRAS v6** | *Core Integrity* | Research-Grade Core & Fidelity | Trace decomposer, analytical sum-check, exact partials, sub-component logging. | $\Delta_{\text{abs}} \le 10^{-6}$, 100% test pass |
| **AHRAS v7** | *Dynamic Stability*| Adaptive Weighting & Drift | Context-gated fusion network, shadow validation, Welford/EWMA drift router. | Adaptation gain MSE $\le 0.02$ |
| **AHRAS v8** | *Open-Set Frontier*| Open-Set Unknown Attack Detection| Mahalanobis OOD latent distance, held-out attack rejection, multi-domain transfer. | Unknown Family Recall $\ge 0.75$, FPR $\le 0.05$ |
| **AHRAS v9** | *Relational Reasoner*| Graph & Provenance Campaigns | Temporal Heterogeneous GNN, Noisy-OR episode clustering, lateral campaign attribution. | Multi-hop Lateral Movement $F1 \ge 0.88$ |
| **AHRAS v10** | *Proactive Horizon*| Causal Early-Warning Prediction | Holt linear smoothing, hazard threshold crossing, proactive quarantine escalation. | Lead Time $\ge 3$ steps, Zero lookahead leakage |
| **AHRAS v11** | *Host Telemetry* | Two-Tier Endpoint Collection | eBPF/ETW kernel collection adapter, Shannon entropy watcher, process tree lineage. | Telemetry ingestion overhead $\le 3\%$ CPU |
| **AHRAS v12** | *Constrained Autonomy*| Conformal Safe Response & RASE | Split conformal nonconformity bounds, action blast-radius matrix, response verification. | False Intervention Cost reduction $\ge 60\%$ |
| **AHRAS v13** | *Continual Multimodal*| Cross-Modal Representation | Self-supervised masked feature encoder ($z_{\text{net}}, z_{\text{proc}}, z_{\text{id}}$), 5-bank replay. | Catastrophic Forgetting rate $\le 2\%$ |
| **AHRAS v14** | *Federated Edge* | Byzantine-Robust Distributed | Coordinate median aggregation, client reputation decay $T_i(t)$, FedKD knowledge sharing. | Retained F1 $\ge 0.95$ under 30% malicious clients |

---

## 3. Core Architectural Subsystems

### Subsystem A: Detection Fabric (Decoupled Evidence Providers)
Detectors never directly decide risk; they output standardized `EvidenceRecord` instances:
* **Contract**: `detect(event: Dict[str, Any]) -> List[EvidenceRecord]`
* **Required Evidence Fields**: `evidence_id`, `event_id`, `entity_id`, `source`, `detector_type`, `raw_score`, `normalized_score`, `confidence`, `uncertainty`, `mitre_mapping`.

### Subsystem B: Multi-Signal Risk Engine
The risk calculation preserves explicit arithmetic operation ordering:
1. **Raw Ingestion & Feature Masking**
2. **Quality & De-Correlation Weighting** ($w_i' = w_i \cdot Q_i / (1 + \sum C_{ij} w_j)$)
3. **Additive Threat Accumulation** ($\sum w_i' E_i$)
4. **Contextual Modulations** (Multiplication by $A_{\text{crit}}$ and $(1 - U_{\text{penalty}})$)
5. **Dynamic Trust Subtraction** ($- w_{\text{trust}} T_{\text{trust}}$)
6. **Clipping & Quantization** ($\text{Clip}_0^1$, rounded to 4 decimals)

### Subsystem C: Closed-Loop Governance & Continual Replay
* **Conformal Abstention**: High uncertainty or OOD inputs abstain from autonomous mitigation and route to SOC analyst queues.
* **5-Bank Memory Management**: Budgeted FIFO pools partitioned by sample difficulty to eliminate catastrophic forgetting.
