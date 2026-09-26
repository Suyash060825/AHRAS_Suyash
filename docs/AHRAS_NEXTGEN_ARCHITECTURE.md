# AHRAS Next-Generation Target Architecture

**Platform**: Adaptive, Hybrid, Risk-Aware Security (AHRAS)  
**Standard**: Modular, Non-Monolithic, Auditable, Uncertainty-Bounded Defense Platform  
**Operational Loop**:  
$\text{OBSERVE} \to \text{DETECT} \to \text{CORRELATE} \to \text{ASSESS} \to \text{EXPLAIN} \to \text{PREDICT} \to \text{RESPOND} \to \text{VERIFY} \to \text{LEARN} \to \text{ADAPT} \to \text{repeat}$

---

## 1. End-to-End System Architecture

```
                                  TELEMETRY SOURCES
       ┌───────────────────────────────┬───────────────────────────────┐
       ▼                               ▼                               ▼
┌──────────────┐               ┌──────────────┐               ┌──────────────┐
│Network Taps  │               │Host Sensors  │               │Cloud IAM /   │
│(PCAP/NetFlow)│               │(Syscall/ETW) │               │SaaS Audits   │
└──────┬───────┘               └──────┬───────┘               └──────┬───────┘
       │                              │                              │
       └───────────────────────┬──────┴──────────────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │ Stage 0: OCSF Normalizer & Enricher  │
            │ (Schema 1.1.0, GeoIP, Asset, Port)   │
            └──────────────────┬───────────────────┘
                               │ Canonical OCSF Event
                               ▼
            ┌──────────────────────────────────────┐
            │ Stage 1: Streaming Sketch Fast-Path  │  (Extension F)
            │ (Count-Min, Heavy Hitters, Entropy)  │
            └──────────┬───────────────────────────┘
                       │
                       ▼
            ┌──────────────────────────────────────┐
            │ Stage 2: Confidence-Based Router     │  (Extension E)
            │  - Stage 1 Detectors (Sigs + Stats)  │
            │  - [High Conf / Low Uncert? EXIT]    │
            │  - Stage 2 ML Ensemble (IF / AE / SVM│
            │  - Stage 3 Encrypted Session Intel   │  (Extension G)
            │  - Stage 4 Multimodal Attention      │
            └──────────────────┬───────────────────┘
                               │ Standardized EvidenceRecords
                               ▼
            ┌──────────────────────────────────────┐
            │ Stage 3: Provenance Attack Reasoner  │  (Extension C)
            │ (Hetero DAG: Host, Proc, Socket, IOC)│
            │ + Temporal GNN Message Passing       │
            └──────────────────┬───────────────────┘
                               │ Relational Graph Energy G_corr
                               ▼
            ┌──────────────────────────────────────┐
            │ Stage 4: Multi-Signal Risk Engine    │
            │  R_t = Clip[(Σ w_i E_i)·A·(1-U) - wT]│
            │  + Temporal Epistemic Instability    │  (Extension D)
            │  + Recidivism Threat Memory Boost    │
            └──────────┬───────────────────┬───────┘
                       │                   │
                       ▼                   ▼
            ┌─────────────────────┐ ┌──────────────────────────────┐
            │ DecisionTrace Ledger│ │ Stage 5: Holt Forecaster     │
            │ (Exact DAG Replay   │ │ (Causal Early Warning h=1,3,5│
            │  Δ <= 1e-4)         │ └──────────────┬───────────────┘
            └──────────┬──────────┘                │
                       │                           │
                       ▼                           ▼
            ┌─────────────────────┐ ┌──────────────────────────────┐
            │ XAI Reliability 2.0 │ │ Stage 6: Conformal Autonomy  │
            │ (Stability, Suffic, │ │ Gate (Split Quantile Bounds) │
            │  Comprehensiveness) │ │ [PASS | MONITOR | DECEPTION |│
            │ (Extension A)       │ │  STAGE | AUTO | ABSTAIN]     │
            └─────────────────────┘ └──────────────┬───────────────┘
                                                   │
                                                   ▼
            ┌──────────────────────────────────────────────────────┐
            │ Stage 7: Active Response & Security Twin Simulation  │
            │  1. Pre-execution Counterfactual Simulation (Twin)   │  (Extension B)
            │  2. Dynamic Lure Selection (Adaptive Deception)      │  (Extension H)
            │  3. Gated Execution (DRY_RUN by default)             │
            │  4. Post-execution Observed Efficacy Learning        │  (Extension I)
            └──────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────────────┐
            │ Stage 8: Safe Continual & Federated Adaptation Loop  │
            │  - Confidence-Gated Pseudo-Label Validation Engine   │  (Extension J)
            │  - 5-Bank Multi-Memory Replay (Provenance Preserved) │  (Extension K)
            │  - Byzantine-Robust Federated Knowledge Distillation │
            └──────────────────────────────────────────────────────┘
```

---

## 2. Core Architectural Contracts

### 2.1 The Evidence Record Contract
All detectors output strictly standardized instances of `EvidenceRecord`:
```python
class EvidenceRecord(BaseModel):
    evidence_id: str
    event_id: str
    entity_id: str
    source: str                 # "signature", "ml_anomaly", "stat_drift", "encrypted_session", "graph", "sketch"
    detector_type: str
    raw_score: float            # Continuous value [0, inf)
    normalized_score: float     # Calibrated score [0.0, 1.0]
    confidence: float           # Confidence in detection [0.0, 1.0]
    uncertainty: float          # Epistemic uncertainty [0.0, 1.0]
    mitre_technique: Optional[str]
    metadata: Dict[str, Any]
    timestamp: float
```

### 2.2 The DecisionTrace Contract
The risk engine records complete computational provenance, making every score deterministically verifiable:
```python
class DecisionTrace(BaseModel):
    trace_id: str
    event_id: str
    entity_key: str
    timestamp: float
    raw_evidence: Dict[str, float]
    normalized_evidence: Dict[str, float]
    active_weights: Dict[str, float]
    additive_risk_pre_context: float
    context_modifiers: Dict[str, float]
    trust_discount: float
    final_risk: float
    epistemic_uncertainty: float
    instability_index: float
    conformal_tau: float
    selected_action: str
    execution_mode: str         # "DRY_RUN" | "AUTONOMOUS" | "ESCALATED"
    model_versions: Dict[str, str]
```

### 2.3 The Security Twin Contract
Before an autonomous response is executed, it passes through the digital twin simulation:
```python
class SimulationResult(BaseModel):
    action_type: str
    target_entity: str
    pre_action_risk: float
    expected_post_action_risk: float
    path_breakage_probability: float
    remaining_attack_steps: int
    blast_radius_score: float
    collateral_disruption_cost: float
    simulation_confidence: float
    validation_status: str      # "VALIDATED" | "POLICY_VIOLATION" | "HIGH_BLAST_RADIUS"
```

---

## 3. Operational Guarantees

1. **Strictly Non-Monolithic**: Detectors and modules are loosely coupled. Any detector or analysis stage can be dynamically bypassed or disabled via configuration (`use_encrypted_session=False`, `use_sketch=False`, `use_security_twin=False`).
2. **Safe By Default**: All active defense mechanisms initialize in `DRY_RUN` mode. No real network block, host isolation, or token revocation occurs unless explicitly configured in production mode with verified analyst policy approval.
3. **Provable Auditability**: 100% of risk assessments, model routing choices, and counterfactual simulations are logged to the tamper-evident cryptographic ledger with SHA-256 state chaining.
