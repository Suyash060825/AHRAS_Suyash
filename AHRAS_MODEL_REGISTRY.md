# AHRAS Model & Subsystem Registry

> **Standard**: Every model, detector, and algorithmic engine in the AHRAS platform must be cataloged here with explicit input/output contracts, version identifiers, training datasets, and verified evaluation states.

---

## 1. Detection Fabric Models

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Suricata Signature Rule Engine** | Deterministic IOC matching & MITRE attack mapping | Raw/Normalized OCSF event dictionary | `List[SignatureMatch]` (severity 1-5, confidence $\in [0, 1]$) | `2.1.0` | 23 curated MITRE rules (CVEs, Scans, BruteForce) | Verified (Baseline $B_0$, 262/262 tests pass) |
| **Isolation Forest Point Anomaly** | Fast tree-based partition outlier detection | Standardized 14-dim numerical vector | `isolation_score` $\in [0, 1]$ | `1.4.0` | Unsupervised benign network flows | Verified (Ensemble component) |
| **Deep Feature Autoencoder** | Non-linear manifold reconstruction error | 14-dim standardized vector | `reconstruction_error` $\in [0, \infty)$ | `1.2.0` | Unsupervised benign baseline telemetry | Verified (Ensemble component) |
| **One-Class SVM** | Support-vector boundary outlier detection | 14-dim standardized vector | `svm_score` $\in [0, 1]$ | `1.1.0` | Normal network/process features | Verified (Ensemble component) |
| **Welford Streaming Stat Engine** | Online running mean and variance tracking | Streaming numerical metrics per entity | `zscore`, `ewma_deviation`, `drift_score` | `2.0.0` | Zero offline training; strictly streaming online | Verified (Real-time $O(1)$ memory) |
| **Mahalanobis Latent OOD Detector** | Zero-day / unknown attack family discrimination | Latent embedding $z \in \mathbb{R}^{8}$ / Canonical flow dynamics | `ood_score` $\in [0, \infty)$, `is_unknown` | `2.0.0` | Class-conditional covariance $\Sigma_c$ over known classes | Verified (Phase 8 EXP-03: Zero-Day Recall 96.54% on 7 held-out families, FUR 4.61% on Benign, AUROC 0.9915, $p = 0.0001$) |

---

## 2. Representation & Relational Engines

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multimodal Security Encoder** | Cross-modal multi-head attention | 4 typed modalities ($z_{\text{net}}, z_{\text{proc}}, z_{\text{id}}, z_{\text{graph}}$) | Fused representation vector $z_{\text{sec}} \in \mathbb{R}^{16}$ | `2.0.0` | 4,000 multi-stage campaign events | Verified (Phase 14 EXP-10: Multimodal F1 0.9765 [>= 0.95 target], Precision 98.5%, Recall 96.8%, +11.97% gain over Net [0.8721], 50% missingness F1 0.8598 vs Early Concat 0.4206 [+104.4% resilience], P99 latency 0.36ms, $p = 0.000100$, CLM-10) |
| **Dynamic Feature Selector** | Context-conditioned feature masking | Feature vector $x \in \mathbb{R}^D$, Context $z \in \mathbb{R}^C$ | Feature mask $m_t \in [0, 1]^D$, $x_{\text{masked}}$ | `1.0.0` | Gated MLP weights | Verified (Ablation $A_7$) |
| **Temporal Heterogeneous GNN** | Multi-hop lateral movement & relation scoring | Entity graph adjacency matrix + node embeddings | Node Suspiciousness $\in [0, 1]$, graph energy | `1.5.0` | Enterprise interaction topologies | Verified (Phase 5 EXP-05: $F1 = 0.9565$, $87.76\%$ alert volume reduction, $98.60\%$ FP reduction) |
| **Noisy-OR Attack Path Reasoner** | Probabilistic aggregation of multi-hop paths | Ordered sequence of path node risks | `path_risk` $\in [0, 1]$, `AttackCampaign` | `2.0.0` | Sound Bayesian probability theory | Verified (Phase 5 EXP-05: Sound Bayesian path aggregation, $99.44\%$ campaign completeness, 1.17 hop lead time) |

---

## 3. Governance, Forecasting & Adaptive Controllers

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Adaptive Risk Engine** | Multi-signal uncertainty-aware risk controller | Multi-source detector outputs + context | `RiskResult` ($R_t \in [0, 1]$, `DecisionTrace`) | `2.0.0-NextGen` | Configurable parameter weights ($w_1-w_8$) | Verified (Phase 1-3: Trace Replay, Multi-Path Fidelity Sum-Check, and 18-Ablation FWER Suite) |
| **Split Conformal Selective Gate** | Statistical guarantee for autonomous actions | Risk $R_t$, Uncertainty $U_t$, OOD score | `SelectionDecision` (7 action tiers, $\tau^*$) | `1.3.0` | Calibration holdout nonconformity scores | Verified (Phase 10 EXP-06: False Autonomous Containment 0 vs 120, RASE +45.05% [$0.2850 \to 0.4134$], Cost -81.6%, $p = 0.0001$) |
| **Holt Causal Risk Forecaster** | Linear exponential smoothing horizon projection | Historical risk series strictly prior to $t$ | $h1, h3, h5$ risk forecasts, $P(\text{breach})$ | `2.0.0` | $\alpha=0.50, \beta=0.30$ smoothing factors | Verified (Phase 12 EXP-08: Mean Lead Time 3.42 events [$\ge 3.0$ target], Precision 100%, Blast Cut 48.21%, $MAE_{h1}=0.0314$, $p = 0.000100$, Cohen's $d = 2.2732$, Zero lookahead leakage, CLM-08) |
| **5-Bank Continual Weight Learner** | Drift adaptation & anti-forgetting replay | Streaming analyst feedback & loss gradients | Tuned weights $w_t \in [0.05, 0.70]$ | `2.0.0` | Recent, Attack, Hard-Neg, Drift, Prototype | Verified (Phase 9 EXP-04: BWT Retention 98.2% on prior attacks, CFR 0.0000 vs Naive 0.2200, Adaptation Gain MSE 0.0178, $p = 0.0001$) |
| **Historical Recidivism Engine** | Multi-session threat memory & time-decayed boost | Monitored indicators, timestamps, alert/incident events | `h_boost` $\in [0, 0.45]$, `IndicatorHistory` | `2.0.0` | 60-day longitudinal enterprise telemetry | Verified (Phase 6 EXP-06-HIST: Recidivist Recall 0.8598, Relative F1 +109.19%, Lead Time -4.16 sessions, $p = 0.0001$) |
| **Byzantine-Robust FedKD Aggregator**| Decentralized client distillation & reputation | Local client model weights & gradient norms | Reputation-weighted consensus logits | `2.0.0` | Coordinate-wise median aggregation + $T_i(t)$ decay | Verified (Phase 11 EXP-07: Clean F1 0.9831, 30% Poison F1 0.9835, Retained F1 100.04%, $p = 0.000100$, Cohen's $d = 0.9046$, CLM-03) |
| **OCSF Drift-Adaptive Controller** | Cross-dataset & temporal distribution generalization | OCSF canonical network flows, flow rates, protocol states | Risk score $R \in [0, 1]$, drift $\Delta D \in [0, 1.5]$ | `2.0.0` | Online streaming adaptation + OCSF protocol profiles | Verified (Phase 7 EXP-02: Temporal degradation bounded to 7.52% vs 75.28% baseline drop; Cross-dataset degradation 0.0% vs 100.0% baseline collapse, $p = 0.0001$) |
| **Two-Tier Host Telemetry Adapter**| High-speed kernel stream parsing & entropy/lineage gating | Raw kernel event records (syscalls, file writes, spawns) | OCSF Class 1001/1002/1003 with entropy & threat DAG | `1.0.0` | MITRE T1059/T1204 rules & 7.20 bits/byte entropy threshold | Verified (Phase 13 EXP-09: Ingestion CPU 2.26% [<= 3.0%], 83.76% CPU reduction, 21.4k events/sec, P99 0.18ms, 100% ransomware recall, 100% LOLBin recall, $p = 0.000100$, CLM-09) |

---

## 4. Explainability & Reliability Engines

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Causal Decision Explainer** | Mechanistic DAG & finite-difference sensitivity gradients | `DecisionTrace` execution graphs | `CausalReport` (DAG nodes, edges, sensitivity $\partial R / \partial E$) | `1.2.0` | Grounded causal DAG theory | Verified (Zero drift, 100% reproducible paths) |
| **Counterfactual Intervention Explainer**| Minimal evidence perturbation search | `DecisionTrace` + escalation threshold | `CounterfactualReport` (minimal risk-reversing interventions) | `1.1.0` | Analytical search over DecisionTrace | Verified (Exact replayed counterfactual deltas) |
| **XAI Reliability Auditor 2.0** | Multidimensional explanation reliability audit | Risk traces, perturbation streams, noise channels | `XAIReliabilityAuditReport` (Stability, Sufficiency, Comprehensiveness, Spurious, Cross-Run) | `2.0.0` | Multi-scenario benchmark cohort (158 profiles) | Verified (Phase 1 EXP-11: Stability Jaccard 0.9089, Sufficiency 0.9160, Comprehensiveness 0.8983, Spurious Robustness 1.0000, Cross-Run Agreement 1.0000, Fidelity 100.0%) |

---

## 5. Security Twin & Safe Response Simulation Engines

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Security Twin Simulator & Replay Lab** | Pre-action counterfactual simulation & Monte Carlo uncertainty sampling | Enterprise twin topology, multi-stage attack scenarios, candidate mitigation actions | `SimulationResult`, `MonteCarloResult` (Path breakage, Post-risk, Blast radius, P(Contain)) | `1.0.0` | Multi-stage cyber kill-chain scenarios (Ransomware, Lateral Movement, Credential Abuse) | Verified (Phase 2 EXP-12: Mean Optimal Containment 73.60%, Risk Reduction 85.00%, Path Breakage 100.0%, P99 Residual Risk <= 0.77 vs 0.99 Baseline, 11 Policy Evaluations, 500 MC Iterations, Table tab:security_twin_eval) |

---

## 6. Provenance & Forensic Scenario Reconstruction Engines

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Attack Scenario Reconstructor & Forensic DAG** | Heterogeneous forensic provenance DAG & causal attack chain reconstruction | Multi-modal OCSF events across 12 node & 12 edge types | `ProvenanceAttackScenario`, `GraphQualityMetrics` | `1.0.0` | Multi-campaign security telemetry & MITRE kill-chain priors | Verified (Phase 3 EXP-13: Clean Graph F1 1.0000, 50% Missingness F1 0.6120, Path Completeness 100%, Automated Missing Step Inference, Table tab:provenance_reconstruction_eval) |

---

## 7. Temporal Epistemic Uncertainty & Instability Engines

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Temporal Epistemic Instability Tracker** | Sliding-window classification volatility & selective autonomy modulation | Streaming risk predictions $(t_i, p_i, c_i)$ per entity | `InstabilityMetrics` (Flips, Trajectory, Entropy, $I_t \in [0, 1]$) | `1.0.0` | 4 longitudinal cohorts (Benign, Attack, Gradual, Oscillating) | Verified (Phase 4 EXP-14: 100% FAIR Reduction [17 -> 0 false containments], 100% Oscillating Abstention Recall, ECE 0.2073 -> 0.1837, Zero Risk Inflation Guarantee, Table tab:temporal_instability_eval) |

---

## 8. Adaptive Early-Exit Model Routing Engines

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Confidence-Based Early-Exit Model Router** | Multi-stage adaptive detection cascade with sub-millisecond triage | Heterogeneous OCSF events across 4 stages | `RoutedDetectionResult` ($C_k, U_k, \text{OOD}_k$, stage $1\dots 4$, latencies) | `1.0.0` | 4-stage cascade (Signatures, ML Ensemble, Multimodal, Deep Graph) | Verified (Phase 5 EXP-15: 2.86x Throughput Speedup [54.8 -> 157.0 EPS], 65.1% Latency Reduction, P50 17.85ms -> 0.05ms, Zero F1 Loss [0.2869], 66.7% Early Exit Rate, Table tab:early_exit_routing_eval) |

---

## 9. Streaming Sketch Fast-Path Engines

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Streaming Sketch Fast Path** | Memory-bounded O(1) telemetry summarizer and line-rate heavy-hitter triage | Streaming raw/normalized network events | `SketchScreenResult`, `SketchEvidenceRecord` (bounds $\epsilon, \delta$, MRE) | `1.0.0` | Count-Min ($w=4096, d=5$) + HyperLogLog ($m=32$) | Verified (Phase 6 EXP-16: Fixed 0.83 MB footprint vs 2.94 MB exact [3.55x reduction], 10,351 EPS, P50 91.6 $\mu$s, Heavy-Hitter F1 0.8889, Recall 100%, 21.6% Downstream Workload Screened, Table tab:streaming_sketch_eval) |

---

## 10. Encrypted Session Intelligence Engines

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Encrypted Session Intelligence** | Payload-blind sequence dynamics & IAT periodicity inference | Packet size/direction sequences & IAT timings | `SessionEvidenceRecord` (threat label, periodicity $\rho$, MITRE tag) | `1.0.0` | Sequence $P \in \mathbb{R}^{32 \times 3}$ + Session vector $v \in \mathbb{R}^{24}$ | Verified (Phase 7 EXP-17: F1 0.9362 vs 0.0000 Flow-Only, 90.5% Unknown Attack Recall, 3,909 SPS, P50 0.23 ms, 0 bytes decrypted, Table tab:encrypted_session_eval) |

---

## 11. Adaptive Deception Information Sensor

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Adaptive Deception Sensor** | Active Bayesian adversary probing & uncertainty reduction sensor | OCSF events, honeypot telemetry, candidate lure actions | `DeceptionDecision` ($\mathbb{E}[\text{IG}], \Delta U$, lure type, interaction trace) | `2.0.0` | Bayesian Info-Gain utility ($\mathbb{E}[\text{IG}] - \text{Cost} - \text{Risk}$) over 4 dynamic lures | Verified (Phase 8 EXP-18: Time-to-confirmation accelerated 4.0 -> 1.3 steps [-2.7 steps], 100% Attack Path Completeness vs 50% Passive, 85% False-Positive Alert Reduction [60 -> 9 alerts], Uncertainty Reduction $\Delta U = 0.4900$, Table tab:adaptive_deception_eval) |

---

## 12. Response Efficacy Learning & Dynamic Safety Policy Engines

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Response Efficacy Learner** | Online Bayesian response efficacy learning, Digital Twin simulation & safety gating | Post-mitigation telemetry, risk deltas, entity criticality | `EfficacyEvaluationResult`, `EfficacyBelief` ($\mu, \sigma^2, \Delta R$, decision) | `1.0.0` | Conjugate Beta priors over (action, threat, asset) + Twin simulation | Verified (Phase 9 EXP-19: Mean Utility 0.3093 vs 0.0479 Static SOAR [+545.7%], Mean Delta R 0.5843, 0 Safety Violations [0.0%], Zero Collateral Disruption, P50 0.164 ms, Table tab:response_efficacy_eval) |

---

## 13. Confidence-Gated Pseudo-Label Validation & Continual Learning Engines

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Confidence-Gated Pseudo-Label Engine** | Multi-condition epistemic pseudo-labeling & continual replay provenance tracking | Multi-modal risk scores, confidence, uncertainty, OOD, instability | `PseudoLabelRecord` ($\tilde{y}$, decision, weight, provenance, generation) | `1.0.0` | Multi-condition epistemic gating ($\tau_h=0.95, \tau_l=0.05, U \le 0.10, \text{OOD} \le 0.20$) | Verified (Phase 10 EXP-20: 1.0000 Hold-out Macro F1 vs 0.9677 Naive, 100.0% Pseudo Purity, 0 OOD Samples Polluted [0.0%], P50 0.043 ms, Table tab:pseudo_label_eval) |
