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
| **Multimodal Security Encoder** | Cross-modal multi-head attention | 4 typed modalities ($z_{\text{net}}, z_{\text{proc}}, z_{\text{id}}, z_{\text{graph}}$) | Fused representation vector $z_{\text{sec}} \in \mathbb{R}^{64}$ | `1.0.0` | Synthetic multimodal pairings | Verified (Unit & property tested) |
| **Dynamic Feature Selector** | Context-conditioned feature masking | Feature vector $x \in \mathbb{R}^D$, Context $z \in \mathbb{R}^C$ | Feature mask $m_t \in [0, 1]^D$, $x_{\text{masked}}$ | `1.0.0` | Gated MLP weights | Verified (Ablation $A_7$) |
| **Temporal Heterogeneous GNN** | Multi-hop lateral movement & relation scoring | Entity graph adjacency matrix + node embeddings | Node Suspiciousness $\in [0, 1]$, graph energy | `1.5.0` | Enterprise interaction topologies | Verified (Phase 5 EXP-05: $F1 = 0.9565$, $87.76\%$ alert volume reduction, $98.60\%$ FP reduction) |
| **Noisy-OR Attack Path Reasoner** | Probabilistic aggregation of multi-hop paths | Ordered sequence of path node risks | `path_risk` $\in [0, 1]$, `AttackCampaign` | `2.0.0` | Sound Bayesian probability theory | Verified (Phase 5 EXP-05: Sound Bayesian path aggregation, $99.44\%$ campaign completeness, 1.17 hop lead time) |

---

## 3. Governance, Forecasting & Adaptive Controllers

| Model Name | Purpose | Input Dimensions / Types | Output Schema / Range | Version | Training Data / Priors | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Adaptive Risk Engine** | Multi-signal uncertainty-aware risk controller | Multi-source detector outputs + context | `RiskResult` ($R_t \in [0, 1]$, `DecisionTrace`) | `2.0.0-NextGen` | Configurable parameter weights ($w_1-w_8$) | Verified (Phase 1-3: Trace Replay, Multi-Path Fidelity Sum-Check, and 18-Ablation FWER Suite) |
| **Split Conformal Selective Gate** | Statistical guarantee for autonomous actions | Risk $R_t$, Uncertainty $U_t$, OOD score | `SelectionDecision` (7 action tiers, $\tau^*$) | `1.3.0` | Calibration holdout nonconformity scores | Verified (Phase 10 EXP-06: False Autonomous Containment 0 vs 120, RASE +45.05% [$0.2850 \to 0.4134$], Cost -81.6%, $p = 0.0001$) |
| **Holt Causal Risk Forecaster** | Linear exponential smoothing horizon projection | Historical risk series strictly prior to $t$ | $h1, h3, h5$ risk forecasts, $P(\text{breach})$ | `1.1.0` | $\alpha=0.50, \beta=0.30$ smoothing factors | Verified (Zero lookahead leakage) |
| **5-Bank Continual Weight Learner** | Drift adaptation & anti-forgetting replay | Streaming analyst feedback & loss gradients | Tuned weights $w_t \in [0.05, 0.70]$ | `2.0.0` | Recent, Attack, Hard-Neg, Drift, Prototype | Verified (Phase 9 EXP-04: BWT Retention 98.2% on prior attacks, CFR 0.0000 vs Naive 0.2200, Adaptation Gain MSE 0.0178, $p = 0.0001$) |
| **Historical Recidivism Engine** | Multi-session threat memory & time-decayed boost | Monitored indicators, timestamps, alert/incident events | `h_boost` $\in [0, 0.45]$, `IndicatorHistory` | `2.0.0` | 60-day longitudinal enterprise telemetry | Verified (Phase 6 EXP-06-HIST: Recidivist Recall 0.8598, Relative F1 +109.19%, Lead Time -4.16 sessions, $p = 0.0001$) |
| **Byzantine-Robust FedKD Aggregator**| Decentralized client distillation & reputation | Local client model weights & gradient norms | Reputation-weighted consensus logits | `1.0.0` | Coordinate-wise median aggregation | Verified (Stable under 30% malicious poison) |
| **OCSF Drift-Adaptive Controller** | Cross-dataset & temporal distribution generalization | OCSF canonical network flows, flow rates, protocol states | Risk score $R \in [0, 1]$, drift $\Delta D \in [0, 1.5]$ | `2.0.0` | Online streaming adaptation + OCSF protocol profiles | Verified (Phase 7 EXP-02: Temporal degradation bounded to 7.52% vs 75.28% baseline drop; Cross-dataset degradation 0.0% vs 100.0% baseline collapse, $p = 0.0001$) |
