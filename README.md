# AHRAS — Adaptive Hybrid Risk-Aware Security Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 243/243 Passed](https://img.shields.io/badge/tests-243%2F243%20passed-brightgreen.svg)]()
[![OCSF Standard Compliant](https://img.shields.io/badge/schema-OCSF%20v1.1-purple.svg)](https://schema.ocsf.io/)
[![Benchmark F1: 0.989 avg](https://img.shields.io/badge/benchmark--f1-0.989-orange.svg)]()

> **AHRAS** is an evidence-driven, closed-loop cyber defense controller. It converts heterogeneous multi-modal detection evidence into uncertainty-aware entity/episode risk, executes safety-gated active response decisions, and maintains complete cryptographic provenance and exact mathematical reconstructibility for all security decisions.

---

## 🏛️ System Architecture Overview

```
                                    ┌──────────────────────────────────────────────┐
                                    │    Suricata / eBPF / CloudWatch Sensors      │
                                    └──────────────────────┬───────────────────────┘
                                                           │ (Raw Telemetry)
                                                           ▼
                                    ┌──────────────────────────────────────────────┐
                                    │       OCSF Schema Normalizer & Enrichment    │
                                    └──────────────────────┬───────────────────────┘
                                                           │ (OCSF Schema Event Dict)
                                                           ▼
                ┌──────────────────────────────────────────┴──────────────────────────────────────────┐
                │                                          │                                          │
                ▼                                          ▼                                          ▼
   ┌──────────────────────────┐               ┌──────────────────────────┐               ┌──────────────────────────┐
   │  Signature Rule Engine   │               │   ML Anomaly Ensemble    │               │   Statistical Engine     │
   │  (23 MITRE Rules)        │               │ (Isolation + AE + SVM)   │               │ (Z-Score, Peer, Trend)   │
   └────────────┬─────────────┘               └────────────┬─────────────┘               └────────────┬─────────────┘
                │                                          │                                          │
                └──────────────────────────────────────────┼──────────────────────────────────────────┘
                                                           │
                                                           ▼
                                    ┌──────────────────────────────────────────────┐
                                    │     Immutable Tamper-Evident Evidence Ledger │
                                    │    (SHA-256 Hashed EvidenceRecord Stream)    │
                                    └──────────────────────┬───────────────────────┘
                                                           │
                                                           ▼
                                    ┌──────────────────────────────────────────────┐
                                    │  Uncertainty-Aware Adaptive Risk Controller  │
                                    │  R_t = F(E_t, C_t, H_t, G_t, U_t, P_t, A_t) │
                                    └──────────────────────┬───────────────────────┘
                                                           │
                                                           ▼
                                    ┌──────────────────────────────────────────────┐
                                    │    Active Defense Safety Policy Engine       │
                                    │ Utility(a) = ΔR·Conf - Blast - RevCost - Unc │
                                    │ (DRY_RUN | SIMULATED | SANDBOX | PRODUCTION) │
                                    └──────────────────────┬───────────────────────┘
                                                           │
                                                           ▼
                                    ┌──────────────────────────────────────────────┐
                                    │    Hardened FastAPI API & SOC Dashboard      │
                                    │ (Rate Limiting, RFC JWT, Security Headers)   │
                                    └──────────────────────────────────────────────┘
```

---

## 🚀 Key Modules & Research Contributions

| Module | Core Capability | Key Technical Innovation |
| :--- | :--- | :--- |
| **Evidence Ledger** | Standardized Multi-Modal Evidence | Immutable, SHA-256 hashed `EvidenceRecord` tracking detector type, version, confidence, uncertainty, MITRE mapping, and provenance. |
| **Hybrid Combiner** | Tri-Engine Detection | Tri-Engine Ensemble (Signature + Isolation Forest + Autoencoder + One-Class SVM) with calibrated probability mapping. |
| **Statistical Engine** | Behavioral Baselines | 13-mechanism engine: Effective Z-score floor, EWMA, Circadian histogram $-\log_2 P$, Port affinity, and Welford's peer cohort algorithm. |
| **Adaptive Risk Engine** | Uncertainty-Aware Risk Scoring | Formal risk equation $R_t = \text{Clip}_0^1 [(w_1 S_{\text{sig}} + w_2 A_{\text{ml}}(1+\Delta D) + w_4 H_{\text{boost}} + w_5 G_{\text{corr}} + w_6 P_{\text{fore}} + w_7 TI) \cdot A_{\text{crit}} \cdot (1-U) - w_3 T_{\text{trust}}]$ with exact XAI reconstructibility ($\Delta \le 10^{-4}$). |
| **Episode Graph** | Temporal Attack Chains | Multi-hop BFS lateral movement detection (`T1021`, `T1078`, `T1059`) with graph corroboration boost to separate isolated anomalies from coordinated campaigns. |
| **Policy Engine** | Safety-Gated Active Defense | Multi-mode execution (`DRY_RUN`, `SIMULATED`, `SANDBOX`, `REAL_PRODUCTION`), action utility optimization, analyst approval queues with 1h expiry, and counterfactual sensitivity analysis. |
| **Causal Forecaster** | Early Warning Escalation | Holt's linear smoothing forecasting ($h=1,3,5$) with Gaussian CDF threshold-crossing probability and zero future leakage. |
| **Adaptive Learning** | Controlled Weight Tuning | Shadow learning mode, rolling validation buffer, stability constraints ($\Delta w \le 0.05$), and automated freeze on validation drift. |
| **Federated IDS** | Multi-Tenant Collaboration | Differentiable parameter averaging over neural Autoencoders with Byzantine gradient norm clipping and NaN/Inf rejection. |
| **Threat Intel & Deception** | High-Information Evidence | Freshness decay $\exp(-\lambda \Delta t)$ for IOCs and isolated honeypot tripwires emitting high-confidence evidence. |

---

## 📊 Empirical Evaluation Benchmark Results

Evaluated across multi-modal threat vectors with 95% bootstrap confidence intervals (`python3 eval/reproduce_paper_experiments.py`):

| Threat Scenario | OCSF Class | Precision | Recall | F1-Score | 95% Bootstrap CI | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Network: Port Scanning** | `network_activity` | **0.975** | **0.975** | **0.975** | [0.950, 1.000] | 2.85 ms |
| **Network: SYN Flood** | `network_activity` | **1.000** | **0.975** | **0.987** | [0.962, 1.000] | 2.92 ms |
| **Network: SSH Brute Force** | `network_activity` | **0.975** | **0.975** | **0.975** | [0.950, 1.000] | 2.68 ms |
| **Host File: Ransomware Entropy** | `file_activity` | **1.000** | **1.000** | **1.000** | [1.000, 1.000] | 2.76 ms |
| **Host Process: Credential Dump** | `process_activity` | **1.000** | **1.000** | **1.000** | [1.000, 1.000] | 2.71 ms |
| **Cloud API: Defense Evasion** | `cloud_api` | **1.000** | **1.000** | **1.000** | [1.000, 1.000] | 2.83 ms |

---

## 💻 Quick Start & Setup Guide

### 1. Requirements & Installation

```bash
# Clone the repository
git clone https://github.com/your-org/AHRAS_Final.git
cd AHRAS_Final

# Install core dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite (228 Tests)

```bash
python3 -m pytest
```

### 3. Run Leakage-Safe Research Matrix (E0–E12 & 12 Ablations)

```bash
python3 evaluation/research_experiments.py
```

### 4. Run Paper Benchmark Reproducibility & LaTeX Export

```bash
python3 eval/reproduce_paper_experiments.py
```

### 5. Launch Hardened SOC API & Web Dashboard

```bash
python3 -c "from api.server import start_api_server; start_api_server(host='127.0.0.1', port=8000)"
```

Access the Web Dashboard at: `http://localhost:8000` (or `http://localhost:8000/health/live` for healthcheck).

---

## 📜 Scientific Claims & Limitations Discipline

- **Exact Reconstruction**: XAI risk reconstructibility is guaranteed within absolute error $\epsilon \le 10^{-4}$ against the discrete evidence components.
- **Leakage Safety**: All evaluation experiments use chronological and entity-disjoint splits with zero future leakage.
- **Federated Scope**: Federated averaging is strictly applied to differentiable Autoencoder parameter vectors; non-differentiable decision tree weights are never averaged.
- **Response Modes**: Active defense commands default to `DRY_RUN` in development and require explicit environment configuration and utility gating for production execution.
