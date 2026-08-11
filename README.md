# AHRAS — Adaptive Hybrid Risk-Aware Security Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 180/180 Passed](https://img.shields.io/badge/tests-180%2F180%20passed-brightgreen.svg)]()
[![OCSF Standard Compliant](https://img.shields.io/badge/schema-OCSF%20v1.1-purple.svg)](https://schema.ocsf.io/)
[![Paper Reproducibility: 1.000 F1](https://img.shields.io/badge/paper--f1-1.000-orange.svg)]()

> **AHRAS** is an enterprise-grade, multi-modal intrusion detection, risk scoring, and active defense framework. Designed for academic research publication (IEEE S&P / USENIX / Computers & Security) and production security operations centers (SOC).

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
                │                                                                                     │
                ▼                                          ▼                                          ▼
   ┌──────────────────────────┐               ┌──────────────────────────┐               ┌──────────────────────────┐
   │  Signature Rule Engine   │               │   ML Anomaly Ensemble    │               │   Statistical Engine     │
   │  (suricata/yara-style)   │               │ (Isolation + AE + SVM)   │               │ (Z-Score, Peer, Trend)   │
   └────────────┬─────────────┘               └────────────┬─────────────┘               └────────────┬─────────────┘
                │                                          │                                          │
                └──────────────────────────────────────────┼──────────────────────────────────────────┘
                                                           │
                                                           ▼
                                    ┌──────────────────────────────────────────────┐
                                    │    Hybrid Combiner & XAI Explainer Engine    │
                                    └──────────────────────┬───────────────────────┘
                                                           │
                                                           ▼
                                    ┌──────────────────────────────────────────────┐
                                    │    Adaptive Risk Engine & Dynamic Trust      │
                                    │ R_t = w1*Ssig + w2*Aml*(1+ΔD) - w3*Ttrust   │
                                    └──────────────────────┬───────────────────────┘
                                                           │
                                                           ▼
                                    ┌──────────────────────────────────────────────┐
                                    │     Active Defense Response Orchestrator     │
                                    │  (Isolate Host / Terminate / Revoke / Block) │
                                    └──────────────────────┬───────────────────────┘
                                                           │
                                                           ▼
                                    ┌──────────────────────────────────────────────┐
                                    │      FastAPI REST API & SOC Web Dashboard    │
                                    └──────────────────────────────────────────────┘
```

---

## 🚀 Key Modules & Novel Innovations

| Module | Core Capability | Key Technical Innovation |
| :--- | :--- | :--- |
| **Module 1** | Telemetry Ingestion & Normalization | Native OCSF Standard Mapping (Classes 1001, 1002, 1003, 4001, 9001) with GeoIP & Threat Intel AbuseIPDB Enrichment. |
| **Module 2** | Hybrid Detection Engine | Tri-Engine Ensemble (Signature + Isolation Forest + Autoencoder + One-Class SVM) hardened via Projected Gradient Descent (PGD) adversarial training. |
| **Module 3** | Statistical Behavioral Baselines | 13-mechanism engine: Effective Z-score floor, EWMA, Circadian histogram $- \log_2 P$, Port affinity, Welford's peer cohort algorithm, multi-day linear trend ramps, and weekend seasonality. |
| **Module 4** | Adaptive Risk & Response | Formal risk scoring equation $R_t = w_1 S_{\text{sig}} + w_2 A_{\text{ml}}(1+\Delta D) - w_3 T_{\text{trust}}$ with dynamic trust decay/recovery and automated active defense. |
| **Module 5** | SOC Dashboard & Paper Reproducibility | Cyber-Glassmorphism Web Dashboard mounted at `GET /` and automated paper benchmark reproduction suite (`eval/reproduce_paper_experiments.py`). |
| **Module 6** | Advanced Commercial Innovations | **1. Federated Learning IDS (FedAvg)** for multi-tenant privacy.<br>**2. LLM Threat Narration** generating CISO breach briefs.<br>**3. Entity Graph Lateral Movement (`T1021`)**.<br>**4. Dynamic Honeypot Engine** for zero-false-positive alerts. |

---

## 📊 Paper Evaluation Benchmark Results

Evaluated across 6 primary threat vectors using `eval/reproduce_paper_experiments.py`:

| Scenario Name | OCSF Class | Precision | Recall | F1-Score | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Network: Port Scanning** | `network_activity` | **1.000** | **1.000** | **1.000** | 40.97 ms |
| **Network: SYN Flood** | `network_activity` | **1.000** | **1.000** | **1.000** | 40.78 ms |
| **Network: SSH Brute Force** | `network_activity` | **1.000** | **1.000** | **1.000** | 41.34 ms |
| **Host File: Ransomware Entropy** | `file_activity` | **1.000** | **1.000** | **1.000** | 39.89 ms |
| **Host Process: Credential Dump** | `process_activity` | **1.000** | **1.000** | **1.000** | 39.27 ms |
| **Cloud API: Defense Evasion** | `cloud_api` | **1.000** | **1.000** | **1.000** | 38.27 ms |

---

## 💻 Quick Start & Setup Guide

### 1. Requirements & Installation

```bash
# Clone the repository
git clone https://github.com/your-org/AHRAS_Final.git
cd AHRAS_Final

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Unit & System Integration Tests (180 Tests)

```bash
# Run complete test discovery across all modules
PYTHONPATH=. python3 -m unittest discover -s tests -p "test_*.py"
```

### 3. Launching the Production SOC REST API & Web Dashboard

```bash
# Launch server on port 8000
python3 -c "from api.server import start_api_server; start_api_server(port=8000)"
```

Navigating to **`http://localhost:8000/`** in your browser will present the real-time SOC Web Dashboard.

### 4. Reproducing Paper Experiments & LaTeX Table Generation

```bash
# Execute evaluation harness
python3 eval/reproduce_paper_experiments.py
```
Output LaTeX code is generated directly in `eval/paper_results_table.tex`.

---

## 📡 REST API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` or `/dashboard` | Interactive Cyber-Glassmorphism SOC Web Dashboard |
| `GET` | `/health` | Health status and dual-mode database engine state |
| `GET` | `/alerts` | Query historical alerts filterable by severity & OCSF class |
| `GET` | `/entities/{key}/report` | Unified per-entity security report (JSON & Markdown) |
| `POST` | `/alerts/{id}/respond` | Approve or reject staged active defense mitigation actions |
| `GET` | `/actions/pending` | List active defense actions awaiting SOC approval |
| `GET` | `/actions/history` | Full audit trail of executed active defense mitigations |
| `POST` | `/analyst/feedback` | Submit false-positive feedback or reset entity baselines |
| `GET` | `/metrics` | Prometheus-compatible SOC operational metrics |

---

## 📜 Citation

If you use the AHRAS framework or dataset generators in your academic research, please cite:

```bibtex
@article{ahras2026framework,
  title={AHRAS: An Adaptive Hybrid Risk-Aware Security Framework for Multi-Modal Cyber Threat Detection},
  author={Pradhan, Suyash and AHRAS Development Team},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026}
}
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
