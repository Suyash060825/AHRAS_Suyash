# AHRAS (Adaptive Hybrid Risk-Aware Security) — Full Scale Startup & Deployment Guide

This guide provides comprehensive, step-by-step instructions for running the **AHRAS Framework** in all environments, from local development to production-scale containerized and Kubernetes deployments.

---

## 📋 Table of Contents
1. [System Overview & Architecture](#-system-overview--architecture)
2. [Prerequisites & Environment Setup](#-prerequisites--environment-setup)
3. [Configuration & Environment Variables](#-configuration--environment-variables)
4. [Deployment Modes](#-deployment-modes)
   - [Mode 1: Local / Standalone Development](#mode-1-local--standalone-development)
   - [Mode 2: Full-Scale Production (Docker Compose)](#mode-2-full-scale-production-docker-compose)
   - [Mode 3: Enterprise Kubernetes Deployment](#mode-3-enterprise-kubernetes-deployment)
5. [Running Sensor Agents & Telemetry Ingestion](#-running-sensor-agents--telemetry-ingestion)
6. [Executing Research, Benchmarks & Test Suites](#-executing-research-benchmarks--test-suites)
7. [SOC Dashboard & REST API Usage](#-soc-dashboard--rest-api-usage)
8. [Active Defense Response Modes & Safety Controls](#-active-defense-response-modes--safety-controls)
9. [Troubleshooting & Maintenance](#-troubleshooting--maintenance)

---

## 🏛 System Overview & Architecture

AHRAS operates as a closed-loop cyber defense controller with the following pipeline:

```
[Sensors: Host / Network / Cloud] 
       │ (Raw Telemetry)
       ▼
[OCSF Normalizer & Enrichment] 
       │ (Standardized Event Dicts)
       ▼
[Multimodal Security Encoder & Dynamic Masking]
       │
       ▼
[Evidence Quality & De-Correlation Engine]
       │
       ▼
[Uncertainty-Aware Adaptive Risk Controller]
       │
       ▼
[Conformal Selective Autonomy Gate] ──► [SOAR Active Mitigation]
       │                                         │
       ▼                                         ▼
[SOC REST API & Web Dashboard] ◄────── [Auditable Cryptographic Ledger]
```

---

## ⚙️ Prerequisites & Environment Setup

### 1. System Requirements
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+) or macOS
- **Python**: Version `3.9` to `3.14`
- **Containerization** (Optional for Prod): Docker 24.0+ & Docker Compose v2+
- **Kubernetes** (Optional for Enterprise): `kubectl` connected to a running cluster (k8s 1.25+)
- **System Packages**: `libpcap-dev` (if capturing live network packets via Scapy), `curl`, `git`

### 2. Python Virtual Environment

```bash
# Clone or navigate to the repository
cd /home/suyashpradhan/Downloads/AHRAS_Suyash-master

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel
```

### 3. Dependency Installation Options

Choose the dependency profile according to your target environment:

- **Core / Development**:
  ```bash
  pip install -r requirements.txt
  pip install -r requirements-dev.txt
  ```

- **Full Scale Production (with Kafka, MongoDB, Scapy, Boto3)**:
  ```bash
  pip install -r requirements-production.txt
  ```

- **Research & Scientific Evaluation**:
  ```bash
  pip install -r requirements-research.txt
  ```

---

## 🔐 Configuration & Environment Variables

Initialize the `.env` file from `.env.example`:

```bash
cp .env.example .env
```

### Key Environment Parameters (`.env`)

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `AHRAS_ENV` | `DEV` | Runtime environment: `DEV`, `STAGING`, `PRODUCTION` |
| `AHRAS_DEV_MODE` | `true` | When `true`, uses in-memory queues & SQLite; when `false`, enables Kafka & MongoDB |
| `AHRAS_SECRET_KEY` | *Hex String* | Mandatory 32+ character high-entropy key for JWT token signing in production |
| `AHRAS_HOST` | `0.0.0.0` | API bind address |
| `AHRAS_PORT` | `8000` | API and SOC dashboard port |
| `AHRAS_RESPONSE_MODE` | `DRY_RUN` | Action execution mode: `DRY_RUN`, `SIMULATED`, `SANDBOX`, `REAL_PRODUCTION` |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string for production persistence |
| `KAFKA_BOOTSTRAP` | `localhost:9092` | Kafka broker endpoints for telemetry streaming |
| `KAFKA_TOPIC_RAW` | `raw.telemetry` | Topic for raw sensor stream |
| `KAFKA_TOPIC_NORM` | `normalized.events` | Topic for OCSF normalized events |

> [!IMPORTANT]
> In `PRODUCTION` mode (`AHRAS_ENV=PRODUCTION`), the server enforces strict secret entropy validation and will fail-closed if default/weak keys are used.

---

## 🚀 Deployment Modes

### Mode 1: Local / Standalone Development

Ideal for rapid testing, local research, and SOC dashboard exploration.

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Verify all unit and integration tests (262 tests)
pytest

# 3. Start the SOC REST API & Dashboard
python3 -c "from api.server import start_api_server; start_api_server(host='127.0.0.1', port=8000)"
```

- **SOC Web Dashboard**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Liveness Probe**: `curl http://127.0.0.1:8000/health/live`

---

### Mode 2: Full-Scale Production (Docker Compose)

Runs the complete distributed stack:
- **AHRAS Core SOC API Server** (Resource limits: 2 CPU, 2048MB RAM)
- **MongoDB 6.0** (Persistent storage)
- **Apache Kafka 7.3.0 & Zookeeper** (High-throughput message bus)

```bash
# 1. Build and launch all production services in the background
docker compose --profile prod up --build -d

# 2. Check container status
docker compose ps

# 3. View streaming logs
docker compose logs -f ahras-api

# 4. Verify system readiness
curl -f http://localhost:8000/health/ready

# 5. Stop the stack when needed
docker compose --profile prod down
```

---

### Mode 3: Enterprise Kubernetes Deployment

For multi-node orchestration and horizontal scaling.

```bash
# 1. Apply ConfigMap
kubectl apply -f k8s/configmap.yaml

# 2. Apply Deployment
kubectl apply -f k8s/deployment.yaml

# 3. Apply Service
kubectl apply -f k8s/service.yaml

# 4. Monitor rollout status
kubectl rollout status deployment/ahras-api

# 5. Check pods and service endpoints
kubectl get pods -l app=ahras-api
kubectl get svc ahras-service
```

---

## 📡 Running Sensor Agents & Telemetry Ingestion

AHRAS supports multi-modal endpoint, network, and cloud telemetry ingestion:

### 1. Host Agent (Process, File Entropy, Network Connections)
Monitors process spawning, Shannon entropy for ransomware detection ($H(X) > 7.2$), and outbound connections.

```bash
# Run standalone daemon
python3 -m sensors.host_agent

# Or launch as background daemon within a Python process:
# from sensors.host_agent import start_host_agent_thread; start_host_agent_thread()
```

### 2. Network Sensor (PCAP & Live Flow Capture)
Captures SYN scans, lateral movement, and high-frequency connection patterns.

```bash
# Requires root/sudo or CAP_NET_RAW for live interface sniffing
sudo python3 -m sensors.network_sensor
```

### 3. Cloud Telemetry Adapter (AWS CloudWatch / CloudTrail / Azure)
Streams cloud audit and identity logs into the normalized OCSF event pipeline.

```bash
python3 -m sensors.cloud_adapter
```

---

## 🧪 Executing Research, Benchmarks & Test Suites

### 1. Unit & Regression Test Suite
Runs all 262 verification and mathematical fidelity tests:
```bash
pytest
```

### 2. Comprehensive Live Research Benchmark
Executes the live computational evaluation without mock data across all 6 OCSF threat categories:
```bash
python3 evaluation/run_comprehensive_research.py
```

### 3. Leakage-Safe Research Matrix (E0–E12 & 12 Ablations)
Runs temporal-split experiments ensuring zero lookahead leakage:
```bash
python3 evaluation/research_experiments.py
```

### 4. Paper Benchmark Reproducibility & LaTeX Export
Reproduces all empirical tables, confidence intervals, and LaTeX outputs:
```bash
python3 eval/reproduce_paper_experiments.py
```

---

## 🖥 SOC Dashboard & REST API Usage

### Key API Endpoints

- **Health Checks**:
  - `GET /health/live` — Instant service liveness probe
  - `GET /health/ready` — Deep readiness probe (DB, models, bus connectivity)
- **Authentication**:
  - `POST /auth/login` — OAuth2 JWT token acquisition
  - `POST /auth/register` — User registration (admin only in prod)
- **Telemetry & Event Ingestion**:
  - `POST /api/v1/events/ingest` — Ingest raw or normalized OCSF event
- **Risk Scoring & Explanations**:
  - `POST /api/v1/risk/score` — Compute uncertainty-aware risk and selective gate decision
  - `GET /api/v1/entity/{entity_id}/report` — Comprehensive risk, trajectory, and causal attribution
- **Active Response Execution**:
  - `POST /api/v1/response/action` — Trigger safety-gated active defense action (isolate host, revoke token, block IP)

---

## 🛡 Active Defense Response Modes & Safety Controls

The `AHRAS_RESPONSE_MODE` setting in `.env` controls how the SOAR engine executes mitigations:

1. `DRY_RUN` *(Default)*: Evaluates policy conditions and records decision trace in the cryptographic ledger without executing actions on the host/network.
2. `SIMULATED`: Generates mock execution metrics and responses for SOC validation.
3. `SANDBOX`: Executes actions within designated virtual sandbox environments.
4. `REAL_PRODUCTION`: Directly applies firewall rules (`iptables`/`nftables`), process terminations, and IAM revocations.

---

## 🔧 Troubleshooting & Maintenance

- **Port 8000 already in use**:
  Change port via `.env`: `AHRAS_PORT=8080` or pass `--port 8080`.
- **Kafka Connection Timeout (Production)**:
  Ensure Zookeeper is healthy and check broker logs: `docker compose logs kafka`.
- **Permission Denied on Network Sniffing**:
  Grant network capabilities to Python binary: `sudo setcap cap_net_raw,cap_net_admin=eip $(which python3)`.
- **Resetting In-Memory / SQLite State**:
  Remove `ahras/logs/ahras_dev.db` to recreate clean local state.
