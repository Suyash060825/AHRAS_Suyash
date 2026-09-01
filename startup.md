# AHRAS Pipeline Startup Guide

This document outlines the final, streamlined process for starting and stopping the **Adaptive Hybrid Risk-Aware Security (AHRAS)** pipeline using Docker. The entire architecture (ML, TGNN, ZTRE, SOAR Orchestrator) has been unified into a single containerized service.

## Prerequisites
- **Docker** and **Docker Compose** installed on your system.

## 1. Configuration
Before starting the system, ensure your environment variables are configured. Open the `.env` file in the root directory:
- `EXECUTION_MODE`: Set to `SIMULATED` (default) for local testing or `REAL_PRODUCTION` to send actual webhooks.
- `SOAR_WEBHOOK_URL`: Your TheHive or Shuffle webhook URL (if using `REAL_PRODUCTION`).
- `LOG_LEVEL`: Adjust logging verbosity (e.g., `INFO`, `DEBUG`).

## 2. Startup (Running the System)
To build and start the entire AHRAS pipeline, simply run:

```bash
docker-compose up -d --build
```
*The `-d` flag runs the container in the background (detached mode). The `--build` flag ensures your latest Python code changes are included.*

**To view the live logs (monitoring incoming events and autonomous decisions):**
```bash
docker-compose logs -f ahras-engine
```

## 3. Shutting Down
To gracefully stop the pipeline and tear down the container, run:

```bash
docker-compose down
```
*This safely stops the `ahras-engine` service and cleans up the docker network without deleting your source code or generated models.*

---

## Under the Hood
When you run `docker-compose up`, the `main.py` entrypoint executes. This spins up:
1. The **Hybrid Combiner** to parse OCSF telemetry.
2. The **Temporal GNN** for lateral movement path analysis.
3. The **Dempster-Shafer Risk Engine** for multi-modal evidence fusion.
4. The **Conformal Autonomy Gate** to evaluate uncertainty.
5. The **ZTRE Engine** to adjust session access scopes.
6. The **SOAR Orchestrator** to handle mitigation.
