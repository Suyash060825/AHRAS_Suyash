# ── Multi-Stage / Hardened Non-Root Container for AHRAS SOC Pipeline ─────────
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AHRAS_DEV_MODE=false \
    AHRAS_ENV=PRODUCTION

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group
RUN groupadd -g 10001 ahras && \
    useradd -u 10001 -g ahras -m -s /bin/bash ahras

COPY requirements-core.txt requirements-production.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure storage directories exist and are owned by non-root user
RUN mkdir -p ahras/logs ahras/detection/models evaluation/results evaluation/data && \
    chown -R ahras:ahras /app

USER ahras

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["python3", "-c", "from api.server import start_api_server; start_api_server(host='0.0.0.0', port=8000)"]
