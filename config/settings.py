from __future__ import annotations
"""
AHRAS Configuration & Environment Profiles
-------------------------------------------
Manages runtime configuration, secrets validation, fail-closed production guarantees,
and environment profiles (DEV, STAGING, PRODUCTION).
"""

import os
import sys
import logging
from typing import List, Set
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── Runtime Mode & Profile ───────────────────────────────────────────────────
AHRAS_ENV = os.getenv("AHRAS_ENV", "").upper()
_DEV_MODE_RAW = os.getenv("AHRAS_DEV_MODE", "").lower()

if AHRAS_ENV:
    DEV_MODE = (AHRAS_ENV == "DEV")
elif _DEV_MODE_RAW:
    DEV_MODE = (_DEV_MODE_RAW in ("true", "1", "yes"))
    AHRAS_ENV = "DEV" if DEV_MODE else "PRODUCTION"
else:
    DEV_MODE = True
    AHRAS_ENV = "DEV"

# ── Secrets & Authentication ──────────────────────────────────────────────────
# Insecure well-known secrets that MUST be blocked in non-DEV profiles
_KNOWN_INSECURE_SECRETS = {
    "ahras-enterprise-secret-key-2026-production-hardening",
    "ahras-production-hardening-secret-2026",
    "secret", "changeme", "admin", "password", "12345678",
    "replace-with-a-secure-random-32-byte-hex-key-in-production"
}

_RAW_SECRET = os.getenv("AHRAS_SECRET_KEY", "")

def validate_secrets_and_environment():
    """
    Validates secret strength and configuration fail-closed behavior.
    Raises RuntimeError on unsafe production startup.
    """
    global _RAW_SECRET
    if not DEV_MODE or AHRAS_ENV in ("PRODUCTION", "STAGING"):
        if not _RAW_SECRET:
            raise RuntimeError(
                "[CRITICAL SECURITY ERROR] Mandatory secret AHRAS_SECRET_KEY is missing in PRODUCTION/STAGING mode. "
                "Startup aborted (Fail-Closed)."
            )
        if _RAW_SECRET in _KNOWN_INSECURE_SECRETS or len(_RAW_SECRET) < 32:
            raise RuntimeError(
                f"[CRITICAL SECURITY ERROR] Insecure or weak AHRAS_SECRET_KEY detected in {AHRAS_ENV} mode "
                f"(length={len(_RAW_SECRET)}, must be >= 32 characters and cryptographically random). "
                "Startup aborted (Fail-Closed)."
            )
    else:
        if not _RAW_SECRET:
            # Safe development fallback only
            _RAW_SECRET = "dev-mode-only-insecure-fallback-key-32-bytes-long!"

validate_secrets_and_environment()
AHRAS_SECRET_KEY = _RAW_SECRET or "dev-mode-only-insecure-fallback-key-32-bytes-long!"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("AHRAS_TOKEN_EXPIRE_MINUTES", "480"))
REFRESH_TOKEN_EXPIRE_MINUTES = int(os.getenv("AHRAS_REFRESH_TOKEN_EXPIRE_MINUTES", "1440"))
JWT_ALGORITHM = "HS256"

# ── API & Network Hardening ───────────────────────────────────────────────────
AHRAS_HOST = os.getenv("AHRAS_HOST", "0.0.0.0")
AHRAS_PORT = int(os.getenv("AHRAS_PORT", "8000"))

_cors_env = os.getenv("AHRAS_ALLOWED_ORIGINS", "")
if _cors_env:
    ALLOWED_ORIGINS: List[str] = [o.strip() for o in _cors_env.split(",") if o.strip()]
elif DEV_MODE:
    ALLOWED_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"]
else:
    ALLOWED_ORIGINS = []  # Fail-closed in prod if not explicitly set

RATE_LIMIT_PER_MINUTE = int(os.getenv("AHRAS_RATE_LIMIT_PER_MINUTE", "120"))
MAX_REQUEST_BYTES = int(os.getenv("AHRAS_MAX_REQUEST_BYTES", str(2 * 1024 * 1024))) # 2MB

# ── Response Execution Mode ───────────────────────────────────────────────────
# Modes: DRY_RUN | SIMULATED | SANDBOX | REAL_PRODUCTION
RESPONSE_MODE = os.getenv("AHRAS_RESPONSE_MODE", "DRY_RUN" if DEV_MODE else "SIMULATED").upper()

# ── Kafka (production message bus) ────────────────────────────────────────────
KAFKA_BOOTSTRAP      = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC_RAW      = "raw.telemetry"
KAFKA_TOPIC_NORM     = "normalized.events"
KAFKA_GROUP_ID       = "ahras-normalizer"

# ── MongoDB (production storage) ─────────────────────────────────────────────
MONGO_URI            = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB             = os.getenv("MONGO_DB", "ahras")
MONGO_COL_EVENTS     = "events"
MONGO_COL_ALERTS     = "alerts"

# ── SQLite (dev storage) ─────────────────────────────────────────────────────
SQLITE_PATH          = os.getenv("SQLITE_PATH", "ahras/logs/ahras_dev.db")

# ── Network capture ──────────────────────────────────────────────────────────
NETWORK_INTERFACE    = os.getenv("NETWORK_INTERFACE", "eth0")
FLOW_WINDOW_SECONDS  = int(os.getenv("FLOW_WINDOW_SECONDS", "10"))

# ── Host agent ───────────────────────────────────────────────────────────────
WATCH_DIRS           = os.getenv("WATCH_DIRS", "/home,/tmp,/var/tmp").split(",")
ENTROPY_THRESHOLD    = float(os.getenv("ENTROPY_THRESHOLD", "7.2"))
PROCESS_POLL_SEC     = int(os.getenv("PROCESS_POLL_SEC", "5"))

# ── Threat intel ─────────────────────────────────────────────────────────────
ABUSEIPDB_KEY        = os.getenv("ABUSEIPDB_KEY", "")
OTX_API_KEY          = os.getenv("OTX_API_KEY", "")
GEO_API_URL          = "http://ip-api.com/json/{ip}?fields=country,city,org,isp,lat,lon"

# ── Suspicious process lineage rules ────────────────────────────────────────
SUSPICIOUS_PARENTS: Set[str] = {
    "libreoffice", "soffice", "winword", "excel",
    "python3", "python", "node", "java", "php", "ruby"
}
SUSPICIOUS_CHILDREN: Set[str] = {
    "bash", "sh", "zsh", "dash", "nc", "ncat", "nmap",
    "wget", "curl", "perl", "python3", "python"
}

# ── Cloud adapter ────────────────────────────────────────────────────────────
CLOUD_SYNTHETIC      = os.getenv("CLOUD_SYNTHETIC", "true").lower() == "true"
CLOUD_INTERVAL_SEC   = float(os.getenv("CLOUD_INTERVAL_SEC", "2.0"))

# ── ML Detection & Paths ─────────────────────────────────────────────────────
AHRAS_MODEL_DIR      = os.getenv("AHRAS_MODEL_DIR", "ahras/detection/models")
IF_CONTAMINATION     = float(os.getenv("IF_CONTAMINATION", "0.05"))
SVM_NU               = float(os.getenv("SVM_NU", "0.02"))
AE_ERROR_PCTILE      = float(os.getenv("AE_ERROR_PCTILE", "97.0"))
ENSEMBLE_THRESH      = float(os.getenv("ENSEMBLE_THRESH", "0.75"))
