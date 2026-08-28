"""
AHRAS Configuration
-------------------
DEV_MODE=True   → uses in-process queue + SQLite  (no external services needed)
DEV_MODE=False  → uses real Kafka + MongoDB        (production deployment)

To switch to production, set DEV_MODE=False in .env and ensure
Kafka (localhost:9092) and MongoDB (localhost:27017) are running.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Runtime mode ────────────────────────────────────────────────────────────
DEV_MODE = os.getenv("AHRAS_DEV_MODE", "true").lower() == "true"

# ── Kafka (production) ───────────────────────────────────────────────────────
KAFKA_BOOTSTRAP      = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC_RAW      = "raw.telemetry"
KAFKA_TOPIC_NORM     = "normalized.events"
KAFKA_GROUP_ID       = "ahras-normalizer"

# ── MongoDB (production) ─────────────────────────────────────────────────────
MONGO_URI            = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB             = "ahras"
MONGO_COL_EVENTS     = "events"
MONGO_COL_ALERTS     = "alerts"

# ── SQLite (dev) ─────────────────────────────────────────────────────────────
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
GEO_API_URL          = "http://ip-api.com/json/{ip}?fields=country,city,org,isp,lat,lon"

# ── Suspicious process lineage rules ────────────────────────────────────────
SUSPICIOUS_PARENTS   = {"libreoffice", "soffice", "winword", "excel",
                         "python3", "python", "node", "java", "php", "ruby"}
SUSPICIOUS_CHILDREN  = {"bash", "sh", "zsh", "dash", "nc", "ncat", "nmap",
                         "wget", "curl", "perl", "python3", "python"}

# ── Cloud adapter ────────────────────────────────────────────────────────────
CLOUD_SYNTHETIC      = os.getenv("CLOUD_SYNTHETIC", "true").lower() == "true"
CLOUD_INTERVAL_SEC   = float(os.getenv("CLOUD_INTERVAL_SEC", "2.0"))
