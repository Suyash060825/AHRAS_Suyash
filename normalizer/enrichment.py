"""
AHRAS Enrichment Engine
------------------------
Provides GeoIP lookup and AbuseIPDB threat intel with:
  - In-memory LRU cache (no repeated API calls for same IP)
  - Private IP detection (RFC 1918 / loopback / link-local)
  - Graceful degradation (returns empty enrichment on failure)
  - Rate limit awareness (stays within free API tiers)
"""

import time
import logging
import requests
from functools import lru_cache

from config.settings import GEO_API_URL, ABUSEIPDB_KEY

log = logging.getLogger(__name__)

# ── Private IP ranges ─────────────────────────────────────────────────────────
_PRIVATE_PREFIXES = (
    "10.", "192.168.", "127.", "::1",
    "172.16.", "172.17.", "172.18.", "172.19.", "172.20.",
    "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
    "172.26.", "172.27.", "172.28.", "172.29.", "172.30.",
    "172.31.", "169.254.", "fc", "fd",
)

_EMPTY_GEO = {"country": "", "city": "", "org": "", "lat": 0.0, "lon": 0.0}
_INTERNAL_GEO = {"country": "internal", "city": "internal",
                  "org": "internal", "lat": 0.0, "lon": 0.0}

# Simple rate limiter: track last N request timestamps
_geo_req_times: list = []
_GEO_RATE_LIMIT = 45   # ip-api.com free tier: 45 req/min


def _is_private(ip: str) -> bool:
    return any(ip.startswith(p) for p in _PRIVATE_PREFIXES)


def _rate_ok() -> bool:
    """Return True if we are within rate limit for GeoIP API."""
    now = time.time()
    # Keep only requests in the last 60 seconds
    _geo_req_times[:] = [t for t in _geo_req_times if now - t < 60]
    if len(_geo_req_times) >= _GEO_RATE_LIMIT:
        return False
    _geo_req_times.append(now)
    return True


@lru_cache(maxsize=2048)
def geoip(ip: str) -> tuple:
    """
    Returns a tuple (country, city, org, lat, lon).
    Cached per IP address — lru_cache requires hashable args.
    """
    if not ip or _is_private(ip):
        return ("internal", "internal", "internal", 0.0, 0.0)

    if not _rate_ok():
        log.debug(f"[ENRICH] GeoIP rate limit — skipping {ip}")
        return ("", "", "", 0.0, 0.0)

    try:
        r = requests.get(GEO_API_URL.format(ip=ip), timeout=3)
        d = r.json()
        if d.get("status") == "fail":
            return ("", "", "", 0.0, 0.0)
        return (
            d.get("country", ""),
            d.get("city", ""),
            d.get("org", ""),
            float(d.get("lat", 0)),
            float(d.get("lon", 0)),
        )
    except Exception as e:
        log.debug(f"[ENRICH] GeoIP failed for {ip}: {e}")
        return ("", "", "", 0.0, 0.0)


def geoip_dict(ip: str) -> dict:
    """Returns geo enrichment as a dict (for embedding in OCSF events)."""
    country, city, org, lat, lon = geoip(ip)
    return {"country": country, "city": city, "org": org, "lat": lat, "lon": lon}


@lru_cache(maxsize=2048)
def abuse_score(ip: str) -> int:
    """
    Returns AbuseIPDB confidence score 0–100.
    Returns 0 if no API key is configured or IP is private.
    """
    if not ABUSEIPDB_KEY or not ip or _is_private(ip):
        return 0
    try:
        r = requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            headers={"Key": ABUSEIPDB_KEY, "Accept": "application/json"},
            params={"ipAddress": ip, "maxAgeInDays": 30},
            timeout=3,
        )
        return int(r.json().get("data", {}).get("abuseConfidenceScore", 0))
    except Exception as e:
        log.debug(f"[ENRICH] AbuseIPDB failed for {ip}: {e}")
        return 0


def enrich_ip(ip: str) -> dict:
    """
    Single call to get all enrichment for an IP.
    Returns dict ready to embed in OCSF enrichment field.
    """
    geo = geoip_dict(ip)
    score = abuse_score(ip)
    return {
        "geo": geo,
        "abuse_score": score,
        "is_private": _is_private(ip),
        "is_threat_intel_hit": score > 50,
        "is_high_risk": score > 80,
    }
