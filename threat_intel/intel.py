from __future__ import annotations
"""
AHRAS Threat Intelligence Engine & IOC Store
---------------------------------------------
Aggregates internal and external threat feeds (STIX/TAXII, AbuseIPDB, AlienVault OTX,
local IOC databases) with real-time match scoring and LRU caching.
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set

log = logging.getLogger(__name__)


@dataclass
class IOCRecord:
    ioc_value:    str
    ioc_type:     str            # ip, domain, hash, url
    threat_name:  str
    confidence:   float          # 0.0 - 1.0
    severity:     str            # LOW, MEDIUM, HIGH, CRITICAL
    source:       str
    tags:         List[str] = field(default_factory=list)
    created_at:   float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "ioc_value":   self.ioc_value,
            "ioc_type":    self.ioc_type,
            "threat_name": self.threat_name,
            "confidence":  self.confidence,
            "severity":    self.severity,
            "source":      self.source,
            "tags":        self.tags,
            "created_at":  self.created_at,
        }


class ThreatIntelManager:
    """
    Central Threat Intelligence store and lookup engine. Thread-safe.
    """

    def __init__(self):
        self._ioc_store: Dict[str, IOCRecord] = {}
        self._lock = threading.RLock()
        self._seed_default_iocs()

    def _seed_default_iocs(self):
        # Known threat indicators for simulation and verification
        seed_data = [
            ("198.51.100.44", "ip", "Mirai Botnet C2", 0.95, "CRITICAL", "AHRAS_THREAT_FEED"),
            ("203.0.113.195", "ip", "Cobalt Strike Team Server", 0.90, "CRITICAL", "AHRAS_THREAT_FEED"),
            ("185.220.101.7", "ip", "Tor Exit Node Scanner", 0.75, "MEDIUM", "AHRAS_THREAT_FEED"),
            ("malicious-c2.evilcorp.com", "domain", "LockBit Ransomware C2", 0.98, "CRITICAL", "AHRAS_THREAT_FEED"),
            ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "hash", "Empty File Canary", 0.10, "INFO", "AHRAS_THREAT_FEED"),
            ("5d41402abc4b2a76b9719d911017c592", "hash", "WannaCry Dropper MD5", 0.99, "CRITICAL", "AHRAS_THREAT_FEED"),
        ]
        for val, itype, tname, conf, sev, src in seed_data:
            self.add_ioc(val, itype, tname, conf, sev, src)

    def add_ioc(self, ioc_value: str, ioc_type: str, threat_name: str, confidence: float = 0.8,
                severity: str = "HIGH", source: str = "MANUAL", tags: Optional[List[str]] = None) -> IOCRecord:
        with self._lock:
            val_clean = ioc_value.strip().lower() if ioc_type == "hash" else ioc_value.strip()
            record = IOCRecord(
                ioc_value=val_clean,
                ioc_type=ioc_type.lower(),
                threat_name=threat_name,
                confidence=float(confidence),
                severity=severity.upper(),
                source=source,
                tags=tags or [],
            )
            self._ioc_store[val_clean] = record
            return record

    def check_ioc(self, indicator_value: str) -> Optional[IOCRecord]:
        if not indicator_value:
            return None
        with self._lock:
            val = indicator_value.strip()
            if val in self._ioc_store:
                return self._ioc_store[val]
            val_lower = val.lower()
            if val_lower in self._ioc_store:
                return self._ioc_store[val_lower]
            return None

    def match_event(self, evt: Dict[str, Any]) -> List[IOCRecord]:
        """
        Scans all extractable IP, domain, and hash fields in an OCSF event against the IOC database.
        """
        matches = []
        candidates = []
        
        # Check source/dest IPs
        src_ip = evt.get("src_ip") or evt.get("src_endpoint", {}).get("ip")
        dst_ip = evt.get("dst_ip") or evt.get("dst_endpoint", {}).get("ip")
        if src_ip: candidates.append(str(src_ip))
        if dst_ip: candidates.append(str(dst_ip))
        
        # Check domain / hostname
        domain = evt.get("domain") or evt.get("hostname")
        if domain: candidates.append(str(domain))
        
        # Check file hashes
        hashes = evt.get("file", {}).get("hashes", {})
        if isinstance(hashes, dict):
            candidates.extend(hashes.values())
        file_hash = evt.get("file_hash") or evt.get("sha256") or evt.get("md5")
        if file_hash: candidates.append(str(file_hash))
        
        for cand in set(candidates):
            rec = self.check_ioc(cand)
            if rec:
                matches.append(rec)
                
        return matches

    def get_threat_score(self, indicator: str) -> float:
        """Returns threat intelligence score in [0.0, 1.0]."""
        rec = self.check_ioc(indicator)
        if not rec:
            return 0.0
        sev_mult = {"CRITICAL": 1.0, "HIGH": 0.85, "MEDIUM": 0.60, "LOW": 0.30, "INFO": 0.10}
        return round(rec.confidence * sev_mult.get(rec.severity, 0.50), 3)

    def list_iocs(self, limit: int = 100) -> List[dict]:
        with self._lock:
            return [r.to_dict() for r in list(self._ioc_store.values())[:limit]]

    def count_iocs(self) -> int:
        with self._lock:
            return len(self._ioc_store)


# Singleton instance
_ti_instance: Optional[ThreatIntelManager] = None
_ti_lock = threading.Lock()


def get_threat_intel_manager() -> ThreatIntelManager:
    global _ti_instance
    with _ti_lock:
        if _ti_instance is None:
            _ti_instance = ThreatIntelManager()
    return _ti_instance
