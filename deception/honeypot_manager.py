from __future__ import annotations
"""
AHRAS Module 13 — Active Deception & Dynamic Honeypot Engine
-------------------------------------------------------------
Deploys and monitors dynamic honey-tokens, decoy listeners, and canary files.

Research Properties:
  1. High-Information Evidence Generation: Any interaction is deterministic confirmation
     of unauthorized actor engagement (near-zero false positive probability under clean deployment).
  2. Isolated Evidence Channel: Deception alerts are registered as discrete EvidenceRecord
     contributions without skewing statistical baseline distributions.
  3. MITRE Technique Attribution: Automatically tags adversary TTPs (T1078, T1021, T1046, T1083).
"""

import time
import uuid
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

from ahras.evidence.models import EvidenceRecord, EvidenceType, EvidenceSource
from ahras.evidence.ledger import get_evidence_ledger

log = logging.getLogger(__name__)

LURE_MITRE_MAP = {
    "HONEY_TOKEN":       "T1078.004",  # Cloud Accounts / Canary Credentials
    "FAKE_PORT":         "T1046",      # Network Service Discovery
    "DECOY_FILE":        "T1083",      # File and Directory Discovery
    "CANARY_CREDENTIAL": "T1552",      # Unsecured Credentials
}


@dataclass
class DeceptionLure:
    """Represents a dynamic honeypot lure token or endpoint."""
    lure_id:          str
    lure_type:        str           # HONEY_TOKEN, FAKE_PORT, DECOY_FILE, CANARY_CREDENTIAL
    target_entity:    str
    lure_key:         str           # Fake AWS key, decoy file path, port number
    deployed_at:      float
    mitre_technique:  str = "T1078"
    is_triggered:     bool = False
    triggered_at:     Optional[float] = None
    attacker_ip:      Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class DeceptionManager:
    """
    Manages deployment and interaction monitoring of dynamic honeypots. Thread-safe.
    """

    def __init__(self):
        self._active_lures: Dict[str, DeceptionLure] = {}
        self._triggered_log: List[DeceptionLure] = []
        self._lock = threading.RLock()

    def deploy_lure_for_entity(self, entity_key: str, risk_score: float, lure_type: str = "HONEY_TOKEN") -> Optional[DeceptionLure]:
        """Deploys dynamic honeypot lure if entity risk crosses high threshold (>= 0.70)."""
        if risk_score < 0.70:
            return None

        with self._lock:
            lure_id = f"LURE-{str(uuid.uuid4())[:8]}"
            lkey = f"AKIAIOSFODNN7EXAMPLE-{lure_id}" if lure_type == "HONEY_TOKEN" else f"/var/secrets/canary_db_backup_{lure_id}.kdbx"
            mtech = LURE_MITRE_MAP.get(lure_type, "T1078")

            lure = DeceptionLure(
                lure_id=lure_id,
                lure_type=lure_type,
                target_entity=entity_key,
                lure_key=lkey,
                deployed_at=time.time(),
                mitre_technique=mtech,
            )
            self._active_lures[lure.lure_key] = lure
            log.info(f"[DECEPTION] Deployed dynamic {lure_type} for suspicious entity '{entity_key}' (lure_id={lure_id})")
            return lure

    def check_interaction(self, accessed_key: str, attacker_ip: Optional[str] = None) -> Optional[DeceptionLure]:
        """
        Evaluates whether an accessed resource matches a deployed honeypot lure.
        Emits an EvidenceRecord into the evidence ledger on trigger.
        """
        with self._lock:
            if accessed_key in self._active_lures:
                lure = self._active_lures[accessed_key]
                lure.is_triggered = True
                lure.triggered_at = time.time()
                lure.attacker_ip = attacker_ip
                self._triggered_log.append(lure)
                
                # Emit high-confidence EvidenceRecord
                ev = EvidenceRecord(
                    event_id=f"EVT-DECEPT-{uuid.uuid4().hex[:8]}",
                    entity_id=attacker_ip or lure.target_entity,
                    source=EvidenceSource.DYNAMIC_HONEYPOT.value,
                    detector_type=EvidenceType.DECEPTION.value,
                    raw_score=1.0,
                    normalized_score=1.0,
                    confidence=0.99,
                    uncertainty=0.01,
                    mitre_mapping=[lure.mitre_technique],
                    explanation=f"High-fidelity honeypot tripwire hit on {lure.lure_type} ({lure.lure_id})",
                )
                get_evidence_ledger().record_evidence(ev)

                log.critical(f"[DECEPTION ALERT] CONFIRMED ADVERSARY ENGAGEMENT! Lure '{lure.lure_id}' triggered by {attacker_ip or 'unknown'}")
                return lure
        return None

    def get_active_lures(self) -> List[dict]:
        with self._lock:
            return [l.to_dict() for l in self._active_lures.values()]

    def get_triggered_lures(self) -> List[dict]:
        with self._lock:
            return [l.to_dict() for l in self._triggered_log]


# Singleton
_deception_instance: Optional[DeceptionManager] = None
_deception_lock = threading.Lock()


def get_deception_manager() -> DeceptionManager:
    global _deception_instance
    with _deception_lock:
        if _deception_instance is None:
            _deception_instance = DeceptionManager()
    return _deception_instance
