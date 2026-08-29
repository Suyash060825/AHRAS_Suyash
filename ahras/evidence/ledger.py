from __future__ import annotations
"""
AHRAS Evidence Ledger & Audit Trail
------------------------------------
In-memory and persistent evidence ledger maintaining chronological,
tamper-evident detection contributions across entities and episodes.
"""

import threading
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

from ahras.evidence.models import EvidenceRecord, EvidenceType

log = logging.getLogger(__name__)


class EvidenceLedger:
    """
    Central tamper-evident repository of detection evidence.
    Supports querying by event, entity, episode, or time window.
    """

    def __init__(self, max_records_per_entity: int = 1000):
        self._records: List[EvidenceRecord] = []
        self._by_event: Dict[str, List[EvidenceRecord]] = defaultdict(list)
        self._by_entity: Dict[str, List[EvidenceRecord]] = defaultdict(list)
        self._max_per_entity = max_records_per_entity
        self._lock = threading.RLock()

    def record_evidence(self, record: EvidenceRecord) -> EvidenceRecord:
        """Appends an evidence record and indexes it by event and entity."""
        with self._lock:
            self._records.append(record)
            self._by_event[record.event_id].append(record)
            
            ent_list = self._by_entity[record.entity_id]
            ent_list.append(record)
            if len(ent_list) > self._max_per_entity:
                ent_list.pop(0)
                
            return record

    def record_batch(self, records: List[EvidenceRecord]) -> List[EvidenceRecord]:
        with self._lock:
            for r in records:
                self.record_evidence(r)
            return records

    def get_event_evidence(self, event_id: str) -> List[EvidenceRecord]:
        with self._lock:
            return list(self._by_event.get(event_id, []))

    def get_entity_evidence(self, entity_id: str, limit: int = 50) -> List[EvidenceRecord]:
        with self._lock:
            records = self._by_entity.get(entity_id, [])
            return list(records[-limit:])

    def verify_all_integrity(self) -> Dict[str, Any]:
        """Audits every evidence record in the ledger for cryptographic hash consistency."""
        with self._lock:
            total = len(self._records)
            valid = 0
            tampered = []
            for r in self._records:
                if r.verify_integrity():
                    valid += 1
                else:
                    tampered.append(r.evidence_id)
            return {
                "total_records": total,
                "valid_records": valid,
                "tampered_records": tampered,
                "integrity_pass": (len(tampered) == 0),
            }

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._by_event.clear()
            self._by_entity.clear()


_evidence_ledger_instance: Optional[EvidenceLedger] = None
_ledger_lock = threading.Lock()


def get_evidence_ledger() -> EvidenceLedger:
    global _evidence_ledger_instance
    with _ledger_lock:
        if _evidence_ledger_instance is None:
            _evidence_ledger_instance = EvidenceLedger()
    return _evidence_ledger_instance
