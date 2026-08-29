from __future__ import annotations
"""
AHRAS Evidence Ledger & Tamper-Evident Append-Only Hash Chain
--------------------------------------------------------------
Maintains a cryptographically linked, tamper-evident hash chain of detection
evidence records across entities, detectors, and temporal attack episodes.

Guarantees:
  - Sequence consistency: seq_num = seq_num_(n-1) + 1
  - Cryptographic linking: prev_hash_n = record_hash_(n-1)
  - Full chain verification: verify_chain() audits insertions, deletions, modifications, and reordering
  - Batch Merkle checkpoints for state-snapshot audits
"""

import hashlib
import json
import threading
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple

from ahras.evidence.models import EvidenceRecord, EvidenceType

log = logging.getLogger(__name__)

GENESIS_PREV_HASH = "0" * 64


class EvidenceLedger:
    """
    Central tamper-evident append-only ledger of detection evidence.
    Thread-safe via RLock.
    """

    def __init__(self, max_records_per_entity: int = 2000):
        self._chain: List[EvidenceRecord] = []
        self._by_event: Dict[str, List[EvidenceRecord]] = defaultdict(list)
        self._by_entity: Dict[str, List[EvidenceRecord]] = defaultdict(list)
        self._by_evidence_id: Dict[str, EvidenceRecord] = {}
        self._max_per_entity = max_records_per_entity
        self._lock = threading.RLock()

    @property
    def latest_hash(self) -> str:
        with self._lock:
            return self._chain[-1].record_hash if self._chain else GENESIS_PREV_HASH

    @property
    def chain_length(self) -> int:
        with self._lock:
            return len(self._chain)

    def record_evidence(self, record: EvidenceRecord) -> EvidenceRecord:
        """
        Appends an evidence record to the cryptographic hash chain.
        Assigns the next monotonic sequence number and predecessor hash link.
        """
        with self._lock:
            # Assign chain linkage
            record.seq_num = len(self._chain)
            record.prev_hash = self.latest_hash
            record.record_hash = record.compute_hash()

            self._chain.append(record)
            self._by_event[record.event_id].append(record)
            self._by_evidence_id[record.evidence_id] = record
            
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

    def get_record_by_id(self, evidence_id: str) -> Optional[EvidenceRecord]:
        with self._lock:
            return self._by_evidence_id.get(evidence_id)

    def verify_chain(self) -> Dict[str, Any]:
        """
        Audits the entire append-only hash chain from genesis (0) to tip (N-1).
        Verifies:
          1. Sequence number monotonicity
          2. Individual record SHA-256 payload integrity
          3. Predecessor hash link continuity (prev_hash_i == record_hash_(i-1))
        """
        with self._lock:
            total = len(self._chain)
            if total == 0:
                return {
                    "total_records": 0,
                    "valid_records": 0,
                    "is_valid": True,
                    "tamper_detected": False,
                    "broken_indices": [],
                    "failure_reasons": [],
                }

            broken_indices: List[int] = []
            failure_reasons: List[str] = []

            for i in range(total):
                rec = self._chain[i]

                # 1. Sequence number check
                if rec.seq_num != i:
                    broken_indices.append(i)
                    failure_reasons.append(f"Seq num mismatch at index {i}: expected {i}, got {rec.seq_num}")
                    continue

                # 2. Previous hash link check
                expected_prev = GENESIS_PREV_HASH if i == 0 else self._chain[i - 1].record_hash
                if rec.prev_hash != expected_prev:
                    broken_indices.append(i)
                    failure_reasons.append(f"Broken prev_hash link at index {i}: expected {expected_prev[:12]}..., got {rec.prev_hash[:12]}...")
                    continue

                # 3. Payload integrity hash check
                if not rec.verify_integrity():
                    broken_indices.append(i)
                    failure_reasons.append(f"Payload tamper at index {i} ({rec.evidence_id}): hash mismatch")
                    continue

            is_valid = (len(broken_indices) == 0)
            return {
                "total_records": total,
                "valid_records": total - len(broken_indices),
                "is_valid": is_valid,
                "tamper_detected": not is_valid,
                "broken_indices": broken_indices,
                "failure_reasons": failure_reasons,
            }

    def verify_all_integrity(self) -> Dict[str, Any]:
        """Backward-compatible audit method returning verification dictionary."""
        audit = self.verify_chain()
        return {
            "total_records": audit["total_records"],
            "valid_records": audit["valid_records"],
            "tampered_records": [self._chain[i].evidence_id for i in audit["broken_indices"] if i < len(self._chain)],
            "integrity_pass": audit["is_valid"],
        }

    def compute_merkle_root(self) -> str:
        """Computes deterministic Merkle root hash across all records in the ledger."""
        with self._lock:
            if not self._chain:
                return hashlib.sha256(b"EMPTY_LEDGER").hexdigest()

            hashes = [r.record_hash for r in self._chain]
            while len(hashes) > 1:
                if len(hashes) % 2 != 0:
                    hashes.append(hashes[-1])
                new_level = []
                for j in range(0, len(hashes), 2):
                    combined = (hashes[j] + hashes[j + 1]).encode('utf-8')
                    new_level.append(hashlib.sha256(combined).hexdigest())
                hashes = new_level
            return hashes[0]

    def clear(self) -> None:
        with self._lock:
            self._chain.clear()
            self._by_event.clear()
            self._by_entity.clear()
            self._by_evidence_id.clear()


# Singleton instance
_evidence_ledger_instance: Optional[EvidenceLedger] = None
_ledger_lock = threading.Lock()


def get_evidence_ledger() -> EvidenceLedger:
    global _evidence_ledger_instance
    with _ledger_lock:
        if _evidence_ledger_instance is None:
            _evidence_ledger_instance = EvidenceLedger()
    return _evidence_ledger_instance
