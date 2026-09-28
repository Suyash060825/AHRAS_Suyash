from __future__ import annotations
"""
AHRAS Module — Cryptographic Decision Provenance Epoch Ledger (Section 27 / Research Frontier P1)
--------------------------------------------------------------------------------------------------
Maintains an immutable, append-only epoch ledger of high-impact automated security decisions:

Each Record:
  - event_hash: SHA-256 digest of normalized input telemetry
  - model_hash: SHA-256 digest of active classifier / ensemble weights
  - config_hash: SHA-256 digest of active system configuration
  - policy_version: Policy identifier and threshold specification
  - risk_trace: Composite risk score, confidence, and epistemic uncertainty
  - xai_hash: SHA-256 digest of explanation feature attribution vector
  - simulation_hash: SHA-256 digest of digital twin pre-flight impact simulation
  - response_action: Playbook action selected
  - timestamp: Microsecond precision timestamp
  - record_hash: Canonical SHA-256 hash of all above fields

Each Epoch:
  - epoch_id: Monotonically increasing epoch sequence index
  - prev_epoch_hash: Cryptographic link to predecessor epoch
  - decision_records: List of decision provenance records
  - decision_hashes: Ordered leaf hashes
  - model_hashes: Set of active model hashes in this epoch
  - policy_hashes: Set of active policy hashes in this epoch
  - merkle_root: Merkle tree root hash computed over decision hashes
  - timestamp: Epoch finalization timestamp
  - epoch_hash: SHA-256(epoch_id || prev_epoch_hash || merkle_root || timestamp)

Guarantees & Audit:
  - 100% deterministic tamper detection if event, model, explanation, policy,
    or response action is modified post-hoc (Section 27.1).
"""

import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

GENESIS_PREV_EPOCH_HASH = "0" * 64


def _sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_merkle_root(leaf_hashes: List[str]) -> str:
    """Computes binary Merkle tree root hash from a list of leaf hashes."""
    if not leaf_hashes:
        return _sha256("EMPTY_TREE")
    if len(leaf_hashes) == 1:
        return leaf_hashes[0]

    current_layer = list(leaf_hashes)
    while len(current_layer) > 1:
        next_layer = []
        for i in range(0, len(current_layer), 2):
            left = current_layer[i]
            right = current_layer[i + 1] if i + 1 < len(current_layer) else left
            combined = _sha256(left + right)
            next_layer.append(combined)
        current_layer = next_layer
    return current_layer[0]


@dataclass
class DecisionProvenanceRecord:
    """Cryptographic audit record for an automated security intervention decision."""
    decision_id: str
    event_hash: str
    model_hash: str
    config_hash: str
    policy_version: str
    risk_trace: Dict[str, float]
    xai_hash: str
    simulation_hash: str
    response_action: str
    timestamp: float = field(default_factory=time.time)
    record_hash: str = ""

    def __post_init__(self) -> None:
        if not self.record_hash:
            self.record_hash = self.compute_hash()

    def compute_hash(self) -> str:
        payload = {
            "decision_id": self.decision_id,
            "event_hash": self.event_hash,
            "model_hash": self.model_hash,
            "config_hash": self.config_hash,
            "policy_version": self.policy_version,
            "risk_trace": {k: round(v, 6) for k, v in sorted(self.risk_trace.items())},
            "xai_hash": self.xai_hash,
            "simulation_hash": self.simulation_hash,
            "response_action": self.response_action,
            "timestamp": round(self.timestamp, 4),
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return _sha256(canonical_str)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityEvidenceEpoch:
    """Checkpoint epoch bundling a block of decision provenance records."""
    epoch_id: int
    prev_epoch_hash: str
    decision_records: List[DecisionProvenanceRecord]
    decision_hashes: List[str]
    model_hashes: List[str]
    policy_hashes: List[str]
    merkle_root: str
    timestamp: float
    epoch_hash: str = ""

    def __post_init__(self) -> None:
        if not self.epoch_hash:
            self.epoch_hash = self.compute_hash()

    def compute_hash(self) -> str:
        payload = f"{self.epoch_id}:{self.prev_epoch_hash}:{self.merkle_root}:{self.timestamp:.4f}"
        return _sha256(payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "prev_epoch_hash": self.prev_epoch_hash,
            "decision_count": len(self.decision_records),
            "decision_hashes": self.decision_hashes,
            "model_hashes": self.model_hashes,
            "policy_hashes": self.policy_hashes,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
            "epoch_hash": self.epoch_hash,
        }


class EpochProvenanceLedger:
    """
    Append-only cryptographic ledger organizing decisions into verifiable epochs.
    Thread-safe via RLock.
    """

    def __init__(self, epoch_capacity: int = 100) -> None:
        self.epoch_capacity = epoch_capacity
        self._lock = threading.RLock()
        self._epochs: List[SecurityEvidenceEpoch] = []
        self._current_records: List[DecisionProvenanceRecord] = []
        self._all_decisions: Dict[str, DecisionProvenanceRecord] = {}

    @property
    def latest_epoch_hash(self) -> str:
        with self._lock:
            return self._epochs[-1].epoch_hash if self._epochs else GENESIS_PREV_EPOCH_HASH

    @property
    def total_epochs(self) -> int:
        with self._lock:
            return len(self._epochs)

    @property
    def total_decisions(self) -> int:
        with self._lock:
            return len(self._all_decisions)

    def append_decision(self, record: DecisionProvenanceRecord) -> DecisionProvenanceRecord:
        """Appends a new decision provenance record. Triggers epoch checkpoint if capacity reached."""
        with self._lock:
            if not record.record_hash:
                record.record_hash = record.compute_hash()

            self._current_records.append(record)
            self._all_decisions[record.decision_id] = record

            if len(self._current_records) >= self.epoch_capacity:
                self.checkpoint_epoch()

            return record

    def checkpoint_epoch(self) -> Optional[SecurityEvidenceEpoch]:
        """Finalizes pending decisions into an immutable epoch linked to predecessor epoch."""
        with self._lock:
            if not self._current_records:
                return None

            epoch_id = len(self._epochs)
            prev_hash = self.latest_epoch_hash
            records = list(self._current_records)
            self._current_records.clear()

            decision_hashes = [r.record_hash for r in records]
            model_hashes = sorted(list({r.model_hash for r in records}))
            policy_hashes = sorted(list({_sha256(r.policy_version) for r in records}))
            merkle_root = compute_merkle_root(decision_hashes)
            now = time.time()

            epoch = SecurityEvidenceEpoch(
                epoch_id=epoch_id,
                prev_epoch_hash=prev_hash,
                decision_records=records,
                decision_hashes=decision_hashes,
                model_hashes=model_hashes,
                policy_hashes=policy_hashes,
                merkle_root=merkle_root,
                timestamp=now,
            )
            self._epochs.append(epoch)
            log.info(f"[PROVENANCE] Checkpointed Epoch {epoch_id} ({len(records)} decisions, hash={epoch.epoch_hash[:12]}...)")
            return epoch

    def verify_epoch(self, epoch: SecurityEvidenceEpoch) -> Tuple[bool, str]:
        """Audits cryptographic integrity of a single epoch."""
        # 1. Verify individual decision hashes
        for r in epoch.decision_records:
            recomputed_hash = r.compute_hash()
            if recomputed_hash != r.record_hash:
                return False, f"Decision {r.decision_id} hash mismatch: computed {recomputed_hash} != {r.record_hash}"

        # 2. Verify Merkle root
        leaf_hashes = [r.record_hash for r in epoch.decision_records]
        recomputed_root = compute_merkle_root(leaf_hashes)
        if recomputed_root != epoch.merkle_root:
            return False, f"Epoch {epoch.epoch_id} Merkle root mismatch: computed {recomputed_root} != {epoch.merkle_root}"

        # 3. Verify epoch header hash
        recomputed_epoch_hash = epoch.compute_hash()
        if recomputed_epoch_hash != epoch.epoch_hash:
            return False, f"Epoch {epoch.epoch_id} header hash mismatch: computed {recomputed_epoch_hash} != {epoch.epoch_hash}"

        return True, "Epoch verified intact"

    def verify_entire_ledger(self) -> Dict[str, Any]:
        """Audits the complete chain of epochs from genesis to tip."""
        with self._lock:
            if not self._epochs:
                return {"valid": True, "epoch_count": 0, "verified_decisions": 0, "errors": []}

            errors: List[str] = []
            expected_prev_hash = GENESIS_PREV_EPOCH_HASH
            verified_decisions = 0

            for i, epoch in enumerate(self._epochs):
                if epoch.epoch_id != i:
                    errors.append(f"Epoch sequence broken at index {i}: epoch_id is {epoch.epoch_id}")

                if epoch.prev_epoch_hash != expected_prev_hash:
                    errors.append(f"Epoch {i} link mismatch: prev_hash {epoch.prev_epoch_hash[:12]} != expected {expected_prev_hash[:12]}")

                is_ok, msg = self.verify_epoch(epoch)
                if not is_ok:
                    errors.append(f"Epoch {i} integrity failure: {msg}")

                verified_decisions += len(epoch.decision_records)
                expected_prev_hash = epoch.epoch_hash

            return {
                "valid": len(errors) == 0,
                "epoch_count": len(self._epochs),
                "verified_decisions": verified_decisions,
                "pending_decisions": len(self._current_records),
                "errors": errors,
            }

    def tamper_detect_test(
        self,
        epoch_id: int,
        field_to_tamper: str,
        tampered_value: Any = "MALICIOUS_TAMPERED_PAYLOAD",
    ) -> bool:
        """
        Executes Section 27.1 Tamper Test:
        Modifies a field in-place within an epoch and confirms that verify_entire_ledger()
        detects the tampering with 100% certainty.
        """
        with self._lock:
            if epoch_id >= len(self._epochs):
                return False

            epoch = self._epochs[epoch_id]
            if not epoch.decision_records:
                return False

            # Deepcopy target record to restore later
            target_record = epoch.decision_records[0]
            orig_record = copy.deepcopy(target_record)

            try:
                # Apply tampering directly to object attribute
                if hasattr(target_record, field_to_tamper):
                    setattr(target_record, field_to_tamper, tampered_value)
                elif field_to_tamper in target_record.risk_trace:
                    target_record.risk_trace[field_to_tamper] = float(tampered_value)
                else:
                    return False

                # Verification MUST fail
                audit = self.verify_entire_ledger()
                detected = not audit["valid"]
                return detected
            finally:
                # Restore original record to maintain integrity
                epoch.decision_records[0] = orig_record
