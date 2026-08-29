from __future__ import annotations
"""
AHRAS Evidence Ledger Test Suite
---------------------------------
Validates evidence model integrity, tamper detection, and ledger querying.
"""

import unittest
import time
from ahras.evidence.models import EvidenceRecord, EvidenceType, EvidenceSource
from ahras.evidence.ledger import EvidenceLedger, get_evidence_ledger


class TestEvidenceLedger(unittest.TestCase):
    def setUp(self):
        self.ledger = EvidenceLedger()

    def test_evidence_record_creation_and_hashing(self):
        ev = EvidenceRecord(
            event_id="EVT-001",
            entity_id="192.168.1.50",
            source=EvidenceSource.SURICATA_ENGINE.value,
            detector_type=EvidenceType.SIGNATURE.value,
            raw_score=4.0,
            normalized_score=0.80,
            confidence=0.95,
            uncertainty=0.05,
            mitre_mapping=["T1046"],
            explanation="Port scan signature detected",
        )
        self.assertTrue(ev.verify_integrity())
        self.assertIsNotNone(ev.record_hash)
        self.assertEqual(len(ev.record_hash), 64) # SHA-256 hex string

    def test_evidence_tamper_detection(self):
        ev = EvidenceRecord(
            event_id="EVT-002",
            entity_id="10.0.0.1",
            source=EvidenceSource.AUTOENCODER.value,
            detector_type=EvidenceType.ML_ANOMALY.value,
            raw_score=0.75,
            normalized_score=0.75,
            confidence=0.85,
        )
        self.assertTrue(ev.verify_integrity())
        
        # Tamper with the normalized score without updating hash
        ev.normalized_score = 0.10
        self.assertFalse(ev.verify_integrity())

    def test_ledger_indexing_and_retrieval(self):
        ev1 = EvidenceRecord(event_id="EVT-100", entity_id="user_alice", detector_type=EvidenceType.IDENTITY_CONTEXT.value, normalized_score=0.2)
        ev2 = EvidenceRecord(event_id="EVT-100", entity_id="user_alice", detector_type=EvidenceType.STATISTICAL_DRIFT.value, normalized_score=0.6)
        ev3 = EvidenceRecord(event_id="EVT-101", entity_id="user_bob", detector_type=EvidenceType.SIGNATURE.value, normalized_score=0.9)

        self.ledger.record_evidence(ev1)
        self.ledger.record_evidence(ev2)
        self.ledger.record_evidence(ev3)

        by_evt = self.ledger.get_event_evidence("EVT-100")
        self.assertEqual(len(by_evt), 2)

        by_ent = self.ledger.get_entity_evidence("user_alice")
        self.assertEqual(len(by_ent), 2)

        audit = self.ledger.verify_all_integrity()
        self.assertTrue(audit["integrity_pass"])
        self.assertEqual(audit["valid_records"], 3)


if __name__ == "__main__":
    unittest.main()
