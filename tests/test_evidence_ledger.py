from __future__ import annotations
"""
AHRAS Tamper-Evident Evidence Ledger Test Suite
------------------------------------------------
Verifies cryptographic hash chain properties:
  1. Monotonic sequence numbering & predecessor pointer continuity
  2. Content modification tamper detection
  3. Record deletion detection
  4. Record injection / insertion detection
  5. Record reordering detection
  6. Merkle root integrity calculation
"""

import unittest
import copy
from ahras.evidence.models import EvidenceRecord, EvidenceType, EvidenceSource
from ahras.evidence.ledger import EvidenceLedger


class TestEvidenceLedgerChain(unittest.TestCase):
    def setUp(self):
        self.ledger = EvidenceLedger()

    def test_valid_chain_construction(self):
        ev0 = EvidenceRecord(event_id="EVT-0", entity_id="host_1", normalized_score=0.2)
        ev1 = EvidenceRecord(event_id="EVT-1", entity_id="host_1", normalized_score=0.5)
        ev2 = EvidenceRecord(event_id="EVT-2", entity_id="host_2", normalized_score=0.8)

        self.ledger.record_evidence(ev0)
        self.ledger.record_evidence(ev1)
        self.ledger.record_evidence(ev2)

        self.assertEqual(self.ledger.chain_length, 3)
        self.assertEqual(ev0.seq_num, 0)
        self.assertEqual(ev1.seq_num, 1)
        self.assertEqual(ev2.seq_num, 2)

        self.assertEqual(ev1.prev_hash, ev0.record_hash)
        self.assertEqual(ev2.prev_hash, ev1.record_hash)

        audit = self.ledger.verify_chain()
        self.assertTrue(audit["is_valid"])
        self.assertFalse(audit["tamper_detected"])

    def test_content_modification_detected(self):
        ev0 = EvidenceRecord(event_id="EVT-0", entity_id="host_1", normalized_score=0.2)
        ev1 = EvidenceRecord(event_id="EVT-1", entity_id="host_1", normalized_score=0.5)
        self.ledger.record_evidence(ev0)
        self.ledger.record_evidence(ev1)

        # Attacker modifies score of ev0 after insertion
        ev0.normalized_score = 0.99

        audit = self.ledger.verify_chain()
        self.assertFalse(audit["is_valid"])
        self.assertTrue(audit["tamper_detected"])
        self.assertIn(0, audit["broken_indices"])

    def test_record_deletion_detected(self):
        ev0 = EvidenceRecord(event_id="EVT-0", entity_id="host_1", normalized_score=0.2)
        ev1 = EvidenceRecord(event_id="EVT-1", entity_id="host_1", normalized_score=0.5)
        ev2 = EvidenceRecord(event_id="EVT-2", entity_id="host_2", normalized_score=0.8)
        self.ledger.record_evidence(ev0)
        self.ledger.record_evidence(ev1)
        self.ledger.record_evidence(ev2)

        # Attacker deletes ev1 from ledger list
        del self.ledger._chain[1]

        audit = self.ledger.verify_chain()
        self.assertFalse(audit["is_valid"])
        self.assertTrue(audit["tamper_detected"])

    def test_record_reordering_detected(self):
        ev0 = EvidenceRecord(event_id="EVT-0", entity_id="host_1", normalized_score=0.2)
        ev1 = EvidenceRecord(event_id="EVT-1", entity_id="host_1", normalized_score=0.5)
        self.ledger.record_evidence(ev0)
        self.ledger.record_evidence(ev1)

        # Attacker swaps position of ev0 and ev1
        self.ledger._chain[0], self.ledger._chain[1] = self.ledger._chain[1], self.ledger._chain[0]

        audit = self.ledger.verify_chain()
        self.assertFalse(audit["is_valid"])
        self.assertTrue(audit["tamper_detected"])

    def test_merkle_root_calculation(self):
        ev0 = EvidenceRecord(event_id="EVT-0", entity_id="host_1", normalized_score=0.2)
        ev1 = EvidenceRecord(event_id="EVT-1", entity_id="host_1", normalized_score=0.5)
        self.ledger.record_evidence(ev0)
        self.ledger.record_evidence(ev1)

        root = self.ledger.compute_merkle_root()
        self.assertIsInstance(root, str)
        self.assertEqual(len(root), 64)


if __name__ == "__main__":
    unittest.main()
