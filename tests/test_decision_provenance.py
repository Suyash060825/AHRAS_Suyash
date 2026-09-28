from __future__ import annotations
"""
Unit tests for Cryptographic Decision Provenance Epoch Ledger (Section 27).
Verifies:
- DecisionProvenanceRecord canonical hashing
- Binary Merkle tree root calculation
- Monotonic epoch sequence & cryptographic predecessor chaining
- Auto-checkpointing on capacity threshold
- Entire ledger cryptographic verification
- Section 27.1 Tamper Detection across all high-impact decision fields
"""

import time
import unittest
from ahras.evidence.decision_provenance import (
    GENESIS_PREV_EPOCH_HASH,
    DecisionProvenanceRecord,
    EpochProvenanceLedger,
    SecurityEvidenceEpoch,
    compute_merkle_root,
)


class TestDecisionProvenance(unittest.TestCase):
    def setUp(self):
        self.ledger = EpochProvenanceLedger(epoch_capacity=5)

    def _create_sample_record(self, decision_id: str, action: str = "CONTAIN_HOST") -> DecisionProvenanceRecord:
        return DecisionProvenanceRecord(
            decision_id=decision_id,
            event_hash="a" * 64,
            model_hash="b" * 64,
            config_hash="c" * 64,
            policy_version="ahras-policy-v2.1",
            risk_trace={"composite_risk": 0.88, "confidence": 0.95, "epistemic_uncertainty": 0.05},
            xai_hash="d" * 64,
            simulation_hash="e" * 64,
            response_action=action,
            timestamp=1700000000.0,
        )

    def test_record_hash_deterministic(self):
        rec1 = self._create_sample_record("dec-001")
        rec2 = self._create_sample_record("dec-001")
        self.assertEqual(rec1.record_hash, rec2.record_hash)
        self.assertTrue(len(rec1.record_hash) == 64)

    def test_merkle_root_computation(self):
        leaves = ["1" * 64, "2" * 64, "3" * 64]
        root = compute_merkle_root(leaves)
        self.assertIsInstance(root, str)
        self.assertEqual(len(root), 64)

        # Empty tree handling
        empty_root = compute_merkle_root([])
        self.assertEqual(len(empty_root), 64)

        # Single leaf handling
        single_leaf = "4" * 64
        self.assertEqual(compute_merkle_root([single_leaf]), single_leaf)

    def test_epoch_checkpointing_and_chaining(self):
        # Insert 12 records (with capacity 5, will create 2 full epochs and leave 2 pending)
        for i in range(12):
            rec = self._create_sample_record(f"dec-{i:03d}")
            self.ledger.append_decision(rec)

        self.assertEqual(self.ledger.total_epochs, 2)
        self.assertEqual(self.ledger.total_decisions, 12)

        # Manual checkpoint of remaining 2 records
        epoch2 = self.ledger.checkpoint_epoch()
        self.assertIsNotNone(epoch2)
        self.assertEqual(self.ledger.total_epochs, 3)

        # Verify predecessor links
        epochs = self.ledger._epochs
        self.assertEqual(epochs[0].prev_epoch_hash, GENESIS_PREV_EPOCH_HASH)
        self.assertEqual(epochs[1].prev_epoch_hash, epochs[0].epoch_hash)
        self.assertEqual(epochs[2].prev_epoch_hash, epochs[1].epoch_hash)

        # Verify entire ledger
        audit = self.ledger.verify_entire_ledger()
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["epoch_count"], 3)
        self.assertEqual(audit["verified_decisions"], 12)
        self.assertEqual(audit["errors"], [])

    def test_section_27_1_tamper_detection(self):
        """Tests that any post-hoc tampering of high-impact fields is 100% detected."""
        for i in range(5):
            rec = self._create_sample_record(f"dec-{i:03d}")
            self.ledger.append_decision(rec)

        self.assertEqual(self.ledger.total_epochs, 1)

        # Tampering event_hash
        detected_event = self.ledger.tamper_detect_test(0, "event_hash", "f" * 64)
        self.assertTrue(detected_event, "Failed to detect event_hash tampering")

        # Tampering model_hash
        detected_model = self.ledger.tamper_detect_test(0, "model_hash", "9" * 64)
        self.assertTrue(detected_model, "Failed to detect model_hash tampering")

        # Tampering policy_version
        detected_policy = self.ledger.tamper_detect_test(0, "policy_version", "malicious-policy-v999")
        self.assertTrue(detected_policy, "Failed to detect policy_version tampering")

        # Tampering risk_trace (composite_risk)
        detected_risk = self.ledger.tamper_detect_test(0, "composite_risk", 0.01)
        self.assertTrue(detected_risk, "Failed to detect composite_risk tampering")

        # Tampering xai_hash
        detected_xai = self.ledger.tamper_detect_test(0, "xai_hash", "0" * 64)
        self.assertTrue(detected_xai, "Failed to detect xai_hash tampering")

        # Tampering simulation_hash
        detected_sim = self.ledger.tamper_detect_test(0, "simulation_hash", "7" * 64)
        self.assertTrue(detected_sim, "Failed to detect simulation_hash tampering")

        # Tampering response_action
        detected_action = self.ledger.tamper_detect_test(0, "response_action", "ALLOW_TRAFFIC_SILENTLY")
        self.assertTrue(detected_action, "Failed to detect response_action tampering")

        # Final audit should pass since tamper_detect_test restores original values
        audit = self.ledger.verify_entire_ledger()
        self.assertTrue(audit["valid"])


if __name__ == "__main__":
    unittest.main()
