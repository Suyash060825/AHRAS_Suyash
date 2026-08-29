from __future__ import annotations
"""
AHRAS Leakage Audit & Data Contamination Test Suite
---------------------------------------------------
Verifies that chronological and entity-disjoint splitting prevent data leakage,
and tests that intentional temporal & entity contamination is strictly caught.
"""

import unittest
from evaluation.leakage_audit import temporal_train_test_split, entity_disjoint_train_test_split, LeakageAuditor


class MockRecord:
    def __init__(self, entity_key: str, timestamp: float, label: int = 0):
        self.src_ip = entity_key
        self.entity_key = entity_key
        self.timestamp = timestamp
        self.label = label


class TestLeakageAudit(unittest.TestCase):
    def setUp(self):
        self.auditor = LeakageAuditor()

    def test_clean_temporal_split_passes_audit(self):
        # 100 records in chronological order
        records = [MockRecord(f"192.168.1.{i%10}", timestamp=1000.0 + i) for i in range(100)]
        train, val, test = temporal_train_test_split(records, train_ratio=0.70, val_ratio=0.15)
        
        audit = self.auditor.audit_splits(train, test, is_entity_disjoint=False)
        self.assertTrue(audit["overall_leakage_audit_pass"])
        self.assertFalse(audit["temporal_leakage_detected"])

    def test_intentional_temporal_leakage_is_detected(self):
        # Test record has timestamp prior to max train timestamp
        train = [MockRecord("host_a", timestamp=2000.0)]
        test = [MockRecord("host_b", timestamp=1500.0)] # Past timestamp in test!

        audit = self.auditor.audit_splits(train, test, is_entity_disjoint=False)
        self.assertFalse(audit["overall_leakage_audit_pass"])
        self.assertTrue(audit["temporal_leakage_detected"])

    def test_clean_entity_disjoint_split_passes_audit(self):
        records = [
            MockRecord("server_alpha", 1000.0), MockRecord("server_alpha", 1001.0),
            MockRecord("server_beta", 1000.0), MockRecord("server_beta", 1001.0),
            MockRecord("server_gamma", 1000.0), MockRecord("server_gamma", 1001.0),
            MockRecord("server_delta", 1000.0), MockRecord("server_delta", 1001.0),
        ]
        train, test = entity_disjoint_train_test_split(records, train_ratio=0.50, seed=42)
        audit = self.auditor.audit_splits(train, test, is_entity_disjoint=True, is_temporal=False)
        self.assertTrue(audit["overall_leakage_audit_pass"])
        self.assertTrue(audit["entity_disjoint_pass"])

    def test_intentional_entity_leakage_is_detected(self):
        train = [MockRecord("server_shared", 1000.0), MockRecord("server_train_only", 1001.0)]
        test = [MockRecord("server_shared", 2000.0), MockRecord("server_test_only", 2001.0)]

        audit = self.auditor.audit_splits(train, test, is_entity_disjoint=True)
        self.assertFalse(audit["overall_leakage_audit_pass"])
        self.assertFalse(audit["entity_disjoint_pass"])
        self.assertEqual(audit["entity_overlap_count"], 1)


if __name__ == "__main__":
    unittest.main()
