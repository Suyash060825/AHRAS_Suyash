from __future__ import annotations
import unittest
from threat_intel.intel import ThreatIntelManager, IOCRecord
from threat_intel.stix_ingestor import STIXIngestor


class TestThreatIntelligence(unittest.TestCase):
    def setUp(self):
        self.ti = ThreatIntelManager()

    def test_01_ioc_lookup(self):
        rec = self.ti.check_ioc("198.51.100.44")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.severity, "CRITICAL")
        self.assertIn("Mirai", rec.threat_name)

    def test_02_unknown_ioc(self):
        rec = self.ti.check_ioc("8.8.8.8")
        self.assertIsNone(rec)
        self.assertEqual(self.ti.get_threat_score("8.8.8.8"), 0.0)

    def test_03_add_custom_ioc(self):
        self.ti.add_ioc("192.0.2.100", "ip", "Custom Test Bot", 0.90, "HIGH", "TEST_SUITE")
        rec = self.ti.check_ioc("192.0.2.100")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.threat_name, "Custom Test Bot")

    def test_04_event_matching(self):
        evt = {"src_ip": "198.51.100.44", "dst_ip": "10.0.0.1", "dst_port": 80}
        matches = self.ti.match_event(evt)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].ioc_value, "198.51.100.44")

    def test_05_stix_bundle_parsing(self):
        stix_bundle = {
            "type": "bundle",
            "id": "bundle--123",
            "objects": [
                {
                    "type": "indicator",
                    "id": "indicator--abc",
                    "name": "Cobalt Strike Beacon",
                    "pattern": "[ipv4-addr:value = '198.51.100.99']",
                    "confidence": 90,
                    "labels": ["malicious-activity", "c2"]
                }
            ]
        }
        iocs = STIXIngestor.parse_bundle(stix_bundle)
        self.assertEqual(len(iocs), 1)
        self.assertEqual(iocs[0]["ioc_value"], "198.51.100.99")
        self.assertEqual(iocs[0]["threat_name"], "Cobalt Strike Beacon")


if __name__ == "__main__":
    unittest.main()
