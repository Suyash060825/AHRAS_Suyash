from __future__ import annotations
"""
AHRAS Full System End-to-End Test Suite (Phase 5)
--------------------------------------------------
Validates complete end-to-end processing pipeline across all 5 modules:
Raw Event -> OCSF Normalizer -> Feature Extractor -> Signature Engine ->
ML Ensemble -> Statistical Engine -> Adaptive Risk Engine -> Response Orchestrator ->
FastAPI REST API Service.
"""

import unittest
import numpy as np
from fastapi.testclient import TestClient

from normalizer.ocsf_normalizer import _norm_network, _norm_process, _norm_file, _norm_cloud
from detection.feature_extractor import extract
from detection.hybrid_engine import get_combiner
from detection.pipeline import bootstrap_ml_models
from detection.risk_engine import run_risk_engine, get_risk_engine
from response import get_response_orchestrator
from api.server import app


class FullSystemIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bootstrap_ml_models()
        cls.combiner = get_combiner()
        cls.client = TestClient(app)
        cls.risk_engine = get_risk_engine()
        cls.orchestrator = get_response_orchestrator()

    def test_01_normal_network_pipeline(self):
        raw = {"src_ip": "192.168.1.10", "dst_ip": "10.0.0.1", "packet_count": 5, "duration_sec": 1.0}
        ocsf = _norm_network(raw)
        vec = extract(ocsf)
        self.assertIsNotNone(vec)
        res = self.combiner.process(ocsf)
        self.assertIsNotNone(res)
        self.assertFalse(res.is_alert)

    def test_02_port_scan_pipeline(self):
        raw = {"src_ip": "192.168.1.99", "unique_dst_ports": 150, "packet_count": 300, "duration_sec": 1.0, "tcp_flags": ["SYN"]}
        ocsf = _norm_network(raw)
        vec = extract(ocsf)
        res = self.combiner.process(ocsf)
        self.assertTrue(res.is_alert)
        rule_ids = [m.get("rule_id") for m in res.signature_matches if isinstance(m, dict)]
        self.assertIn("NET-001", rule_ids)

    def test_03_syn_flood_pipeline(self):
        raw = {"src_ip": "172.16.0.4", "packet_count": 15000, "duration_sec": 1.0, "tcp_flags": ["SYN"]}
        ocsf = _norm_network(raw)
        res = self.combiner.process(ocsf)
        self.assertTrue(res.is_alert)

    def test_04_ssh_brute_pipeline(self):
        raw = {"src_ip": "198.51.100.44", "dst_port": 22, "packet_count": 500, "duration_sec": 3.0}
        ocsf = _norm_network(raw)
        res = self.combiner.process(ocsf)
        self.assertTrue(res.is_alert)

    def test_05_ransomware_pipeline_and_auto_response(self):
        raw = {"filepath": "/home/user/document.docx.locked", "entropy": 7.9, "high_entropy": True, "hostname": "finance-pc"}
        ocsf = _norm_file(raw)
        res = self.combiner.process(ocsf)
        self.assertTrue(res.is_alert)
        
        # Risk Evaluation with multi-engine alignment
        from detection.anomaly_engine.ml_engine import AnomalyResult
        from detection.statistical_engine.stat_engine import StatResult
        ml_anom = AnomalyResult(ocsf_class="file_activity", is_anomaly=True, confidence=0.9, isolation_score=0.9, reconstruction_error=0.9, svm_score=0.9, ensemble_score=0.9, n_models_fired=3, model_trained=True)
        stat_anom = StatResult(entity_key="finance-pc", is_anomaly=True, confidence=0.9, zscore=5.0, ewma_deviation=4.0, temporal_density=100, behavioral_drift=2.0)
        
        risk = run_risk_engine("finance-pc", res.signature_matches, ml_anom, stat_anom, ocsf)
        actions = self.orchestrator.evaluate_and_respond(risk, ocsf)
        self.assertGreater(len(actions), 0)
        self.assertEqual(actions[0].action_type, "ISOLATE_HOST")

    def test_06_credential_dump_pipeline(self):
        raw = {"name": "lsass.exe", "cmdline": "mimikatz.exe privilege::debug", "pid": 600}
        ocsf = _norm_process(raw)
        res = self.combiner.process(ocsf)
        self.assertTrue(res.is_alert)

    def test_07_cloud_defense_evasion_pipeline(self):
        raw = {"user_identity": "rogue_admin", "action": "cloudtrail:StopLogging", "severity_hint": "critical"}
        ocsf = _norm_cloud(raw)
        res = self.combiner.process(ocsf)
        self.assertTrue(res.is_alert)

    def test_08_api_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)

    def test_09_api_dashboard(self):
        r = self.client.get("/dashboard")
        self.assertEqual(r.status_code, 200)
        self.assertIn("AHRAS SOC Operations", r.text)

    def test_10_api_alerts(self):
        r = self.client.get("/alerts?limit=5")
        self.assertEqual(r.status_code, 200)

    def test_11_api_entity_report(self):
        r = self.client.get("/entities/192.168.1.99/report?ocsf_class=network_activity")
        self.assertEqual(r.status_code, 200)

    def test_12_api_metrics(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["system_status"], "OPERATIONAL")

    def test_13_api_pending_actions(self):
        r = self.client.get("/actions/pending")
        self.assertEqual(r.status_code, 200)

    def test_14_api_action_history(self):
        r = self.client.get("/actions/history")
        self.assertEqual(r.status_code, 200)

    def test_15_api_analyst_feedback(self):
        payload = {"ocsf_class": "network_activity", "entity_key": "192.168.1.99", "action": "mark_false_positive"}
        r = self.client.post("/analyst/feedback", json=payload)
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
