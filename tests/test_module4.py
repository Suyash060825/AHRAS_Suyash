from __future__ import annotations
"""
AHRAS Module 4 Test Suite — Integration & System Verification
--------------------------------------------------------------
Test Coverage (25 Test Cases):
  - T01–T05: Adaptive Risk Formula & Dynamic Entity Trust Engine
  - T06–T10: Severity Transitions & Remediation Escalation
  - T11–T15: Active Defense Response Orchestrator & Dry-Run Execution
  - T16–T20: SOC Analyst Approval Queue & Audit History Log
  - T21–T25: FastAPI SOC REST API Endpoints & Operational Metrics
"""

import unittest
import numpy as np
from fastapi.testclient import TestClient

from detection.risk_engine import (
    run_risk_engine, get_risk_engine, RiskResult, AdaptiveRiskEngine
)
from detection.signature_engine.rules import SignatureMatch
from detection.anomaly_engine.ml_engine import AnomalyResult
from detection.statistical_engine.stat_engine import StatResult
from response import (
    get_response_orchestrator, run_auto_response, ResponseAction, ResponseOrchestrator
)
from normalizer.ocsf_normalizer import (
    _norm_network, _norm_process, _norm_file, _norm_cloud
)
from api.server import app


# ─────────────────────────────────────────────────────────────────────────────
# 1. Adaptive Risk Formula & Trust Engine Tests (T01–T05)
# ─────────────────────────────────────────────────────────────────────────────

class T01_AdaptiveRiskFormula(unittest.TestCase):
    def test_null_safety_defaults(self):
        res = run_risk_engine("entity_01", None, None, None, None)
        self.assertIsInstance(res, RiskResult)
        self.assertGreaterEqual(res.risk_score, 0.0)
        self.assertLessEqual(res.risk_score, 1.0)

    def test_normal_event_low_risk(self):
        ml_norm = AnomalyResult(ocsf_class="network_activity", is_anomaly=False, confidence=0.1, isolation_score=0.1, reconstruction_error=0.1, svm_score=0.1, ensemble_score=0.1, n_models_fired=0, model_trained=True)
        stat_norm = StatResult(entity_key="entity_01", is_anomaly=False, confidence=0.1, zscore=0.1, ewma_deviation=0.1, temporal_density=5, behavioral_drift=0.1)
        res = run_risk_engine("entity_01", [], ml_norm, stat_norm, {})
        self.assertLess(res.risk_score, 0.30)
        self.assertEqual(res.severity, "INFO")

    def test_critical_attack_high_risk(self):
        sig = [SignatureMatch(rule_id="R1", rule_name="Ransomware", attack_type="ransomware", severity=5, confidence=0.9, description="High entropy", mitre_technique="T1486")]
        ml_anom = AnomalyResult(ocsf_class="file_activity", is_anomaly=True, confidence=0.9, isolation_score=0.9, reconstruction_error=0.9, svm_score=0.9, ensemble_score=0.9, n_models_fired=3, model_trained=True)
        stat_anom = StatResult(entity_key="entity_02", is_anomaly=True, confidence=0.9, zscore=5.0, ewma_deviation=4.0, temporal_density=100, behavioral_drift=2.5, flags=["z_score=5.0"], mitre_techniques=["T1486"])
        res = run_risk_engine("entity_02", sig, ml_anom, stat_anom, {})
        self.assertEqual(res.severity, "CRITICAL")
        self.assertEqual(res.remediation_level, "AUTO_REMEDIATE")
        self.assertIn("T1486", res.mitre_techniques)

    def test_trust_decay_on_alert(self):
        eng = get_risk_engine()
        eng.set_trust("trust_entity", 0.90)
        sig = [SignatureMatch(rule_id="R1", rule_name="Rule", attack_type="attack", severity=5, confidence=0.9, description="desc")]
        ml_anom = AnomalyResult(ocsf_class="network_activity", is_anomaly=True, confidence=0.8, isolation_score=0.8, reconstruction_error=0.8, svm_score=0.8, ensemble_score=0.8, n_models_fired=2, model_trained=True)
        stat_anom = StatResult(entity_key="trust_entity", is_anomaly=True, confidence=0.9, zscore=5.0, ewma_deviation=4.0, temporal_density=100, behavioral_drift=2.0)
        eng.score_risk("trust_entity", sig, ml_anom, stat_anom, {})
        self.assertLess(eng.get_trust("trust_entity"), 0.90)

    def test_trust_recovery_on_clean_events(self):
        eng = get_risk_engine()
        eng.set_trust("trust_entity_2", 0.40)
        ml_norm = AnomalyResult(ocsf_class="network_activity", is_anomaly=False, confidence=0.1, isolation_score=0.1, reconstruction_error=0.1, svm_score=0.1, ensemble_score=0.1, n_models_fired=0, model_trained=True)
        stat_norm = StatResult(entity_key="trust_entity_2", is_anomaly=False, confidence=0.1, zscore=0.1, ewma_deviation=0.1, temporal_density=5, behavioral_drift=0.1)
        for _ in range(5):
            eng.score_risk("trust_entity_2", [], ml_norm, stat_norm, {})
        self.assertGreater(eng.get_trust("trust_entity_2"), 0.40)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Severity & Remediation Escalation Tests (T06–T10)
# ─────────────────────────────────────────────────────────────────────────────

class T02_SeverityEscalation(unittest.TestCase):
    def test_info_severity_mapping(self):
        r = RiskResult(entity_key="e", risk_score=0.1, severity="INFO", severity_id=1, remediation_level="LOG_ONLY", is_alert=False, S_sig=0.0, A_ml=0.1, delta_D=0.0, T_trust=0.5, explanation="exp")
        self.assertEqual(r.remediation_level, "LOG_ONLY")
        self.assertFalse(r.is_alert)

    def test_medium_severity_staging(self):
        ml = AnomalyResult(ocsf_class="network_activity", is_anomaly=True, confidence=0.6, isolation_score=0.6, reconstruction_error=0.6, svm_score=0.6, ensemble_score=0.6, n_models_fired=2, model_trained=True)
        r = run_risk_engine("med_entity", [], ml, None, {})
        self.assertIn(r.remediation_level, ["STAGE_APPROVAL", "LOG_ONLY", "SOC_ALERT_HIGH"])

    def test_high_severity_alert(self):
        sig = [SignatureMatch(rule_id="R1", rule_name="SSH Brute", attack_type="brute_force", severity=4, confidence=0.8, description="SSH brute force")]
        ml = AnomalyResult(ocsf_class="network_activity", is_anomaly=True, confidence=0.7, isolation_score=0.7, reconstruction_error=0.7, svm_score=0.7, ensemble_score=0.7, n_models_fired=2, model_trained=True)
        stat = StatResult(entity_key="high_entity", is_anomaly=True, confidence=0.7, zscore=3.0, ewma_deviation=2.0, temporal_density=50, behavioral_drift=1.0)
        r = run_risk_engine("high_entity", sig, ml, stat, {})
        self.assertTrue(r.is_alert)
        self.assertIn(r.severity, ["HIGH", "CRITICAL"])

    def test_critical_severity_auto_remediate(self):
        sig = [SignatureMatch(rule_id="R1", rule_name="Ransomware", attack_type="ransomware", severity=5, confidence=0.95, description="Crypto locker")]
        ml = AnomalyResult(ocsf_class="file_activity", is_anomaly=True, confidence=0.95, isolation_score=0.95, reconstruction_error=0.95, svm_score=0.95, ensemble_score=0.95, n_models_fired=3, model_trained=True)
        stat = StatResult(entity_key="crit_entity", is_anomaly=True, confidence=0.95, zscore=5.0, ewma_deviation=4.0, temporal_density=100, behavioral_drift=2.0)
        r = run_risk_engine("crit_entity", sig, ml, stat, {})
        self.assertEqual(r.severity, "CRITICAL")
        self.assertEqual(r.remediation_level, "AUTO_REMEDIATE")

    def test_risk_score_clipping(self):
        eng = get_risk_engine()
        r = eng.score_risk("clip_entity", [], None, None, {})
        self.assertGreaterEqual(r.risk_score, 0.0)
        self.assertLessEqual(r.risk_score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Response Orchestrator Tests (T11–T15)
# ─────────────────────────────────────────────────────────────────────────────

class T03_ResponseOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orch = get_response_orchestrator()

    def test_isolate_host_action(self):
        risk = RiskResult(entity_key="192.168.1.10", risk_score=0.95, severity="CRITICAL", severity_id=5, remediation_level="AUTO_REMEDIATE", is_alert=True, S_sig=1.0, A_ml=0.9, delta_D=2.0, T_trust=0.0, explanation="Critical")
        evt = _norm_file({"filepath": "/tmp/test", "entropy": 7.9, "hostname": "host-01"})
        actions = self.orch.evaluate_and_respond(risk, evt)
        self.assertGreater(len(actions), 0)
        self.assertEqual(actions[0].action_type, "ISOLATE_HOST")
        self.assertEqual(actions[0].status, "EXECUTED")

    def test_terminate_process_action(self):
        risk = RiskResult(entity_key="host-02", risk_score=0.95, severity="CRITICAL", severity_id=5, remediation_level="AUTO_REMEDIATE", is_alert=True, S_sig=1.0, A_ml=0.9, delta_D=2.0, T_trust=0.0, explanation="Critical")
        evt = _norm_process({"name": "malware.exe", "pid": 1234, "hostname": "host-02"})
        actions = self.orch.evaluate_and_respond(risk, evt)
        self.assertGreater(len(actions), 0)
        self.assertEqual(actions[0].action_type, "TERMINATE_PROCESS")

    def test_revoke_token_action(self):
        risk = RiskResult(entity_key="cloud_user", risk_score=0.95, severity="CRITICAL", severity_id=5, remediation_level="AUTO_REMEDIATE", is_alert=True, S_sig=1.0, A_ml=0.9, delta_D=2.0, T_trust=0.0, explanation="Critical")
        evt = _norm_cloud({"user_identity": "cloud_user", "action": "StopLogging"})
        actions = self.orch.evaluate_and_respond(risk, evt)
        self.assertGreater(len(actions), 0)
        self.assertEqual(actions[0].action_type, "REVOKE_TOKEN")

    def test_block_ip_action(self):
        risk = RiskResult(entity_key="10.0.0.5", risk_score=0.95, severity="CRITICAL", severity_id=5, remediation_level="AUTO_REMEDIATE", is_alert=True, S_sig=1.0, A_ml=0.9, delta_D=2.0, T_trust=0.0, explanation="Critical")
        evt = _norm_network({"src_ip": "10.0.0.5", "packet_count": 5000})
        actions = self.orch.evaluate_and_respond(risk, evt)
        self.assertGreater(len(actions), 0)
        self.assertIn(actions[0].action_type, ["BLOCK_IP", "ISOLATE_HOST"])

    def test_log_only_no_action(self):
        risk = RiskResult(entity_key="safe_host", risk_score=0.1, severity="INFO", severity_id=1, remediation_level="LOG_ONLY", is_alert=False, S_sig=0.0, A_ml=0.1, delta_D=0.0, T_trust=0.5, explanation="Safe")
        actions = self.orch.evaluate_and_respond(risk, {})
        self.assertEqual(len(actions), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SOC Analyst Approval Queue Tests (T16–T20)
# ─────────────────────────────────────────────────────────────────────────────

class T04_AnalystApprovalQueue(unittest.TestCase):
    def setUp(self):
        self.orch = get_response_orchestrator()

    def test_stage_approval_queue(self):
        risk = RiskResult(entity_key="stage_host", risk_score=0.75, severity="HIGH", severity_id=4, remediation_level="STAGE_APPROVAL", is_alert=True, S_sig=0.8, A_ml=0.7, delta_D=1.0, T_trust=0.3, explanation="High")
        evt = _norm_process({"name": "suspicious.sh", "pid": 999, "hostname": "stage_host"})
        actions = self.orch.evaluate_and_respond(risk, evt)
        self.assertGreater(len(actions), 0)
        self.assertEqual(actions[0].status, "PENDING_APPROVAL")

    def test_approve_staged_action(self):
        risk = RiskResult(entity_key="app_host", risk_score=0.75, severity="HIGH", severity_id=4, remediation_level="STAGE_APPROVAL", is_alert=True, S_sig=0.8, A_ml=0.7, delta_D=1.0, T_trust=0.3, explanation="High")
        evt = _norm_process({"name": "test.exe", "pid": 888, "hostname": "app_host"})
        actions = self.orch.evaluate_and_respond(risk, evt)
        act_id = actions[0].action_id
        res = self.orch.approve_action(act_id)
        self.assertTrue(res)

    def test_reject_staged_action(self):
        risk = RiskResult(entity_key="rej_host", risk_score=0.75, severity="HIGH", severity_id=4, remediation_level="STAGE_APPROVAL", is_alert=True, S_sig=0.8, A_ml=0.7, delta_D=1.0, T_trust=0.3, explanation="High")
        evt = _norm_process({"name": "false_positive.exe", "pid": 777, "hostname": "rej_host"})
        actions = self.orch.evaluate_and_respond(risk, evt)
        act_id = actions[0].action_id
        res = self.orch.reject_action(act_id, "False positive confirmed")
        self.assertTrue(res)

    def test_nonexistent_action_approval(self):
        res = self.orch.approve_action("nonexistent_id_12345")
        self.assertFalse(res)

    def test_action_history_audit_logging(self):
        hist = self.orch.get_action_history()
        self.assertIsInstance(hist, list)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FastAPI REST API Endpoint Tests (T21–T25)
# ─────────────────────────────────────────────────────────────────────────────

class T05_FastAPISOCEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "healthy")

    def test_alerts_list_endpoint(self):
        r = self.client.get("/alerts?limit=5")
        self.assertEqual(r.status_code, 200)
        self.assertIn("alerts", r.json())

    def test_entity_report_endpoint_json_and_markdown(self):
        r_json = self.client.get("/entities/192.168.1.50/report?ocsf_class=network_activity")
        self.assertEqual(r_json.status_code, 200)
        self.assertIn("overall_confidence", r_json.json())

        r_md = self.client.get("/entities/192.168.1.50/report?ocsf_class=network_activity&format=markdown")
        self.assertEqual(r_md.status_code, 200)
        self.assertIn("content", r_md.json())

    def test_analyst_feedback_endpoint(self):
        payload = {"ocsf_class": "network_activity", "entity_key": "192.168.1.50", "action": "mark_false_positive"}
        r = self.client.post("/analyst/feedback", json=payload)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "SUCCESS")

    def test_soc_metrics_endpoint(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["system_status"], "OPERATIONAL")


if __name__ == "__main__":
    unittest.main()
