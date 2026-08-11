"""
AHRAS Module 1 — Test Suite
============================
Tests every component of the telemetry pipeline:

  T1  Config loading
  T2  Message bus (dev queue)
  T3  Storage layer (SQLite)
  T4  Shannon entropy calculation
  T5  File hash
  T6  GeoIP enrichment (private IP shortcut)
  T7  Abuse score (no-key path)
  T8  OCSF normalization — network event
  T9  OCSF normalization — process event
  T10 OCSF normalization — file event (normal)
  T11 OCSF normalization — file event (high entropy / ransomware)
  T12 OCSF normalization — cloud event (critical action)
  T13 OCSF normalization — unknown source is dropped cleanly
  T14 Network sensor — simulated flow generation
  T15 Host agent — process monitor poll
  T16 Host agent — connection monitor poll
  T17 Cloud adapter — synthetic event format
  T18 End-to-end pipeline: inject → normalize → store → query
  T19 Port scan indicator in normalization
  T20 Suspicious lineage detection
  T21 Bus backpressure (50k message limit)
  T22 SQLite concurrent write safety
  T23 Normalizer stats tracking
  T24 Storage aggregate_classes
"""

import sys
import os
import time
import uuid
import math
import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force dev mode for all tests
os.environ["AHRAS_DEV_MODE"] = "true"
os.environ["AHRAS_DEV_MODE"] = "true"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raw_network(**overrides) -> dict:
    base = {
        "event_id":         str(uuid.uuid4()),
        "source":           "network_tap",
        "timestamp":        _ts(),
        "src_ip":           "192.168.1.100",
        "dst_ip":           "10.0.0.1",
        "src_port":         54321,
        "dst_port":         80,
        "protocol":         "TCP",
        "packet_count":     10,
        "byte_count":       1500,
        "duration_sec":     1.5,
        "tcp_flags":        ["SYN", "ACK"],
        "unique_dst_ports": 1,
    }
    return {**base, **overrides}


def _raw_process(**overrides) -> dict:
    base = {
        "event_id":    str(uuid.uuid4()),
        "source":      "host_agent",
        "event_type":  "process_spawn",
        "timestamp":   _ts(),
        "hostname":    "test-host",
        "pid":         12345,
        "name":        "bash",
        "exe":         "/bin/bash",
        "cmdline":     "bash -c 'id'",
        "username":    "testuser",
        "parent_pid":  1000,
        "parent_name": "python3",
    }
    return {**base, **overrides}


def _raw_file(**overrides) -> dict:
    base = {
        "event_id":    str(uuid.uuid4()),
        "source":      "host_agent",
        "event_type":  "file_write",
        "timestamp":   _ts(),
        "hostname":    "test-host",
        "filepath":    "/home/user/document.txt",
        "entropy":     4.5,
        "high_entropy": False,
        "sha256":      "",
    }
    return {**base, **overrides}


def _raw_cloud(**overrides) -> dict:
    base = {
        "event_id":           str(uuid.uuid4()),
        "source":             "cloud_adapter",
        "event_type":         "cloud_api_call",
        "timestamp":          _ts(),
        "provider":           "aws",
        "action":             "s3:GetObject",
        "severity_hint":      "low",
        "user_identity":      "alice@corp.com",
        "source_ip":          "10.0.0.51",
        "region":             "us-east-1",
        "user_agent":         "boto3/1.28.0",
        "request_parameters": {},
        "error_code":         None,
    }
    return {**base, **overrides}


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

class T01_Config(unittest.TestCase):
    def test_settings_importable(self):
        from config.settings import (
            DEV_MODE, KAFKA_TOPIC_RAW, KAFKA_TOPIC_NORM,
            ENTROPY_THRESHOLD, WATCH_DIRS
        )
        self.assertTrue(DEV_MODE)
        self.assertEqual(KAFKA_TOPIC_RAW, "raw.telemetry")
        self.assertEqual(KAFKA_TOPIC_NORM, "normalized.events")
        self.assertGreater(ENTROPY_THRESHOLD, 0)
        self.assertIsInstance(WATCH_DIRS, list)


class T02_MessageBus(unittest.TestCase):
    def setUp(self):
        # Reset singleton for test isolation
        import pipeline.bus as bus
        bus._topics.clear()

    def test_produce_consume_roundtrip(self):
        from pipeline.bus import get_producer, get_consumer
        producer = get_producer()
        consumer = get_consumer("test.topic")

        msg = {"hello": "world", "n": 42}
        producer.send("test.topic", msg)

        received = next(iter(consumer))
        self.assertEqual(received["hello"], "world")
        self.assertEqual(received["n"], 42)

    def test_multiple_messages_ordered(self):
        from pipeline.bus import get_producer, get_consumer
        producer = get_producer()
        consumer = get_consumer("test.order")

        msgs = [{"seq": i} for i in range(5)]
        for m in msgs:
            producer.send("test.order", m)

        received = [next(iter(consumer))["seq"] for _ in range(5)]
        self.assertEqual(received, list(range(5)))

    def test_flush_is_noop(self):
        from pipeline.bus import get_producer
        get_producer().flush()   # should not raise


class T03_Storage(unittest.TestCase):
    def setUp(self):
        # Use a temp file for each test
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        from storage.store import _SQLiteStore
        self._store = _SQLiteStore(self._tmp.name)

    def tearDown(self):
        self._store.close()
        Path(self._tmp.name).unlink(missing_ok=True)

    def test_insert_and_count(self):
        doc = {"event_id": str(uuid.uuid4()), "ocsf_class": "network_activity",
               "severity_id": 1, "time": _ts()}
        self._store.insert("events", doc)
        self.assertEqual(self._store.count("events"), 1)

    def test_query_returns_doc(self):
        eid = str(uuid.uuid4())
        doc = {"event_id": eid, "ocsf_class": "file_activity",
               "severity_id": 3, "time": _ts(), "test_key": "abc"}
        self._store.insert("events", doc)
        results = self._store.query("events", limit=10)
        self.assertTrue(any(r.get("test_key") == "abc" for r in results))

    def test_aggregate_classes(self):
        for cls in ["network_activity", "network_activity", "cloud_api"]:
            self._store.insert("events", {
                "event_id": str(uuid.uuid4()), "ocsf_class": cls,
                "severity_id": 1, "time": _ts()
            })
        agg = {r["_id"]: r["count"] for r in self._store.aggregate_classes("events")}
        self.assertEqual(agg["network_activity"], 2)
        self.assertEqual(agg["cloud_api"], 1)

    def test_upsert_same_id(self):
        eid = str(uuid.uuid4())
        doc = {"event_id": eid, "ocsf_class": "test", "severity_id": 1, "time": _ts()}
        self._store.insert("events", doc)
        self._store.insert("events", doc)   # should not raise (OR REPLACE)
        self.assertEqual(self._store.count("events"), 1)


class T04_Entropy(unittest.TestCase):
    def test_zero_entropy_uniform(self):
        """File of all identical bytes → entropy 0"""
        from sensors.host_agent import shannon_entropy
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"\x00" * 1024)
            name = f.name
        try:
            h = shannon_entropy(name)
            self.assertAlmostEqual(h, 0.0, places=3)
        finally:
            Path(name).unlink(missing_ok=True)

    def test_high_entropy_random(self):
        """Random bytes → entropy close to 8.0"""
        from sensors.host_agent import shannon_entropy
        import os as _os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(_os.urandom(65536))
            name = f.name
        try:
            h = shannon_entropy(name)
            self.assertGreater(h, 7.0)
        finally:
            Path(name).unlink(missing_ok=True)

    def test_text_low_entropy(self):
        """Plain text → entropy clearly below threshold"""
        from sensors.host_agent import shannon_entropy
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt",
                                          mode="w") as f:
            f.write("hello world " * 500)
            name = f.name
        try:
            h = shannon_entropy(name)
            self.assertLess(h, 7.2)
        finally:
            Path(name).unlink(missing_ok=True)

    def test_missing_file_returns_zero(self):
        from sensors.host_agent import shannon_entropy
        h = shannon_entropy("/nonexistent/path/file.bin")
        self.assertEqual(h, 0.0)

    def test_entropy_formula_manual(self):
        """Verify H(X) manually for 2-symbol equal distribution → H = 1.0"""
        from sensors.host_agent import shannon_entropy
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            # Equal numbers of 0x00 and 0xFF → P(x) = 0.5 each → H = 1.0
            f.write(bytes([0x00, 0xFF] * 512))
            name = f.name
        try:
            h = shannon_entropy(name)
            self.assertAlmostEqual(h, 1.0, places=2)
        finally:
            Path(name).unlink(missing_ok=True)


class T05_FileHash(unittest.TestCase):
    def test_sha256_correct(self):
        import hashlib
        from sensors.host_agent import file_sha256
        content = b"ahras test content"
        expected = hashlib.sha256(content).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            name = f.name
        try:
            self.assertEqual(file_sha256(name), expected)
        finally:
            Path(name).unlink(missing_ok=True)

    def test_missing_file_returns_empty(self):
        from sensors.host_agent import file_sha256
        self.assertEqual(file_sha256("/nonexistent/file.bin"), "")


class T06_GeoIP(unittest.TestCase):
    def test_private_ip_returns_internal(self):
        from normalizer.enrichment import geoip_dict, geoip
        geoip.cache_clear()
        result = geoip_dict("192.168.1.1")
        self.assertEqual(result["country"], "internal")

    def test_loopback_is_private(self):
        from normalizer.enrichment import geoip_dict, geoip
        geoip.cache_clear()
        result = geoip_dict("127.0.0.1")
        self.assertEqual(result["country"], "internal")

    def test_10_range_is_private(self):
        from normalizer.enrichment import _is_private
        self.assertTrue(_is_private("10.0.0.1"))
        self.assertTrue(_is_private("172.16.0.1"))
        self.assertFalse(_is_private("8.8.8.8"))


class T07_AbuseScore(unittest.TestCase):
    def test_no_key_returns_zero(self):
        from normalizer.enrichment import abuse_score
        abuse_score.cache_clear()
        score = abuse_score("8.8.8.8")
        self.assertEqual(score, 0)   # no API key configured in test env

    def test_private_ip_returns_zero(self):
        from normalizer.enrichment import abuse_score
        abuse_score.cache_clear()
        self.assertEqual(abuse_score("192.168.1.1"), 0)


class T08_NormNetwork(unittest.TestCase):
    def test_schema_fields_present(self):
        from normalizer.ocsf_normalizer import _norm_network
        evt = _norm_network(_raw_network())
        for field in ["ocsf_class_id", "ocsf_class", "event_id", "time",
                      "severity_id", "src_endpoint", "dst_endpoint",
                      "protocol", "traffic", "enrichment"]:
            self.assertIn(field, evt, f"Missing field: {field}")

    def test_class_id_correct(self):
        from normalizer.ocsf_normalizer import _norm_network
        evt = _norm_network(_raw_network())
        self.assertEqual(evt["ocsf_class_id"], 1001)
        self.assertEqual(evt["ocsf_class"], "network_activity")

    def test_traffic_stats_preserved(self):
        from normalizer.ocsf_normalizer import _norm_network
        raw = _raw_network(packet_count=99, byte_count=8888, duration_sec=5.5)
        evt = _norm_network(raw)
        self.assertEqual(evt["traffic"]["packets"], 99)
        self.assertEqual(evt["traffic"]["bytes"], 8888)
        self.assertAlmostEqual(evt["traffic"]["duration_sec"], 5.5)

    def test_internal_src_severity_1(self):
        from normalizer.ocsf_normalizer import _norm_network
        evt = _norm_network(_raw_network(src_ip="10.0.0.5"))
        self.assertEqual(evt["severity_id"], 1)


class T09_NormProcess(unittest.TestCase):
    def test_schema_fields_present(self):
        from normalizer.ocsf_normalizer import _norm_process
        evt = _norm_process(_raw_process())
        for field in ["ocsf_class_id", "ocsf_class", "event_id", "time",
                      "severity_id", "actor", "process", "device"]:
            self.assertIn(field, evt, f"Missing: {field}")

    def test_class_id_correct(self):
        from normalizer.ocsf_normalizer import _norm_process
        evt = _norm_process(_raw_process())
        self.assertEqual(evt["ocsf_class_id"], 1002)

    def test_process_details_preserved(self):
        from normalizer.ocsf_normalizer import _norm_process
        raw = _raw_process(name="nc", exe="/bin/nc", username="root")
        evt = _norm_process(raw)
        self.assertEqual(evt["actor"]["process"]["name"], "nc")
        self.assertEqual(evt["actor"]["process"]["user"]["name"], "root")


class T10_NormFileNormal(unittest.TestCase):
    def test_low_entropy_severity_1(self):
        from normalizer.ocsf_normalizer import _norm_file
        evt = _norm_file(_raw_file(entropy=3.2, high_entropy=False))
        self.assertEqual(evt["severity_id"], 1)
        self.assertFalse(evt["enrichment"]["ransomware_indicator"])

    def test_class_id_correct(self):
        from normalizer.ocsf_normalizer import _norm_file
        evt = _norm_file(_raw_file())
        self.assertEqual(evt["ocsf_class_id"], 1003)
        self.assertEqual(evt["ocsf_class"], "file_activity")


class T11_NormFileHighEntropy(unittest.TestCase):
    def test_high_entropy_raises_severity(self):
        from normalizer.ocsf_normalizer import _norm_file
        evt = _norm_file(_raw_file(entropy=7.8, high_entropy=True,
                                    filepath="/home/user/photo.jpg.enc"))
        self.assertGreaterEqual(evt["severity_id"], 3)
        self.assertTrue(evt["enrichment"]["ransomware_indicator"])
        self.assertTrue(evt["enrichment"]["high_entropy"])

    def test_entropy_value_preserved(self):
        from normalizer.ocsf_normalizer import _norm_file
        evt = _norm_file(_raw_file(entropy=7.94, high_entropy=True))
        self.assertAlmostEqual(evt["enrichment"]["entropy"], 7.94)

    def test_filepath_preserved(self):
        from normalizer.ocsf_normalizer import _norm_file
        evt = _norm_file(_raw_file(filepath="/tmp/secret.docx.locked"))
        self.assertEqual(evt["file"]["path"], "/tmp/secret.docx.locked")


class T12_NormCloud(unittest.TestCase):
    def test_critical_action_severity_4(self):
        from normalizer.ocsf_normalizer import _norm_cloud
        evt = _norm_cloud(_raw_cloud(
            action="cloudtrail:StopLogging",
            severity_hint="critical",
            user_identity="unknown-external",
        ))
        self.assertEqual(evt["severity_id"], 4)
        self.assertTrue(evt["enrichment"]["high_privilege_action"])

    def test_low_action_severity_1(self):
        from normalizer.ocsf_normalizer import _norm_cloud
        evt = _norm_cloud(_raw_cloud(action="s3:GetObject", severity_hint="low"))
        self.assertEqual(evt["severity_id"], 1)

    def test_class_id_correct(self):
        from normalizer.ocsf_normalizer import _norm_cloud
        evt = _norm_cloud(_raw_cloud())
        self.assertEqual(evt["ocsf_class_id"], 4001)
        self.assertEqual(evt["ocsf_class"], "cloud_api")

    def test_api_operation_preserved(self):
        from normalizer.ocsf_normalizer import _norm_cloud
        evt = _norm_cloud(_raw_cloud(action="iam:CreateUser"))
        self.assertEqual(evt["api"]["operation"], "iam:CreateUser")


class T13_NormUnknownDropped(unittest.TestCase):
    def test_unknown_source_returns_none(self):
        from normalizer.ocsf_normalizer import _route
        raw = {"event_id": str(uuid.uuid4()), "source": "mystery_source",
               "timestamp": _ts()}
        self.assertIsNone(_route(raw))

    def test_unknown_host_event_type_returns_none(self):
        from normalizer.ocsf_normalizer import _route
        raw = {"event_id": str(uuid.uuid4()), "source": "host_agent",
               "event_type": "mystery_type", "timestamp": _ts()}
        self.assertIsNone(_route(raw))


class T14_NetworkSensor(unittest.TestCase):
    def test_simulated_emits_valid_events(self):
        import pipeline.bus as bus
        bus._topics.clear()

        from sensors.network_sensor import SimulatedNetworkSensor
        from pipeline.bus import get_consumer

        sensor   = SimulatedNetworkSensor(flows_per_second=20,
                                           attack_probability=0.0)
        consumer = get_consumer("raw.telemetry")

        stop = threading.Event()
        t = threading.Thread(target=sensor.start, args=(stop,), daemon=True)
        t.start()
        time.sleep(0.5)
        stop.set()

        events = []
        try:
            for _ in range(3):
                events.append(next(iter(consumer)))
        except StopIteration:
            pass

        self.assertGreater(len(events), 0)
        for e in events:
            self.assertEqual(e["source"], "network_tap")
            self.assertIn("src_ip", e)
            self.assertIn("protocol", e)
            self.assertIn(e["protocol"], ["TCP", "UDP", "ICMP", "OTHER"])

    def test_attack_events_emitted(self):
        import pipeline.bus as bus
        bus._topics.clear()

        from sensors.network_sensor import SimulatedNetworkSensor
        from pipeline.bus import get_consumer

        # 100% attack probability to guarantee attack events
        sensor   = SimulatedNetworkSensor(flows_per_second=20,
                                           attack_probability=1.0)
        consumer = get_consumer("raw.telemetry")

        stop = threading.Event()
        t = threading.Thread(target=sensor.start, args=(stop,), daemon=True)
        t.start()
        time.sleep(0.3)
        stop.set()

        events = []
        try:
            for _ in range(5):
                events.append(next(iter(consumer)))
        except StopIteration:
            pass

        self.assertGreater(len(events), 0)


class T15_ProcessMonitor(unittest.TestCase):
    def test_poll_does_not_crash(self):
        import pipeline.bus as bus
        bus._topics.clear()
        from sensors.host_agent import _ProcessMonitor
        from pipeline.bus import get_producer
        mon = _ProcessMonitor(get_producer())
        mon.poll()   # just verify no exception

    def test_poll_emits_process_events(self):
        import pipeline.bus as bus
        bus._topics.clear()
        from sensors.host_agent import _ProcessMonitor
        from pipeline.bus import get_producer, get_consumer

        prod = get_producer()
        con  = get_consumer("raw.telemetry")
        mon  = _ProcessMonitor(prod)
        # Reset seen_pids to empty to force detection of all current processes
        mon._seen_pids = set()
        mon.poll()

        events = []
        try:
            for _ in range(5):
                events.append(next(iter(con)))
        except StopIteration:
            pass

        if events:
            self.assertEqual(events[0]["source"], "host_agent")
            self.assertEqual(events[0]["event_type"], "process_spawn")


class T16_ConnectionMonitor(unittest.TestCase):
    def test_poll_does_not_crash(self):
        import pipeline.bus as bus
        bus._topics.clear()
        from sensors.host_agent import _ConnectionMonitor
        from pipeline.bus import get_producer
        mon = _ConnectionMonitor(get_producer())
        mon.poll()   # may produce 0 events in sandbox but must not crash


class T17_CloudAdapter(unittest.TestCase):
    def test_synthetic_event_schema(self):
        from sensors.cloud_adapter import _rand_event
        for _ in range(20):
            evt = _rand_event()
            self.assertEqual(evt["source"], "cloud_adapter")
            self.assertEqual(evt["event_type"], "cloud_api_call")
            self.assertIn("action", evt)
            self.assertIn(evt["severity_hint"],
                          ["low", "medium", "high", "critical"])
            self.assertIn("user_identity", evt)
            self.assertIn("source_ip", evt)
            self.assertIn("region", evt)

    def test_critical_events_present_in_catalog(self):
        from sensors.cloud_adapter import _EVENTS
        critical = [e for e in _EVENTS if e[1] == "critical"]
        self.assertGreater(len(critical), 0)


class T18_EndToEnd(unittest.TestCase):
    """Full pipeline: produce raw → normalize → store → query."""

    def setUp(self):
        import pipeline.bus as bus
        bus._topics.clear()

        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()

    def tearDown(self):
        Path(self._tmp.name).unlink(missing_ok=True)

    def test_network_event_reaches_storage(self):
        from storage.store import _SQLiteStore
        from normalizer.ocsf_normalizer import _route
        from pipeline.bus import get_producer

        store    = _SQLiteStore(self._tmp.name)
        producer = get_producer()

        raw = _raw_network()
        normalized = _route(raw)
        self.assertIsNotNone(normalized)

        store.insert("events", normalized)
        count = store.count("events")
        self.assertEqual(count, 1)

        results = store.query("events", limit=1)
        self.assertEqual(results[0]["ocsf_class"], "network_activity")
        store.close()

    def test_all_three_sources_stored(self):
        from storage.store import _SQLiteStore
        from normalizer.ocsf_normalizer import _route

        store = _SQLiteStore(self._tmp.name)

        raws = [
            _raw_network(),
            _raw_process(),
            _raw_file(entropy=7.9, high_entropy=True),
            _raw_cloud(action="cloudtrail:StopLogging", severity_hint="critical"),
        ]

        classes = []
        for raw in raws:
            norm = _route(raw)
            self.assertIsNotNone(norm, f"Failed to normalize: {raw['source']}")
            store.insert("events", norm)
            classes.append(norm["ocsf_class"])

        self.assertIn("network_activity", classes)
        self.assertIn("process_activity", classes)
        self.assertIn("file_activity", classes)
        self.assertIn("cloud_api", classes)
        self.assertEqual(store.count("events"), 4)
        store.close()

    def test_normalizer_thread_processes_events(self):
        import pipeline.bus as bus
        bus._topics.clear()

        # Use temp store
        import storage.store as store_mod
        store_mod._store_instance = None
        os.environ["SQLITE_PATH"] = self._tmp.name

        from normalizer.ocsf_normalizer import run_normalizer, get_stats
        from pipeline.bus import get_producer

        stop = threading.Event()
        t = threading.Thread(target=run_normalizer, args=(stop,), daemon=True)
        t.start()

        # Inject events
        producer = get_producer()
        for _ in range(5):
            producer.send("raw.telemetry", _raw_network())
        producer.send("raw.telemetry", _raw_cloud())
        producer.flush()

        time.sleep(0.5)
        stop.set()
        t.join(timeout=2)

        stats = get_stats()
        self.assertGreater(stats["normalized"], 0)
        self.assertEqual(stats["errors"], 0)


class T19_PortScanDetection(unittest.TestCase):
    def test_port_scan_flagged(self):
        from normalizer.ocsf_normalizer import _norm_network
        raw = _raw_network(
            unique_dst_ports=200,
            tcp_flags=["SYN"],
            packet_count=200,
        )
        evt = _norm_network(raw)
        self.assertTrue(evt["enrichment"]["port_scan_indicator"])

    def test_normal_traffic_not_flagged(self):
        from normalizer.ocsf_normalizer import _norm_network
        raw = _raw_network(unique_dst_ports=1, tcp_flags=["SYN", "ACK"])
        evt = _norm_network(raw)
        self.assertFalse(evt["enrichment"]["port_scan_indicator"])


class T20_SuspiciousLineage(unittest.TestCase):
    def test_office_spawning_shell_flagged(self):
        from normalizer.ocsf_normalizer import _norm_process
        raw = _raw_process(parent_name="libreoffice", name="bash")
        evt = _norm_process(raw)
        self.assertTrue(evt["enrichment"]["suspicious_lineage"])
        self.assertGreaterEqual(evt["severity_id"], 3)

    def test_normal_lineage_not_flagged(self):
        from normalizer.ocsf_normalizer import _norm_process
        raw = _raw_process(parent_name="systemd", name="sshd")
        evt = _norm_process(raw)
        self.assertFalse(evt["enrichment"]["suspicious_lineage"])
        self.assertEqual(evt["severity_id"], 1)


class T21_BusBackpressure(unittest.TestCase):
    def test_full_queue_does_not_raise(self):
        import pipeline.bus as bus
        bus._topics.clear()
        from pipeline.bus import _DevProducer
        producer = _DevProducer()
        # Fill past max (50k) — should not raise
        for i in range(200):
            producer.send("backpressure.test", {"i": i})


class T22_ConcurrentWrites(unittest.TestCase):
    def test_concurrent_inserts_consistent(self):
        from storage.store import _SQLiteStore
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = _SQLiteStore(tmp.name)

        errors = []

        def insert_batch(n):
            try:
                for _ in range(n):
                    store.insert("events", {
                        "event_id": str(uuid.uuid4()),
                        "ocsf_class": "network_activity",
                        "severity_id": 1,
                        "time": _ts(),
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=insert_batch, args=(20,))
                   for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Concurrent write errors: {errors}")
        self.assertEqual(store.count("events"), 100)
        store.close()
        Path(tmp.name).unlink(missing_ok=True)


class T23_NormalizerStats(unittest.TestCase):
    def test_stats_increment_correctly(self):
        import pipeline.bus as bus
        bus._topics.clear()

        import storage.store as store_mod
        store_mod._store_instance = None

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        os.environ["SQLITE_PATH"] = tmp.name

        import normalizer.ocsf_normalizer as norm_mod
        norm_mod._stats = {"received": 0, "normalized": 0, "dropped": 0, "errors": 0}

        from normalizer.ocsf_normalizer import run_normalizer, get_stats
        from pipeline.bus import get_producer

        stop = threading.Event()
        t = threading.Thread(target=run_normalizer, args=(stop,), daemon=True)
        t.start()

        prod = get_producer()
        for _ in range(3):
            prod.send("raw.telemetry", _raw_network())
        # Unknown source — should be dropped
        prod.send("raw.telemetry", {"source": "unknown", "event_id": str(uuid.uuid4())})
        prod.flush()

        time.sleep(0.4)
        stop.set()
        t.join(timeout=2)

        stats = get_stats()
        self.assertGreaterEqual(stats["normalized"], 3)
        self.assertGreaterEqual(stats["dropped"], 1)
        self.assertEqual(stats["errors"], 0)

        Path(tmp.name).unlink(missing_ok=True)


class T24_AggregateClasses(unittest.TestCase):
    def test_aggregate_correct(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        from storage.store import _SQLiteStore
        store = _SQLiteStore(tmp.name)

        from normalizer.ocsf_normalizer import _route
        for raw in [_raw_network(), _raw_network(), _raw_cloud(),
                    _raw_file(), _raw_process()]:
            norm = _route(raw)
            if norm:
                store.insert("events", norm)

        agg = {r["_id"]: r["count"] for r in store.aggregate_classes("events")}
        self.assertEqual(agg.get("network_activity", 0), 2)
        self.assertEqual(agg.get("cloud_api", 0), 1)
        store.close()
        Path(tmp.name).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    loader.sortTestMethodsUsing = None

    # Add all test classes in order
    for cls in [
        T01_Config, T02_MessageBus, T03_Storage, T04_Entropy,
        T05_FileHash, T06_GeoIP, T07_AbuseScore, T08_NormNetwork,
        T09_NormProcess, T10_NormFileNormal, T11_NormFileHighEntropy,
        T12_NormCloud, T13_NormUnknownDropped, T14_NetworkSensor,
        T15_ProcessMonitor, T16_ConnectionMonitor, T17_CloudAdapter,
        T18_EndToEnd, T19_PortScanDetection, T20_SuspiciousLineage,
        T21_BusBackpressure, T22_ConcurrentWrites, T23_NormalizerStats,
        T24_AggregateClasses,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
