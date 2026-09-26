"""
AHRAS Unit & Integration Tests — Streaming Sketch Fast Path (Extension F)
"""
import uuid
import pytest
import numpy as np

from detection.streaming_sketch import (
    StreamingSketchEngine,
    SketchConfig,
    SketchEvidenceRecord,
    SketchScreenResult,
    get_sketch_engine,
)


@pytest.fixture
def clean_engine():
    cfg = SketchConfig(
        width=2048,
        depth=4,
        heavy_hitter_threshold=0.02,
        fanout_threshold=10,
        burst_multiplier=2.5,
    )
    return StreamingSketchEngine(config=cfg)


def _gen_event(src_ip="192.168.1.10", dst_ip="10.0.0.1", pkts=10, bytes_cnt=1000):
    return {
        "event_id": str(uuid.uuid4()),
        "ocsf_class": "network_activity",
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": 50000,
        "dst_port": 80,
        "protocol": "TCP",
        "packet_count": pkts,
        "byte_count": bytes_cnt,
        "timestamp": 1700000000.0,
    }


# ── Tests ────────────────────────────────────────────────────────────────────

def test_sketch_config_and_singleton():
    cfg = SketchConfig(width=1024, depth=3)
    engine = get_sketch_engine(cfg)
    assert engine.config.width == 1024
    assert engine.config.depth == 3
    assert engine.epsilon > 0.0
    assert engine.delta > 0.0
    stats = engine.get_stats()
    assert "memory_footprint_kb" in stats
    assert stats["width"] == 1024


def test_count_min_no_undercount_invariant(clean_engine):
    """
    Fundamental Count-Min guarantee: hat{a}_i >= a_i always (never undercounts).
    """
    target_ip = "192.168.1.99"
    exact_count = 150

    for _ in range(exact_count):
        clean_engine.update(_gen_event(src_ip=target_ip))

    # Also insert background noise
    for i in range(300):
        clean_engine.update(_gen_event(src_ip=f"10.0.{i % 20}.{i % 250}"))

    est_count, err_bound = clean_engine.query_source_frequency(target_ip)
    assert est_count >= exact_count
    # Error should be within theoretical bounds: hat{a} - a <= epsilon * N
    total_n = clean_engine.total_events
    assert est_count - exact_count <= clean_engine.epsilon * total_n + 1.0


def test_volume_accumulation(clean_engine):
    src_ip = "172.16.0.5"
    pkts_per_evt = 25
    bytes_per_evt = 1500
    n = 20

    for _ in range(n):
        clean_engine.update(_gen_event(src_ip=src_ip, pkts=pkts_per_evt, bytes_cnt=bytes_per_evt))

    est_pkts, est_bytes = clean_engine.query_source_volume(src_ip)
    assert est_pkts >= n * pkts_per_evt
    assert est_bytes >= n * bytes_per_evt


def test_fanout_cardinality_estimation(clean_engine):
    """
    Verifies HyperLogLog estimation of distinct destinations contacted by a single scanner.
    """
    scanner_ip = "10.10.10.50"
    num_distinct_targets = 30

    for i in range(num_distinct_targets):
        dst = f"192.168.100.{i + 1}"
        clean_engine.update(_gen_event(src_ip=scanner_ip, dst_ip=dst))

    est_fanout = clean_engine.query_source_fanout(scanner_ip)
    # HyperLogLog with m=32 has ~18% standard error. Should be within [15, 45].
    assert 15 <= est_fanout <= 45


def test_heavy_hitter_screening_triage(clean_engine):
    # 1. Normal traffic should NOT be flagged
    normal_evt = _gen_event(src_ip="192.168.1.5", dst_ip="8.8.8.8")
    res_normal = clean_engine.screen(normal_evt)
    assert res_normal.is_flagged is False
    assert len(res_normal.evidence_records) == 0

    # 2. Inject heavy-hitter burst from attacker IP
    attacker_ip = "10.0.66.66"
    for _ in range(50):
        clean_engine.screen(_gen_event(src_ip=attacker_ip, dst_ip="192.168.1.1"))

    # The next event from this attacker should be flagged as heavy-hitter
    res_flagged = clean_engine.screen(_gen_event(src_ip=attacker_ip, dst_ip="192.168.1.1"))
    assert res_flagged.is_flagged is True
    assert any("src_heavy_hitter" in r for r in res_flagged.reasons)
    assert len(res_flagged.evidence_records) > 0

    rec = res_flagged.evidence_records[0]
    assert isinstance(rec, SketchEvidenceRecord)
    assert rec.source == "sketch"
    assert rec.detector_type == "streaming_sketch"
    assert rec.mitre_technique == "T1498"
    assert rec.exact_value_available is False  # Contract adherence
    assert rec.error_bound > 0.0


def test_high_fanout_port_scan_screening(clean_engine):
    scanner_ip = "192.168.5.200"
    # Contact 25 distinct destinations to trigger fan-out scanner threshold (10)
    for i in range(25):
        clean_engine.screen(_gen_event(src_ip=scanner_ip, dst_ip=f"10.0.1.{i}"))

    res = clean_engine.screen(_gen_event(src_ip=scanner_ip, dst_ip="10.0.1.99"))
    assert res.is_flagged is True
    assert any("high_fanout_scan" in r for r in res.reasons)
    fanout_records = [r for r in res.evidence_records if r.anomaly_type == "high_fanout_scan"]
    assert len(fanout_records) > 0
    assert fanout_records[0].mitre_technique == "T1046"


def test_strict_o1_memory_invariance(clean_engine):
    """
    Verifies that memory footprint does not grow with unique cardinality (O(1) space).
    """
    mem_init = clean_engine.get_memory_footprint_bytes()

    # Stream 2,000 distinct entities
    for i in range(2000):
        clean_engine.update(_gen_event(src_ip=f"10.{i // 256}.{i % 256}.1", dst_ip=f"172.16.{i // 256}.{i % 256}"))

    mem_after = clean_engine.get_memory_footprint_bytes()
    assert mem_init == mem_after  # Strictly exact O(1) memory!


def test_concurrent_streaming_thread_safety(clean_engine):
    import concurrent.futures

    events = [_gen_event(src_ip=f"10.0.0.{i % 10}") for i in range(100)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(clean_engine.screen, events))

    assert len(results) == 100
    assert clean_engine.total_events == 100
