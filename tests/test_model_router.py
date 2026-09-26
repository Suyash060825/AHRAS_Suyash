"""
AHRAS Unit & Integration Tests — Confidence-Based Early-Exit Model Router (Extension E)
"""
import uuid
import pytest
import numpy as np
from datetime import datetime, timezone

from detection.model_router import (
    ConfidenceModelRouter,
    RouterConfig,
    RoutedDetectionResult,
    get_model_router,
)
from detection.hybrid_engine import DetectionResult


@pytest.fixture
def clean_router():
    cfg = RouterConfig(
        conf_threshold=0.85,
        uncertainty_threshold=0.20,
        margin_threshold=0.35,
        ood_threshold=0.30,
        decision_threshold=0.50,
        enabled=True,
    )
    return ConfidenceModelRouter(config=cfg)


def _benign_network_evt():
    return {
        "event_id": str(uuid.uuid4()),
        "ocsf_class": "network_activity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "src_ip": "192.168.1.50",
        "dst_ip": "8.8.8.8",
        "src_port": 49152,
        "dst_port": 443,
        "protocol": "TCP",
        "bytes_in": 1200,
        "bytes_out": 450,
        "packet_count": 8,
        "pkts_in": 5,
        "pkts_out": 3,
        "duration": 0.45,
        "tcp_flags": ["SYN", "ACK"],
        "severity_id": 1,
    }


def _signature_attack_evt():
    # SSH brute force / port scan signature trigger
    return {
        "event_id": str(uuid.uuid4()),
        "ocsf_class": "network_activity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "src_ip": "10.0.0.99",
        "dst_ip": "192.168.1.10",
        "src_port": 55123,
        "dst_port": 22,
        "protocol": "TCP",
        "packet_count": 1500,
        "byte_count": 90000,
        "duration_sec": 0.5,
        "tcp_flags": ["SYN"],
        "unique_dst_ports": 25,  # Port scan signature
        "severity_id": 4,
    }


def _stealthy_anomaly_evt():
    # Subtle behavioral anomaly: non-standard port, unusual byte ratios, no explicit signature match
    return {
        "event_id": str(uuid.uuid4()),
        "ocsf_class": "network_activity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "src_ip": "192.168.1.105",
        "dst_ip": "198.51.100.42",
        "src_port": 39482,
        "dst_port": 8443,
        "protocol": "TCP",
        "bytes_in": 45000,
        "bytes_out": 125000,
        "packet_count": 650,
        "pkts_in": 250,
        "pkts_out": 400,
        "duration": 120.0,
        "tcp_flags": ["ACK", "PSH"],
        "severity_id": 2,
    }


# ── Tests ────────────────────────────────────────────────────────────────────

def test_router_config_and_singleton():
    cfg = RouterConfig(conf_threshold=0.80, uncertainty_threshold=0.15)
    assert cfg.conf_threshold == 0.80
    assert cfg.uncertainty_threshold == 0.15
    assert cfg.enabled is True

    router = get_model_router(cfg)
    assert router.config.conf_threshold == 0.80
    assert isinstance(router, ConfidenceModelRouter)


def test_stage_1_early_exit_on_signature(clean_router):
    evt = _signature_attack_evt()
    res = clean_router.route(evt)

    assert res is not None
    assert isinstance(res, RoutedDetectionResult)
    assert isinstance(res, DetectionResult)  # Backward compatibility contract
    assert res.stage_exited == 1
    assert res.stage_name == "fast_path"
    assert res.early_exited is True
    assert res.is_alert is True
    assert res.confidence >= 0.85
    assert len(res.stages_evaluated) == 1
    assert "fast_path" in res.per_stage_latencies
    assert res.latency_ms > 0.0


def test_stage_1_early_exit_on_clear_benign(clean_router):
    evt = _benign_network_evt()
    res = clean_router.route(evt)

    assert res is not None
    assert res.stage_exited == 1
    assert res.stage_name == "fast_path"
    assert res.early_exited is True
    assert res.is_alert is False
    assert res.confidence <= 0.20
    assert "clear_benign" in res.exit_reason


def test_stage_2_or_later_progression_on_ambiguous(clean_router):
    evt = _stealthy_anomaly_evt()
    res = clean_router.route(evt)

    assert res is not None
    assert res.stage_exited >= 2
    assert 2 in res.stages_evaluated
    assert len(res.confidence_trajectory) >= 2
    assert len(res.uncertainty_trajectory) >= 2


def test_force_full_pipeline(clean_router):
    evt = _signature_attack_evt()
    # Force full execution across all 4 stages even if Stage 1 is decisive
    res = clean_router.route(evt, force_full=True)

    assert res is not None
    assert res.stage_exited == 4
    assert res.stage_name == "deep_graph"
    assert res.early_exited is False
    assert res.stages_evaluated == [1, 2, 3, 4]
    assert len(res.confidence_trajectory) == 4
    assert len(res.uncertainty_trajectory) == 4
    assert "fast_path" in res.per_stage_latencies
    assert "ml_ensemble" in res.per_stage_latencies
    assert "multimodal" in res.per_stage_latencies
    assert "deep_graph" in res.per_stage_latencies


def test_force_specific_stage(clean_router):
    evt = _benign_network_evt()
    res = clean_router.route(evt, force_stage=3)

    assert res is not None
    assert res.stage_exited == 3
    assert res.stage_name == "multimodal"
    assert res.stages_evaluated == [1, 2, 3]


def test_all_ocsf_classes_routing(clean_router):
    # Process activity
    proc_evt = {
        "event_id": str(uuid.uuid4()),
        "ocsf_class": "process_activity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "/usr/bin/cat /etc/hosts",
        "process_name": "cat",
        "is_elevated": False,
        "severity_id": 1,
    }
    r_proc = clean_router.route(proc_evt)
    assert r_proc is not None
    assert r_proc.ocsf_class == "process_activity"

    # File activity
    file_evt = {
        "event_id": str(uuid.uuid4()),
        "ocsf_class": "file_activity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filepath": "/home/user/document.txt",
        "entropy": 4.2,
        "severity_id": 1,
    }
    r_file = clean_router.route(file_evt)
    assert r_file is not None
    assert r_file.ocsf_class == "file_activity"

    # Cloud API
    cloud_evt = {
        "event_id": str(uuid.uuid4()),
        "ocsf_class": "cloud_api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_name": "DescribeInstances",
        "severity_id": 1,
    }
    r_cloud = clean_router.route(cloud_evt)
    assert r_cloud is not None
    assert r_cloud.ocsf_class == "cloud_api"


def test_threshold_tuning_affects_exit():
    # Strict router: requires 0.99 confidence to exit
    strict_cfg = RouterConfig(conf_threshold=0.99, margin_threshold=0.49)
    strict_router = ConfidenceModelRouter(config=strict_cfg)

    # Permissive router: requires only 0.70 confidence to exit
    perm_cfg = RouterConfig(conf_threshold=0.70, margin_threshold=0.20)
    perm_router = ConfidenceModelRouter(config=perm_cfg)

    evt = _stealthy_anomaly_evt()
    res_strict = strict_router.route(evt)
    res_perm = perm_router.route(evt)

    assert res_strict is not None
    assert res_perm is not None
    # Strict router should evaluate at least as many stages as permissive router
    assert len(res_strict.stages_evaluated) >= len(res_perm.stages_evaluated)


def test_routed_result_dictionary_serialization(clean_router):
    evt = _signature_attack_evt()
    res = clean_router.route(evt)
    d = res.to_dict()
    assert isinstance(d, dict)
    assert "stage_exited" in d
    assert "stage_name" in d
    assert "early_exited" in d
    assert "latency_ms" in d
    assert "confidence_trajectory" in d
    assert "uncertainty_trajectory" in d


def test_router_telemetry_and_stats(clean_router):
    evt_sig = _signature_attack_evt()
    evt_benign = _benign_network_evt()
    evt_ambig = _stealthy_anomaly_evt()

    clean_router.route(evt_sig)
    clean_router.route(evt_benign)
    clean_router.route(evt_ambig)

    stats = clean_router.get_stats()
    assert stats["total_routed"] == 3
    assert stats["early_exit_rate"] > 0.0
    assert "f1" in stats["exit_fractions"]
    assert "f2" in stats["exit_fractions"]
    assert "f3" in stats["exit_fractions"]
    assert "f4" in stats["exit_fractions"]
    assert stats["avg_latency_ms"] >= 0.0


def test_concurrent_routing_thread_safety(clean_router):
    import concurrent.futures

    events = [_benign_network_evt() if i % 2 == 0 else _signature_attack_evt() for i in range(20)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(clean_router.route, events))

    assert len(results) == 20
    assert all(r is not None for r in results)
    stats = clean_router.get_stats()
    assert stats["total_routed"] == 20

