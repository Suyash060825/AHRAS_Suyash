"""
AHRAS Unit & Integration Tests — Encrypted Session Intelligence (Extension G)
"""
import pytest
import numpy as np

from detection.encrypted_session import (
    EncryptedSessionIntelligence,
    PacketMetadata,
    SessionMetadata,
    SessionEvidenceRecord,
    get_encrypted_session_intel,
    compute_autocorrelation,
    compute_shannon_entropy,
)


@pytest.fixture
def intel():
    return EncryptedSessionIntelligence(sequence_len=32, periodicity_threshold=0.75)


def _make_c2_beacon_packets(n=20, interval=5.0):
    """Generates synthetic periodic C2 heartbeat beaconing telemetry."""
    pkts = []
    t = 1000.0
    for i in range(n):
        # Client poll (outbound, small ~120B)
        pkts.append(PacketMetadata(size=120, direction=1, timestamp=t))
        # Server response (inbound, small ~200B)
        pkts.append(PacketMetadata(size=200, direction=-1, timestamp=t + 0.05))
        # Fixed interval heartbeat with tiny 0.01s jitter
        t += interval + np.random.uniform(-0.02, 0.02)
    return pkts


def _make_exfiltration_packets(n=30):
    """Generates synthetic bulk encrypted data exfiltration telemetry."""
    pkts = []
    t = 1000.0
    for i in range(n):
        # Heavy MTU-sized data uploads
        pkts.append(PacketMetadata(size=1460, direction=1, timestamp=t))
        if i % 5 == 0:
            # Sparse TCP ACKs from server
            pkts.append(PacketMetadata(size=54, direction=-1, timestamp=t + 0.01))
        t += 0.005
    return pkts


def _make_interactive_shell_packets(n=25):
    """Generates keystroke interactive reverse-shell telemetry."""
    pkts = []
    t = 1000.0
    for i in range(n):
        # Small keystroke packet (outbound, 80B)
        pkts.append(PacketMetadata(size=80, direction=1, timestamp=t))
        # Immediate terminal echo response (inbound, 95B)
        pkts.append(PacketMetadata(size=95, direction=-1, timestamp=t + 0.08))
        # Human inter-keystroke pause (0.2s - 0.8s)
        t += np.random.uniform(0.2, 0.8)
    return pkts


def _make_normal_https_packets(n=40):
    """Generates typical web browsing traffic (mixed sizes, variable bursty IATs)."""
    pkts = []
    t = 1000.0
    for i in range(n):
        size = int(np.random.choice([150, 450, 800, 1420, 60]))
        direction = 1 if np.random.random() > 0.65 else -1
        pkts.append(PacketMetadata(size=size, direction=direction, timestamp=t))
        t += np.random.exponential(scale=0.15)
    return pkts


# ── Tests ────────────────────────────────────────────────────────────────────

def test_autocorrelation_periodicity():
    # Periodic series: delta = 5.0s exactly
    periodic_iats = np.array([5.0, 5.01, 4.99, 5.0, 5.02, 4.98, 5.0, 5.01, 5.0])
    rho_periodic = compute_autocorrelation(periodic_iats)
    assert rho_periodic >= 0.75

    # Random noisy series: exponential intervals
    rng = np.random.default_rng(42)
    random_iats = rng.exponential(scale=2.0, size=50)
    rho_random = compute_autocorrelation(random_iats)
    assert rho_random < 0.50


def test_shannon_entropy():
    # Constant string -> 0 entropy
    assert compute_shannon_entropy("aaaaaaa") == 0.0
    # High entropy random hex
    assert compute_shannon_entropy("a8f3b9c1d4e7") > 3.0


def test_sequence_representation(intel):
    pkts = _make_c2_beacon_packets(n=10)
    seq = intel.extract_sequence(pkts)
    assert seq.shape == (32, 3)
    # Norm sizes in [0, 1]
    assert np.all((seq[:, 0] >= 0.0) & (seq[:, 0] <= 1.0))
    # Directions in {-1, 0, 1}
    assert np.all(np.isin(seq[:, 1], [-1.0, 0.0, 1.0]))


def test_session_vector_representation(intel):
    pkts = _make_normal_https_packets(n=25)
    meta = SessionMetadata(sni="www.wikipedia.org", alpn="h2")
    vec = intel.extract_session_vector(pkts, meta)
    assert vec.shape == (24,)
    assert not np.any(np.isnan(vec))
    assert not np.any(np.isinf(vec))


def test_c2_beaconing_detection(intel):
    pkts = _make_c2_beacon_packets(n=15, interval=4.0)
    rec = intel.analyze_session(
        event_id="evt-c2-01",
        entity_id="192.168.1.55",
        packets=pkts,
        meta=SessionMetadata(sni="evil-c2.duckdns.org"),
    )
    assert isinstance(rec, SessionEvidenceRecord)
    assert rec.threat_label == "C2_BEACONING"
    assert rec.mitre_technique == "T1071.001"
    assert rec.normalized_score >= 0.75
    assert rec.beacon_periodicity >= 0.75
    assert rec.confidence >= 0.90


def test_data_exfiltration_detection(intel):
    pkts = _make_exfiltration_packets(n=35)
    rec = intel.analyze_session(
        event_id="evt-exfil-01",
        entity_id="192.168.1.100",
        packets=pkts,
    )
    assert rec.threat_label == "DATA_EXFILTRATION"
    assert rec.mitre_technique == "T1041"
    assert rec.normalized_score >= 0.80
    assert rec.metadata["outbound_ratio"] >= 0.80


def test_interactive_shell_detection(intel):
    pkts = _make_interactive_shell_packets(n=20)
    rec = intel.analyze_session(
        event_id="evt-shell-01",
        entity_id="192.168.1.42",
        packets=pkts,
    )
    assert rec.threat_label == "INTERACTIVE_SHELL"
    assert rec.mitre_technique == "T1059"
    assert rec.normalized_score >= 0.65


def test_benign_https_classification(intel):
    pkts = _make_normal_https_packets(n=30)
    rec = intel.analyze_session(
        event_id="evt-benign-01",
        entity_id="192.168.1.20",
        packets=pkts,
        meta=SessionMetadata(sni="cdn.cloudflare.net", alpn="h2"),
    )
    assert rec.threat_label == "BENIGN"
    assert rec.normalized_score <= 0.20
    assert rec.mitre_technique is None


def test_empty_session_handling(intel):
    rec = intel.analyze_session("evt-empty", "10.0.0.1", [])
    assert rec.threat_label == "BENIGN"
    assert rec.raw_score == 0.0
    assert rec.metadata["reason"] == "empty_session"
