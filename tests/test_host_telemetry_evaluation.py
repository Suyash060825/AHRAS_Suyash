from __future__ import annotations
"""
AHRAS Verification Suite — Phase 13 / RQ9: Two-Tier Host Telemetry & Endpoint Collection
-----------------------------------------------------------------------------------------
Validates:
  1. Shannon entropy calculation across flat, text, and pseudo-random byte buffers.
  2. ProcessLineageTracker rule-based LOLBin detection and deep subshell heuristic.
  3. Process tree ancestor traversal and history bounding.
  4. TwoTierTelemetryAdapter OCSF event normalization (Classes 1001, 1002, 1003).
  5. Conditional hashing: zero SHA-256 computation on low-entropy benign files.
  6. Empirical benchmark run correctness and CPU overhead bound (<= 3.0% CPU).
  7. Paired permutation testing statistical rigor.
  8. Synchronized claim verification for CLM-09 in CLAIMS_MANIFEST_FINAL.json.
"""

import os
import sys
import json
import pytest
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sensors.two_tier_telemetry import (
    TwoTierTelemetryAdapter,
    compute_block_shannon_entropy,
    ProcessLineageTracker,
    RANSOMWARE_ENTROPY_THRESHOLD,
    SUSPICIOUS_LINEAGE_RULES,
)
from evaluation.host_telemetry_experiment import (
    generate_endpoint_event_suite,
    paired_permutation_test,
    HostTelemetryExperiment,
)


def test_shannon_entropy_calculation():
    """Verifies theoretical bounds and accuracy of Shannon entropy calculation."""
    # 1. Empty buffer -> 0.0
    assert compute_block_shannon_entropy(b"") == 0.0

    # 2. Monotonous buffer -> 0.0
    assert compute_block_shannon_entropy(b"A" * 1000) == 0.0

    # 3. Low-entropy plain text -> between 2.5 and 5.0
    sample_text = b"The quick brown fox jumps over the lazy dog. Repetitive structured log text." * 20
    text_entropy = compute_block_shannon_entropy(sample_text)
    assert 2.5 <= text_entropy <= 5.0

    # 4. Uniform pseudo-random bytes -> close to 8.0 bits/byte
    rng = np.random.default_rng(42)
    rand_bytes = bytes(rng.integers(0, 256, size=4096, dtype=np.uint8))
    rand_entropy = compute_block_shannon_entropy(rand_bytes)
    assert rand_entropy >= 7.80, f"Expected near 8.0 entropy for uniform random bytes, got {rand_entropy}"


def test_process_lineage_tracker_benign():
    """Verifies benign process hierarchies are not falsely flagged."""
    tracker = ProcessLineageTracker(max_history=100)
    is_susp, threat, depth = tracker.register_process(
        pid=100, name="systemd", cmdline="/sbin/init", parent_pid=None
    )
    assert not is_susp
    assert threat is None
    assert depth == 1

    is_susp, threat, depth = tracker.register_process(
        pid=101, name="sshd", cmdline="/usr/sbin/sshd", parent_pid=100
    )
    assert not is_susp
    assert threat is None
    assert depth == 2


def test_process_lineage_tracker_lolbin_rules():
    """Verifies known MITRE ATT&CK suspicious lineages are detected."""
    tracker = ProcessLineageTracker()

    # Word spawning cmd.exe (Office Macro exploit)
    tracker.register_process(pid=500, name="winword.exe", cmdline="winword.exe invoice.docx", parent_pid=1)
    is_susp, threat, depth = tracker.register_process(
        pid=501, name="cmd.exe", cmdline="cmd.exe /c powershell...", parent_pid=500
    )
    assert is_susp
    assert "Office macro" in threat

    # Web server spawning bash (Web shell / reverse shell)
    tracker.register_process(pid=600, name="httpd", cmdline="/usr/sbin/httpd", parent_pid=1)
    is_susp, threat, depth = tracker.register_process(
        pid=601, name="bash", cmdline="/bin/bash -i", parent_pid=600
    )
    assert is_susp
    assert "reverse shell" in threat.lower()

    # Powershell calling vssadmin (Shadow copy deletion)
    tracker.register_process(pid=700, name="powershell.exe", cmdline="powershell.exe", parent_pid=1)
    is_susp, threat, depth = tracker.register_process(
        pid=701, name="vssadmin.exe", cmdline="vssadmin delete shadows /all", parent_pid=700
    )
    assert is_susp
    assert "shadow copy" in threat.lower()


def test_process_lineage_deep_subshell_and_ancestors():
    """Verifies ancestor chain resolution and deep subshell anomaly detection."""
    tracker = ProcessLineageTracker()
    tracker.register_process(pid=10, name="init", cmdline="init", parent_pid=None)
    tracker.register_process(pid=20, name="service", cmdline="service", parent_pid=10)
    tracker.register_process(pid=30, name="helper", cmdline="helper", parent_pid=20)
    # Depth 4 subshell
    is_susp, threat, depth = tracker.register_process(
        pid=40, name="bash", cmdline="/bin/bash", parent_pid=30
    )
    assert is_susp
    assert depth == 4
    assert "Abnormal deep subshell" in threat

    chain = tracker.get_ancestor_chain(40)
    assert len(chain) == 4
    assert "bash(40)" in chain[0]
    assert "init(10)" in chain[-1]


def test_process_lineage_pruning():
    """Verifies that the lineage tracker enforces max_history without memory leaks."""
    tracker = ProcessLineageTracker(max_history=50)
    for i in range(120):
        tracker.register_process(pid=1000 + i, name=f"proc_{i}", cmdline="", parent_pid=None)
    assert len(tracker._tree) <= 50


def test_two_tier_telemetry_conditional_hashing():
    """Verifies conditional hashing: low-entropy files skip SHA-256; high-entropy files compute SHA-256."""
    adapter = TwoTierTelemetryAdapter(entropy_threshold=7.20)

    # Low entropy file write
    benign_ev = {
        "event_type": "file_write",
        "filepath": "/var/log/syslog.txt",
        "content_sample": b"0123456789" * 100, # repetitive digits, entropy < 4
    }
    norm_benign = adapter.process_raw_event(benign_ev)
    assert norm_benign["class_uid"] == 1003
    assert norm_benign["file"]["entropy"] < 7.20
    assert norm_benign["file"]["is_encrypted_risk"] is False
    assert norm_benign["file"]["sha256"] == ""  # Zero-cost: no hash computed
    assert adapter.high_entropy_events == 0

    # High entropy file write (ransomware encryption simulation)
    rng = np.random.default_rng(123)
    rand_data = bytes(rng.integers(0, 256, size=2048, dtype=np.uint8))
    ransom_ev = {
        "event_type": "file_write",
        "filepath": "/home/user/vault.dat.enc",
        "content_sample": rand_data,
    }
    norm_ransom = adapter.process_raw_event(ransom_ev)
    assert norm_ransom["class_uid"] == 1003
    assert norm_ransom["file"]["entropy"] >= 7.20
    assert norm_ransom["file"]["is_encrypted_risk"] is True
    assert len(norm_ransom["file"]["sha256"]) == 64  # SHA-256 computed on demand
    assert adapter.high_entropy_events == 1


def test_two_tier_telemetry_network_event():
    """Verifies OCSF Class 1001 mapping for network connection events."""
    adapter = TwoTierTelemetryAdapter()
    net_ev = {
        "event_type": "network_conn",
        "local_ip": "192.168.1.50",
        "local_port": 54321,
        "remote_ip": "104.244.42.1",
        "remote_port": 443,
        "protocol": "TCP",
        "pid": 1234,
        "process_name": "curl",
    }
    norm = adapter.process_raw_event(net_ev)
    assert norm["class_uid"] == 1001
    assert norm["class_name"] == "network_activity"
    assert norm["connection"]["src_ip"] == "192.168.1.50"
    assert norm["connection"]["dst_port"] == 443


def test_paired_permutation_test_logic():
    """Verifies paired permutation test correctly detects significant differences."""
    a = np.ones(500)
    b = np.zeros(500)
    p_val, d, (ci_low, ci_high) = paired_permutation_test(a, b, n_permutations=1000, seed=42)
    assert p_val <= 0.001
    assert ci_low > 0.90


def test_host_telemetry_experiment_metrics():
    """Verifies end-to-end host telemetry experiment satisfies performance and safety criteria."""
    exp = HostTelemetryExperiment(seed=42)
    report = exp.run_experiment()

    sm = report["summary_metrics"]
    assert sm["ahras_cpu_overhead_pct"] <= 3.0, f"CPU overhead {sm['ahras_cpu_overhead_pct']} exceeds 3.0%"
    assert sm["ransomware_detection_recall"] >= 0.98, "Ransomware recall below 98%"
    assert sm["ransomware_detection_f1"] >= 0.95, "Ransomware F1 below 0.95"
    assert sm["lolbin_lineage_detection_recall"] >= 0.95, "LOLBin recall below 95%"
    assert sm["latency_p99_ms"] <= 0.50, f"P99 latency {sm['latency_p99_ms']} exceeds 0.50 ms"
    assert sm["paired_permutation_p_value"] <= 0.001, "Permutation p-value not significant"


def test_claims_manifest_sync():
    """Verifies that CLM-09 is present and consistent in both claims manifests."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    c1_path = os.path.join(root, "CLAIMS_MANIFEST_FINAL.json")
    c2_path = os.path.join(root, "publication", "CLAIMS_MANIFEST_FINAL.json")

    for p in (c1_path, c2_path):
        assert os.path.exists(p), f"Missing {p}"
        with open(p, "r", encoding="utf-8") as f:
            claims = json.load(f)
        assert "CLM-09" in claims, f"CLM-09 not found in {p}"
        c = claims["CLM-09"]
        assert c["status"] == "SUPPORTED"
        assert c["value"] <= 3.0
        assert c["ransomware_recall"] >= 0.98
        assert c["lolbin_lineage_recall"] >= 0.95
