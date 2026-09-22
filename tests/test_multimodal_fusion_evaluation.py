from __future__ import annotations
"""
AHRAS Verification Suite — Phase 14 / RQ10: Cross-Modal Representation & Multimodal Attention Fusion
-----------------------------------------------------------------------------------------------------
Validates:
  1. Multi-modal campaign dataset generation (dimensions, stages, feature integrity).
  2. ModalityMLP LayerNorm numerical scale stability across varying magnitudes.
  3. CrossModalAttention mathematical projection consistency for single vs multiple modalities.
  4. TemporalAttention sequence pooling and softmax weighting.
  5. Dynamic modality masking for partial telemetry availability (subsets of {net, proc, id, graph}).
  6. Flat feature extraction for early concatenation baseline.
  7. Paired permutation testing statistical rigor and bootstrap intervals.
  8. End-to-end multimodal fusion experiment execution and missingness degradation monotonicity.
  9. Synchronized claim verification for CLM-10 in CLAIMS_MANIFEST_FINAL.json.
"""

import os
import sys
import json
import pytest
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from detection.multimodal_encoder import (
    MultimodalSecurityEncoder,
    ModalityMLP,
    CrossModalAttention,
    TemporalAttention,
    ModalityVectors,
)
from evaluation.multimodal_fusion_experiment import (
    generate_multimodal_campaign_dataset,
    extract_flat_vector,
    paired_permutation_test,
    MultimodalFusionExperiment,
)


def test_multimodal_dataset_generation():
    """Verifies that synthetic multi-modal dataset generates exact schema and distributions."""
    events, labels = generate_multimodal_campaign_dataset(
        n_benign=200, n_campaigns=10, steps_per_campaign=10, seed=42
    )
    assert len(events) == 300
    assert len(labels) == 300
    assert sum(labels == 0) == 200
    assert sum(labels == 1) == 100

    required_keys = [
        "bytes_in", "bytes_out", "packet_count", "port_entropy", "duration",
        "cmd_length", "path_depth", "is_elevated", "cpu_pct",
        "privilege_level", "failed_auth_count", "session_age_sec",
        "in_degree", "out_degree", "neighbor_anomaly_mean", "local_clustering",
    ]
    for ev in events[:20]:
        for k in required_keys:
            assert k in ev, f"Missing key {k} in event"
            assert not np.isnan(float(ev[k])), f"NaN value for {k}"


def test_modality_mlp_layernorm():
    """Verifies that ModalityMLP LayerNorm bounds output representations regardless of input magnitude."""
    mlp = ModalityMLP(in_dim=4, hidden_dim=8, out_dim=16, seed=42)

    # 1. Normal scale inputs
    x_small = np.array([1.0, 2.0, 3.0, 4.0])
    out_small = mlp.forward(x_small)
    assert out_small.shape == (16,)
    assert abs(float(np.mean(out_small))) < 1e-4
    assert abs(float(np.std(out_small)) - 1.0) < 1e-3

    # 2. Huge scale inputs (e.g. unscaled bytes in tens of millions)
    x_huge = np.array([1e7, 5e6, 2e5, 1e4])
    out_huge = mlp.forward(x_huge)
    assert out_huge.shape == (16,)
    assert not np.isnan(out_huge).any()
    assert abs(float(np.mean(out_huge))) < 1e-4
    assert abs(float(np.std(out_huge)) - 1.0) < 1e-3


def test_cross_modal_attention_single_vs_multi_consistency():
    """Verifies that CrossModalAttention applies W_v @ W_o consistently on single and multi-modality inputs."""
    attn = CrossModalAttention(embed_dim=16, num_heads=2, seed=42)
    x = np.random.randn(1, 16)

    # Single modality pass
    fused_single, weights_single = attn.forward(x)
    assert fused_single.shape == (16,)
    assert weights_single.shape == (1, 1)
    assert weights_single[0, 0] == 1.0

    # Multi-modality pass with two identical copies
    stack_double = np.vstack([x, x])
    fused_double, weights_double = attn.forward(stack_double)
    assert fused_double.shape == (16,)
    assert weights_double.shape == (2, 2)

    # Due to symmetry, attention over identical rows must yield mathematically identical fused output
    assert np.allclose(fused_single, fused_double, atol=1e-5)


def test_temporal_attention_aggregation():
    """Verifies temporal sequence pooling and attention weight softmax property."""
    temp_attn = TemporalAttention(embed_dim=8, seed=42)
    seq = np.random.randn(5, 8)
    z_ep, alphas = temp_attn.forward(seq)

    assert z_ep.shape == (8,)
    assert alphas.shape == (5,)
    assert np.isclose(np.sum(alphas), 1.0, atol=1e-5)
    assert not np.isnan(z_ep).any()


def test_modality_masking_subsets():
    """Verifies that MultimodalSecurityEncoder handles arbitrary subsets of active modalities."""
    enc = MultimodalSecurityEncoder(embed_dim=8, seed=42)
    ev = {
        "bytes_in": 5000.0, "bytes_out": 2000.0, "packet_count": 25.0,
        "cmd_length": 30.0, "path_depth": 2.0, "privilege_level": 1.0,
        "in_degree": 2.0, "neighbor_anomaly_mean": 0.05,
    }

    # Full set
    v_full = enc.encode(ev)
    assert v_full.fused.shape == (8,)
    assert v_full.attention_matrix.shape == (4, 4)

    # 3 modalities
    v_3 = enc.encode(ev, active_modalities={"network", "process", "identity"})
    assert v_3.fused.shape == (8,)
    assert v_3.attention_matrix.shape == (3, 3)

    # 2 modalities
    v_2 = enc.encode(ev, active_modalities={"network", "identity"})
    assert v_2.fused.shape == (8,)
    assert v_2.attention_matrix.shape == (2, 2)

    # 1 modality
    v_1 = enc.encode(ev, active_modalities={"network"})
    assert v_1.fused.shape == (8,)
    assert v_1.attention_matrix.shape == (1, 1)


def test_flat_vector_extraction():
    """Verifies 18-dim numerical flat feature extraction."""
    ev = {
        "bytes_in": 5000.0, "bytes_out": 2000.0, "packet_count": 25.0,
        "pkts_out": 20.0, "port_entropy": 0.5, "duration": 2.0,
        "cmd_length": 30.0, "path_depth": 2.0, "is_elevated": True, "cpu_pct": 50.0,
        "privilege_level": 2.0, "failed_auth_count": 0.0, "session_age_sec": 300.0,
        "concurrent_logins": 1.0, "in_degree": 3.0, "out_degree": 3.0,
        "neighbor_anomaly_mean": 0.1, "local_clustering": 0.3,
    }
    vec = extract_flat_vector(ev)
    assert vec.shape == (18,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_paired_permutation_test_logic():
    """Verifies statistical correctness of paired permutation testing."""
    a = np.array([1.0] * 500)
    b = np.array([0.0] * 500)
    p_val, d, (ci_low, ci_high) = paired_permutation_test(a, b, n_permutations=1000, seed=42)
    assert p_val <= 0.001
    assert d > 0.0
    assert ci_low > 0.90


def test_multimodal_experiment_metrics_and_degradation():
    """Verifies that the multimodal fusion experiment satisfies F1 and graceful degradation requirements."""
    exp = MultimodalFusionExperiment(seed=42, embed_dim=16)
    report = exp.run_experiment()

    sm = report["summary_metrics"]
    assert sm["ahras_multimodal_f1"] >= 0.95, f"AHRAS F1 {sm['ahras_multimodal_f1']} below 0.95"
    assert sm["p99_fusion_latency_ms"] <= 0.50, f"P99 latency {sm['p99_fusion_latency_ms']} exceeds 0.50 ms"
    assert sm["permutation_p_vs_net"] <= 0.001, "Permutation p-value not significant"

    # Missingness Degradation monotonicity: 100% >= 75% >= 50% >= 25%
    miss = report["modality_missingness_degradation"]
    f1_100 = miss["100pct_all_modalities"]["f1"]
    f1_75 = miss["75pct_missing_graph"]["f1"]
    f1_50 = miss["50pct_missing_graph_process"]["f1"]
    f1_25 = miss["25pct_network_only"]["f1"]

    assert f1_100 >= f1_75, f"Expected f1_100 ({f1_100}) >= f1_75 ({f1_75})"
    assert f1_75 >= f1_50, f"Expected f1_75 ({f1_75}) >= f1_50 ({f1_50})"
    assert f1_50 >= f1_25, f"Expected f1_50 ({f1_50}) >= f1_25 ({f1_25})"
    assert f1_50 >= 0.70, f"Expected 50% missingness F1 ({f1_50}) >= 0.70"


def test_claims_manifest_sync_clm10():
    """Verifies that CLM-10 is present and consistent in both claims manifests."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    c1_path = os.path.join(root, "CLAIMS_MANIFEST_FINAL.json")
    c2_path = os.path.join(root, "publication", "CLAIMS_MANIFEST_FINAL.json")

    for p in (c1_path, c2_path):
        assert os.path.exists(p), f"Missing {p}"
        with open(p, "r", encoding="utf-8") as f:
            claims = json.load(f)
        assert "CLM-10" in claims, f"CLM-10 not found in {p}"
        c = claims["CLM-10"]
        assert c["status"] == "SUPPORTED"
        assert c["value"] >= 0.95
        assert c["missingness_50pct_f1"] >= 0.70
