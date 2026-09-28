from __future__ import annotations
"""
AHRAS Unit Tests — Phase 10: Energy Efficiency, Model Compression, Edge Profiles & Privacy Research
-----------------------------------------------------------------------------------------------------
Verifies:
  1. Energy-Aware Security Profiling (Section 52):
     - Empirical CPU latency and thermodynamic energy estimation (microjoules/event)
     - Security-Per-Watt (SPW) calculation
     - Resource-aware vs Monolithic savings
  2. Model Compression Suite (Section 53):
     - INT8 Quantization (scale/zero-point) and memory reduction
     - Magnitude weight pruning with sparsity masks
     - Distilled student model with soft targets and fallback to teacher
  3. Edge Deployment Profiles (Section 54):
     - CENTRAL, EDGE, ENDPOINT, HYBRID architecture parameter validation
     - Throughput and latency envelope bounds
  4. Federated Privacy-Utility Research (Section 51):
     - Differential Privacy Gaussian mechanism under varying epsilon
     - Communication compression and rare-attack trade-off frontier
  5. REST API endpoints for Phase 10
"""

import math
import pytest
import numpy as np
from fastapi.testclient import TestClient

from performance.energy_profiler import EnergyProfiler, get_energy_profiler
from models.compression import (
    QuantizedLinearLayer,
    PrunedLinearLayer,
    DistilledStudentModel,
    ModelCompressor,
)
from deployment.edge_profiles import (
    DeploymentTier,
    DeploymentProfileManager,
    get_deployment_profile_manager,
)
from federated.privacy_utility import (
    FederatedPrivacyUtilityResearcher,
    get_federated_privacy_researcher,
)
from api.server import app


@pytest.fixture
def profiler():
    return EnergyProfiler(tdp_watts=65.0, idle_watts=12.0)


@pytest.fixture
def client():
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Energy-Aware Security Profiling Tests (Section 52)
# ─────────────────────────────────────────────────────────────────────────────

def test_energy_profiler_measurement(profiler):
    samples = list(range(200))
    rec = profiler.measure_tier_energy(
        tier_name="TEST_SKETCH",
        execution_fn=lambda x: x * 3 + 1,
        samples=samples,
        detection_f1=0.98,
    )
    assert rec.event_count == 200
    assert rec.execution_time_sec > 0.0
    assert rec.microjoules_per_event > 0.0
    assert rec.security_per_watt > 0.0


def test_monolithic_vs_resource_aware_comparison(profiler):
    workload = list(range(100))
    # Monolithic does complex math; resource aware does early exit
    comp = profiler.compare_monolithic_vs_resource_aware(
        monolithic_fn=lambda x: sum(math.sin(x + i) for i in range(50)),
        resource_aware_fn=lambda x: x if x < 50 else sum(math.sin(x + i) for i in range(10)),
        workload=workload,
        f1_monolithic=0.985,
        f1_resource_aware=0.980,
    )
    assert comp["energy_saved_percent"] > 0.0
    assert comp["latency_speedup_ratio"] > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Compression Suite Tests (Section 53)
# ─────────────────────────────────────────────────────────────────────────────

def test_quantized_int8_linear_layer():
    rng = np.random.default_rng(42)
    W = rng.normal(0.0, 0.5, size=(16, 8)).astype(np.float32)
    b = rng.normal(0.0, 0.1, size=(8,)).astype(np.float32)
    X = rng.normal(0.0, 1.0, size=(10, 16)).astype(np.float32)

    q_layer = QuantizedLinearLayer(W, b)
    assert q_layer.weights_int8.dtype == np.int8
    assert q_layer.byte_size < (W.nbytes + b.nbytes)

    out = q_layer.forward(X)
    assert out.shape == (10, 8)
    assert np.all(np.isfinite(out))


def test_pruned_sparse_linear_layer():
    rng = np.random.default_rng(42)
    W = rng.normal(0.0, 0.5, size=(20, 10)).astype(np.float32)
    b = np.zeros(10, dtype=np.float32)
    X = rng.normal(0.0, 1.0, size=(5, 20)).astype(np.float32)

    pruned = PrunedLinearLayer(W, b, sparsity_percentile=60.0)
    # Approx 60% should be zeroed
    zeros = np.sum(pruned.pruned_weights == 0.0)
    assert zeros >= 100

    out = pruned.forward(X)
    assert out.shape == (5, 10)


def test_distilled_student_with_fallback():
    class DummyTeacher:
        def predict(self, X):
            return np.ones(len(X)) * 0.99

    teacher = DummyTeacher()
    student = DistilledStudentModel(in_dim=8, hidden_dim=4, seed=42, fallback_teacher=teacher)

    rng = np.random.default_rng(42)
    X = rng.normal(0.5, 0.2, size=(20, 8))
    y = np.array([0] * 10 + [1] * 10)
    teacher_probs = np.array([0.1] * 10 + [0.9] * 10)

    loss = student.train_distillation(X, y, teacher_probs, epochs=15)
    assert isinstance(loss, float)

    # Test prediction near boundary triggering fallback
    # We pass an artificial input that produces uncertainty
    x_boundary = np.zeros(8)
    score, used_fallback = student.predict_with_fallback(x_boundary, uncertainty_thresh=0.01)
    assert used_fallback is True
    assert score == 0.99


def test_model_compression_benchmark_suite():
    rng = np.random.default_rng(42)
    X_train = rng.normal(0.5, 0.2, size=(60, 8))
    y_train = np.array([0] * 30 + [1] * 30)
    X_test = rng.normal(0.5, 0.2, size=(20, 8))
    y_test = np.array([0] * 10 + [1] * 10)
    ood_test = rng.normal(1.5, 0.2, size=(10, 8))

    suite = ModelCompressor.benchmark_compression_suite(X_train, y_train, X_test, y_test, ood_test)
    assert "ORIGINAL_FP32" in suite
    assert "QUANTIZED_INT8" in suite
    assert "PRUNED_SPARSE" in suite
    assert "DISTILLED_STUDENT" in suite

    assert suite["QUANTIZED_INT8"].memory_reduction_pct > 0.0
    assert suite["DISTILLED_STUDENT"].fallback_available is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Edge Deployment Profiles Tests (Section 54)
# ─────────────────────────────────────────────────────────────────────────────

def test_deployment_profile_manager():
    mgr = DeploymentProfileManager()
    profiles = mgr.benchmark_deployment_profiles()

    assert "CENTRAL" in profiles
    assert "EDGE" in profiles
    assert "ENDPOINT" in profiles
    assert "HYBRID" in profiles

    # CENTRAL has maximum memory allocation
    assert profiles["CENTRAL"]["memory_limit_mb"] >= profiles["EDGE"]["memory_limit_mb"]
    assert profiles["EDGE"]["memory_limit_mb"] >= profiles["ENDPOINT"]["memory_limit_mb"]

    # ENDPOINT has lowest latency budget
    assert profiles["ENDPOINT"]["max_latency_ms"] < profiles["EDGE"]["max_latency_ms"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Federated Privacy-Utility Research Tests (Section 51)
# ─────────────────────────────────────────────────────────────────────────────

def test_federated_privacy_gaussian_noise():
    researcher = FederatedPrivacyUtilityResearcher(delta=1e-5, clip_norm=1.0, seed=42)
    weights = np.ones(50)
    rng = np.random.default_rng(42)

    # Infinite epsilon -> No noise added
    clean = researcher.apply_differential_privacy(weights, epsilon=math.inf, rng=rng)
    assert np.allclose(clean, weights)

    # Finite epsilon -> Calibrated Gaussian noise added
    noisy = researcher.apply_differential_privacy(weights, epsilon=1.0, rng=rng)
    assert not np.allclose(noisy, weights)
    assert np.all(np.isfinite(noisy))


def test_federated_privacy_utility_frontier():
    researcher = FederatedPrivacyUtilityResearcher()
    frontier = researcher.evaluate_privacy_utility_frontier(epsilons=[1.0, 5.0, math.inf])

    assert len(frontier) == 3
    # More privacy (lower epsilon) should exhibit lower rare-attack recall or higher sigma
    assert frontier[0].sigma_noise > frontier[1].sigma_noise
    assert frontier[2].sigma_noise == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. REST API Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_api_phase10_endpoints(client):
    res_en = client.get("/api/performance/energy-profile")
    assert res_en.status_code == 200
    assert "microjoules_per_event" in res_en.json()

    res_dep = client.get("/api/deployment/profiles")
    assert res_dep.status_code == 200
    assert "CENTRAL" in res_dep.json()
    assert "EDGE" in res_dep.json()

    res_fed = client.get("/api/federated/privacy-utility")
    assert res_fed.status_code == 200
    assert len(res_fed.json()["frontier"]) >= 1
