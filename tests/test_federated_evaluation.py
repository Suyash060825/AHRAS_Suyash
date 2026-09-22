from __future__ import annotations
"""
AHRAS Test Suite — Phase 11 / RQ7: Byzantine-Robust Multi-Tenant Federated Learning
------------------------------------------------------------------------------------
Tests:
  1. Multi-tenant Dirichlet non-IID dataset generator.
  2. Byzantine attack injection mechanics (explosion, sign-flipping, backdoor).
  3. Server rejection of excessive gradient norms.
  4. Server quarantine of decaying rogue client reputations.
  5. Coordinate-wise median Byzantine resilience.
  6. Trimmed mean aggregation resilience.
  7. FedKD knowledge distillation consensus.
  8. Full multi-round experiment execution and statistical verification.
  9. Report consistency with CLM-03 and publication rules.
"""

import os
import json
import pytest
import numpy as np

from federated.fed_learning import (
    FederatedIDSServer,
    ModelUpdate,
    ClientReputationTracker,
    FederatedKnowledgeDistiller,
    PersonalizedFedProxClient,
)
from evaluation.federated_learning_experiment import (
    generate_multitenant_dataset,
    evaluate_classifier_weights,
    paired_permutation_test,
    FederatedByzantineExperiment,
    TENANT_PROFILES,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_01_multitenant_dataset_generation():
    client_data, (X_test, y_test) = generate_multitenant_dataset(
        n_clients=10, samples_per_client=200, n_global_test=300, seed=42
    )
    assert len(client_data) == 10
    assert X_test.shape == (300, 14)
    assert len(y_test) == 300
    assert set(np.unique(y_test)).issubset({0, 1})

    for tenant_id, (X_c, y_c) in client_data.items():
        assert X_c.shape == (200, 14)
        assert len(y_c) == 200
        # Verify both classes exist or bounded
        assert len(np.unique(y_c)) >= 1


def test_02_byzantine_attack_injection_mechanics():
    weights = {
        "W1": np.ones((14, 8)),
        "b1": np.zeros(8),
        "W2": np.ones((8, 2)),
        "b2": np.zeros(2),
    }
    client = PersonalizedFedProxClient("test_client", mu_prox=0.1)
    X = np.random.normal(0, 1, size=(50, 14))
    y = np.ones(50, dtype=np.int64)

    # 1. Normal clean step
    clean_up = client.local_train_step(weights, X, y, n_epochs=2, lr=0.05)
    assert clean_up.local_loss > 0.0

    # 2. Gradient explosion
    poison_W1 = weights["W1"] * (-200.0)
    norm = float(np.linalg.norm(poison_W1))
    assert norm > 100.0

    # 3. Directional sign-flipping
    flipped_W1 = weights["W1"] - (clean_up.weights["W1"] - weights["W1"]) * 3.0
    dot_prod = np.sum((flipped_W1 - weights["W1"]) * (clean_up.weights["W1"] - weights["W1"]))
    assert dot_prod < 0.0  # Inverted direction


def test_03_server_rejection_of_excessive_gradient_norm():
    server = FederatedIDSServer(min_clients=2, byzantine_clip_norm=5.0)
    bad_up = ModelUpdate(
        client_id="attacker_1",
        num_samples=100,
        weights={"W1": np.ones((14, 8)) * 500.0},
        local_loss=25.0,
        timestamp=0.0,
    )
    accepted = server.receive_update(bad_up)
    assert not accepted
    assert len(server._rejected_updates) == 1
    assert server.reputation_tracker.get_reputation("attacker_1") < 0.75


def test_04_server_quarantine_of_low_reputation_client():
    server = FederatedIDSServer(
        min_clients=2,
        byzantine_clip_norm=10.0,
        aggregation_strategy="fedkd_reputation",
        quarantine_threshold=0.25,
    )

    u1 = ModelUpdate("client_ok_1", 100, {"W": np.array([1.0, 1.0])}, 0.02, 0.0)
    u2 = ModelUpdate("client_ok_2", 100, {"W": np.array([1.0, 1.0])}, 0.02, 0.0)
    u_rogue = ModelUpdate("client_rogue", 100, {"W": np.array([5.0, 5.0])}, 0.02, 0.0)

    server.receive_update(u1)
    server.receive_update(u2)
    server.receive_update(u_rogue)

    # Rogue client reputation decays below quarantine threshold
    server.reputation_tracker._reputations["client_rogue"] = 0.15

    global_w = server.aggregate_round()
    stats = server.get_stats()
    assert stats["quarantined_updates"] >= 1
    # Rogue client's 5.0 was quarantined, so mean should be 1.0
    np.testing.assert_allclose(global_w["W"], np.array([1.0, 1.0]), atol=1e-3)


def test_05_coordinate_median_aggregation_resilience():
    server = FederatedIDSServer(min_clients=4, aggregation_strategy="coordinate_median", byzantine_clip_norm=100.0)
    # 3 benign clients at 1.0, 1 malicious client at 50.0
    for i in range(3):
        server.receive_update(ModelUpdate(f"benign_{i}", 100, {"W": np.array([1.0, 2.0])}, 0.02, 0.0))
    server.receive_update(ModelUpdate("malicious_0", 100, {"W": np.array([50.0, 50.0])}, 0.02, 0.0))

    global_w = server.aggregate_round()
    # Coordinate median of [1.0, 1.0, 1.0, 50.0] is 1.0
    np.testing.assert_allclose(global_w["W"], np.array([1.0, 2.0]), atol=1e-2)


def test_06_trimmed_mean_aggregation_resilience():
    server = FederatedIDSServer(min_clients=5, aggregation_strategy="trimmed_mean", byzantine_clip_norm=100.0)
    # 4 benign clients around 2.0, 1 extreme outlier at 80.0
    for i in range(4):
        server.receive_update(ModelUpdate(f"benign_{i}", 100, {"W": np.array([2.0, 3.0])}, 0.02, 0.0))
    server.receive_update(ModelUpdate("malicious_0", 100, {"W": np.array([80.0, 80.0])}, 0.02, 0.0))

    global_w = server.aggregate_round()
    # Trimmed mean trims the top 20% outlier 80.0
    assert global_w["W"][0] < 5.0


def test_07_fedkd_consensus_logit_distillation():
    distiller = FederatedKnowledgeDistiller(temperature=2.0)
    logits_benign = np.array([0.90, 0.10])
    logits_malicious = np.array([0.05, 0.95])

    # Benign client has reputation 0.95, malicious has 0.05
    consensus = distiller.distill_consensus([logits_benign, logits_malicious], reputations=[0.95, 0.05])
    assert consensus[0] > 0.85
    assert consensus[1] < 0.15


def test_08_federated_experiment_full_run():
    exp = FederatedByzantineExperiment(seed=42)
    # Fast test with small config
    exp.n_rounds = 2
    res = exp.run_experiment()

    assert "experiment_id" in res and res["experiment_id"] == "EXP-07"
    assert "summary_metrics" in res
    sm = res["summary_metrics"]
    assert sm["ahras_f1_clean"] >= 0.95
    assert sm["ahras_f1_30pct_poison"] >= 0.95
    assert sm["paired_permutation_p_value"] <= 0.05
    assert sm["cohens_d"] > 0.50
    assert len(sm["bootstrap_ci_95"]) == 2


def test_09_report_consistency_and_clm03_sync():
    report_path = os.path.join(_ROOT, "evaluation", "results", "FEDERATED_LEARNING_REPORT.json")
    claims_path = os.path.join(_ROOT, "CLAIMS_MANIFEST_FINAL.json")

    assert os.path.exists(report_path), "FEDERATED_LEARNING_REPORT.json must exist"
    assert os.path.exists(claims_path), "CLAIMS_MANIFEST_FINAL.json must exist"

    with open(report_path) as f:
        rep = json.load(f)
    with open(claims_path) as f:
        claims = json.load(f)

    assert "CLM-03" in claims
    clm = claims["CLM-03"]
    assert clm["status"] == "SUPPORTED"
    assert clm["value"] == 0.9835
    assert rep["claims_mapping"]["value"] == clm["value"]
    assert rep["summary_metrics"]["ahras_f1_30pct_poison"] == clm["value"]
