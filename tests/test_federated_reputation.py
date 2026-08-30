import pytest
import numpy as np
from federated.fed_learning import (
    ClientReputationTracker,
    FederatedKnowledgeDistiller,
    FederatedIDSServer,
    ModelUpdate
)

def test_client_reputation_decay_and_update():
    tracker = ClientReputationTracker(alpha=0.50, default_rep=0.80)
    
    # Good update
    rep_good = tracker.update_reputation("client-1", update_valid=True, local_loss=0.05)
    assert rep_good > 0.80
    
    # Poisoned / invalid update
    rep_bad = tracker.update_reputation("client-2", update_valid=False, local_loss=1.0, is_byzantine=True)
    assert rep_bad < 0.50

def test_federated_distiller():
    distiller = FederatedKnowledgeDistiller(temperature=2.0)
    l1 = np.array([0.8, 0.2])
    l2 = np.array([0.6, 0.4])
    
    consensus = distiller.distill_consensus([l1, l2], reputations=[0.9, 0.1])
    assert consensus.shape == (2,)
    # l1 had 90% weight, so consensus should be closer to l1
    assert consensus[0] > 0.75

def test_reputation_weighted_fed_server():
    server = FederatedIDSServer(min_clients=2, enable_robust_aggregation=False, enable_reputation_weighting=True)
    
    # Client 1: good update
    u1 = ModelUpdate(
        client_id="c1",
        num_samples=100,
        weights={"W": np.ones((2, 2))},
        local_loss=0.02,
        timestamp=0.0
    )
    # Client 2: higher loss update
    u2 = ModelUpdate(
        client_id="c2",
        num_samples=100,
        weights={"W": np.zeros((2, 2))},
        local_loss=0.40,
        timestamp=0.0
    )
    
    server.receive_update(u1)
    server.receive_update(u2)
    
    global_w = server.aggregate_round()
    assert "W" in global_w
    # c1 has higher reputation, so global W should be > 0.5
    assert np.mean(global_w["W"]) > 0.50
