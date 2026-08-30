import pytest
import numpy as np
from detection.attack_path import AttackPathReasoner, AttackPath

def test_noisy_or_aggregation():
    reasoner = AttackPathReasoner()
    
    # Empty path -> 0.0
    assert reasoner.score_path_noisy_or([]) == 0.0
    
    # Single node
    assert reasoner.score_path_noisy_or([0.4]) == 0.40
    
    # Two independent nodes with 0.5 risk: 1 - (1-0.5)*(1-0.5) = 0.75
    assert pytest.approx(reasoner.score_path_noisy_or([0.5, 0.5]), 0.01) == 0.75
    
    # Monotonicity test: adding a hop cannot decrease path risk
    r1 = reasoner.score_path_noisy_or([0.3, 0.4])
    r2 = reasoner.score_path_noisy_or([0.3, 0.4, 0.5])
    assert r2 >= r1

def test_evaluate_path():
    reasoner = AttackPathReasoner()
    risks = {"host-A": 0.30, "host-B": 0.80, "host-C": 0.50}
    path = reasoner.evaluate_path(["host-A", "host-B", "host-C"], risks)
    
    assert isinstance(path, AttackPath)
    assert path.hop_count == 2
    assert path.critical_node == "host-B"
    assert path.path_risk > 0.80

def test_score_episode():
    reasoner = AttackPathReasoner(embed_dim=8)
    embeds = {
        "host-A": np.random.randn(8),
        "host-B": np.random.randn(8),
    }
    risks = {"host-A": 0.60, "host-B": 0.70}
    ep_score = reasoner.score_episode(["host-A", "host-B"], embeds, risks)
    assert 0.0 <= ep_score <= 1.0
