import pytest
from adaptive_learning.active_learner import ActiveLearner, ActiveLearningRequest

def test_active_learner_acquisition_and_budget():
    learner = ActiveLearner(budget_per_window=3, window_sec=3600.0)
    
    # 1. High uncertainty qualifies for query
    should_q = learner.should_query(uncertainty=0.55, ood_score=0.70, abstain_action="ABSTAIN")
    assert should_q is True
    
    # 2. Create requests
    req1 = learner.create_request("EVT-1", "HOST-1", uncertainty=0.6, ood_score=0.7, risk_score=0.55)
    req2 = learner.create_request("EVT-2", "HOST-2", uncertainty=0.5, ood_score=0.6, risk_score=0.45)
    req3 = learner.create_request("EVT-3", "HOST-3", uncertainty=0.7, ood_score=0.8, risk_score=0.60)
    
    assert len(learner.get_pending()) == 3
    
    # 4th request should fail query due to budget = 3
    should_q_4 = learner.should_query(uncertainty=0.9, ood_score=0.9, abstain_action="ABSTAIN")
    assert should_q_4 is False
    
    # Resolve one label
    resolved = learner.resolve_label(req1.request_id, ground_truth_label=1)
    assert resolved is not None
    assert resolved.status == "LABELED"
    assert resolved.analyst_label == 1
    assert len(learner.get_pending()) == 2
