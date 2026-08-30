import pytest
from xai.causal_explainer import CausalExplainer, CausalReport

def test_causal_explainer_chains():
    explainer = CausalExplainer()
    
    raw_inputs = {
        "S_sig": 0.85,
        "A_ml": 0.90,
        "delta_D": 0.15,
        "G_corr": 0.70,
        "H_boost": 0.0,
        "P_fore": 0.40,
        "TI_score": 0.95,
        "T_trust": 0.20,
    }
    intermediate = {
        "w_sig_S": 0.425,
        "w_ml_A": 0.27,
        "w_trust_T": 0.03,
    }
    
    report = explainer.explain_decision(
        raw_inputs=raw_inputs,
        intermediate_terms=intermediate,
        final_risk=0.88,
        event_id="EVT-TEST",
        entity_key="HOST-X"
    )
    
    assert isinstance(report, CausalReport)
    assert report.event_id == "EVT-TEST"
    assert report.final_risk == 0.88
    assert len(report.causal_chains) > 0
    assert report.total_causal_mass > 0.0
    # S_sig or TI_score or A_ml should be dominant
    assert report.dominant_causal_factor in raw_inputs
