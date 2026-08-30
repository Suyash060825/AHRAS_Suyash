import pytest
import numpy as np
from detection.selective_gate import (
    ConformalRiskGate,
    SelectionDecision,
    ACTION_AUTONOMOUS_CONTAINMENT,
    ACTION_AUTONOMOUS_PASS,
    ACTION_ESCALATE_ANALYST,
    ACTION_DECEPTION,
    ACTION_MONITOR,
    ACTION_STAGED_CONTAINMENT,
    ACTION_ABSTAIN
)

def test_conformal_gate_calibration():
    gate = ConformalRiskGate(target_coverage=0.90)
    risk_scores = [0.1, 0.2, 0.8, 0.9, 0.3, 0.85]
    labels = [0, 0, 1, 1, 0, 1]
    
    tau = gate.calibrate(risk_scores, labels)
    assert gate.is_calibrated
    assert 0.0 < tau <= 1.0

def test_conformal_gate_decisions():
    gate = ConformalRiskGate(target_coverage=0.90, uncertainty_threshold=0.35, risk_action_threshold=0.70)
    gate.calibrated_tau = 0.20
    
    # 1. Confident attack -> AUTONOMOUS_CONTAINMENT
    d1 = gate.evaluate_gate(risk_score=0.95, uncertainty=0.10, ood_score=0.10)
    assert d1.action in (ACTION_AUTONOMOUS_CONTAINMENT, ACTION_STAGED_CONTAINMENT)
    assert d1.is_autonomous is True
    
    # 2. Confident benign -> AUTONOMOUS_PASS
    d2 = gate.evaluate_gate(risk_score=0.10, uncertainty=0.05, ood_score=0.05)
    assert d2.action == ACTION_AUTONOMOUS_PASS
    assert d2.is_autonomous is True
    
    # 3. High uncertainty alert -> ESCALATE_ANALYST
    d3 = gate.evaluate_gate(risk_score=0.85, uncertainty=0.60, ood_score=0.10)
    assert d3.action == ACTION_ESCALATE_ANALYST
    assert d3.is_autonomous is False
    
    # 4. Zero-day OOD -> DECEPTION
    d4 = gate.evaluate_gate(risk_score=0.50, uncertainty=0.20, ood_score=0.80)
    assert d4.action == ACTION_DECEPTION
    assert d4.is_autonomous is True
