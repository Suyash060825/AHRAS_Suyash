from __future__ import annotations
"""
AHRAS Module 12 — Counterfactual XAI Auditability & Sensitivity Engine
----------------------------------------------------------------------
Provides mathematically grounded, evidence-based counterfactual explanations:
  1. Root-Cause Decomposition: Pinpoints exact evidence drivers of risk escalation.
  2. Minimal Counterfactual Interventions: Computes minimal evidence perturbations
     that would reduce the risk below critical/high response thresholds.
  3. Strict Audit Grounding: Zero LLM hallucinations; all counterfactual statements
     are derived directly from the deterministic DecisionTrace.
"""

import copy
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from detection.risk_engine import DecisionTrace, RiskConfig, replay_decision_trace

log = logging.getLogger(__name__)


@dataclass
class CounterfactualIntervention:
    evidence_name:       str            # e.g., 'signature', 'anomaly', 'graph', 'drift'
    original_value:      float
    counterfactual_val:  float
    original_risk:       float
    resulting_risk:      float
    risk_delta:          float
    prevents_escalation: bool
    explanation_text:    str

    def to_dict(self) -> dict:
        return {
            "evidence_name":       self.evidence_name,
            "original_value":      round(self.original_value, 4),
            "counterfactual_val":  round(self.counterfactual_val, 4),
            "original_risk":       round(self.original_risk, 4),
            "resulting_risk":      round(self.resulting_risk, 4),
            "risk_delta":          round(self.risk_delta, 4),
            "prevents_escalation": self.prevents_escalation,
            "explanation_text":    self.explanation_text,
        }


@dataclass
class CounterfactualReport:
    event_id:             str
    entity_key:           str
    original_risk:        float
    target_threshold:     float
    escalation_cause:     str
    interventions:        List[CounterfactualIntervention]
    minimal_intervention: Optional[CounterfactualIntervention] = None

    def to_dict(self) -> dict:
        return {
            "event_id":             self.event_id,
            "entity_key":           self.entity_key,
            "original_risk":        round(self.original_risk, 4),
            "target_threshold":     round(self.target_threshold, 4),
            "escalation_cause":     self.escalation_cause,
            "interventions":        [i.to_dict() for i in self.interventions],
            "minimal_intervention": self.minimal_intervention.to_dict() if self.minimal_intervention else None,
        }


class CounterfactualExplainer:
    """
    Performs analytical counterfactual search over a recorded DecisionTrace.
    """

    def analyze_trace(self, trace: DecisionTrace, target_threshold: float = 0.70) -> CounterfactualReport:
        orig_risk = trace.final_clamped_score
        inputs = trace.raw_inputs
        cfg = trace.config
        
        # 1. Identify primary escalation driver
        terms = trace.intermediate_terms
        numeric_terms = {k: float(v) for k, v in terms.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        sorted_terms = sorted(numeric_terms.items(), key=lambda kv: kv[1], reverse=True)
        top_driver_name, top_driver_val = sorted_terms[0] if sorted_terms else ("none", 0.0)
        
        escalation_cause = (
            f"Primary risk driver is '{top_driver_name}' contributing {top_driver_val:.3f} to threat sum."
        )

        interventions: List[CounterfactualIntervention] = []

        # 2. Test single-variable ablation interventions
        evidence_keys = [
            ("S_sig", "signature", "term_sig"),
            ("A_ml", "ml_anomaly", "term_ml"),
            ("delta_D", "behavioral_drift", "term_ml"),
            ("G_corr", "graph_corroboration", "term_graph"),
            ("H_boost", "history_recidivism", "term_hist"),
            ("TI_score", "threat_intelligence", "term_ti"),
        ]

        for input_key, human_name, term_key in evidence_keys:
            orig_val = inputs.get(input_key, 0.0)
            if orig_val <= 0.0:
                continue

            # Counterfactual: setting evidence to zero
            sim_trace = copy.deepcopy(trace)
            sim_trace.raw_inputs[input_key] = 0.0
            if input_key == "delta_D":
                sim_trace.raw_inputs["delta_D"] = 0.0

            sim_risk = replay_decision_trace(sim_trace)
            r_delta = orig_risk - sim_risk
            prevents = sim_risk < target_threshold

            exp_text = (
                f"Removing {human_name} (reducing from {orig_val:.2f} to 0.0) "
                f"reduces risk by {r_delta:.3f} (from {orig_risk:.3f} to {sim_risk:.3f}), "
                f"{'preventing' if prevents else 'insufficient to prevent'} escalation beyond {target_threshold:.2f}."
            )

            interventions.append(CounterfactualIntervention(
                evidence_name=human_name,
                original_value=orig_val,
                counterfactual_val=0.0,
                original_risk=orig_risk,
                resulting_risk=sim_risk,
                risk_delta=r_delta,
                prevents_escalation=prevents,
                explanation_text=exp_text,
            ))

        # Select minimal intervention that prevents escalation
        preventing = [i for i in interventions if i.prevents_escalation]
        preventing_sorted = sorted(preventing, key=lambda x: x.original_value)
        minimal_int = preventing_sorted[0] if preventing_sorted else None

        return CounterfactualReport(
            event_id=trace.event_id,
            entity_key=trace.entity_key,
            original_risk=orig_risk,
            target_threshold=target_threshold,
            escalation_cause=escalation_cause,
            interventions=interventions,
            minimal_intervention=minimal_int,
        )
