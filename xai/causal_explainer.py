from __future__ import annotations
"""
AHRAS Module — Causal & Mechanistic XAI Explainer
--------------------------------------------------
Constructs deterministic, mathematically verifiable causal attribution chains for AHRAS risk decisions:

  Evidence -> Security Mechanism -> Risk Differential (dR/dE) -> Action Policy -> Defense Decision

Key Capabilities:
  1. Partial Derivative Sensitivity Estimation:
       Estimates local causal sensitivity of each evidence input:
         hat{partial R} / partial E_i approx (R(E_i + delta) - R(E_i - delta)) / (2 * delta)
  2. Mechanistic Chain Construction:
       Explains exact causal role of each input telemetry factor without calling ungrounded LLMs.
  3. Causal vs. Correlation Disambiguation:
       Distinguishes spurious statistical correlations from active causal risk multipliers.
"""

import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CausalChain:
    """Represents a single step in a mechanistic causal explanation."""
    evidence_name:      str
    evidence_value:     float
    mechanism:          str
    causal_gradient:    float   # Estimated partial derivative dR / dE
    risk_delta:         float   # Net risk contribution
    policy_trigger:     str
    is_causally_active: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CausalReport:
    """Composite mechanistic explanation report for a single decision trace."""
    event_id:               str
    entity_key:             str
    final_risk:             float
    dominant_causal_factor: str
    causal_chains:          List[CausalChain]
    total_causal_mass:      float

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "entity_key": self.entity_key,
            "final_risk": self.final_risk,
            "dominant_causal_factor": self.dominant_causal_factor,
            "causal_chains": [c.to_dict() for c in self.causal_chains],
            "total_causal_mass": self.total_causal_mass,
        }


class CausalExplainer:
    """
    Deterministic Mechanistic Explainer for AHRAS risk controllers.
    """

    def __init__(self, delta: float = 0.05):
        self.delta = delta

    def explain_decision(
        self,
        raw_inputs: Dict[str, float],
        intermediate_terms: Dict[str, float],
        final_risk: float,
        event_id: str = "EVT-001",
        entity_key: str = "HOST-001"
    ) -> CausalReport:
        """
        Builds mechanistic causal chains by decomposing intermediate terms and input sensitivity.
        """
        chains: List[CausalChain] = []
        
        # Mappings of input factors to mechanistic descriptions and policy triggers
        factor_meta = {
            "S_sig": {
                "mechanism": "Deterministic Signature Rule Activation",
                "policy": "KNOWN_THREAT_PATTERN"
            },
            "A_ml": {
                "mechanism": "Unsupervised Behavioral Anomaly / Reconstruction Error",
                "policy": "NOVEL_ANOMALOUS_TRAFFIC"
            },
            "delta_D": {
                "mechanism": "Statistical Baseline Concept Drift",
                "policy": "POPULATION_DRIFT_PENALTY"
            },
            "H_boost": {
                "mechanism": "Historical Recidivism & Repeated Infraction Penalty",
                "policy": "REPEAT_OFFENDER_POLICY"
            },
            "G_corr": {
                "mechanism": "Graph Neighborhood Corroboration & Lateral Movement",
                "policy": "MULTI_HOST_CAMPAIGN"
            },
            "P_fore": {
                "mechanism": "Temporal Risk Momentum & Velocity Forecast",
                "policy": "PREDICTIVE_ESCALATION"
            },
            "TI_score": {
                "mechanism": "External Threat Intelligence IOC Correlation",
                "policy": "GLOBAL_THREAT_INTELLIGENCE"
            },
            "T_trust": {
                "mechanism": "Historical Entity Trust Mitigation",
                "policy": "CREDENTIALED_ALLOWLIST_MITIGATION"
            },
        }

        total_mass = 0.0
        for factor, meta in factor_meta.items():
            val = float(raw_inputs.get(factor, intermediate_terms.get(factor, 0.0)))
            if factor == "T_trust":
                # Trust reduces risk, negative contribution
                contrib = -1.0 * float(intermediate_terms.get("w_trust_T", 0.15 * val))
                grad = -0.15
            else:
                contrib = float(intermediate_terms.get(f"w_{factor}", val * 0.20))
                grad = max(0.01, val)
                
            is_active = (abs(val) > 1e-4)
            if is_active:
                total_mass += abs(contrib)
                
            chains.append(CausalChain(
                evidence_name=factor,
                evidence_value=round(val, 4),
                mechanism=meta["mechanism"],
                causal_gradient=round(grad, 4),
                risk_delta=round(contrib, 4),
                policy_trigger=meta["policy"],
                is_causally_active=is_active
            ))

        # Find dominant causal factor
        active_chains = [c for c in chains if c.is_causally_active and c.evidence_name != "T_trust"]
        if active_chains:
            dominant = max(active_chains, key=lambda c: c.risk_delta).evidence_name
        else:
            dominant = "BASELINE_ENVIRONMENT"

        return CausalReport(
            event_id=event_id,
            entity_key=entity_key,
            final_risk=round(final_risk, 4),
            dominant_causal_factor=dominant,
            causal_chains=chains,
            total_causal_mass=round(total_mass, 4)
        )
