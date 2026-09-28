from __future__ import annotations
"""
AHRAS Human-AI Learning-to-Defer Engine (Section 42)
------------------------------------------------------
Arbitrates collaborative decision handoffs between autonomous AI policies and human SOC analysts.

Core Invariant:
  "Do NOT learn authorization rights. Authorization remains policy-controlled."
  Learning-to-defer optimizes operational routing based on model uncertainty vs human cognitive burden,
  but can NEVER grant authorization for an action prohibited by deterministic security policy.

Arbitration States:
  - AUTOMATE: High model confidence, routine threat, conformal safety verified. Executed autonomously.
  - RECOMMEND: Clear signal, but low-to-medium blast radius; staged for quick analyst one-click sign-off.
  - ESCALATE: High risk with high uncertainty, novel attack technique, or critical asset involved;
              escalated to Senior SOC Lead with full investigation dossier.
  - ABSTAIN: Ambiguous near-threshold telemetry without actionable evidence; routes to baseline monitoring.

Multi-Objective Cost Optimization:
  Minimizes operational loss L:
    L = C_workload * (Is_Human) + C_false_auto * (False_Automation) + C_breach * (Uncontained_Attack)
"""

import time
import math
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple

log = logging.getLogger(__name__)


class DeferralAction(str, Enum):
    AUTOMATE  = "AUTOMATE"
    RECOMMEND = "RECOMMEND"
    ESCALATE  = "ESCALATE"
    ABSTAIN   = "ABSTAIN"


@dataclass
class DeferralDecision:
    """Output of the learning-to-defer optimization."""
    event_id:             str
    selected_action:      DeferralAction
    risk_score:           float
    uncertainty:          float
    asset_criticality:    float    # [0.0, 1.0] (1.0 = Crown Jewel)
    policy_permitted:     bool     # Whether automated execution is legally allowed by policy
    estimated_loss:       float    # Operational loss value
    rationale:            str
    requires_human_touch: bool
    escalation_priority:  str      # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    timestamp:            float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["selected_action"] = self.selected_action.value
        return d


class LearningToDeferEngine:
    """
    Cost-sensitive learning-to-defer optimizer.
    Balances automated mitigation velocity against analyst fatigue and false-positive containment disasters.
    """

    def __init__(
        self,
        c_workload: float = 1.0,         # Cost per human analyst review
        c_false_auto: float = 25.0,      # Severe penalty for catastrophic false automation
        c_breach: float = 50.0,          # Catastrophic penalty for uncontained breach
        uncertainty_escalate: float = 0.30,
        asset_crown_jewel_threshold: float = 0.80,
    ):
        self.c_workload = c_workload
        self.c_false_auto = c_false_auto
        self.c_breach = c_breach
        self.uncertainty_escalate = uncertainty_escalate
        self.asset_crown_jewel_threshold = asset_crown_jewel_threshold

    def evaluate_decision(
        self,
        event_id: str,
        risk_score: float,
        uncertainty: float,
        asset_criticality: float = 0.50,
        policy_permits_automation: bool = True,
        is_novel_technique: bool = False,
    ) -> DeferralDecision:
        """
        Determines the optimal collaboration mode: AUTOMATE, RECOMMEND, ESCALATE, or ABSTAIN.
        """
        now = time.time()

        # Invariant 1: If policy strictly forbids automation, it CANNOT be AUTOMATE
        if not policy_permits_automation:
            if risk_score >= 0.70:
                return DeferralDecision(
                    event_id=event_id,
                    selected_action=DeferralAction.ESCALATE,
                    risk_score=round(risk_score, 4),
                    uncertainty=round(uncertainty, 4),
                    asset_criticality=round(asset_criticality, 4),
                    policy_permitted=False,
                    estimated_loss=round(self.c_workload, 4),
                    rationale="Policy prohibits automated response for this asset/action class. Escalated to human responder.",
                    requires_human_touch=True,
                    escalation_priority="HIGH" if risk_score < 0.85 else "CRITICAL",
                    timestamp=now,
                )
            elif risk_score >= 0.40:
                return DeferralDecision(
                    event_id=event_id,
                    selected_action=DeferralAction.RECOMMEND,
                    risk_score=round(risk_score, 4),
                    uncertainty=round(uncertainty, 4),
                    asset_criticality=round(asset_criticality, 4),
                    policy_permitted=False,
                    estimated_loss=round(self.c_workload * 0.5, 4),
                    rationale="Policy requires human confirmation. Mitigation staged as recommendation.",
                    requires_human_touch=True,
                    escalation_priority="MEDIUM",
                    timestamp=now,
                )
            else:
                return DeferralDecision(
                    event_id=event_id,
                    selected_action=DeferralAction.ABSTAIN,
                    risk_score=round(risk_score, 4),
                    uncertainty=round(uncertainty, 4),
                    asset_criticality=round(asset_criticality, 4),
                    policy_permitted=False,
                    estimated_loss=0.0,
                    rationale="Low risk and automated intervention disabled. Monitoring passively.",
                    requires_human_touch=False,
                    escalation_priority="LOW",
                    timestamp=now,
                )

        # Invariant 2: Crown Jewel Assets or Novel Zero-Days under uncertainty MUST be ESCALATED
        if (asset_criticality >= self.asset_crown_jewel_threshold and risk_score >= 0.60) or \
           (is_novel_technique and risk_score >= 0.60) or \
           (uncertainty >= self.uncertainty_escalate and risk_score >= 0.60):
            loss = self.c_workload
            return DeferralDecision(
                event_id=event_id,
                selected_action=DeferralAction.ESCALATE,
                risk_score=round(risk_score, 4),
                uncertainty=round(uncertainty, 4),
                asset_criticality=round(asset_criticality, 4),
                policy_permitted=True,
                estimated_loss=round(loss, 4),
                rationale="High-stakes asset or epistemic uncertainty warrants mandatory senior analyst escalation.",
                requires_human_touch=True,
                escalation_priority="CRITICAL" if asset_criticality >= 0.90 else "HIGH",
                timestamp=now,
            )

        # Evaluate expected losses for AUTOMATE vs RECOMMEND vs ABSTAIN
        # P(Error) estimated proportional to uncertainty and (1 - risk)
        p_fp = uncertainty * (1.0 - risk_score)
        p_fn = (1.0 - risk_score) if risk_score < 0.5 else 0.0

        loss_auto = (p_fp * self.c_false_auto) + (p_fn * self.c_breach)
        loss_rec = self.c_workload * 0.60 + (p_fn * self.c_breach * 0.50)
        loss_abstain = p_fn * self.c_breach

        # Decision boundary
        if risk_score >= 0.75 and uncertainty < self.uncertainty_escalate and loss_auto < loss_rec:
            return DeferralDecision(
                event_id=event_id,
                selected_action=DeferralAction.AUTOMATE,
                risk_score=round(risk_score, 4),
                uncertainty=round(uncertainty, 4),
                asset_criticality=round(asset_criticality, 4),
                policy_permitted=True,
                estimated_loss=round(loss_auto, 4),
                rationale="High confidence breach with negligible false-automation risk. Closed-loop autonomous containment authorized.",
                requires_human_touch=False,
                escalation_priority="NONE",
                timestamp=now,
            )
        elif risk_score >= 0.45:
            return DeferralDecision(
                event_id=event_id,
                selected_action=DeferralAction.RECOMMEND,
                risk_score=round(risk_score, 4),
                uncertainty=round(uncertainty, 4),
                asset_criticality=round(asset_criticality, 4),
                policy_permitted=True,
                estimated_loss=round(loss_rec, 4),
                rationale="Suspicious threat with moderate confidence. Staged for analyst approval queue.",
                requires_human_touch=True,
                escalation_priority="MEDIUM",
                timestamp=now,
            )
        else:
            return DeferralDecision(
                event_id=event_id,
                selected_action=DeferralAction.ABSTAIN,
                risk_score=round(risk_score, 4),
                uncertainty=round(uncertainty, 4),
                asset_criticality=round(asset_criticality, 4),
                policy_permitted=True,
                estimated_loss=round(loss_abstain, 4),
                rationale="Low risk telemetry within normal baseline variance. Retained in passive stream.",
                requires_human_touch=False,
                escalation_priority="LOW",
                timestamp=now,
            )


_global_defer_engine: Optional[LearningToDeferEngine] = None

def get_learning_to_defer_engine() -> LearningToDeferEngine:
    global _global_defer_engine
    if _global_defer_engine is None:
        _global_defer_engine = LearningToDeferEngine()
    return _global_defer_engine
