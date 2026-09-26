"""
AHRAS Cost-Sensitive Response Policy
------------------------------------
Unifies Conformal Safety Guarantees, Bayesian Operational Loss Minimization,
and Deterministic Safety Invariants into a safe, bounded autonomous response engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from response.safety_invariants import SafetyInvariantChecker, SafetyVerdict
from response.conformal_controller import ConformalResponseController, ConformalSafetyDecision


@dataclass
class ResponseActionVerdict:
    """
    Final operational response decision dispatched by the engine.
    """
    event_id: str
    target_entity: str
    dispatched_action: str
    is_autonomous_executed: bool
    expected_loss: float
    conformal_decision: ConformalSafetyDecision
    safety_verdict: SafetyVerdict
    audit_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "target_entity": self.target_entity,
            "dispatched_action": self.dispatched_action,
            "is_autonomous_executed": self.is_autonomous_executed,
            "expected_loss": round(self.expected_loss, 4),
            "conformal_decision": self.conformal_decision.to_dict(),
            "safety_verdict": self.safety_verdict.to_dict(),
            "audit_notes": self.audit_notes,
        }


# Operational cost parameters: (C_intervention, C_fp, Efficacy)
ACTION_COST_PROFILES = {
    "AUTONOMOUS_CONTAINMENT": {"c_intervene": 15.0, "c_fp": 120.0, "efficacy": 0.98},
    "STAGED_CONTAINMENT":     {"c_intervene": 8.0,  "c_fp": 30.0,  "efficacy": 0.85},
    "DECEPTION":              {"c_intervene": 2.0,  "c_fp": 5.0,   "efficacy": 0.70},
    "MONITOR":                {"c_intervene": 0.5,  "c_fp": 0.0,   "efficacy": 0.10},
    "ABSTAIN":                {"c_intervene": 1.0,  "c_fp": 0.0,   "efficacy": 0.05},
    "ESCALATE_ANALYST":       {"c_intervene": 10.0, "c_fp": 10.0,  "efficacy": 0.95},
    "AUTONOMOUS_PASS":        {"c_intervene": 0.0,  "c_fp": 0.0,   "efficacy": 0.00},
}


class CostSensitiveResponseEngine:
    """
    Arbitrates autonomous cyber defense actions under conformal risk bounds and hard safety invariants.
    """
    def __init__(
        self,
        conformal_controller: Optional[ConformalResponseController] = None,
        safety_checker: Optional[SafetyInvariantChecker] = None,
        c_breach: float = 500.0,
    ) -> None:
        self.conformal = conformal_controller or ConformalResponseController()
        self.safety = safety_checker or SafetyInvariantChecker()
        self.c_breach = c_breach

    def compute_expected_loss(self, action: str, p_attack: float) -> float:
        """
        Loss(a, p) = C_intervene(a) + (1 - p) * C_fp(a) + p * (1 - Efficacy(a)) * C_breach
        """
        prof = ACTION_COST_PROFILES.get(action, ACTION_COST_PROFILES["MONITOR"])
        p_benign = max(0.0, 1.0 - p_attack)
        c_int = prof["c_intervene"]
        c_fp = prof["c_fp"]
        eff = prof["efficacy"]
        return float(c_int + (p_benign * c_fp) + (p_attack * (1.0 - eff) * self.c_breach))

    def arbitrate_response(
        self,
        event_id: str,
        target_entity: str,
        p_attack: float,
        epistemic_uncertainty: float = 0.05,
        entity_metadata: Optional[Dict[str, Any]] = None,
    ) -> ResponseActionVerdict:
        """
        Determines the optimal response action balancing conformal coverage, expected loss, and safety barriers.
        """
        # 1. Conformal prediction set evaluation
        conf_decision = self.conformal.evaluate_instance(p_attack, epistemic_uncertainty)

        # 2. Select initial action recommendation from conformal gate
        rec_action = conf_decision.gating_recommendation

        # 3. Check hard deterministic safety invariants
        safety_verdict = self.safety.evaluate_action_safety(
            action=rec_action,
            target_entity=target_entity,
            entity_metadata=entity_metadata,
        )

        # 4. Arbitration logic
        is_autonomous_exec = False
        dispatched_action = rec_action
        audit_note = "Conformal recommendation approved by safety invariants."

        if not safety_verdict.passed:
            # Hard safety barrier triggered -> Downgrade destructive action to STAGED or ESCALATE
            if rec_action in {"AUTONOMOUS_CONTAINMENT", "ISOLATE_HOST"}:
                dispatched_action = "STAGED_CONTAINMENT"
                audit_note = f"Automated containment downgraded to STAGED due to safety invariant violations: {'; '.join(safety_verdict.violations)}"
            else:
                dispatched_action = "ESCALATE_ANALYST"
                audit_note = f"Action downgraded due to safety violation: {'; '.join(safety_verdict.violations)}"
        else:
            if conf_decision.is_autonomous_candidate:
                is_autonomous_exec = True
                audit_note = f"Autonomous action '{dispatched_action}' executed safely within conformal bound."

        loss = self.compute_expected_loss(dispatched_action, p_attack)

        return ResponseActionVerdict(
            event_id=event_id,
            target_entity=target_entity,
            dispatched_action=dispatched_action,
            is_autonomous_executed=is_autonomous_exec,
            expected_loss=loss,
            conformal_decision=conf_decision,
            safety_verdict=safety_verdict,
            audit_notes=audit_note,
        )
