from __future__ import annotations
"""
AHRAS Module 4 / SOAR — Response Efficacy Learning & Adaptive Policy Engine
----------------------------------------------------------------------------
Implements continuous, empirical response efficacy learning and safety-gated
counterfactual utility optimization:
  - Records post-mitigation observations: (Risk_before - Risk_after), latency,
    residual activity, collateral impact.
  - Maintains Bayesian conjugate Beta beliefs for (action, threat, asset) tuples.
  - Integrates with Digital Twin pre-execution counterfactual simulation.
  - Enforces hard safety invariants: no learned utility can override asset criticality
    and human-in-the-loop policies for Tier-1 assets or high epistemic uncertainty.
"""

import copy
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    EFFICACY_EXPLORATION_EPSILON,
    EFFICACY_PRIOR_WEIGHT,
    MAX_AUTONOMOUS_BLAST_RADIUS,
    USE_RESPONSE_EFFICACY_LEARNER,
)
from response.orchestrator import ACTION_COST_MATRIX, ResponseAction

log = logging.getLogger(__name__)


@dataclass
class ResponseExecutionRecord:
    """Record of an executed or staged mitigation action and its observed outcome."""
    record_id: str
    action_type: str
    threat_family: str
    asset_class: str
    entity_key: str
    target_identifier: str
    risk_before: float
    risk_after: float
    observed_risk_reduction: float
    uncertainty_before: float
    uncertainty_after: float
    time_to_effect_sec: float
    residual_activity: bool
    blast_radius_observed: float
    collateral_cost_observed: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    twin_simulated: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EfficacyBelief:
    """Conjugate Beta distribution over expected risk reduction for an action tuple."""
    action_type: str
    threat_family: str
    asset_class: str
    alpha: float
    beta_param: float
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def mean(self) -> float:
        total = self.alpha + self.beta_param
        return self.alpha / total if total > 0 else 0.5

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta_param
        if total <= 0:
            return 0.05
        return (self.alpha * self.beta_param) / ((total ** 2) * (total + 1.0))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def credible_interval_95(self) -> Tuple[float, float]:
        m = self.mean
        s = self.std
        return (max(0.0, m - 1.96 * s), min(1.0, m + 1.96 * s))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "threat_family": self.threat_family,
            "asset_class": self.asset_class,
            "alpha": round(self.alpha, 4),
            "beta_param": round(self.beta_param, 4),
            "sample_count": self.sample_count,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "credible_interval_95": [round(x, 4) for x in self.credible_interval_95],
            "last_updated": self.last_updated,
        }


@dataclass
class EfficacyEvaluationResult:
    """Outcome of evaluating candidate response action utility with learned prior & safety gates."""
    action_type: str
    threat_family: str
    asset_class: str
    static_utility: float
    learned_utility: float
    expected_risk_reduction: float
    efficacy_mean: float
    efficacy_std: float
    twin_simulated: bool
    twin_breakage_probability: float
    twin_expected_post_risk: Optional[float]
    safety_override_triggered: bool
    safety_reason: str
    recommended_decision: str  # AUTO_EXECUTE, STAGE_FOR_SOC, ABSTAIN

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default empirical domain priors for (action, threat, asset_class)
# Prior format: (prior_mean, prior_weight) -> alpha = mean * weight, beta = (1 - mean) * weight
DOMAIN_EFFICACY_PRIORS: Dict[Tuple[str, str, str], float] = {
    # (action, threat_family, asset_class) -> expected reduction
    ("BLOCK_IP", "BOTNET_C2", "ALL"): 0.85,
    ("BLOCK_IP", "BRUTE_FORCE", "ALL"): 0.80,
    ("BLOCK_IP", "RANSOMWARE", "WORKSTATION"): 0.20,  # Ineffective once running locally
    ("BLOCK_IP", "RANSOMWARE", "SERVER"): 0.30,
    ("BLOCK_IP", "DATA_EXFILTRATION", "ALL"): 0.75,
    ("ISOLATE_HOST", "RANSOMWARE", "WORKSTATION"): 0.92,
    ("ISOLATE_HOST", "RANSOMWARE", "DOMAIN_CONTROLLER"): 0.90,
    ("ISOLATE_HOST", "LATERAL_MOVEMENT", "WORKSTATION"): 0.88,
    ("ISOLATE_HOST", "LATERAL_MOVEMENT", "DOMAIN_CONTROLLER"): 0.85,
    ("TERMINATE_PROCESS", "RANSOMWARE", "ALL"): 0.82,
    ("TERMINATE_PROCESS", "BOTNET_C2", "ALL"): 0.70,
    ("TERMINATE_PROCESS", "LATERAL_MOVEMENT", "ALL"): 0.65,
    ("REVOKE_TOKEN", "CREDENTIAL_ABUSE", "ALL"): 0.90,
    ("REVOKE_TOKEN", "CLOUD_COMPROMISE", "ALL"): 0.92,
    ("REVOKE_TOKEN", "PRIVILEGE_ESCALATION", "ALL"): 0.80,
}


class ResponseEfficacyLearner:
    """
    Online Bayesian learner that records observed response outcomes, updates
    action-threat efficacy distributions, conducts pre-simulation via the Digital Twin,
    and enforces hard safety constraints.
    """

    def __init__(
        self,
        prior_strength: float = EFFICACY_PRIOR_WEIGHT,
        twin_simulator: Optional[Any] = None,
        dry_run: bool = True,
    ):
        self.prior_strength = prior_strength
        self.twin_simulator = twin_simulator
        self.dry_run = dry_run
        self._beliefs: Dict[Tuple[str, str, str], EfficacyBelief] = {}
        self._execution_history: List[ResponseExecutionRecord] = []
        self._lock = threading.RLock()
        self._init_default_priors()

    def _init_default_priors(self) -> None:
        """Initializes default Bayesian priors for known (action, threat, asset) tuples."""
        w = max(1.0, self.prior_strength)
        for (action, threat, asset), mean_val in DOMAIN_EFFICACY_PRIORS.items():
            mean_clamped = max(0.05, min(0.95, mean_val))
            alpha = mean_clamped * w
            beta_p = (1.0 - mean_clamped) * w
            key = (action.upper(), threat.upper(), asset.upper())
            self._beliefs[key] = EfficacyBelief(
                action_type=key[0],
                threat_family=key[1],
                asset_class=key[2],
                alpha=alpha,
                beta_param=beta_p,
                sample_count=0,
                last_updated=time.time(),
            )

    def _get_belief_key(self, action_type: str, threat_family: str, asset_class: str) -> Tuple[str, str, str]:
        act = action_type.upper()
        if act == "BLOCK_SOURCE":
            act = "BLOCK_IP"
        return (act, threat_family.upper(), asset_class.upper())

    def get_belief(self, action_type: str, threat_family: str, asset_class: str) -> EfficacyBelief:
        """
        Retrieves the efficacy belief for a given tuple, falling back gracefully:
          1. Exact match (action, threat, asset)
          2. Asset wildcard (action, threat, 'ALL')
          3. Threat wildcard (action, 'ALL', 'ALL')
          4. Base static cost matrix fallback
        """
        with self._lock:
            act = action_type.upper()
            if act == "BLOCK_SOURCE":
                act = "BLOCK_IP"
            tf = threat_family.upper()
            ac = asset_class.upper()

            for key in [(act, tf, ac), (act, tf, "ALL"), (act, "ALL", "ALL")]:
                if key in self._beliefs:
                    return self._beliefs[key]

            # Construct new default belief from ACTION_COST_MATRIX
            base_meta = ACTION_COST_MATRIX.get(act, {"expected_risk_reduction": 0.50})
            base_mean = base_meta.get("expected_risk_reduction", 0.50)
            w = self.prior_strength
            alpha = base_mean * w
            beta_p = (1.0 - base_mean) * w
            new_belief = EfficacyBelief(
                action_type=act,
                threat_family=tf,
                asset_class=ac,
                alpha=alpha,
                beta_param=beta_p,
                sample_count=0,
                last_updated=time.time(),
            )
            self._beliefs[(act, tf, ac)] = new_belief
            return new_belief

    def record_outcome(
        self,
        action_type: str,
        threat_family: str,
        asset_class: str,
        entity_key: str,
        target_identifier: str,
        risk_before: float,
        risk_after: float,
        uncertainty_before: float = 0.15,
        uncertainty_after: float = 0.10,
        time_to_effect_sec: float = 1.0,
        residual_activity: bool = False,
        blast_radius_observed: Optional[float] = None,
        collateral_cost_observed: Optional[float] = None,
        success: bool = True,
        twin_simulated: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> ResponseExecutionRecord:
        """
        Records the real-world post-execution outcome of a defense action and
        updates the Bayesian conjugate Beta distribution for that (action, threat, asset) context.
        """
        with self._lock:
            act = action_type.upper()
            if act == "BLOCK_SOURCE":
                act = "BLOCK_IP"
            tf = threat_family.upper()
            ac = asset_class.upper()

            # Calculate observed risk reduction clamped to [0.0, 1.0]
            raw_reduction = risk_before - risk_after
            observed_reduction = max(0.0, min(1.0, raw_reduction))

            meta = ACTION_COST_MATRIX.get(act, {"blast_radius": 0.20, "reversibility_cost": 0.10})
            blast = blast_radius_observed if blast_radius_observed is not None else meta.get("blast_radius", 0.20)
            collateral = collateral_cost_observed if collateral_cost_observed is not None else meta.get("reversibility_cost", 0.10)

            record = ResponseExecutionRecord(
                record_id=str(uuid.uuid4())[:8],
                action_type=act,
                threat_family=tf,
                asset_class=ac,
                entity_key=entity_key,
                target_identifier=target_identifier,
                risk_before=round(float(risk_before), 4),
                risk_after=round(float(risk_after), 4),
                observed_risk_reduction=round(float(observed_reduction), 4),
                uncertainty_before=round(float(uncertainty_before), 4),
                uncertainty_after=round(float(uncertainty_after), 4),
                time_to_effect_sec=round(float(time_to_effect_sec), 3),
                residual_activity=bool(residual_activity),
                blast_radius_observed=round(float(blast), 4),
                collateral_cost_observed=round(float(collateral), 4),
                success=bool(success),
                timestamp=time.time(),
                twin_simulated=bool(twin_simulated),
                context=context or {},
            )
            self._execution_history.append(record)

            # Update Bayesian Belief
            # If residual activity was detected or action failed, penalize alpha and reward beta
            belief = self.get_belief(act, tf, ac)
            
            delta_alpha = observed_reduction
            delta_beta = (1.0 - observed_reduction)
            
            if residual_activity or not success:
                delta_beta += 0.50
                delta_alpha = max(0.0, delta_alpha - 0.20)

            belief.alpha += delta_alpha
            belief.beta_param += delta_beta
            belief.sample_count += 1
            belief.last_updated = time.time()

            log.info(
                f"[EFFICACY LEARNER] Updated ({act}, {tf}, {ac}): "
                f"ObservedReduction={observed_reduction:.3f}, Residual={residual_activity} "
                f"-> Mean={belief.mean:.3f}, Std={belief.std:.3f}, Samples={belief.sample_count}"
            )
            return record

    def evaluate_action_utility(
        self,
        action_type: str,
        threat_family: str,
        asset_class: str,
        current_risk: float,
        confidence: float,
        uncertainty: float,
        twin_scenario: Optional[Any] = None,
        twin_target: Optional[str] = None,
    ) -> EfficacyEvaluationResult:
        """
        Computes the complete, safety-gated utility of a candidate response action:
          1. Computes static baseline utility.
          2. Retrieves learned Bayesian efficacy prior/posterior.
          3. Simulates counterfactual intervention in the Digital Twin if scenario provided.
          4. Fuses empirical belief with digital twin path breakage.
          5. Applies HARD SAFETY INVARIANTS (Tier-1 Crown Jewel and high uncertainty gating).
        """
        act = action_type.upper()
        if act == "BLOCK_SOURCE":
            act = "BLOCK_IP"
        tf = threat_family.upper()
        ac = asset_class.upper()

        meta = ACTION_COST_MATRIX.get(act, {"blast_radius": 0.25, "reversibility_cost": 0.15, "expected_risk_reduction": 0.50})
        static_expected_red = meta.get("expected_risk_reduction", 0.50)
        blast_cost = meta.get("blast_radius", 0.25)
        rev_cost = meta.get("reversibility_cost", 0.15)
        unc_penalty = uncertainty * 0.25

        # 1. Static utility
        static_utility = (static_expected_red * (current_risk / 1.0) * confidence) - blast_cost - rev_cost - unc_penalty

        # 2. Learned belief
        belief = self.get_belief(act, tf, ac)
        efficacy_mean = belief.mean
        efficacy_std = belief.std

        # 3. Pre-execution Digital Twin counterfactual simulation
        twin_simulated = False
        twin_breakage = 1.0
        twin_post_risk: Optional[float] = None

        if self.twin_simulator is not None and twin_scenario is not None and twin_target:
            try:
                sim_res = self.twin_simulator.simulate_action(
                    scenario=twin_scenario,
                    action_type=act,
                    target_entity=twin_target,
                    current_risk=current_risk,
                )
                twin_simulated = True
                twin_breakage = sim_res.path_breakage_probability
                twin_post_risk = sim_res.expected_post_action_risk
                twin_delta = max(0.0, (sim_res.pre_action_risk - sim_res.expected_post_action_risk) / max(0.01, sim_res.pre_action_risk))
                # Fuse empirical belief with digital twin simulation
                fused_reduction = (0.50 * efficacy_mean) + (0.50 * twin_delta * twin_breakage)
            except Exception as e:
                log.warning(f"[EFFICACY LEARNER] Twin simulation failed: {e}. Falling back to empirical belief.")
                fused_reduction = efficacy_mean
        else:
            fused_reduction = efficacy_mean

        # 4. Learned dynamic utility
        # Penalize utility if belief variance is high (epistemic risk penalty)
        epistemic_penalty = efficacy_std * 0.15
        learned_utility = (
            (fused_reduction * (current_risk / 1.0) * confidence)
            - blast_cost
            - rev_cost
            - unc_penalty
            - epistemic_penalty
        )

        # 5. HARD SAFETY INVARIANTS:
        safety_override_triggered = False
        safety_reason = ""
        recommended_decision = "AUTO_EXECUTE"

        # Invariant 1: High Criticality / Tier-1 Assets (e.g. Domain Controller, Core DB, Gateway)
        # Destructive actions (ISOLATE_HOST) on Tier-1 assets CANNOT be executed autonomously.
        if ac in ("DOMAIN_CONTROLLER", "CRITICAL_SERVER", "CORE_DATABASE", "TIER_1") and act in ("ISOLATE_HOST", "REBOOT", "WIPE"):
            safety_override_triggered = True
            safety_reason = f"Hard safety rule: Autonomous {act} prohibited on Tier-1 asset ({ac}). Requires analyst approval."
            recommended_decision = "STAGE_FOR_SOC"

        # Invariant 2: Blast Radius Gate
        # Actions with high blast radius on Tier-2 assets require extreme confidence (> 0.90) to auto-execute.
        elif blast_cost > MAX_AUTONOMOUS_BLAST_RADIUS and confidence < 0.90:
            safety_override_triggered = True
            safety_reason = f"Blast radius {blast_cost:.2f} exceeds threshold {MAX_AUTONOMOUS_BLAST_RADIUS:.2f} with confidence {confidence:.2f} < 0.90."
            recommended_decision = "STAGE_FOR_SOC"

        # Invariant 3: Epistemic Uncertainty Gate
        # If model uncertainty + belief uncertainty is excessive, do not auto-execute
        elif (uncertainty + efficacy_std) > 0.45:
            safety_override_triggered = True
            safety_reason = f"Combined epistemic uncertainty ({uncertainty + efficacy_std:.2f}) exceeds safe autonomous threshold 0.45."
            recommended_decision = "STAGE_FOR_SOC"

        # Invariant 4: Negative Learned Utility Gate
        elif learned_utility <= 0.0:
            recommended_decision = "ABSTAIN"
            safety_reason = f"Learned action utility {learned_utility:.3f} is non-positive."

        # Invariant 5: Dry-Run Mode Enforcement
        if self.dry_run and recommended_decision == "AUTO_EXECUTE":
            recommended_decision = "SIMULATED_AUTO_EXECUTE"

        return EfficacyEvaluationResult(
            action_type=act,
            threat_family=tf,
            asset_class=ac,
            static_utility=round(float(static_utility), 4),
            learned_utility=round(float(learned_utility), 4),
            expected_risk_reduction=round(float(fused_reduction), 4),
            efficacy_mean=round(float(efficacy_mean), 4),
            efficacy_std=round(float(efficacy_std), 4),
            twin_simulated=twin_simulated,
            twin_breakage_probability=round(float(twin_breakage), 4),
            twin_expected_post_risk=round(float(twin_post_risk), 4) if twin_post_risk is not None else None,
            safety_override_triggered=safety_override_triggered,
            safety_reason=safety_reason,
            recommended_decision=recommended_decision,
        )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Returns comprehensive summary of learned efficacy profiles and feedback stats."""
        with self._lock:
            beliefs_summary = [b.to_dict() for b in self._beliefs.values()]
            total_observations = len(self._execution_history)
            mean_reduction = (
                float(np.mean([r.observed_risk_reduction for r in self._execution_history]))
                if total_observations > 0
                else 0.0
            )
            success_rate = (
                float(np.mean([1.0 if r.success else 0.0 for r in self._execution_history]))
                if total_observations > 0
                else 0.0
            )
            residual_rate = (
                float(np.mean([1.0 if r.residual_activity else 0.0 for r in self._execution_history]))
                if total_observations > 0
                else 0.0
            )
            return {
                "total_profiles_tracked": len(self._beliefs),
                "total_observations_recorded": total_observations,
                "mean_observed_risk_reduction": round(mean_reduction, 4),
                "overall_success_rate": round(success_rate, 4),
                "residual_threat_rate": round(residual_rate, 4),
                "profiles": beliefs_summary,
            }


# Singleton accessor
_learner_instance: Optional[ResponseEfficacyLearner] = None
_learner_lock = threading.Lock()


def get_response_efficacy_learner() -> ResponseEfficacyLearner:
    global _learner_instance
    with _learner_lock:
        if _learner_instance is None:
            _learner_instance = ResponseEfficacyLearner(dry_run=True)
    return _learner_instance
