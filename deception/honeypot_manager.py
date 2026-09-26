from __future__ import annotations
"""
AHRAS Module 13 — Active Deception & Dynamic Information-Theoretic Honeypot Engine
---------------------------------------------------------------------------------
Elevates honeypots and honey-tokens from static tripwires into active Bayesian
information-gathering sensors (Extension H).

Mathematical Optimization:
  DeceptionValue(L) = ExpectedInformationGain(L) - DeploymentCost(L) - OperationalRisk(L)

Where:
  ExpectedInformationGain = Uncertainty * (0.40 * Risk + 0.30 * AttackPathImportance + 0.30 * AssetCriticality) * ContextRelevance
  DeploymentCost: Resource & maintenance overhead per lure type
  OperationalRisk: Potential collateral disruption or benign scanner attraction risk

Dynamic Lure Types:
  - HONEY_TOKEN: Cloud IAM canary tokens, AWS fake credentials (T1078.004)
  - FAKE_PORT: High-interaction emulated listening port (T1046)
  - DECOY_FILE: High-entropy ransomware bait file / DB dump (T1083)
  - CANARY_CREDENTIAL: Shadow password file or memory credential bait (T1552)

Deception Feedback Loop:
  On lure trigger:
    1. Emit high-confidence EvidenceRecord (confidence=0.99, uncertainty=0.01)
    2. Attach event/entity metadata
    3. Update security graph with confirmed adversary edge
    4. Update incident status
    5. Elevate entity risk to critical (R >= 0.95) with collapsed uncertainty
    6. Confirm attack-chain confidence (TTP ground-truth confirmation)
"""

import time
import uuid
import math
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from config.settings import USE_ADAPTIVE_DECEPTION, DECEPTION_VALUE_THRESHOLD
from ahras.evidence.models import EvidenceRecord, EvidenceType, EvidenceSource
from ahras.evidence.ledger import get_evidence_ledger

log = logging.getLogger(__name__)

LURE_MITRE_MAP = {
    "HONEY_TOKEN":       "T1078.004",  # Cloud Accounts / Canary Credentials
    "FAKE_PORT":         "T1046",      # Network Service Discovery
    "DECOY_FILE":        "T1083",      # File and Directory Discovery
    "CANARY_CREDENTIAL": "T1552",      # Unsecured Credentials
}

LURE_PROFILES = {
    "HONEY_TOKEN": {
        "deployment_cost": 0.04,
        "operational_risk": 0.02,
        "mitre_technique": "T1078.004",
        "preferred_contexts": ["cloud_api", "identity", "credential_access"],
    },
    "FAKE_PORT": {
        "deployment_cost": 0.14,
        "operational_risk": 0.12,
        "mitre_technique": "T1046",
        "preferred_contexts": ["network_activity", "reconnaissance", "lateral_movement"],
    },
    "DECOY_FILE": {
        "deployment_cost": 0.08,
        "operational_risk": 0.05,
        "mitre_technique": "T1083",
        "preferred_contexts": ["file_activity", "ransomware", "collection"],
    },
    "CANARY_CREDENTIAL": {
        "deployment_cost": 0.06,
        "operational_risk": 0.03,
        "mitre_technique": "T1552",
        "preferred_contexts": ["process_activity", "credential_access", "privilege_escalation"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeceptionLure:
    """Represents a dynamic honeypot lure token or endpoint."""
    lure_id:          str
    lure_type:        str           # HONEY_TOKEN, FAKE_PORT, DECOY_FILE, CANARY_CREDENTIAL
    target_entity:    str
    lure_key:         str           # Fake AWS key, decoy file path, port number
    deployed_at:      float
    mitre_technique:  str = "T1078"
    is_triggered:     bool = False
    triggered_at:     Optional[float] = None
    attacker_ip:      Optional[str] = None
    expected_ig:      float = 0.0
    deception_value:  float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeceptionUtilityEvaluation:
    """Evaluation of expected information gain vs deployment cost for a candidate lure."""
    entity_key: str
    lure_type: str
    expected_information_gain: float
    deployment_cost: float
    operational_risk: float
    deception_value: float
    is_recommended: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeceptionTriggerFeedback:
    """Structured feedback produced when an adversary interacts with a deployed lure."""
    lure_id: str
    lure_type: str
    attacker_entity: str
    evidence_id: str
    pre_trigger_risk: float
    post_trigger_risk: float
    pre_trigger_uncertainty: float
    post_trigger_uncertainty: float
    uncertainty_reduction: float
    attack_chain_confirmed: bool
    graph_update: Dict[str, Any]
    incident_update: Dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Deception Manager
# ─────────────────────────────────────────────────────────────────────────────

class DeceptionManager:
    """
    Manages Bayesian utility-driven deployment, monitoring, and forensic feedback
    of dynamic honeypots and canary sensors. Thread-safe.
    """

    def __init__(self):
        self._active_lures: Dict[str, DeceptionLure] = {}
        self._triggered_log: List[DeceptionLure] = []
        self._feedback_log: List[DeceptionTriggerFeedback] = []
        self._entity_states: Dict[str, Dict[str, float]] = {}  # entity -> {"risk": R, "uncertainty": U}
        self._lock = threading.RLock()

    # ── Utility Formulation & Dynamic Lure Selection ─────────────────────────

    def evaluate_deception_utility(
        self,
        entity_key: str,
        risk_score: float,
        uncertainty: float = 0.50,
        attack_path_importance: float = 0.50,
        asset_criticality: float = 0.50,
        candidate_lure_type: str = "HONEY_TOKEN",
        context: Optional[str] = None,
    ) -> DeceptionUtilityEvaluation:
        """
        Calculates DeceptionValue = ExpectedInformationGain - DeploymentCost - OperationalRisk.
        """
        profile = LURE_PROFILES.get(candidate_lure_type, LURE_PROFILES["HONEY_TOKEN"])
        cost = profile["deployment_cost"]
        op_risk = profile["operational_risk"]

        # Context relevance multiplier (1.3x if matching observed activity, 0.7x if misaligned)
        ctx = (context or "").lower()
        pref_contexts = profile["preferred_contexts"]
        if any(p in ctx for p in pref_contexts):
            context_mult = 1.30
        elif ctx:
            context_mult = 0.70
        else:
            context_mult = 1.00

        # Information Gain is high when:
        # 1. Epistemic uncertainty is high (tripwire provides decisive clarity)
        # 2. Risk and attack-path importance are significant
        # 3. Target asset has high enterprise criticality
        base_threat = (
            0.40 * float(np.clip(risk_score, 0.0, 1.0))
            + 0.30 * float(np.clip(attack_path_importance, 0.0, 1.0))
            + 0.30 * float(np.clip(asset_criticality, 0.0, 1.0))
        )
        expected_ig = float(np.clip(uncertainty * base_threat * context_mult, 0.0, 1.0))

        # DeceptionValue calculation
        deception_value = round(expected_ig - cost - op_risk, 4)
        is_recommended = bool(deception_value >= DECEPTION_VALUE_THRESHOLD)

        reason = (
            f"DeceptionValue={deception_value:.3f} >= {DECEPTION_VALUE_THRESHOLD:.2f} "
            f"(E[IG]={expected_ig:.3f}, Cost={cost:.2f}, OpRisk={op_risk:.2f})"
            if is_recommended
            else f"DeceptionValue={deception_value:.3f} below threshold {DECEPTION_VALUE_THRESHOLD:.2f}"
        )

        return DeceptionUtilityEvaluation(
            entity_key=entity_key,
            lure_type=candidate_lure_type,
            expected_information_gain=round(expected_ig, 4),
            deployment_cost=cost,
            operational_risk=op_risk,
            deception_value=deception_value,
            is_recommended=is_recommended,
            reason=reason,
        )

    def select_optimal_lure(
        self,
        entity_key: str,
        risk_score: float,
        uncertainty: float = 0.50,
        attack_path_importance: float = 0.50,
        asset_criticality: float = 0.50,
        context: Optional[str] = None,
    ) -> Optional[DeceptionUtilityEvaluation]:
        """
        Evaluates all candidate lure types and selects the lure maximizing DeceptionValue.
        Returns None if no lure achieves DeceptionValue >= threshold.
        """
        evaluations = [
            self.evaluate_deception_utility(
                entity_key=entity_key,
                risk_score=risk_score,
                uncertainty=uncertainty,
                attack_path_importance=attack_path_importance,
                asset_criticality=asset_criticality,
                candidate_lure_type=lure_type,
                context=context,
            )
            for lure_type in LURE_PROFILES.keys()
        ]

        # Filter recommended lures
        recommended = [e for e in evaluations if e.is_recommended]
        if not recommended:
            return None

        # Select lure with maximum net DeceptionValue
        optimal = max(recommended, key=lambda e: e.deception_value)
        return optimal

    # ── Lure Deployment ──────────────────────────────────────────────────────

    def deploy_optimal_lure(
        self,
        entity_key: str,
        risk_score: float,
        uncertainty: float = 0.50,
        attack_path_importance: float = 0.50,
        asset_criticality: float = 0.50,
        context: Optional[str] = None,
    ) -> Optional[DeceptionLure]:
        """
        Dynamically selects and deploys the information-theoretically optimal lure.
        """
        eval_result = self.select_optimal_lure(
            entity_key=entity_key,
            risk_score=risk_score,
            uncertainty=uncertainty,
            attack_path_importance=attack_path_importance,
            asset_criticality=asset_criticality,
            context=context,
        )
        if eval_result is None:
            log.debug(f"[DECEPTION] No lure meets utility threshold for '{entity_key}'")
            return None

        return self._deploy_concrete_lure(
            entity_key=entity_key,
            lure_type=eval_result.lure_type,
            expected_ig=eval_result.expected_information_gain,
            deception_value=eval_result.deception_value,
        )

    def deploy_lure_for_entity(
        self,
        entity_key: str,
        risk_score: float,
        lure_type: str = "HONEY_TOKEN",
    ) -> Optional[DeceptionLure]:
        """
        Backward-compatible deployment interface.
        If USE_ADAPTIVE_DECEPTION is True, verifies non-negative utility before deploying.
        """
        if USE_ADAPTIVE_DECEPTION:
            eval_res = self.evaluate_deception_utility(
                entity_key=entity_key,
                risk_score=risk_score,
                uncertainty=0.45,
                attack_path_importance=0.60,
                asset_criticality=0.60,
                candidate_lure_type=lure_type,
            )
            if not eval_res.is_recommended and risk_score < 0.70:
                return None
            return self._deploy_concrete_lure(
                entity_key, lure_type, eval_res.expected_information_gain, eval_res.deception_value
            )

        # Legacy static threshold fallback
        if risk_score < 0.70:
            return None
        return self._deploy_concrete_lure(entity_key, lure_type, 0.50, 0.35)

    def _deploy_concrete_lure(
        self,
        entity_key: str,
        lure_type: str,
        expected_ig: float,
        deception_value: float,
    ) -> DeceptionLure:
        with self._lock:
            lure_id = f"LURE-{str(uuid.uuid4())[:8]}"
            if lure_type == "HONEY_TOKEN":
                lkey = f"AKIAIOSFODNN7EXAMPLE-{lure_id}"
            elif lure_type == "FAKE_PORT":
                lkey = f"port:8443-{lure_id}"
            elif lure_type == "DECOY_FILE":
                lkey = f"/var/secrets/canary_db_backup_{lure_id}.kdbx"
            else:  # CANARY_CREDENTIAL
                lkey = f"cred:admin_backup_{lure_id}"

            mtech = LURE_MITRE_MAP.get(lure_type, "T1078")

            lure = DeceptionLure(
                lure_id=lure_id,
                lure_type=lure_type,
                target_entity=entity_key,
                lure_key=lkey,
                deployed_at=time.time(),
                mitre_technique=mtech,
                expected_ig=expected_ig,
                deception_value=deception_value,
            )
            self._active_lures[lure.lure_key] = lure
            self._entity_states[entity_key] = {"risk": 0.65, "uncertainty": 0.50}
            log.info(
                f"[DECEPTION] Deployed optimal {lure_type} for '{entity_key}' "
                f"(lure_id={lure_id}, E[IG]={expected_ig:.2f}, Value={deception_value:.2f})"
            )
            return lure

    # ── Interaction & Deception Feedback Loop ────────────────────────────────

    def check_interaction(
        self, accessed_key: str, attacker_ip: Optional[str] = None
    ) -> Optional[DeceptionLure]:
        """Backward-compatible interaction check wrapping the full feedback loop."""
        feedback = self.handle_lure_trigger(accessed_key, attacker_ip)
        if feedback is not None:
            with self._lock:
                return self._active_lures.get(accessed_key)
        return None

    def handle_lure_trigger(
        self,
        accessed_key: str,
        attacker_ip: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[DeceptionTriggerFeedback]:
        """
        Executes the formal 6-stage Deception Feedback Loop (Section 9.2):
          1. Create high-confidence EvidenceRecord
          2. Attach event/entity
          3. Update security graph
          4. Update incident status
          5. Update risk & collapse uncertainty
          6. Update attack-chain confidence
        """
        with self._lock:
            if accessed_key not in self._active_lures:
                return None

            lure = self._active_lures[accessed_key]
            now = time.time()
            lure.is_triggered = True
            lure.triggered_at = now
            lure.attacker_ip = attacker_ip
            self._triggered_log.append(lure)

            target_entity = attacker_ip or lure.target_entity
            pre_state = self._entity_states.get(target_entity, {"risk": 0.65, "uncertainty": 0.50})
            pre_risk = pre_state["risk"]
            pre_unc = pre_state["uncertainty"]

            # 1. Create high-confidence EvidenceRecord
            evidence_id = f"EVT-DECEPT-{uuid.uuid4().hex[:8]}"
            ev = EvidenceRecord(
                event_id=evidence_id,
                entity_id=target_entity,
                source=EvidenceSource.DYNAMIC_HONEYPOT.value,
                detector_type=EvidenceType.DECEPTION.value,
                raw_score=1.0,
                normalized_score=1.0,
                confidence=0.99,
                uncertainty=0.01,
                mitre_mapping=[lure.mitre_technique],
                explanation=(
                    f"High-fidelity honeypot tripwire hit on {lure.lure_type} ({lure.lure_id}). "
                    f"Confirmed unauthorized adversary interaction."
                ),
            )
            try:
                get_evidence_ledger().record_evidence(ev)
            except Exception as e:
                log.warning(f"[DECEPTION] Could not write to evidence ledger: {e}")

            # 3. Update security graph (compromised entity node + tripwire edge)
            graph_update = {
                "action": "ADD_COMPROMISE_EDGE",
                "source_node": target_entity,
                "target_node": lure.lure_id,
                "edge_type": "INTERACTS_WITH_HONEYPOT",
                "adversary_confirmed": True,
                "mitre_technique": lure.mitre_technique,
            }

            # 4. Update incident
            incident_update = {
                "incident_id": f"INC-{uuid.uuid4().hex[:6]}",
                "severity": "CRITICAL",
                "attack_stage": "CREDENTIAL_ACCESS_CONFIRMED",
                "analyst_escalation": False,  # Autonomously confirmed, no manual triage needed
            }

            # 5. Update risk & collapse uncertainty
            post_risk = 0.98
            post_unc = 0.01
            uncertainty_reduction = round(pre_unc - post_unc, 4)
            self._entity_states[target_entity] = {"risk": post_risk, "uncertainty": post_unc}

            # 6. Update attack-chain confidence
            attack_chain_confirmed = True

            feedback = DeceptionTriggerFeedback(
                lure_id=lure.lure_id,
                lure_type=lure.lure_type,
                attacker_entity=target_entity,
                evidence_id=evidence_id,
                pre_trigger_risk=pre_risk,
                post_trigger_risk=post_risk,
                pre_trigger_uncertainty=pre_unc,
                post_trigger_uncertainty=post_unc,
                uncertainty_reduction=uncertainty_reduction,
                attack_chain_confirmed=attack_chain_confirmed,
                graph_update=graph_update,
                incident_update=incident_update,
                timestamp=now,
            )
            self._feedback_log.append(feedback)

            log.critical(
                f"[DECEPTION ALERT] CONFIRMED ADVERSARY ENGAGEMENT! "
                f"Lure '{lure.lure_id}' ({lure.lure_type}) triggered by {target_entity}. "
                f"Uncertainty collapsed: {pre_unc:.2f} -> {post_unc:.2f}, Risk: {pre_risk:.2f} -> {post_risk:.2f}"
            )
            return feedback

    # ── Inspection & Telemetry ───────────────────────────────────────────────

    def get_active_lures(self) -> List[dict]:
        with self._lock:
            return [l.to_dict() for l in self._active_lures.values()]

    def get_triggered_lures(self) -> List[dict]:
        with self._lock:
            return [l.to_dict() for l in self._triggered_log]

    def get_feedback_log(self) -> List[dict]:
        with self._lock:
            return [f.to_dict() for f in self._feedback_log]


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Accessor
# ─────────────────────────────────────────────────────────────────────────────

_deception_instance: Optional[DeceptionManager] = None
_deception_lock = threading.Lock()


def get_deception_manager() -> DeceptionManager:
    """Returns thread-safe singleton instance of DeceptionManager."""
    global _deception_instance
    with _deception_lock:
        if _deception_instance is None:
            _deception_instance = DeceptionManager()
    return _deception_instance
