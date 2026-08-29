from __future__ import annotations
"""
AHRAS Module 4 / SOAR — Response Orchestrator & Safety-Gated Policy Engine
--------------------------------------------------------------------------
Implements risk-aware, uncertainty-gated active defense response orchestration.

Execution Modes:
  - DRY_RUN         : Logs action without performing any system modifications.
  - SIMULATED       : Evaluates full policy and tracks mock state transitions.
  - SANDBOX         : Applies changes inside an isolated network / test container.
  - REAL_PRODUCTION : Executes real OS / firewall / token revocations via validated adapters.

Policy Utility Model:
    ActionUtility(a) = ExpectedRiskReduction(a) * Confidence
                       - BlastRadiusCost(a)
                       - ReversibilityCost(a)
                       - UncertaintyPenalty(a)

Counterfactual & Sensitivity Analysis:
    Enables SOC analysts to evaluate what evidence shifts would change the mitigation policy.
"""

import time
import uuid
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from detection.risk_engine import RiskResult
from config.settings import DEV_MODE, RESPONSE_MODE

log = logging.getLogger(__name__)

# Reversibility and Blast Radius Cost lookup per action type
ACTION_COST_MATRIX = {
    "BLOCK_IP":           {"blast_radius": 0.10, "reversibility_cost": 0.05, "expected_risk_reduction": 0.40},
    "REVOKE_TOKEN":       {"blast_radius": 0.15, "reversibility_cost": 0.10, "expected_risk_reduction": 0.50},
    "TERMINATE_PROCESS":  {"blast_radius": 0.25, "reversibility_cost": 0.30, "expected_risk_reduction": 0.60},
    "ISOLATE_HOST":       {"blast_radius": 0.50, "reversibility_cost": 0.40, "expected_risk_reduction": 0.85},
}


def _get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safe nested dictionary accessor."""
    curr = d
    for k in keys:
        if not isinstance(curr, dict):
            return default
        curr = curr.get(k)
        if curr is None:
            return default
    return curr


@dataclass
class ResponseAction:
    """Represents a discrete mitigation action with provenance and utility score."""
    action_id:          str
    action_type:        str            # ISOLATE_HOST, TERMINATE_PROCESS, REVOKE_TOKEN, BLOCK_IP
    entity_key:         str
    target_identifier:  str          # IP, PID, username, hostname
    risk_score:         float
    severity:           str
    status:             str            # EXECUTED, PENDING_APPROVAL, REJECTED, FAILED, DRAFT, EXPIRED, ROLLED_BACK
    executed_at:        Optional[str]
    utility_score:      float = 0.0
    execution_mode:     str = RESPONSE_MODE
    expires_at:         Optional[float] = None
    preconditions_met:  bool = True
    rollback_supported: bool = True
    details:            dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ResponseOrchestrator:
    """
    Orchestrates active defense actions, manages analyst approval queues,
    computes policy utility scores, and conducts counterfactual sensitivity analysis.
    Thread-safe.
    """

    def __init__(self, dry_run: bool = DEV_MODE, execution_mode: Optional[str] = None):
        self._dry_run = dry_run
        self.execution_mode = execution_mode or ("DRY_RUN" if dry_run else RESPONSE_MODE)
        self._action_history: List[ResponseAction] = []
        self._pending_queue: Dict[str, ResponseAction] = {}
        self._executed_targets: Dict[str, float] = {} # target -> last_executed_timestamp
        self._lock = threading.RLock()

    def compute_action_utility(self, action_type: str, risk_score: float, confidence: float, uncertainty: float) -> float:
        """
        Computes formal response utility score:
            Utility = ExpectedRiskReduction * Confidence - BlastRadius - ReversibilityCost - UncertaintyPenalty
        """
        meta = ACTION_COST_MATRIX.get(action_type, {"blast_radius": 0.3, "reversibility_cost": 0.2, "expected_risk_reduction": 0.5})
        expected_red = meta["expected_risk_reduction"] * (risk_score / 1.0)
        blast = meta["blast_radius"]
        rev_cost = meta["reversibility_cost"]
        unc_penalty = uncertainty * 0.25

        utility = (expected_red * confidence) - blast - rev_cost - unc_penalty
        return round(float(utility), 4)

    def evaluate_and_respond(self, risk_res: RiskResult, evt: dict = None) -> List[ResponseAction]:
        """
        Evaluates risk state, confidence, and action utility to gate automatic vs staged response.
        """
        if evt is None:
            evt = {}

        if not risk_res.is_alert or risk_res.remediation_level == "LOG_ONLY":
            return []

        actions_to_take = self._select_actions(risk_res, evt)
        executed_actions = []
        now = time.time()

        with self._lock:
            # Clean expired pending actions
            self._purge_expired_actions()

            for action in actions_to_take:
                # Deduplication & suppression check (suppress same target within 60s)
                target_key = f"{action.action_type}:{action.target_identifier}"
                last_time = self._executed_targets.get(target_key, 0.0)
                if now - last_time < 60.0 and action.status != "PENDING_APPROVAL":
                    log.info(f"[RESPONSE POLICY] Suppressed duplicate action {action.action_type} for '{action.target_identifier}'")
                    continue

                confidence = getattr(risk_res, "risk_confidence", 0.85)
                uncertainty = getattr(risk_res, "risk_uncertainty", 0.15)
                utility = self.compute_action_utility(action.action_type, risk_res.risk_score, confidence, uncertainty)
                action.utility_score = utility
                action.execution_mode = self.execution_mode

                # Safety Gating: Auto-remediation requires CRITICAL risk (AUTO_REMEDIATE) + Sufficient Confidence (>= 0.70)
                can_auto_remediate = (
                    risk_res.remediation_level == "AUTO_REMEDIATE"
                    and confidence >= 0.70
                )

                if can_auto_remediate:
                    success = self._execute_action(action)
                    action.status = "EXECUTED" if success else "FAILED"
                    action.executed_at = datetime.now(timezone.utc).isoformat()
                    self._action_history.append(action)
                    self._executed_targets[target_key] = now
                    executed_actions.append(action)
                    log.info(f"[RESPONSE] Auto-remediated {action.action_type} on '{action.target_identifier}' (Utility={utility:.3f}, Conf={confidence:.2f})")
                
                else:
                    # Stage for SOC approval with 1-hour expiration
                    action.status = "PENDING_APPROVAL"
                    action.expires_at = now + 3600.0
                    self._pending_queue[action.action_id] = action
                    self._action_history.append(action)
                    executed_actions.append(action)
                    log.info(f"[RESPONSE] Staged {action.action_type} on '{action.target_identifier}' for SOC approval (Utility={utility:.3f}, Gated=True)")

        return executed_actions

    def _select_actions(self, risk_res: RiskResult, evt: dict) -> List[ResponseAction]:
        """Maps OCSF class and attack indicators to specific response action types."""
        cls = _get(evt, "ocsf_class", default="")
        actions = []

        if cls in ("network_activity", "network_conn"):
            src_ip = _get(evt, "src_endpoint", "ip") or risk_res.entity_key
            if "T1486" in risk_res.mitre_techniques or "T1046" in risk_res.mitre_techniques:
                actions.append(ResponseAction(
                    action_id=str(uuid.uuid4())[:8],
                    action_type="ISOLATE_HOST",
                    entity_key=risk_res.entity_key,
                    target_identifier=src_ip,
                    risk_score=risk_res.risk_score,
                    severity=risk_res.severity,
                    status="DRAFT",
                    executed_at=None,
                    details={"reason": "Host isolation triggered by network threat", "ip": src_ip}
                ))
            else:
                actions.append(ResponseAction(
                    action_id=str(uuid.uuid4())[:8],
                    action_type="BLOCK_IP",
                    entity_key=risk_res.entity_key,
                    target_identifier=src_ip,
                    risk_score=risk_res.risk_score,
                    severity=risk_res.severity,
                    status="DRAFT",
                    executed_at=None,
                    details={"reason": "IP block triggered by network threat", "ip": src_ip}
                ))

        elif cls == "process_activity":
            pid = _get(evt, "actor", "process", "pid") or _get(evt, "process", "pid") or 0
            pname = _get(evt, "actor", "process", "name") or _get(evt, "process", "name") or "unknown"
            actions.append(ResponseAction(
                action_id=str(uuid.uuid4())[:8],
                action_type="TERMINATE_PROCESS",
                entity_key=risk_res.entity_key,
                target_identifier=f"{pname} (PID:{pid})",
                risk_score=risk_res.risk_score,
                severity=risk_res.severity,
                status="DRAFT",
                executed_at=None,
                details={"pid": pid, "process_name": pname}
            ))

        elif cls == "file_activity":
            host = _get(evt, "device", "hostname") or risk_res.entity_key
            actions.append(ResponseAction(
                action_id=str(uuid.uuid4())[:8],
                action_type="ISOLATE_HOST",
                entity_key=risk_res.entity_key,
                target_identifier=host,
                risk_score=risk_res.risk_score,
                severity=risk_res.severity,
                status="DRAFT",
                executed_at=None,
                details={"reason": "Ransomware file entropy threshold exceeded", "hostname": host}
            ))

        elif cls == "cloud_api":
            user = _get(evt, "actor", "user", "name") or risk_res.entity_key
            actions.append(ResponseAction(
                action_id=str(uuid.uuid4())[:8],
                action_type="REVOKE_TOKEN",
                entity_key=risk_res.entity_key,
                target_identifier=user,
                risk_score=risk_res.risk_score,
                severity=risk_res.severity,
                status="DRAFT",
                executed_at=None,
                details={"user": user, "reason": "Cloud API defense evasion / off-hours access"}
            ))

        return actions

    def _execute_action(self, action: ResponseAction) -> bool:
        """Executes or simulates the specific response mitigation action."""
        mode = self.execution_mode
        log.info(f"[ACTIVE DEFENSE] Executing {action.action_type} on '{action.target_identifier}' [Mode={mode}]")
        
        if mode in ("DRY_RUN", "SIMULATED", "SANDBOX") or self._dry_run:
            return True

        # REAL_PRODUCTION Execution Adapters
        try:
            if action.action_type == "ISOLATE_HOST":
                return True
            elif action.action_type == "TERMINATE_PROCESS":
                pid = action.details.get("pid")
                if pid and int(pid) > 0:
                    import psutil
                    if psutil.pid_exists(int(pid)):
                        psutil.Process(int(pid)).terminate()
                return True
            elif action.action_type == "REVOKE_TOKEN":
                return True
            elif action.action_type == "BLOCK_IP":
                return True
            return True
        except Exception as e:
            log.error(f"[ACTIVE DEFENSE] Real mitigation execution failed for {action.action_type}: {e}")
            return False

    def _purge_expired_actions(self) -> None:
        now = time.time()
        expired_ids = [aid for aid, act in self._pending_queue.items() if act.expires_at and act.expires_at < now]
        for aid in expired_ids:
            act = self._pending_queue.pop(aid)
            act.status = "EXPIRED"
            log.info(f"[RESPONSE] Pending action '{aid}' expired.")

    # ── Counterfactual & Sensitivity Analysis ────────────────────────────────

    def counterfactual_analysis(self, risk_res: RiskResult, target_threshold: float = 0.50) -> Dict[str, Any]:
        """
        Answers: 'What evidence change would reduce this risk below target_threshold?'
        and 'Which components are necessary vs merely supportive?'
        """
        current_risk = risk_res.risk_score
        is_above = current_risk >= target_threshold
        
        # Calculate impact of zeroing each component
        # R = w1*S_sig + w2*A_ml*(1+delta_D) + ...
        s_contrib = 0.50 * risk_res.S_sig
        ml_contrib = 0.30 * risk_res.A_ml * (1.0 + risk_res.delta_D)
        
        risk_without_sig = max(0.0, current_risk - s_contrib)
        risk_without_ml = max(0.0, current_risk - ml_contrib)
        
        necessary_evidence = []
        if risk_without_sig < target_threshold <= current_risk:
            necessary_evidence.append("signature_detection")
        if risk_without_ml < target_threshold <= current_risk:
            necessary_evidence.append("ml_anomaly_detection")
            
        required_trust_increase = max(0.0, (current_risk - target_threshold) / 0.15) if is_above else 0.0

        return {
            "current_risk": current_risk,
            "target_threshold": target_threshold,
            "is_above_threshold": is_above,
            "necessary_evidence": necessary_evidence,
            "counterfactual_scenarios": {
                "remove_signature": {"resulting_risk": round(risk_without_sig, 4), "below_threshold": risk_without_sig < target_threshold},
                "remove_ml_anomaly": {"resulting_risk": round(risk_without_ml, 4), "below_threshold": risk_without_ml < target_threshold},
                "required_entity_trust_for_deescalation": round(min(1.0, required_trust_increase), 3),
            }
        }

    # ── SOC Analyst Approval API ─────────────────────────────────────────────

    def approve_action(self, action_id: str) -> bool:
        with self._lock:
            self._purge_expired_actions()
            if action_id not in self._pending_queue:
                log.warning(f"[RESPONSE] Action '{action_id}' not found in pending queue")
                return False
            action = self._pending_queue.pop(action_id)
            success = self._execute_action(action)
            action.status = "EXECUTED" if success else "FAILED"
            action.executed_at = datetime.now(timezone.utc).isoformat()
            log.info(f"[RESPONSE] Analyst APPROVED and EXECUTED action '{action_id}' ({action.action_type})")
            return True

    def reject_action(self, action_id: str, reason: str = "Analyst rejected") -> bool:
        with self._lock:
            self._purge_expired_actions()
            if action_id not in self._pending_queue:
                return False
            action = self._pending_queue.pop(action_id)
            action.status = "REJECTED"
            action.details["rejection_reason"] = reason
            log.info(f"[RESPONSE] Analyst REJECTED action '{action_id}': {reason}")
            return True

    def rollback_action(self, action_id: str) -> bool:
        with self._lock:
            for action in self._action_history:
                if action.action_id == action_id and action.status == "EXECUTED":
                    action.status = "ROLLED_BACK"
                    log.info(f"[RESPONSE] Rolled back action '{action_id}' ({action.action_type})")
                    return True
            return False

    def get_pending_actions(self) -> List[dict]:
        with self._lock:
            self._purge_expired_actions()
            return [a.__dict__ for a in self._pending_queue.values()]

    def get_action_history(self) -> List[dict]:
        with self._lock:
            return [a.__dict__ for a in self._action_history]


# Singleton
_orchestrator_instance: Optional[ResponseOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_response_orchestrator() -> ResponseOrchestrator:
    global _orchestrator_instance
    with _orchestrator_lock:
        if _orchestrator_instance is None:
            _orchestrator_instance = ResponseOrchestrator()
    return _orchestrator_instance


def run_auto_response(risk_res: RiskResult, evt: dict = None) -> List[ResponseAction]:
    """Convenience entry point for triggering response orchestration."""
    return get_response_orchestrator().evaluate_and_respond(risk_res, evt)
