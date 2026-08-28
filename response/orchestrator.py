from __future__ import annotations
"""
AHRAS Response Orchestrator & Active Defense System
--------------------------------------------------
Automates threat mitigation and active defense response based on risk evaluation
and MITRE ATT&CK mitigation mappings.

Supported Active Defense Actions:
  - ISOLATE_HOST       : Network interface isolation / iptables firewall block (T1046, T1486)
  - TERMINATE_PROCESS  : Process PID kill / process termination (T1059, T1003)
  - REVOKE_TOKEN       : Cloud API session token revocation / user lock (T1078, T1562)
  - BLOCK_IP           : Network ACL entry / IP blacklist insertion (T1071, T1571)

Policy Execution Matrix:
  - CRITICAL (R_t >= 0.90) : AUTO_REMEDIATE (Immediate execution)
  - HIGH (0.70 <= R_t < 0.90) : STAGE_APPROVAL (Queued for SOC Analyst confirmation)
  - MEDIUM (0.50 <= R_t < 0.70) : STAGE_APPROVAL (Queued for SOC Analyst confirmation)
  - LOW / INFO (R_t < 0.50)   : LOG_ONLY (Recorded in audit log, no action taken)
"""

import time
import uuid
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from detection.risk_engine import RiskResult
from config.settings import DEV_MODE

log = logging.getLogger(__name__)


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
    """Represents a discrete mitigation action."""
    action_id:       str
    action_type:     str            # ISOLATE_HOST, TERMINATE_PROCESS, REVOKE_TOKEN, BLOCK_IP
    entity_key:      str
    target_identifier: str          # IP, PID, username, hostname
    risk_score:      float
    severity:        str
    status:          str            # EXECUTED, PENDING_APPROVAL, REJECTED, FAILED
    executed_at:     Optional[str]
    details:         dict = field(default_factory=dict)


class ResponseOrchestrator:
    """
    Orchestrates active defense actions, manages analyst approval queues,
    and maintains an audit log of all response executions. Thread-safe.
    """

    def __init__(self, dry_run: bool = DEV_MODE):
        self._dry_run = dry_run
        self._action_history: List[ResponseAction] = []
        self._pending_queue: Dict[str, ResponseAction] = {}
        self._lock = threading.RLock()

    def evaluate_and_respond(self, risk_res: RiskResult, evt: dict = None) -> List[ResponseAction]:
        """
        Determines necessary mitigation actions from RiskResult & event details,
        executing immediately if CRITICAL or queuing if HIGH/MEDIUM.
        """
        if evt is None:
            evt = {}

        if not risk_res.is_alert or risk_res.remediation_level == "LOG_ONLY":
            return []

        actions_to_take = self._select_actions(risk_res, evt)
        executed_actions = []

        with self._lock:
            for action in actions_to_take:
                if risk_res.remediation_level == "AUTO_REMEDIATE":
                    success = self._execute_action(action)
                    action.status = "EXECUTED" if success else "FAILED"
                    action.executed_at = datetime.now(timezone.utc).isoformat()
                    self._action_history.append(action)
                    executed_actions.append(action)
                    log.info(f"[RESPONSE] Auto-remediated {action.action_type} on '{action.target_identifier}'")
                
                elif risk_res.remediation_level in ("STAGE_APPROVAL", "SOC_ALERT_HIGH"):
                    action.status = "PENDING_APPROVAL"
                    self._pending_queue[action.action_id] = action
                    self._action_history.append(action)
                    executed_actions.append(action)
                    log.info(f"[RESPONSE] Staged {action.action_type} on '{action.target_identifier}' for SOC approval")

        return executed_actions

    def _select_actions(self, risk_res: RiskResult, evt: dict) -> List[ResponseAction]:
        """Maps OCSF class and attack indicators to specific response action types."""
        cls = _get(evt, "ocsf_class", default="")
        actions = []

        # 1. Network Activity / Network Conn -> ISOLATE_HOST or BLOCK_IP
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

        # 2. Process Activity -> TERMINATE_PROCESS
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

        # 3. File Activity (Ransomware) -> ISOLATE_HOST & TERMINATE_PROCESS
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

        # 4. Cloud API -> REVOKE_TOKEN
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
        """Executes the specific response mitigation action (or simulates in dry-run mode)."""
        log.info(f"[ACTIVE DEFENSE] Executing {action.action_type} on target '{action.target_identifier}' (Dry-Run={self._dry_run})")
        if self._dry_run:
            return True

        try:
            if action.action_type == "ISOLATE_HOST":
                # Simulated firewall rule insertion
                return True
            elif action.action_type == "TERMINATE_PROCESS":
                pid = action.details.get("pid")
                if pid and pid > 0:
                    import psutil
                    if psutil.pid_exists(pid):
                        psutil.Process(pid).terminate()
                return True
            elif action.action_type == "REVOKE_TOKEN":
                # Simulated OAuth/IAM token revocation
                return True
            elif action.action_type == "BLOCK_IP":
                # Simulated ACL block entry
                return True
            return True
        except Exception as e:
            log.error(f"[ACTIVE DEFENSE] Action {action.action_type} failed: {e}")
            return False

    # ── SOC Analyst Approval API ─────────────────────────────────────────────

    def approve_action(self, action_id: str) -> bool:
        """Approves and executes a staged response action from the pending queue."""
        with self._lock:
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
        """Rejects a staged response action."""
        with self._lock:
            if action_id not in self._pending_queue:
                return False
            action = self._pending_queue.pop(action_id)
            action.status = "REJECTED"
            action.details["rejection_reason"] = reason
            log.info(f"[RESPONSE] Analyst REJECTED action '{action_id}': {reason}")
            return True

    def get_pending_actions(self) -> List[dict]:
        with self._lock:
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
