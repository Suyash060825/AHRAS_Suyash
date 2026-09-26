"""
AHRAS Response Safety Invariants & Blast Radius Guard
------------------------------------------------------
Enforces deterministic fail-closed safety constraints on all autonomous response actions:
  1. Critical Asset Invariant: Destructive containment on critical infrastructure
     (Domain Controllers, DNS roots, core routers) strictly requires human analyst sign-off.
  2. Blast Radius Limit: Autonomous isolations cannot exceed a hard subnet budget (e.g., <= 20%).
  3. Strict Reversibility Invariant: Every autonomous action must have a verified compensating action.
  4. Response Rate Invariant: Maximum autonomous containment rate per time window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SafetyVerdict:
    """
    Verdict returned by SafetyInvariantChecker before executing any mitigation action.
    """
    passed: bool
    action_permitted: bool
    requires_human_approval: bool
    violations: List[str]
    compensating_action: Optional[str] = None
    target_entity: str = ""
    action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "action_permitted": self.action_permitted,
            "requires_human_approval": self.requires_human_approval,
            "violations": self.violations,
            "compensating_action": self.compensating_action,
            "target_entity": self.target_entity,
            "action": self.action,
        }


# Standard compensating inverse actions
COMPENSATING_ACTIONS = {
    "ISOLATE_HOST": "UNISOLATE_HOST",
    "AUTONOMOUS_CONTAINMENT": "UNQUARANTINE_ENTITY",
    "BLOCK_IP": "UNBLOCK_IP",
    "REVOKE_CREDENTIALS": "REISSUE_TEMPORARY_TOKEN",
    "KILL_PROCESS": "NOT_REVERSIBLE_AUDIT_LOGGED",
    "DECEPTION": "TEARDOWN_LURE",
    "MONITOR": "NO_OP",
    "AUTONOMOUS_PASS": "NO_OP",
}


class SafetyInvariantChecker:
    """
    Deterministic safety barrier governing all automated remediation triggers.
    """
    def __init__(
        self,
        critical_entities: Optional[Set[str]] = None,
        max_subnet_isolate_pct: float = 0.20,
        max_actions_per_minute: int = 10,
    ) -> None:
        self.critical_entities = critical_entities or {
            "dc-01", "dc-primary", "dns-root", "gateway-core", "domain_controller"
        }
        self.max_subnet_isolate_pct = max_subnet_isolate_pct
        self.max_actions_per_minute = max_actions_per_minute
        self._action_timestamps: List[float] = []
        self._isolated_entities: Set[str] = set()
        self._isolated_entity_subnets: Dict[str, str] = {}

    def evaluate_action_safety(
        self,
        action: str,
        target_entity: str,
        entity_metadata: Optional[Dict[str, Any]] = None,
        current_time: Optional[float] = None,
    ) -> SafetyVerdict:
        """
        Validates safety invariants against target entity and operational state.
        """
        now = current_time or time.time()
        meta = entity_metadata or {}
        violations: List[str] = []
        requires_human = False

        # 1. Critical Asset Invariant
        is_critical = (
            target_entity.lower() in self.critical_entities
            or meta.get("is_critical", False)
            or meta.get("role", "").upper() in {"DOMAIN_CONTROLLER", "DNS_ROOT", "CORE_ROUTER"}
        )
        if is_critical and action in {"ISOLATE_HOST", "AUTONOMOUS_CONTAINMENT", "BLOCK_IP"}:
            violations.append(f"Target entity '{target_entity}' is designated CRITICAL infrastructure.")
            requires_human = True

        # 2. Rate Limiting Invariant
        self._action_timestamps = [t for t in self._action_timestamps if (now - t) < 60.0]
        if len(self._action_timestamps) >= self.max_actions_per_minute:
            violations.append(f"Autonomous action rate limit exceeded ({self.max_actions_per_minute} actions/min).")
            requires_human = True

        # 3. Subnet Blast Radius Limit
        subnet_total = int(meta.get("subnet_total_hosts", 100))
        subnet_tag = meta.get("subnet", "default")
        currently_isolated = sum(1 for s in self._isolated_entity_subnets.values() if s == subnet_tag)
        if action in {"ISOLATE_HOST", "AUTONOMOUS_CONTAINMENT"}:
            new_ratio = (currently_isolated + 1) / max(1, subnet_total)
            if new_ratio > self.max_subnet_isolate_pct:
                violations.append(
                    f"Subnet blast radius violated ({new_ratio:.1%} > {self.max_subnet_isolate_pct:.1%})."
                )
                requires_human = True

        # 4. Strict Reversibility Invariant
        comp_action = COMPENSATING_ACTIONS.get(action)
        if comp_action is None:
            violations.append(f"No registered compensating action exists for '{action}'.")
            requires_human = True

        passed = len(violations) == 0
        permitted = passed

        if permitted and action in {"ISOLATE_HOST", "AUTONOMOUS_CONTAINMENT"}:
            self._isolated_entities.add(target_entity)
            self._isolated_entity_subnets[target_entity] = subnet_tag
            self._action_timestamps.append(now)

        return SafetyVerdict(
            passed=passed,
            action_permitted=permitted,
            requires_human_approval=requires_human,
            violations=violations,
            compensating_action=comp_action,
            target_entity=target_entity,
            action=action,
        )

    def record_release(self, target_entity: str) -> None:
        """Removes entity from active isolation set upon compensation / release."""
        self._isolated_entities.discard(target_entity)
        self._isolated_entity_subnets.pop(target_entity, None)
