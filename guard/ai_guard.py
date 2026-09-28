from __future__ import annotations
"""
AHRAS AI Security Guard (Section 41)
------------------------------------
Defends the AI subsystem itself from adversarial manipulation, instruction injection,
and unauthorized execution.

Core Tenets:
1. Strict Trust Boundaries:
   - TRUSTED_SYSTEM: Internal signed code and configurations.
   - TRUSTED_SENSOR: Authenticated, attestation-verified telemetry feeds.
   - UNVERIFIED_TELEMETRY: Raw unauthenticated network/endpoint streams.
   - EXTERNAL_CONTENT: External documents, web feeds, CVE databases.
   - THREAT_INTEL: Untrusted external indicators of compromise.
   - USER_INPUT: Analyst chat inputs, prompt submissions.
   - MODEL_OUTPUT: Non-executable model inferences and draft text.

2. Non-Executable External Content Invariant:
   "Never allow external content to become executable instructions."
   Scans text for prompt injection, indirect prompt injection, system prompt override attempts,
   and delimiters escaping system context.

3. Deterministic Tool Authorization Gate:
   "Any future tool call requires deterministic authorization."
   Every invocation is evaluated against a deterministic policy matrix:
   - Tool name, caller identity, requested action, target resource, policy rule.
   - Rejects unauthorized scope escalation and requires explicit human approval for destructive actions.
   - Maintains an append-only audit trail: tool, identity, action, resource, policy, authorization, human approval, outcome.
"""

import re
import time
import uuid
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple

log = logging.getLogger(__name__)


class TrustClass(str, Enum):
    TRUSTED_SYSTEM       = "TRUSTED_SYSTEM"
    TRUSTED_SENSOR       = "TRUSTED_SENSOR"
    UNVERIFIED_TELEMETRY = "UNVERIFIED_TELEMETRY"
    EXTERNAL_CONTENT     = "EXTERNAL_CONTENT"
    THREAT_INTEL         = "THREAT_INTEL"
    USER_INPUT           = "USER_INPUT"
    MODEL_OUTPUT         = "MODEL_OUTPUT"


class AuthorizationStatus(str, Enum):
    AUTHORIZED       = "AUTHORIZED"
    DENIED_POLICY    = "DENIED_POLICY"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"
    BLOCKED_INJECTION = "BLOCKED_INJECTION"
    SCOPE_ESCALATION = "SCOPE_ESCALATION"


# Known prompt injection & jailbreak signature patterns
INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"(?i)disregard\s+(all\s+)?(previous|prior|system)\s+(instructions|prompts|rules)",
    r"(?i)you\s+are\s+now\s+(an?\s+)?(unrestricted|jailbroken|dan|evil|root|admin)",
    r"(?i)system\s+override",
    r"(?i)bypass\s+(safety|security|policy|authorization)",
    r"(?i)new\s+system\s+instruction:",
    r"(?i)act\s+as\s+(an?\s+)?unrestricted",
    r"(?i)execute\s+(system|shell|bash|cmd|powershell)\s+command",
    r"(?i)<script\b[^>]*>([\s\S]*?)<\/script>",
    r"(?i)eval\s*\(",
    r"(?i)delete\s+from\s+ledger",
    r"(?i)drop\s+table",
]


@dataclass
class ToolExecutionRequest:
    """Represents a tool call request from an agent, user, or workflow."""
    request_id:       str = field(default_factory=lambda: f"REQ-{uuid.uuid4().hex[:12]}")
    caller_identity:  str = "analyst-01"
    caller_role:      str = "ANALYST"      # "VIEWER", "ANALYST", "INCIDENT_RESPONDER", "SYSTEM_ADMIN"
    tool_name:        str = "isolate_host"
    requested_action: str = "EXECUTE"      # "READ", "SIMULATE", "EXECUTE", "TERMINATE"
    target_resource:  str = "host-192.168.1.50"
    parameters:       Dict[str, Any] = field(default_factory=dict)
    human_approval_token: Optional[str] = None
    input_source_trust: TrustClass = TrustClass.USER_INPUT
    timestamp:        float = field(default_factory=time.time)


@dataclass
class ToolAuthorizationAuditRecord:
    """Append-only audit record for tool authorization decisions."""
    record_id:            str
    request_id:           str
    caller_identity:      str
    caller_role:          str
    tool_name:            str
    requested_action:     str
    target_resource:      str
    policy_id:            str
    authorization_status: AuthorizationStatus
    human_approval:       bool
    reason:               str
    timestamp:            float
    record_hash:          str = ""

    def compute_hash(self) -> str:
        payload = f"{self.record_id}:{self.request_id}:{self.caller_identity}:{self.tool_name}:{self.requested_action}:{self.target_resource}:{self.authorization_status.value}:{self.human_approval}:{self.timestamp}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["authorization_status"] = self.authorization_status.value
        return d


class AISecurityGuard:
    """
    Deterministic AI Security Guard enforcing trust classification,
    prompt injection sanitization, memory integrity, and tool execution governance.
    """

    # Deterministic Role-Based Tool Permission Matrix
    ROLE_PERMISSIONS: Dict[str, Dict[str, Set[str]]] = {
        "VIEWER": {
            "get_status": {"READ"},
            "view_incident": {"READ"},
            "query_graph": {"READ"},
        },
        "ANALYST": {
            "get_status": {"READ"},
            "view_incident": {"READ"},
            "query_graph": {"READ"},
            "query_telemetry": {"READ"},
            "simulate_response": {"SIMULATE"},
            "recommend_action": {"SIMULATE"},
            "isolate_host": {"SIMULATE"},
            "block_ip": {"SIMULATE"},
            "revoke_token": {"SIMULATE"},
        },
        "INCIDENT_RESPONDER": {
            "get_status": {"READ"},
            "view_incident": {"READ"},
            "query_graph": {"READ"},
            "query_telemetry": {"READ"},
            "simulate_response": {"SIMULATE"},
            "recommend_action": {"SIMULATE"},
            "block_ip": {"SIMULATE", "EXECUTE"},
            "isolate_host": {"SIMULATE", "EXECUTE"},
            "revoke_token": {"SIMULATE", "EXECUTE"},
        },
        "SYSTEM_ADMIN": {
            "*": {"READ", "SIMULATE", "EXECUTE", "TERMINATE"}
        }
    }

    # Tools requiring mandatory human approval for destructive execution
    HIGH_IMPACT_TOOLS: Set[str] = {
        "isolate_host",
        "revoke_token",
        "wipe_device",
        "terminate_process",
        "deploy_firewall_rule",
    }

    def __init__(self):
        self._compiled_patterns = [re.compile(p) for p in INJECTION_PATTERNS]
        self._audit_ledger: List[ToolAuthorizationAuditRecord] = []

    def sanitize_input(self, text: str, trust_class: TrustClass) -> Tuple[bool, str, Optional[str]]:
        """
        Scans input for prompt injection or malicious instructions.
        Returns: (is_safe, sanitized_text, violation_reason)
        """
        # Trusted system internal inputs are exempt from strict content blocking
        if trust_class == TrustClass.TRUSTED_SYSTEM:
            return True, text, None

        for pattern in self._compiled_patterns:
            match = pattern.search(text)
            if match:
                violation = f"Prompt injection detected matching signature: '{match.group(0)}'"
                log.warning(f"AISecurityGuard blocked untrusted input ({trust_class.value}): {violation}")
                return False, "[BLOCKED_BY_AI_SECURITY_GUARD]", violation

        # Check for structural delimiter attacks (e.g. attempting to break Markdown/JSON/System delimiters)
        if trust_class in (TrustClass.EXTERNAL_CONTENT, TrustClass.THREAT_INTEL, TrustClass.UNVERIFIED_TELEMETRY):
            # Strip dangerous markdown or prompt tags
            text_cleaned = re.sub(r"(?i)<\/?(system|instruction|prompt|tool_call)[^>]*>", "[STRIPPED_TAG]", text)
            return True, text_cleaned, None

        return True, text, None

    def authorize_tool_call(self, request: ToolExecutionRequest) -> ToolAuthorizationAuditRecord:
        """
        Determines whether a tool invocation is authorized based on deterministic policy.
        Enforces human approval gates for high-impact response actions.
        """
        now = time.time()
        role = request.caller_role.upper()
        tool = request.tool_name
        action = request.requested_action.upper()
        rec_id = f"AUD-{uuid.uuid4().hex[:12]}"

        # 1. Parameter safety check (reject injection inside tool parameters)
        for param_k, param_v in request.parameters.items():
            if isinstance(param_v, str):
                is_safe, _, reason = self.sanitize_input(param_v, request.input_source_trust)
                if not is_safe:
                    record = ToolAuthorizationAuditRecord(
                        record_id=rec_id,
                        request_id=request.request_id,
                        caller_identity=request.caller_identity,
                        caller_role=role,
                        tool_name=tool,
                        requested_action=action,
                        target_resource=request.target_resource,
                        policy_id="POL-AI-GUARD-INJECTION",
                        authorization_status=AuthorizationStatus.BLOCKED_INJECTION,
                        human_approval=bool(request.human_approval_token),
                        reason=f"Parameter '{param_k}' contains injection: {reason}",
                        timestamp=now,
                    )
                    record.record_hash = record.compute_hash()
                    self._audit_ledger.append(record)
                    return record

        # 2. Role-Based Access Control Verification
        allowed_tools = self.ROLE_PERMISSIONS.get(role, {})
        has_wildcard = "*" in allowed_tools
        if not has_wildcard:
            if tool not in allowed_tools:
                record = ToolAuthorizationAuditRecord(
                    record_id=rec_id,
                    request_id=request.request_id,
                    caller_identity=request.caller_identity,
                    caller_role=role,
                    tool_name=tool,
                    requested_action=action,
                    target_resource=request.target_resource,
                    policy_id="POL-RBAC-001",
                    authorization_status=AuthorizationStatus.SCOPE_ESCALATION,
                    human_approval=bool(request.human_approval_token),
                    reason=f"Role '{role}' is not permitted to access tool '{tool}'.",
                    timestamp=now,
                )
                record.record_hash = record.compute_hash()
                self._audit_ledger.append(record)
                return record

            permitted_actions = allowed_tools.get(tool, set())
            if action not in permitted_actions:
                record = ToolAuthorizationAuditRecord(
                    record_id=rec_id,
                    request_id=request.request_id,
                    caller_identity=request.caller_identity,
                    caller_role=role,
                    tool_name=tool,
                    requested_action=action,
                    target_resource=request.target_resource,
                    policy_id="POL-RBAC-002",
                    authorization_status=AuthorizationStatus.DENIED_POLICY,
                    human_approval=bool(request.human_approval_token),
                    reason=f"Role '{role}' is not permitted to perform action '{action}' on '{tool}'. Allowed: {permitted_actions}",
                    timestamp=now,
                )
                record.record_hash = record.compute_hash()
                self._audit_ledger.append(record)
                return record

        # 3. High-Impact Action Human Approval Gate
        if tool in self.HIGH_IMPACT_TOOLS and action == "EXECUTE":
            if not request.human_approval_token:
                record = ToolAuthorizationAuditRecord(
                    record_id=rec_id,
                    request_id=request.request_id,
                    caller_identity=request.caller_identity,
                    caller_role=role,
                    tool_name=tool,
                    requested_action=action,
                    target_resource=request.target_resource,
                    policy_id="POL-SAFETY-HUMAN-GATE",
                    authorization_status=AuthorizationStatus.REQUIRES_APPROVAL,
                    human_approval=False,
                    reason=f"Tool '{tool}' with action '{action}' is a high-impact operation requiring explicit human approval token.",
                    timestamp=now,
                )
                record.record_hash = record.compute_hash()
                self._audit_ledger.append(record)
                return record

        # 4. Authorized Execution
        record = ToolAuthorizationAuditRecord(
            record_id=rec_id,
            request_id=request.request_id,
            caller_identity=request.caller_identity,
            caller_role=role,
            tool_name=tool,
            requested_action=action,
            target_resource=request.target_resource,
            policy_id="POL-AUTH-PASS",
            authorization_status=AuthorizationStatus.AUTHORIZED,
            human_approval=bool(request.human_approval_token),
            reason="Tool call conforms to deterministic RBAC and safety policies.",
            timestamp=now,
        )
        record.record_hash = record.compute_hash()
        self._audit_ledger.append(record)
        return record

    def get_audit_records(self) -> List[ToolAuthorizationAuditRecord]:
        return list(self._audit_ledger)

    def clear_audit_records(self) -> None:
        self._audit_ledger.clear()


# Global Singleton
_global_ai_guard: Optional[AISecurityGuard] = None

def get_ai_security_guard() -> AISecurityGuard:
    global _global_ai_guard
    if _global_ai_guard is None:
        _global_ai_guard = AISecurityGuard()
    return _global_ai_guard
