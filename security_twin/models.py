from __future__ import annotations
"""
AHRAS Module / Digital Twin — Entity, Scenario, and Simulation Models
---------------------------------------------------------------------
Defines formal data contracts for the AHRAS Security Twin and Attack Replay Lab:
  - Enterprise digital assets, identities, hosts, processes, and controls.
  - Multi-stage attack scenario representations with preconditions/postconditions.
  - Counterfactual response simulation contracts and Monte Carlo result containers.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class EntityType(str, Enum):
    HOST = "HOST"
    USER = "USER"
    PROCESS = "PROCESS"
    FILE = "FILE"
    NETWORK = "NETWORK"
    ASSET = "ASSET"
    SERVICE = "SERVICE"
    IDENTITY = "IDENTITY"
    SECURITY_CONTROL = "SECURITY_CONTROL"


class AttackStage(str, Enum):
    RECON = "RECON"
    INITIAL_ACCESS = "INITIAL_ACCESS"
    EXECUTION = "EXECUTION"
    PERSISTENCE = "PERSISTENCE"
    CREDENTIAL_ACCESS = "CREDENTIAL_ACCESS"
    LATERAL_MOVEMENT = "LATERAL_MOVEMENT"
    COLLECTION = "COLLECTION"
    EXFILTRATION = "EXFILTRATION"


class ActionType(str, Enum):
    NO_ACTION = "NO_ACTION"
    BLOCK_SOURCE = "BLOCK_SOURCE"
    ISOLATE_HOST = "ISOLATE_HOST"
    REVOKE_TOKEN = "REVOKE_TOKEN"
    TERMINATE_PROCESS = "TERMINATE_PROCESS"


class SimulationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    HIGH_BLAST_RADIUS = "HIGH_BLAST_RADIUS"
    INFEASIBLE = "INFEASIBLE"


@dataclass
class Host:
    host_id: str
    hostname: str
    ip_address: str
    os_type: str = "linux"
    criticality: float = 0.5            # [0.0, 1.0] business criticality
    is_isolated: bool = False
    services: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    compromise_level: float = 0.0       # [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class User:
    user_id: str
    username: str
    role: str = "standard"
    is_privileged: bool = False
    credentials_revoked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Process:
    pid: int
    process_name: str
    host_id: str
    user_id: str
    cmdline: str = ""
    parent_pid: Optional[int] = None
    is_terminated: bool = False
    hash_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class File:
    path: str
    host_id: str
    is_encrypted: bool = False
    entropy: float = 4.0
    permissions: str = "0644"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Network:
    network_id: str
    cidr: str
    zone: str = "INTERNAL"              # "DMZ", "INTERNAL", "PROD", "EXTERNAL"
    blocked_ips: List[str] = field(default_factory=list)
    ingress_rules: List[str] = field(default_factory=list)
    egress_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Asset:
    asset_id: str
    asset_type: str                     # "DATABASE", "API_GATEWAY", "FILE_SERVER", "SENSITIVE_DATA"
    business_criticality: float = 0.8
    host_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Service:
    service_id: str
    name: str
    port: int
    protocol: str = "TCP"
    host_id: str = ""
    running: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Identity:
    token_id: str
    user_id: str
    expiry: float = 0.0
    is_valid: bool = True
    permissions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityControl:
    control_id: str
    control_type: str                   # "FIREWALL", "EDR", "IAM", "HONEYPOT"
    target_entity: str
    is_active: bool = True
    rules: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityTwinSnapshot:
    snapshot_id: str
    timestamp: float
    entities: Dict[str, Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    active_controls: Dict[str, Any]
    risk_state: Dict[str, float]
    model_versions: Dict[str, str] = field(default_factory=dict)
    response_policies: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttackStep:
    step_id: str
    stage: AttackStage
    timestamp: float
    source: str
    destination: str
    technique: str                      # MITRE ATT&CK ID e.g. "T1046", "T1021"
    technique_name: str
    preconditions: Dict[str, Any] = field(default_factory=dict)
    postconditions: Dict[str, Any] = field(default_factory=dict)
    expected_evidence: Dict[str, Any] = field(default_factory=dict)
    success_prob: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["stage"] = self.stage.value if isinstance(self.stage, AttackStage) else str(self.stage)
        return res


@dataclass
class AttackScenario:
    scenario_id: str
    name: str
    description: str
    steps: List[AttackStep] = field(default_factory=list)
    current_step_idx: int = 0
    completed: bool = False
    aborted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "current_step_idx": self.current_step_idx,
            "completed": self.completed,
            "aborted": self.aborted,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class SimulationResult:
    action_type: str
    target_entity: str
    pre_action_risk: float
    expected_post_action_risk: float
    path_breakage_probability: float
    remaining_attack_steps: int
    blast_radius_score: float
    collateral_disruption_cost: float
    simulation_confidence: float
    validation_status: str              # "VALIDATED", "POLICY_VIOLATION", "HIGH_BLAST_RADIUS", "INFEASIBLE"
    time_to_containment: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MonteCarloResult:
    action_type: str
    num_iterations: int
    mean_residual_risk: float
    median_residual_risk: float
    std_residual_risk: float
    p10_risk: float
    p50_risk: float
    p90_risk: float
    p99_risk: float
    containment_probability: float
    escalation_probability: float
    blast_radius_mean: float
    risk_samples: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        # Omit large raw samples list from dictionary if over 50 items to keep serialization lightweight
        if len(res.get("risk_samples", [])) > 50:
            res["risk_samples"] = res["risk_samples"][:50]
        return res
