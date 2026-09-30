"""
Scenario Definition and Composition Schema for AHRAS Synthetic and Simulation Evaluation.
Explicitly distinguishes synthetic, simulation, and real metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
import time
import uuid

class EnvironmentType(str, Enum):
    ENTERPRISE_IT = "enterprise_it"
    CLOUD_HYBRID = "cloud_hybrid"
    OT_ICS = "ot_ics"
    SMART_OFFICE_IOT = "smart_office_iot"
    TELECOMMUNICATIONS = "telecommunications"

class AttackFamily(str, Enum):
    RECONNAISSANCE = "reconnaissance"
    SCANNING = "scanning"
    BRUTE_FORCE = "brute_force"
    CREDENTIAL_ABUSE = "credential_abuse"
    EXPLOITATION = "exploitation"
    WEB_ATTACK = "web_attack"
    DENIAL_OF_SERVICE = "denial_of_service"
    DISTRIBUTED_DENIAL_OF_SERVICE = "distributed_denial_of_service"
    BOTNET = "botnet"
    COMMAND_AND_CONTROL = "command_and_control"
    MALWARE = "malware"
    LATERAL_MOVEMENT = "lateral_movement"
    PERSISTENCE = "persistence"
    DISCOVERY = "discovery"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"
    IOT_SPECIFIC_BEHAVIOR = "iot_specific_behavior"

class NoiseLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class TelemetryQuality(str, Enum):
    PRISTINE = "pristine"
    DEGRADED_5 = "degraded_5pct"
    DEGRADED_10 = "degraded_10pct"
    DEGRADED_20 = "degraded_20pct"
    DEGRADED_50 = "degraded_50pct"
    INTERMITTENT = "intermittent_sensor"

@dataclass
class ScenarioIdentityContext:
    user_id: str
    role: str
    credential_level: str
    is_compromised: bool = False
    anomalous_login: bool = False

@dataclass
class ScenarioAssetContext:
    asset_id: str
    ip_address: str
    hostname: str
    criticality: float = 0.5  # 0.0 to 1.0
    zone: str = "internal"
    os_type: str = "linux"

@dataclass
class ScenarioStage:
    stage_index: int
    attack_family: Optional[AttackFamily]
    description: str
    duration_seconds: float
    event_rate_eps: float
    source_asset_id: str
    target_asset_id: str
    ttp_code: str = "T1000"
    is_malicious: bool = False

@dataclass
class ScenarioSpecification:
    scenario_id: str = field(default_factory=lambda: f"SCEN-{uuid.uuid4().hex[:8]}")
    name: str = "Generic Scenario"
    scenario_type: str = "SYNTHETIC"  # SYNTHETIC | SIMULATION | BENCHMARK
    environment_type: EnvironmentType = EnvironmentType.ENTERPRISE_IT
    stages: List[ScenarioStage] = field(default_factory=list)
    assets: List[ScenarioAssetContext] = field(default_factory=list)
    identities: List[ScenarioIdentityContext] = field(default_factory=list)
    noise_level: NoiseLevel = NoiseLevel.LOW
    telemetry_quality: TelemetryQuality = TelemetryQuality.PRISTINE
    duration_total_seconds: float = 300.0
    random_seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "scenario_type": self.scenario_type,
            "environment_type": self.environment_type.value,
            "noise_level": self.noise_level.value,
            "telemetry_quality": self.telemetry_quality.value,
            "duration_total_seconds": self.duration_total_seconds,
            "random_seed": self.random_seed,
            "stages_count": len(self.stages),
            "assets_count": len(self.assets),
            "identities_count": len(self.identities),
            "metadata": self.metadata,
        }
