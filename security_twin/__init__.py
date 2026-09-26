from __future__ import annotations
"""
AHRAS Security Twin & Attack Replay Lab Package
-----------------------------------------------
Provides digital twin state management, multi-stage attack scenario generation,
OCSF event synthesis, pre-action counterfactual response simulation,
and Monte Carlo risk distribution sampling.
"""

from security_twin.models import (
    EntityType,
    AttackStage,
    ActionType,
    SimulationStatus,
    Host,
    User,
    Process,
    File,
    Network,
    Asset,
    Service,
    Identity,
    SecurityControl,
    SecurityTwinSnapshot,
    AttackStep,
    AttackScenario,
    SimulationResult,
    MonteCarloResult,
)
from security_twin.state import SecurityTwin, create_enterprise_test_twin
from security_twin.scenario import (
    check_step_preconditions,
    apply_step_postconditions,
    emit_ocsf_event,
    build_ransomware_burst_scenario,
    build_lateral_movement_scenario,
    build_credential_abuse_scenario,
)
from security_twin.simulation import SecurityTwinSimulator

__all__ = [
    "EntityType",
    "AttackStage",
    "ActionType",
    "SimulationStatus",
    "Host",
    "User",
    "Process",
    "File",
    "Network",
    "Asset",
    "Service",
    "Identity",
    "SecurityControl",
    "SecurityTwinSnapshot",
    "AttackStep",
    "AttackScenario",
    "SimulationResult",
    "MonteCarloResult",
    "SecurityTwin",
    "create_enterprise_test_twin",
    "SecurityTwinSimulator",
    "check_step_preconditions",
    "apply_step_postconditions",
    "emit_ocsf_event",
    "build_ransomware_burst_scenario",
    "build_lateral_movement_scenario",
    "build_credential_abuse_scenario",
]
