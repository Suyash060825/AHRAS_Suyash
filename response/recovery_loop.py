from __future__ import annotations
"""
AHRAS Module — Resilience & Recovery Closed Loop (Section 34 / Master Prompt)
-----------------------------------------------------------------------------
Implements the formal 6-stage post-incident containment and recovery lifecycle:
  DETECT -> CONTAIN -> ERADICATE -> RESTORE -> VERIFY -> RECOVER

Post-Recovery Monitoring & Recurrence Invariant:
  - Measures residual risk continuously after recovery.
  - Monitors for adversary reinfection / lateral recurrence in a time window.
  - Automatically reopens the incident and triggers containment re-evaluation
    if entity risk escalates post-recovery.
  - Records full KPI telemetry: Time-to-Containment (TTC), Time-to-Recovery (TTR),
    residual risk, and recurrence rates.
"""

import enum
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class RecoveryStage(str, enum.Enum):
    DETECTED = "DETECTED"
    CONTAINED = "CONTAINED"
    ERADICATED = "ERADICATED"
    RESTORED = "RESTORED"
    VERIFIED = "VERIFIED"
    RECOVERED = "RECOVERED"
    REOPENED_ON_RECURRENCE = "REOPENED_ON_RECURRENCE"


@dataclass
class RecoveryTransition:
    """Audit log entry for an incident recovery state transition."""
    from_stage: str
    to_stage: str
    timestamp: float
    actor: str
    notes: str
    risk_at_transition: float


@dataclass
class IncidentRecoveryRecord:
    """Maintains state, timing, and residual risk across the recovery lifecycle."""
    incident_id: str
    entity_id: str
    current_stage: RecoveryStage = RecoveryStage.DETECTED
    initial_risk: float = 0.85
    residual_risk: float = 0.85
    detected_at: float = field(default_factory=time.time)
    contained_at: Optional[float] = None
    eradicated_at: Optional[float] = None
    restored_at: Optional[float] = None
    verified_at: Optional[float] = None
    recovered_at: Optional[float] = None
    reopened_at: Optional[float] = None
    recurrence_monitored_until: Optional[float] = None
    recurrence_detected: bool = False
    transitions: List[RecoveryTransition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_to_containment_sec(self) -> Optional[float]:
        if self.contained_at and self.detected_at:
            return max(0.0, self.contained_at - self.detected_at)
        return None

    @property
    def time_to_recovery_sec(self) -> Optional[float]:
        if self.recovered_at and self.detected_at:
            return max(0.0, self.recovered_at - self.detected_at)
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["current_stage"] = self.current_stage.value
        d["time_to_containment_sec"] = self.time_to_containment_sec
        d["time_to_recovery_sec"] = self.time_to_recovery_sec
        return d


class ResilienceRecoveryEngine:
    """
    Stateful orchestrator for the AHRAS Resilience and Recovery Loop.
    Enforces monotonic progression through verification gates before marking an incident recovered.
    """

    def __init__(self, safe_residual_risk_threshold: float = 0.15, recurrence_window_sec: float = 300.0):
        self.safe_threshold = safe_residual_risk_threshold
        self.recurrence_window = recurrence_window_sec
        self._incidents: Dict[str, IncidentRecoveryRecord] = {}
        self._lock = threading.RLock()

    def register_incident(self, incident_id: str, entity_id: str, initial_risk: float = 0.85) -> IncidentRecoveryRecord:
        """Initializes a new incident in DETECTED stage."""
        with self._lock:
            rec = IncidentRecoveryRecord(
                incident_id=incident_id,
                entity_id=entity_id,
                current_stage=RecoveryStage.DETECTED,
                initial_risk=initial_risk,
                residual_risk=initial_risk,
                detected_at=time.time(),
            )
            rec.transitions.append(RecoveryTransition(
                from_stage="NONE",
                to_stage=RecoveryStage.DETECTED.value,
                timestamp=rec.detected_at,
                actor="AHRAS_ALERT_INTELLIGENCE",
                notes="Incident detected and ingested into recovery tracking",
                risk_at_transition=initial_risk,
            ))
            self._incidents[incident_id] = rec
            return rec

    def get_incident(self, incident_id: str) -> Optional[IncidentRecoveryRecord]:
        with self._lock:
            return self._incidents.get(incident_id)

    def advance_to_contained(
        self,
        incident_id: str,
        action_applied: str,
        post_containment_risk: float,
        actor: str = "AHRAS_SOAR_ORCHESTRATOR",
    ) -> IncidentRecoveryRecord:
        """Moves incident from DETECTED to CONTAINED."""
        with self._lock:
            rec = self._get_or_raise(incident_id)
            now = time.time()
            rec.contained_at = now
            rec.residual_risk = post_containment_risk
            prev_stage = rec.current_stage.value
            rec.current_stage = RecoveryStage.CONTAINED
            rec.metadata["containment_action"] = action_applied
            rec.transitions.append(RecoveryTransition(
                from_stage=prev_stage,
                to_stage=RecoveryStage.CONTAINED.value,
                timestamp=now,
                actor=actor,
                notes=f"Containment action '{action_applied}' executed successfully",
                risk_at_transition=post_containment_risk,
            ))
            log.info(f"[RECOVERY] Incident '{incident_id}' CONTAINED via {action_applied} (risk={post_containment_risk:.3f})")
            return rec

    def advance_to_eradicated(
        self,
        incident_id: str,
        cleared_artifacts: List[str],
        actor: str = "ANALYST_REMEDIATION",
    ) -> IncidentRecoveryRecord:
        """Moves incident from CONTAINED to ERADICATED."""
        with self._lock:
            rec = self._get_or_raise(incident_id)
            if rec.current_stage != RecoveryStage.CONTAINED:
                raise ValueError(f"Cannot eradicate incident in stage {rec.current_stage.value}; must be CONTAINED")
            now = time.time()
            rec.eradicated_at = now
            rec.residual_risk = max(0.05, rec.residual_risk * 0.70)
            prev_stage = rec.current_stage.value
            rec.current_stage = RecoveryStage.ERADICATED
            rec.metadata["cleared_artifacts"] = cleared_artifacts
            rec.transitions.append(RecoveryTransition(
                from_stage=prev_stage,
                to_stage=RecoveryStage.ERADICATED.value,
                timestamp=now,
                actor=actor,
                notes=f"Eradicated {len(cleared_artifacts)} artifacts: {cleared_artifacts[:3]}",
                risk_at_transition=rec.residual_risk,
            ))
            log.info(f"[RECOVERY] Incident '{incident_id}' ERADICATED ({len(cleared_artifacts)} items removed)")
            return rec

    def advance_to_restored(
        self,
        incident_id: str,
        restore_reference: str,
        actor: str = "SYSADMIN_RESTORE",
    ) -> IncidentRecoveryRecord:
        """Moves incident from ERADICATED to RESTORED."""
        with self._lock:
            rec = self._get_or_raise(incident_id)
            if rec.current_stage != RecoveryStage.ERADICATED:
                raise ValueError(f"Cannot restore incident in stage {rec.current_stage.value}; must be ERADICATED")
            now = time.time()
            rec.restored_at = now
            rec.residual_risk = max(0.02, rec.residual_risk * 0.50)
            prev_stage = rec.current_stage.value
            rec.current_stage = RecoveryStage.RESTORED
            rec.metadata["restore_reference"] = restore_reference
            rec.transitions.append(RecoveryTransition(
                from_stage=prev_stage,
                to_stage=RecoveryStage.RESTORED.value,
                timestamp=now,
                actor=actor,
                notes=f"Restored from clean baseline snapshot {restore_reference}",
                risk_at_transition=rec.residual_risk,
            ))
            log.info(f"[RECOVERY] Incident '{incident_id}' RESTORED from {restore_reference}")
            return rec

    def advance_to_verified(
        self,
        incident_id: str,
        observed_telemetry_risk: float,
        actor: str = "AHRAS_VERIFICATION_GATE",
    ) -> Tuple[bool, IncidentRecoveryRecord]:
        """
        Executes formal postcondition verification:
        Checks if observed post-restoration risk <= safe threshold (0.15).
        """
        with self._lock:
            rec = self._get_or_raise(incident_id)
            if rec.current_stage != RecoveryStage.RESTORED:
                raise ValueError(f"Cannot verify incident in stage {rec.current_stage.value}; must be RESTORED")

            now = time.time()
            rec.residual_risk = observed_telemetry_risk

            if observed_telemetry_risk <= self.safe_threshold:
                rec.verified_at = now
                prev_stage = rec.current_stage.value
                rec.current_stage = RecoveryStage.VERIFIED
                rec.transitions.append(RecoveryTransition(
                    from_stage=prev_stage,
                    to_stage=RecoveryStage.VERIFIED.value,
                    timestamp=now,
                    actor=actor,
                    notes=f"Verification successful: residual risk {observed_telemetry_risk:.3f} <= safe threshold {self.safe_threshold}",
                    risk_at_transition=observed_telemetry_risk,
                ))
                log.info(f"[RECOVERY] Incident '{incident_id}' VERIFIED clean (risk={observed_telemetry_risk:.3f})")
                return True, rec
            else:
                rec.transitions.append(RecoveryTransition(
                    from_stage=rec.current_stage.value,
                    to_stage=rec.current_stage.value,
                    timestamp=now,
                    actor=actor,
                    notes=f"Verification failed: residual risk {observed_telemetry_risk:.3f} exceeds threshold {self.safe_threshold}",
                    risk_at_transition=observed_telemetry_risk,
                ))
                log.warning(f"[RECOVERY] Verification failed for '{incident_id}': risk {observed_telemetry_risk:.3f} > {self.safe_threshold}")
                return False, rec

    def advance_to_recovered(
        self,
        incident_id: str,
        actor: str = "SOC_LEAD_CLOSE",
    ) -> IncidentRecoveryRecord:
        """Marks incident as fully recovered and opens recurrence monitoring window."""
        with self._lock:
            rec = self._get_or_raise(incident_id)
            if rec.current_stage != RecoveryStage.VERIFIED:
                raise ValueError(f"Cannot mark RECOVERED from stage {rec.current_stage.value}; must be VERIFIED")
            now = time.time()
            rec.recovered_at = now
            rec.recurrence_monitored_until = now + self.recurrence_window
            prev_stage = rec.current_stage.value
            rec.current_stage = RecoveryStage.RECOVERED
            rec.transitions.append(RecoveryTransition(
                from_stage=prev_stage,
                to_stage=RecoveryStage.RECOVERED.value,
                timestamp=now,
                actor=actor,
                notes=f"Incident closed. Monitoring recurrence for {self.recurrence_window}s",
                risk_at_transition=rec.residual_risk,
            ))
            log.info(f"[RECOVERY] Incident '{incident_id}' RECOVERED. Active recurrence monitoring active.")
            return rec

    def evaluate_recurrence(
        self,
        incident_id: str,
        current_entity_risk: float,
        risk_spike_threshold: float = 0.35,
    ) -> Tuple[bool, IncidentRecoveryRecord]:
        """
        Monitors for adversary recurrence post-recovery.
        If risk spikes above threshold within monitoring window, automatically reopens incident.
        """
        with self._lock:
            rec = self._get_or_raise(incident_id)
            if rec.current_stage != RecoveryStage.RECOVERED:
                return False, rec

            now = time.time()
            if rec.recurrence_monitored_until and now <= rec.recurrence_monitored_until:
                if current_entity_risk >= risk_spike_threshold:
                    rec.reopened_at = now
                    rec.recurrence_detected = True
                    rec.residual_risk = current_entity_risk
                    prev_stage = rec.current_stage.value
                    rec.current_stage = RecoveryStage.REOPENED_ON_RECURRENCE
                    rec.transitions.append(RecoveryTransition(
                        from_stage=prev_stage,
                        to_stage=RecoveryStage.REOPENED_ON_RECURRENCE.value,
                        timestamp=now,
                        actor="AHRAS_RECURRENCE_WATCHDOG",
                        notes=f"Recurrence detected: risk spiked to {current_entity_risk:.3f} (>= {risk_spike_threshold}) within window",
                        risk_at_transition=current_entity_risk,
                    ))
                    log.warning(f"[RECOVERY] REOPENED '{incident_id}' due to recurrence! (risk={current_entity_risk:.3f})")
                    return True, rec

            return False, rec

    def _get_or_raise(self, incident_id: str) -> IncidentRecoveryRecord:
        rec = self._incidents.get(incident_id)
        if not rec:
            raise KeyError(f"Incident '{incident_id}' not tracked in recovery engine")
        return rec
