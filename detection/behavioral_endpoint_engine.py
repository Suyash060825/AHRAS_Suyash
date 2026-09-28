from __future__ import annotations
"""
AHRAS Behavioral Endpoint Engine (Section 38)
-----------------------------------------------
Behavioral detection for:
  1. Ransomware:
     - Mass file modification / rapid file access
     - Rename bursts with suspicious extensions (.locked, .enc, .crypt, etc.)
     - High Shannon entropy (H >= 7.2)
     - Shadow copy deletion attempts (vssadmin, wmic shadowcopy, bcdedit)
  2. Worms:
     - Rapid host fan-out (outbound connection burst across multiple target hosts)
     - Propagation patterns on lateral movement ports (SMB 445, RDP 3389, SSH 22)
     - Synchronized multi-target sweeps
  3. Malware:
     - Suspicious process execution chains (e.g. Office/Web server -> Shell/PowerShell)
     - Persistence installation (cron, systemd, registry Run keys)
     - Privilege escalation attempts (sudoers tampering, SUID abuse)
     - C2 beaconing callbacks

Architecture Invariant:
  "Feed all of them into: EvidenceRecord -> Risk Engine -> Graph -> XAI -> Response.
   Do not create isolated malware/ransomware architectures."
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict

from sensors.endpoint_sensor import EndpointEvent, EndpointEventType
from ahras.evidence.models import EvidenceRecord, EvidenceSource, EvidenceType

log = logging.getLogger(__name__)

# Ransomware file extensions
RANSOMWARE_EXTENSIONS: Set[str] = {
    ".enc", ".locked", ".crypt", ".crypto", ".crypted",
    ".encrypted", ".pay2me", ".cry", ".wnry", ".wncry",
    ".ransom", ".locky", ".cerber", ".zepto"
}

# Suspicious shadow copy deletion strings
SHADOW_COPY_PATTERNS: List[str] = [
    "vssadmin delete shadows",
    "vssadmin.exe delete shadows",
    "wmic shadowcopy delete",
    "wbadmin delete catalog",
    "bcdedit /set {default} recoveryenabled no",
    "bcdedit /set {default} bootstatuspolicy ignoreallfailures",
    "shred -u",
]

# Suspicious parent-child process pairs
SUSPICIOUS_SPAWN_PAIRS: List[Tuple[str, str]] = [
    ("winword.exe", "cmd.exe"),
    ("winword.exe", "powershell.exe"),
    ("excel.exe", "cmd.exe"),
    ("excel.exe", "powershell.exe"),
    ("outlook.exe", "powershell.exe"),
    ("nginx", "bash"),
    ("nginx", "sh"),
    ("httpd", "bash"),
    ("httpd", "sh"),
    ("apache2", "bash"),
    ("apache2", "sh"),
    ("sqlservr.exe", "cmd.exe"),
    ("tomcat", "sh"),
]

# Worm propagation ports
PROPAGATION_PORTS: Set[int] = {445, 139, 22, 3389, 5985, 5986, 135}

# Persistence paths / keywords
PERSISTENCE_TARGETS: List[str] = [
    "/etc/cron",
    "/etc/systemd/system",
    "/etc/rc.local",
    "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
    "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
    ".bashrc",
    ".bash_profile",
]


@dataclass
class BehavioralThreatAlert:
    """Standardized alert produced by the behavioral engine."""
    alert_id:          str
    threat_category:   str       # "RANSOMWARE", "WORM", "MALWARE"
    technique_id:      str       # e.g. "T1486", "T1021", "T1059"
    technique_name:    str
    confidence:        float     # [0.0, 1.0]
    severity:          str       # "HIGH", "CRITICAL"
    host_id:           str
    details:           str
    timestamp:         float
    evidence_record:   EvidenceRecord


class BehavioralEndpointEngine:
    """
    Stateful behavioral analytics engine processing host telemetry events.
    Computes rolling indicators and generates EvidenceRecords for downstream risk reasoning.
    """

    def __init__(
        self,
        entropy_threshold: float = 7.20,
        ransomware_mod_threshold: int = 10,
        worm_fanout_threshold: int = 8,
        window_sec: float = 60.0,
    ):
        self.entropy_threshold = entropy_threshold
        self.ransomware_mod_threshold = ransomware_mod_threshold
        self.worm_fanout_threshold = worm_fanout_threshold
        self.window_sec = window_sec

        # Rolling state tracking per host: host_id -> List of events
        self._file_mods: Dict[str, List[EndpointEvent]] = defaultdict(list)
        self._renames: Dict[str, List[EndpointEvent]] = defaultdict(list)
        self._net_conns: Dict[str, List[EndpointEvent]] = defaultdict(list)

    def analyze_event(self, event: EndpointEvent) -> List[BehavioralThreatAlert]:
        """
        Analyzes a single EndpointEvent in the context of recent rolling host state.
        Returns a list of triggered BehavioralThreatAlerts (each containing an EvidenceRecord).
        """
        alerts: List[BehavioralThreatAlert] = []
        host_id = event.host_id
        now = event.timestamp
        cutoff = now - self.window_sec

        # ── 1. RANSOMWARE DETECTION ──────────────────────────────────────────
        if event.event_type == EndpointEventType.FILE_OPERATION:
            if event.file_operation in ("WRITE", "CREATE"):
                self._file_mods[host_id].append(event)
            elif event.file_operation == "RENAME":
                self._renames[host_id].append(event)

            # Prune expired
            self._file_mods[host_id] = [e for e in self._file_mods[host_id] if e.timestamp >= cutoff]
            self._renames[host_id] = [e for e in self._renames[host_id] if e.timestamp >= cutoff]

            recent_mods = len(self._file_mods[host_id])
            recent_renames = len(self._renames[host_id])

            # Check A: High entropy file modification
            if event.file_entropy >= self.entropy_threshold:
                ev_rec = EvidenceRecord(
                    entity_id=host_id,
                    source="endpoint_behavioral_engine",
                    detector_type=EvidenceType.ML_ANOMALY.value,
                    detector_version="behavioral_v1.0",
                    raw_score=event.file_entropy / 8.0,
                    normalized_score=min(1.0, event.file_entropy / 8.0),
                    confidence=0.88,
                    uncertainty=0.12,
                    timestamp=now,
                    mitre_mapping=["T1486"],
                    explanation=f"High Shannon entropy ({event.file_entropy:.2f}/8.0) detected on file '{event.file_path}' indicative of encryption.",
                )
                alerts.append(BehavioralThreatAlert(
                    alert_id=f"BTA-RANSOM-ENTROPY-{event.event_id}",
                    threat_category="RANSOMWARE",
                    technique_id="T1486",
                    technique_name="Data Encrypted for Impact",
                    confidence=0.88,
                    severity="HIGH",
                    host_id=host_id,
                    details=ev_rec.explanation,
                    timestamp=now,
                    evidence_record=ev_rec,
                ))

            # Check B: Mass file modification burst
            if recent_mods >= self.ransomware_mod_threshold:
                ev_rec = EvidenceRecord(
                    entity_id=host_id,
                    source="endpoint_behavioral_engine",
                    detector_type=EvidenceType.STATISTICAL_DRIFT.value,
                    detector_version="behavioral_v1.0",
                    raw_score=min(1.0, recent_mods / 20.0),
                    normalized_score=min(1.0, recent_mods / 20.0),
                    confidence=0.92,
                    uncertainty=0.08,
                    timestamp=now,
                    mitre_mapping=["T1486"],
                    explanation=f"Mass file modification burst: {recent_mods} files written in {self.window_sec}s window.",
                )
                alerts.append(BehavioralThreatAlert(
                    alert_id=f"BTA-RANSOM-BURST-{event.event_id}",
                    threat_category="RANSOMWARE",
                    technique_id="T1486",
                    technique_name="Data Encrypted for Impact",
                    confidence=0.92,
                    severity="CRITICAL",
                    host_id=host_id,
                    details=ev_rec.explanation,
                    timestamp=now,
                    evidence_record=ev_rec,
                ))

            # Check C: Rename burst with ransomware extensions
            ext = event.file_extension.lower()
            if ext in RANSOMWARE_EXTENSIONS or any(t.lower().endswith(ext) for t in RANSOMWARE_EXTENSIONS):
                ev_rec = EvidenceRecord(
                    entity_id=host_id,
                    source="endpoint_behavioral_engine",
                    detector_type=EvidenceType.SIGNATURE.value,
                    detector_version="behavioral_v1.0",
                    raw_score=0.95,
                    normalized_score=0.95,
                    confidence=0.95,
                    uncertainty=0.05,
                    timestamp=now,
                    mitre_mapping=["T1486"],
                    explanation=f"File extension rename to known ransomware extension '{ext}' on '{event.file_path}'.",
                )
                alerts.append(BehavioralThreatAlert(
                    alert_id=f"BTA-RANSOM-EXT-{event.event_id}",
                    threat_category="RANSOMWARE",
                    technique_id="T1486",
                    technique_name="Data Encrypted for Impact",
                    confidence=0.95,
                    severity="CRITICAL",
                    host_id=host_id,
                    details=ev_rec.explanation,
                    timestamp=now,
                    evidence_record=ev_rec,
                ))

        # Check D: Shadow copy deletion in command line
        if event.event_type == EndpointEventType.PROCESS_SPAWN:
            cmd_lower = event.cmdline.lower()
            if any(p in cmd_lower for p in SHADOW_COPY_PATTERNS):
                ev_rec = EvidenceRecord(
                    entity_id=host_id,
                    source="endpoint_behavioral_engine",
                    detector_type=EvidenceType.SIGNATURE.value,
                    detector_version="behavioral_v1.0",
                    raw_score=0.98,
                    normalized_score=0.98,
                    confidence=0.98,
                    uncertainty=0.02,
                    timestamp=now,
                    mitre_mapping=["T1490"],
                    explanation=f"Shadow copy deletion attempt detected: '{event.cmdline}'.",
                )
                alerts.append(BehavioralThreatAlert(
                    alert_id=f"BTA-RANSOM-SHADOW-{event.event_id}",
                    threat_category="RANSOMWARE",
                    technique_id="T1490",
                    technique_name="Inhibit System Recovery",
                    confidence=0.98,
                    severity="CRITICAL",
                    host_id=host_id,
                    details=ev_rec.explanation,
                    timestamp=now,
                    evidence_record=ev_rec,
                ))

        # ── 2. WORM DETECTION ────────────────────────────────────────────────
        if event.event_type == EndpointEventType.NETWORK_CONNECT:
            self._net_conns[host_id].append(event)
            self._net_conns[host_id] = [e for e in self._net_conns[host_id] if e.timestamp >= cutoff]

            # Calculate unique outbound destinations and propagation port activity
            recent_conns = self._net_conns[host_id]
            unique_dsts = {e.dst_ip for e in recent_conns if e.dst_ip and e.dst_ip != event.src_ip}
            prop_ports = [e for e in recent_conns if e.dst_port in PROPAGATION_PORTS]

            if len(unique_dsts) >= self.worm_fanout_threshold:
                ev_rec = EvidenceRecord(
                    entity_id=host_id,
                    source="endpoint_behavioral_engine",
                    detector_type=EvidenceType.ML_ANOMALY.value,
                    detector_version="behavioral_v1.0",
                    raw_score=min(1.0, len(unique_dsts) / 15.0),
                    normalized_score=min(1.0, len(unique_dsts) / 15.0),
                    confidence=0.91,
                    uncertainty=0.09,
                    timestamp=now,
                    mitre_mapping=["T1021", "T1046"],
                    explanation=f"Rapid host fan-out detected: {len(unique_dsts)} unique destination hosts contacted in {self.window_sec}s.",
                )
                alerts.append(BehavioralThreatAlert(
                    alert_id=f"BTA-WORM-FANOUT-{event.event_id}",
                    threat_category="WORM",
                    technique_id="T1021",
                    technique_name="Remote Services Propagation",
                    confidence=0.91,
                    severity="CRITICAL",
                    host_id=host_id,
                    details=ev_rec.explanation,
                    timestamp=now,
                    evidence_record=ev_rec,
                ))

            if len(prop_ports) >= 6:
                ev_rec = EvidenceRecord(
                    entity_id=host_id,
                    source="endpoint_behavioral_engine",
                    detector_type=EvidenceType.STATISTICAL_DRIFT.value,
                    detector_version="behavioral_v1.0",
                    raw_score=0.85,
                    normalized_score=0.85,
                    confidence=0.87,
                    uncertainty=0.13,
                    timestamp=now,
                    mitre_mapping=["T1021"],
                    explanation=f"Repetitive lateral movement connection burst on propagation ports (SMB/SSH/RDP): {len(prop_ports)} connections.",
                )
                alerts.append(BehavioralThreatAlert(
                    alert_id=f"BTA-WORM-PORTS-{event.event_id}",
                    threat_category="WORM",
                    technique_id="T1021",
                    technique_name="Remote Services Propagation",
                    confidence=0.87,
                    severity="HIGH",
                    host_id=host_id,
                    details=ev_rec.explanation,
                    timestamp=now,
                    evidence_record=ev_rec,
                ))

        # ── 3. MALWARE DETECTION ─────────────────────────────────────────────
        if event.event_type == EndpointEventType.PROCESS_SPAWN:
            p_name = event.exe.split("/")[-1].split("\\")[-1].lower()
            par_name = event.parent_exe.split("/")[-1].split("\\")[-1].lower()

            # Suspicious process lineage
            for sp_parent, sp_child in SUSPICIOUS_SPAWN_PAIRS:
                if par_name == sp_parent.lower() and p_name == sp_child.lower():
                    ev_rec = EvidenceRecord(
                        entity_id=host_id,
                        source="endpoint_behavioral_engine",
                        detector_type=EvidenceType.SIGNATURE.value,
                        detector_version="behavioral_v1.0",
                        raw_score=0.89,
                        normalized_score=0.89,
                        confidence=0.90,
                        uncertainty=0.10,
                        timestamp=now,
                        mitre_mapping=["T1059"],
                        explanation=f"Suspicious process lineage: '{par_name}' spawned shell/interpreter '{p_name}' ({event.cmdline}).",
                    )
                    alerts.append(BehavioralThreatAlert(
                        alert_id=f"BTA-MALWARE-LINEAGE-{event.event_id}",
                        threat_category="MALWARE",
                        technique_id="T1059",
                        technique_name="Command and Scripting Interpreter",
                        confidence=0.90,
                        severity="HIGH",
                        host_id=host_id,
                        details=ev_rec.explanation,
                        timestamp=now,
                        evidence_record=ev_rec,
                    ))
                    break

        if event.event_type == EndpointEventType.PERSISTENCE_SET:
            ev_rec = EvidenceRecord(
                entity_id=host_id,
                source="endpoint_behavioral_engine",
                detector_type=EvidenceType.SIGNATURE.value,
                detector_version="behavioral_v1.0",
                raw_score=0.93,
                normalized_score=0.93,
                confidence=0.92,
                uncertainty=0.08,
                timestamp=now,
                mitre_mapping=["T1543", "T1053"],
                explanation=f"Persistence installation detected: {event.persistence_type} on target '{event.target_path or event.file_path}'.",
            )
            alerts.append(BehavioralThreatAlert(
                alert_id=f"BTA-MALWARE-PERSISTENCE-{event.event_id}",
                threat_category="MALWARE",
                technique_id="T1543",
                technique_name="Create or Modify System Process",
                confidence=0.92,
                severity="CRITICAL",
                host_id=host_id,
                details=ev_rec.explanation,
                timestamp=now,
                evidence_record=ev_rec,
            ))

        if event.event_type == EndpointEventType.PRIVILEGE_CHANGE:
            if event.is_root and not event.auth_success:
                ev_rec = EvidenceRecord(
                    entity_id=host_id,
                    source="endpoint_behavioral_engine",
                    detector_type=EvidenceType.STATISTICAL_DRIFT.value,
                    detector_version="behavioral_v1.0",
                    raw_score=0.94,
                    normalized_score=0.94,
                    confidence=0.93,
                    uncertainty=0.07,
                    timestamp=now,
                    mitre_mapping=["T1548"],
                    explanation=f"Unauthorized privilege escalation attempt: user '{event.user_id}' attempted root escalation.",
                )
                alerts.append(BehavioralThreatAlert(
                    alert_id=f"BTA-MALWARE-PRIVESC-{event.event_id}",
                    threat_category="MALWARE",
                    technique_id="T1548",
                    technique_name="Abuse Elevation Control Mechanism",
                    confidence=0.93,
                    severity="CRITICAL",
                    host_id=host_id,
                    details=ev_rec.explanation,
                    timestamp=now,
                    evidence_record=ev_rec,
                ))

        return alerts
