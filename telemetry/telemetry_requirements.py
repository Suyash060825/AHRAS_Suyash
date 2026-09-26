"""
AHRAS Telemetry Requirements Engine
------------------------------------
Defines minimal necessary, optional/redundant, and false-positive reduction fields
for each ATT&CK technique and concrete execution vector, along with empirical
telemetry volume metrics (bytes/event, events/sec).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from coverage.implementation_catalog import ImplementationCatalog, get_default_catalog


@dataclass
class TelemetryVolumeMetric:
    """
    Volume footprint of telemetry for a specific event class / vector.
    """
    raw_bytes_per_event: int           # Size of raw unstructured log / pcap frame
    normalized_bytes_per_event: int    # Full OCSF normalized JSON event size
    minimal_bytes_per_event: int       # Stripped minimal OCSF schema containing only required fields
    estimated_events_per_sec: float    # Expected fleet-wide ingestion rate (EPS)

    @property
    def volume_reduction_ratio(self) -> float:
        """Percentage reduction in event size by pruning to minimal schema."""
        if self.normalized_bytes_per_event <= 0:
            return 0.0
        saved = self.normalized_bytes_per_event - self.minimal_bytes_per_event
        return round((saved / self.normalized_bytes_per_event) * 100.0, 2)


@dataclass
class TelemetryRequirementProfile:
    """
    Detailed telemetry requirements for an attack technique implementation.
    """
    technique_id: str
    implementation_id: str
    vector_name: str
    tactic: str
    ocsf_class: str
    minimal_necessary_fields: List[str]   # Indispensable fields (detection collapses without them)
    optional_redundant_fields: List[str]  # Ancillary fields not evaluated by detection logic
    fp_reduction_fields: List[str]        # Fields required to disambiguate benign noise & reduce FPR
    volume: TelemetryVolumeMetric


class TelemetryRequirementsRegistry:
    """
    Registry of telemetry requirement profiles across the enterprise matrix.
    """
    def __init__(self) -> None:
        self._profiles: Dict[str, TelemetryRequirementProfile] = {}

    def register(self, profile: TelemetryRequirementProfile) -> None:
        self._profiles[profile.implementation_id] = profile

    def get_profile(self, implementation_id: str) -> Optional[TelemetryRequirementProfile]:
        return self._profiles.get(implementation_id)

    def get_profiles_for_technique(self, technique_id: str) -> List[TelemetryRequirementProfile]:
        return [p for p in self._profiles.values() if p.technique_id == technique_id]

    def get_profiles_for_tactic(self, tactic: str) -> List[TelemetryRequirementProfile]:
        return [p for p in self._profiles.values() if p.tactic.lower() == tactic.lower()]

    def get_all_profiles(self) -> List[TelemetryRequirementProfile]:
        return list(self._profiles.values())

    def get_all_indispensable_fields(self) -> Set[str]:
        """Returns union of all minimal necessary fields across all techniques."""
        fields = set()
        for p in self._profiles.values():
            fields.update(p.minimal_necessary_fields)
        return fields

    def get_all_redundant_fields(self) -> Set[str]:
        """Returns union of all optional/redundant fields."""
        fields = set()
        for p in self._profiles.values():
            fields.update(p.optional_redundant_fields)
        return fields

    def get_all_fp_reduction_fields(self) -> Set[str]:
        """Returns union of all false positive reduction fields."""
        fields = set()
        for p in self._profiles.values():
            fields.update(p.fp_reduction_fields)
        return fields


def get_default_requirements_registry() -> TelemetryRequirementsRegistry:
    """
    Builds empirical telemetry requirements across detected ATT&CK vectors in AHRAS.
    """
    registry = TelemetryRequirementsRegistry()

    # 1. Execution: PowerShell encoded command
    registry.register(TelemetryRequirementProfile(
        technique_id="T1059.001",
        implementation_id="T1059.001-IMPL-01",
        vector_name="powershell_encoded_command",
        tactic="Execution",
        ocsf_class="process_activity",
        minimal_necessary_fields=["actor.process.cmd_line", "actor.process.name"],
        optional_redundant_fields=["device.hostname", "raw_source", "actor.process.pid", "time", "severity_id"],
        fp_reduction_fields=["actor.process.user.name", "process.parent_name"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=620,
            normalized_bytes_per_event=480,
            minimal_bytes_per_event=110,
            estimated_events_per_sec=250.0,
        )
    ))

    # 2. Execution: Unix Shell reverse shell
    registry.register(TelemetryRequirementProfile(
        technique_id="T1059.004",
        implementation_id="T1059.004-IMPL-01",
        vector_name="bash_dev_tcp_reverse_shell",
        tactic="Execution",
        ocsf_class="process_activity",
        minimal_necessary_fields=["actor.process.cmd_line"],
        optional_redundant_fields=["device.hostname", "actor.process.pid", "actor.process.exe"],
        fp_reduction_fields=["process.parent_name", "actor.process.user.name"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=450,
            normalized_bytes_per_event=390,
            minimal_bytes_per_event=85,
            estimated_events_per_sec=400.0,
        )
    ))

    # 3. Privilege Escalation: Root child spawn
    registry.register(TelemetryRequirementProfile(
        technique_id="T1548.001",
        implementation_id="T1548.001-IMPL-01",
        vector_name="root_child_spawn_from_suid",
        tactic="Privilege Escalation",
        ocsf_class="process_activity",
        minimal_necessary_fields=["actor.process.name", "actor.process.user.name"],
        optional_redundant_fields=["device.hostname", "raw_source", "actor.process.pid"],
        fp_reduction_fields=["process.parent_name", "actor.process.cmd_line"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=410,
            normalized_bytes_per_event=370,
            minimal_bytes_per_event=75,
            estimated_events_per_sec=320.0,
        )
    ))

    # 4. Privilege Escalation: Cloud unauthorized external user
    registry.register(TelemetryRequirementProfile(
        technique_id="T1078.004",
        implementation_id="T1078.004-IMPL-01",
        vector_name="external_unknown_cloud_actor",
        tactic="Privilege Escalation",
        ocsf_class="cloud_api",
        minimal_necessary_fields=["actor.user.name", "enrichment.is_private"],
        optional_redundant_fields=["device.hostname", "raw_source", "time"],
        fp_reduction_fields=["src_endpoint.ip", "api.operation"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=780,
            normalized_bytes_per_event=540,
            minimal_bytes_per_event=95,
            estimated_events_per_sec=180.0,
        )
    ))

    # 5. Defense Evasion: Shadow copy deletion
    registry.register(TelemetryRequirementProfile(
        technique_id="T1070.004",
        implementation_id="T1070.004-IMPL-01",
        vector_name="vssadmin_shadow_copy_deletion",
        tactic="Defense Evasion",
        ocsf_class="process_activity",
        minimal_necessary_fields=["actor.process.cmd_line"],
        optional_redundant_fields=["device.hostname", "raw_source", "actor.process.pid"],
        fp_reduction_fields=["actor.process.user.name"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=510,
            normalized_bytes_per_event=420,
            minimal_bytes_per_event=90,
            estimated_events_per_sec=150.0,
        )
    ))

    # 6. Defense Evasion: High Shannon entropy file write
    registry.register(TelemetryRequirementProfile(
        technique_id="T1027",
        implementation_id="T1027-IMPL-02",
        vector_name="high_shannon_entropy_file_write",
        tactic="Defense Evasion",
        ocsf_class="file_activity",
        minimal_necessary_fields=["file.path", "enrichment.entropy", "enrichment.ransomware_indicator"],
        optional_redundant_fields=["device.hostname", "file.sha256", "raw_source"],
        fp_reduction_fields=["file.extension", "file.action"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=490,
            normalized_bytes_per_event=410,
            minimal_bytes_per_event=110,
            estimated_events_per_sec=600.0,
        )
    ))

    # 7. Credential Access: LSASS memory dump
    registry.register(TelemetryRequirementProfile(
        technique_id="T1003.001",
        implementation_id="T1003.001-IMPL-01",
        vector_name="mimikatz_cli_execution",
        tactic="Credential Access",
        ocsf_class="process_activity",
        minimal_necessary_fields=["actor.process.cmd_line"],
        optional_redundant_fields=["device.hostname", "actor.process.pid", "time"],
        fp_reduction_fields=["actor.process.user.name", "actor.process.name"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=580,
            normalized_bytes_per_event=460,
            minimal_bytes_per_event=80,
            estimated_events_per_sec=120.0,
        )
    ))

    # 8. Credential Access: SSH brute force
    registry.register(TelemetryRequirementProfile(
        technique_id="T1110.001",
        implementation_id="T1110.001-IMPL-01",
        vector_name="ssh_volumetric_brute_force",
        tactic="Credential Access",
        ocsf_class="network_activity",
        minimal_necessary_fields=["dst_endpoint.port", "traffic.packets"],
        optional_redundant_fields=["traffic.bytes", "src_endpoint.port", "protocol_name", "time"],
        fp_reduction_fields=["src_endpoint.ip", "enrichment.is_private"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=320,
            normalized_bytes_per_event=280,
            minimal_bytes_per_event=65,
            estimated_events_per_sec=1500.0,
        )
    ))

    # 9. Discovery: SYN Port Scan
    registry.register(TelemetryRequirementProfile(
        technique_id="T1046",
        implementation_id="T1046-IMPL-01",
        vector_name="syn_port_scan_sweep",
        tactic="Discovery",
        ocsf_class="network_activity",
        minimal_necessary_fields=["unique_dst_ports", "tcp_flags", "traffic.packets"],
        optional_redundant_fields=["traffic.bytes", "dst_endpoint.port", "src_endpoint.port"],
        fp_reduction_fields=["src_endpoint.ip"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=310,
            normalized_bytes_per_event=290,
            minimal_bytes_per_event=70,
            estimated_events_per_sec=2200.0,
        )
    ))

    # 10. Lateral Movement: SMB port 445 lateral transfer
    registry.register(TelemetryRequirementProfile(
        technique_id="T1021.002",
        implementation_id="T1021.002-IMPL-01",
        vector_name="smb_port_445_high_volume_transfer",
        tactic="Lateral Movement",
        ocsf_class="network_activity",
        minimal_necessary_fields=["dst_endpoint.port", "traffic.packets", "enrichment.is_private"],
        optional_redundant_fields=["traffic.pps", "src_endpoint.port", "protocol_name"],
        fp_reduction_fields=["src_endpoint.ip", "traffic.bytes"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=340,
            normalized_bytes_per_event=300,
            minimal_bytes_per_event=75,
            estimated_events_per_sec=800.0,
        )
    ))

    # 11. Lateral Movement: Inbound RDP access
    registry.register(TelemetryRequirementProfile(
        technique_id="T1021.001",
        implementation_id="T1021.001-IMPL-01",
        vector_name="rdp_inbound_external_connection",
        tactic="Lateral Movement",
        ocsf_class="network_activity",
        minimal_necessary_fields=["dst_endpoint.port", "enrichment.is_private"],
        optional_redundant_fields=["traffic.bytes", "traffic.packets", "protocol_name"],
        fp_reduction_fields=["src_endpoint.ip"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=330,
            normalized_bytes_per_event=290,
            minimal_bytes_per_event=60,
            estimated_events_per_sec=450.0,
        )
    ))

    # 12. Collection: Sensitive file read
    registry.register(TelemetryRequirementProfile(
        technique_id="T1005",
        implementation_id="T1005-IMPL-01",
        vector_name="sensitive_credential_file_access",
        tactic="Collection",
        ocsf_class="file_activity",
        minimal_necessary_fields=["file.path"],
        optional_redundant_fields=["device.hostname", "file.sha256", "file.entropy"],
        fp_reduction_fields=["actor.user.name", "file.action"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=480,
            normalized_bytes_per_event=390,
            minimal_bytes_per_event=65,
            estimated_events_per_sec=500.0,
        )
    ))

    # 13. Command and Control: Encrypted TLS session beaconing
    registry.register(TelemetryRequirementProfile(
        technique_id="T1573.002",
        implementation_id="T1573.002-IMPL-01",
        vector_name="encrypted_session_beaconing",
        tactic="Command and Control",
        ocsf_class="encrypted_session",
        minimal_necessary_fields=["packet_lengths", "inter_arrival_times"],
        optional_redundant_fields=["flow_id", "sni", "alpn", "tls_version"],
        fp_reduction_fields=["dst_endpoint.port"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=1250,
            normalized_bytes_per_event=850,
            minimal_bytes_per_event=210,
            estimated_events_per_sec=650.0,
        )
    ))

    # 14. Impact: Ransomware extension rename
    registry.register(TelemetryRequirementProfile(
        technique_id="T1486",
        implementation_id="T1486-IMPL-01",
        vector_name="ransomware_extension_rename",
        tactic="Impact",
        ocsf_class="file_activity",
        minimal_necessary_fields=["file.path", "enrichment.ransomware_extension"],
        optional_redundant_fields=["device.hostname", "file.sha256", "time"],
        fp_reduction_fields=["enrichment.ransomware_indicator", "file.action"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=460,
            normalized_bytes_per_event=380,
            minimal_bytes_per_event=70,
            estimated_events_per_sec=350.0,
        )
    ))

    # 15. Impact: Volumetric SYN flood
    registry.register(TelemetryRequirementProfile(
        technique_id="T1498.001",
        implementation_id="T1498.001-IMPL-01",
        vector_name="syn_flood_volumetric",
        tactic="Impact",
        ocsf_class="network_activity",
        minimal_necessary_fields=["tcp_flags", "traffic.pps"],
        optional_redundant_fields=["traffic.bytes", "src_endpoint.port", "dst_endpoint.port"],
        fp_reduction_fields=["src_endpoint.ip"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=290,
            normalized_bytes_per_event=260,
            minimal_bytes_per_event=55,
            estimated_events_per_sec=5000.0,
        )
    ))

    # 16. Impact: HTTP GET flood
    registry.register(TelemetryRequirementProfile(
        technique_id="T1499",
        implementation_id="T1499-IMPL-02",
        vector_name="http_volumetric_get_flood",
        tactic="Impact",
        ocsf_class="network_activity",
        minimal_necessary_fields=["dst_endpoint.port", "traffic.packets", "traffic.duration_sec"],
        optional_redundant_fields=["src_endpoint.port", "protocol_name", "time"],
        fp_reduction_fields=["src_endpoint.ip", "traffic.bytes"],
        volume=TelemetryVolumeMetric(
            raw_bytes_per_event=360,
            normalized_bytes_per_event=310,
            minimal_bytes_per_event=70,
            estimated_events_per_sec=4000.0,
        )
    ))

    return registry
