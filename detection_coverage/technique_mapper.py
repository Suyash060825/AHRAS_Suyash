"""
AHRAS Technique Mapper
----------------------
Maps active AHRAS detection engines (signatures, anomaly detectors,
encrypted session analyzers, statistical baselines, graph correlation)
to specific ATT&CK techniques and concrete implementation vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from detection_coverage.implementation_catalog import ImplementationCatalog, TechniqueImplementation


@dataclass
class MappedDetector:
    """
    Representation of an active detector mapped to techniques/implementations.
    """
    engine_name: str  # signature, encrypted_session, ml_anomaly, statistical, graph_tgnn
    detector_id: str  # e.g., "NET-001", "PROC-002", "EncryptedSessionAnalyzer"
    target_technique_id: str
    target_implementation_ids: List[str]
    description: str
    confidence_weight: float = 1.0


class TechniqueMapper:
    """
    Manages mappings between AHRAS detection capabilities and cataloged implementations.
    """
    def __init__(self) -> None:
        self._detectors: List[MappedDetector] = []
        self._init_default_mappings()

    def register_detector(self, detector: MappedDetector) -> None:
        self._detectors.append(detector)

    def get_detectors_for_technique(self, technique_id: str) -> List[MappedDetector]:
        return [d for d in self._detectors if d.target_technique_id == technique_id]

    def get_detectors_for_implementation(self, implementation_id: str) -> List[MappedDetector]:
        return [d for d in self._detectors if implementation_id in d.target_implementation_ids]

    def get_all_mapped_techniques(self) -> Set[str]:
        return {d.target_technique_id for d in self._detectors}

    def get_all_mapped_implementations(self) -> Set[str]:
        impls = set()
        for d in self._detectors:
            impls.update(d.target_implementation_ids)
        return impls

    def _init_default_mappings(self) -> None:
        """
        Initializes concrete mappings for all active AHRAS detection engines.
        """
        # =====================================================================
        # 1. Execution
        # =====================================================================
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="PROC-002",
            target_technique_id="T1059.001",
            target_implementation_ids=["T1059.001-IMPL-01", "T1059.001-IMPL-02"],
            description="Signature rule detecting base64/cradle execution patterns in command line",
            confidence_weight=0.85
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="PROC-002",
            target_technique_id="T1059.004",
            target_implementation_ids=["T1059.004-IMPL-01", "T1059.004-IMPL-02"],
            description="Signature rule detecting /dev/tcp and base64 decode piped to shell",
            confidence_weight=0.90
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="PROC-001",
            target_technique_id="T1059.004",
            target_implementation_ids=["T1059.004-IMPL-01"],
            description="Process lineage anomaly (web server apache2 spawning interactive bash shell)",
            confidence_weight=0.85
        ))

        # =====================================================================
        # 2. Persistence
        # =====================================================================
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="FILE-004",
            target_technique_id="T1543.002",
            target_implementation_ids=["T1543.002-IMPL-02"],
            description="Sensitive system directory write into /etc/systemd/system/",
            confidence_weight=0.80
        ))

        # =====================================================================
        # 3. Privilege Escalation
        # =====================================================================
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="PROC-003",
            target_technique_id="T1548.001",
            target_implementation_ids=["T1548.001-IMPL-01"],
            description="Root child process spawn from non-root parent process",
            confidence_weight=0.90
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="CLOUD-004",
            target_technique_id="T1078.004",
            target_implementation_ids=["T1078.004-IMPL-01"],
            description="Cloud API access from external non-private IP with unknown user",
            confidence_weight=0.80
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="CLOUD-002",
            target_technique_id="T1078.004",
            target_implementation_ids=["T1078.004-IMPL-02"],
            description="High-privilege IAM policy modification (AttachUserPolicy, PutUserPolicy)",
            confidence_weight=0.85
        ))

        # =====================================================================
        # 4. Defense Evasion
        # =====================================================================
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="FILE-003",
            target_technique_id="T1070.004",
            target_implementation_ids=["T1070.004-IMPL-01", "T1070.004-IMPL-02"],
            description="Shadow copy deletion commands and shadow storage target removal",
            confidence_weight=0.90
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="PROC-002",
            target_technique_id="T1027",
            target_implementation_ids=["T1027-IMPL-01"],
            description="Encoded command execution and obfuscated script strings",
            confidence_weight=0.85
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="FILE-001",
            target_technique_id="T1027",
            target_implementation_ids=["T1027-IMPL-02"],
            description="High Shannon entropy payload write detection (entropy >= 7.2)",
            confidence_weight=0.90
        ))

        # =====================================================================
        # 5. Credential Access
        # =====================================================================
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="PROC-004",
            target_technique_id="T1003.001",
            target_implementation_ids=["T1003.001-IMPL-01", "T1003.001-IMPL-02"],
            description="LSASS memory dump via Mimikatz / Procdump command patterns",
            confidence_weight=0.95
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="NET-005",
            target_technique_id="T1110.001",
            target_implementation_ids=["T1110.001-IMPL-01"],
            description="Volumetric SSH brute-force packet burst on port 22",
            confidence_weight=0.90
        ))
        self.register_detector(MappedDetector(
            engine_name="ml_anomaly",
            detector_id="MLDetectionEngine",
            target_technique_id="T1110.001",
            target_implementation_ids=["T1110.001-IMPL-01"],
            description="Isolation Forest flow anomaly for elevated connection bursts",
            confidence_weight=0.85
        ))

        # =====================================================================
        # 6. Discovery
        # =====================================================================
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="NET-001",
            target_technique_id="T1046",
            target_implementation_ids=["T1046-IMPL-01"],
            description="TCP SYN port scan across multiple destination ports",
            confidence_weight=0.90
        ))
        self.register_detector(MappedDetector(
            engine_name="statistical",
            detector_id="StatEngine",
            target_technique_id="T1046",
            target_implementation_ids=["T1046-IMPL-01"],
            description="Port fan-out z-score statistical threshold deviation",
            confidence_weight=0.85
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="CLOUD-003",
            target_technique_id="T1526",
            target_implementation_ids=["T1526-IMPL-01"],
            description="Cloud API AccessDenied / UnauthorizedOperation error rate surge",
            confidence_weight=0.80
        ))

        # =====================================================================
        # 7. Lateral Movement
        # =====================================================================
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="NET-006",
            target_technique_id="T1021.002",
            target_implementation_ids=["T1021.002-IMPL-01"],
            description="High-volume lateral SMB transfer on port 445",
            confidence_weight=0.85
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="NET-007",
            target_technique_id="T1021.001",
            target_implementation_ids=["T1021.001-IMPL-01"],
            description="External inbound RDP access attempt on port 3389",
            confidence_weight=0.85
        ))

        # =====================================================================
        # 8. Collection
        # =====================================================================
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="FILE-004",
            target_technique_id="T1005",
            target_implementation_ids=["T1005-IMPL-01"],
            description="Access to sensitive system paths (/etc/shadow, id_rsa, aws credentials)",
            confidence_weight=0.85
        ))

        # =====================================================================
        # 9. Command and Control
        # =====================================================================
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="NET-010",
            target_technique_id="T1071.001",
            target_implementation_ids=["T1071.001-IMPL-01"],
            description="C2 periodic beaconing detection with low timing variance",
            confidence_weight=0.85
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="NET-009",
            target_technique_id="T1071.001",
            target_implementation_ids=["T1071.001-IMPL-01"],
            description="Threat intelligence IP hit matching known C2 infrastructure",
            confidence_weight=0.95
        ))
        self.register_detector(MappedDetector(
            engine_name="encrypted_session",
            detector_id="EncryptedSessionAnalyzer",
            target_technique_id="T1573.002",
            target_implementation_ids=["T1573.002-IMPL-01", "T1573.002-IMPL-02"],
            description="Encrypted TLS session timing & packet size behavioral classifier",
            confidence_weight=0.93
        ))

        # =====================================================================
        # 10. Impact
        # =====================================================================
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="FILE-002",
            target_technique_id="T1486",
            target_implementation_ids=["T1486-IMPL-01"],
            description="Ransomware extension rename (.locked, .crypto, .wncry)",
            confidence_weight=0.95
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="FILE-001",
            target_technique_id="T1486",
            target_implementation_ids=["T1486-IMPL-02"],
            description="High-entropy file modification indicative of encryption",
            confidence_weight=0.90
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="NET-002",
            target_technique_id="T1498.001",
            target_implementation_ids=["T1498.001-IMPL-01"],
            description="Direct TCP SYN flood (>1000 pps)",
            confidence_weight=0.90
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="NET-003",
            target_technique_id="T1498.001",
            target_implementation_ids=["T1498.001-IMPL-02"],
            description="Direct UDP packet flood (>3000 packets)",
            confidence_weight=0.90
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="NET-004",
            target_technique_id="T1498.001",
            target_implementation_ids=["T1498.001-IMPL-03"],
            description="DNS reflection amplification flood (>100KB bytes from port 53)",
            confidence_weight=0.95
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="NET-011",
            target_technique_id="T1499",
            target_implementation_ids=["T1499-IMPL-01"],
            description="Slowloris / Slow HTTP connection starvation",
            confidence_weight=0.90
        ))
        self.register_detector(MappedDetector(
            engine_name="signature",
            detector_id="NET-012",
            target_technique_id="T1499",
            target_implementation_ids=["T1499-IMPL-02"],
            description="Volumetric HTTP GET flood targeting web server",
            confidence_weight=0.90
        ))
