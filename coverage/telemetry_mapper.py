"""
AHRAS Telemetry Mapper & Observability Evaluator
------------------------------------------------
Computes telemetry observability:
    O(i) = 1(R_req(i) ⊆ R_avail)

Determines whether an implementation vector can be observed by the active
telemetry sensors and OCSF normalizers before any detector logic is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from coverage.implementation_catalog import ImplementationCatalog, TechniqueImplementation


# Standard OCSF & Telemetry fields natively ingested and normalized by AHRAS sensors
DEFAULT_AHRAS_AVAILABLE_TELEMETRY: Set[str] = {
    # Process activity
    "process.cmd",
    "process.name",
    "process.pid",
    "actor.user.name",
    "actor.user.uid",
    "parent_process.name",
    "parent_process.pid",
    "parent_process.uid",
    # File activity
    "file.path",
    "file.action",
    "file.entropy",
    "file.extension",
    # Network activity
    "src_endpoint.ip",
    "src_endpoint.port",
    "dst_endpoint.ip",
    "dst_endpoint.port",
    "traffic.bytes",
    "traffic.packets",
    "traffic.pps",
    "traffic.interval_variance",
    "tcp_flags",
    "protocol_name",
    "unique_dst_ports",
    "enrichment.is_private",
    "enrichment.is_known_c2",
    # Cloud API activity
    "api.operation",
    "response.error",
    # Encrypted session intelligence
    "flow_id",
    "flow.packet_lengths",
    "flow.inter_arrival_times",
}


@dataclass
class TelemetryObservationResult:
    """
    Observation status for a single technique implementation.
    """
    implementation_id: str
    technique_id: str
    tactic: str
    vector_name: str
    observable: bool
    required_fields: List[str]
    available_fields: List[str]
    missing_fields: List[str]
    adequacy_ratio: float  # |R_req ∩ R_avail| / |R_req|


class TelemetryMapper:
    """
    Evaluates telemetry adequacy and concrete observability for MITRE ATT&CK implementations.
    """
    def __init__(self, available_telemetry: Optional[Set[str]] = None) -> None:
        self.available_telemetry: Set[str] = (
            set(available_telemetry) if available_telemetry is not None
            else set(DEFAULT_AHRAS_AVAILABLE_TELEMETRY)
        )

    def register_telemetry_field(self, field_path: str) -> None:
        """Register an additional telemetry field available in the sensor pipeline."""
        self.available_telemetry.add(field_path)

    def remove_telemetry_field(self, field_path: str) -> None:
        """Remove a telemetry field (e.g. simulating sensor outage or degraded telemetry)."""
        self.available_telemetry.discard(field_path)

    def is_observable(self, required_fields: List[str]) -> bool:
        """
        True iff all required telemetry fields are present in available telemetry.
        O(i) = 1(R_req(i) ⊆ R_avail)
        """
        if not required_fields:
            return False
        return all(f in self.available_telemetry for f in required_fields)

    def evaluate_implementation(self, impl: TechniqueImplementation) -> TelemetryObservationResult:
        """
        Evaluates observability and field gaps for a given implementation.
        """
        req_set = set(impl.required_telemetry_fields)
        avail = [f for f in impl.required_telemetry_fields if f in self.available_telemetry]
        missing = [f for f in impl.required_telemetry_fields if f not in self.available_telemetry]
        
        ratio = len(avail) / len(req_set) if req_set else 0.0
        observable = (len(missing) == 0) and (len(req_set) > 0)

        return TelemetryObservationResult(
            implementation_id=impl.implementation_id,
            technique_id=impl.technique_id,
            tactic=impl.tactic,
            vector_name=impl.vector_name,
            observable=observable,
            required_fields=list(impl.required_telemetry_fields),
            available_fields=avail,
            missing_fields=missing,
            adequacy_ratio=round(ratio, 4),
        )

    def evaluate_catalog(self, catalog: ImplementationCatalog) -> Dict[str, TelemetryObservationResult]:
        """
        Evaluates observability for all implementations in the catalog.
        Returns mapping from implementation_id to TelemetryObservationResult.
        """
        results: Dict[str, TelemetryObservationResult] = {}
        for impl in catalog.get_all_implementations():
            results[impl.implementation_id] = self.evaluate_implementation(impl)
        return results

    def get_telemetry_gaps(self, catalog: ImplementationCatalog) -> Dict[str, List[str]]:
        """
        Returns technique_id -> list of missing telemetry fields across unobservable implementations.
        """
        gaps: Dict[str, Set[str]] = {}
        for impl in catalog.get_all_implementations():
            res = self.evaluate_implementation(impl)
            if not res.observable:
                gaps.setdefault(impl.technique_id, set()).update(res.missing_fields)
        return {k: sorted(list(v)) for k, v in gaps.items()}

    def compute_telemetry_coverage(self, implementations: List[TechniqueImplementation]) -> float:
        """
        TC = (observable implementations) / (total implementations)
        """
        if not implementations:
            return 0.0
        observable_count = sum(1 for i in implementations if self.is_observable(i.required_telemetry_fields))
        return round(observable_count / len(implementations), 4)
