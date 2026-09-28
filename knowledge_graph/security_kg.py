from __future__ import annotations
"""
AHRAS Module — Security & Detection Knowledge Graph (Section 21 / Research Frontier P1)
----------------------------------------------------------------------------------------
Implements the high-level heterogeneous security knowledge graph connecting:

Nodes:
  TECHNIQUE, IMPLEMENTATION, TELEMETRY, SENSOR, DETECTION_RULE, ML_MODEL,
  EVIDENCE, ASSET, VULNERABILITY, THREAT_INTEL, RESPONSE_ACTION.

Relationships:
  requires, observed_by, detected_by, affects, mitigated_by, depends_on,
  validated_by, blocked_by, exposed_by.

Answers core operational and architectural queries:
  1. "What currently enables detection of this behavior?"
  2. "What sensor is missing to observe this technique?"
  3. "Which detections depend on an unhealthy sensor?"
  4. "Which response actions mitigate this multi-stage attack path?"
"""

import collections
import enum
import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


class SecurityNodeType(str, enum.Enum):
    TECHNIQUE = "TECHNIQUE"                # MITRE ATT&CK Technique (e.g. T1059)
    IMPLEMENTATION = "IMPLEMENTATION"      # Concrete tool / command line invocation
    TELEMETRY = "TELEMETRY"                # Data modality (e.g. process_exec, flow_record)
    SENSOR = "SENSOR"                      # Ingestion source (e.g. eBPF, Zeek, CloudTrail)
    DETECTION_RULE = "DETECTION_RULE"      # Signature or heuristic rule (e.g. NET-005)
    ML_MODEL = "ML_MODEL"                  # Machine learning / GNN model
    EVIDENCE = "EVIDENCE"                  # Cryptographic evidence record
    ASSET = "ASSET"                        # Host, database, or identity
    VULNERABILITY = "VULNERABILITY"        # CVE or configuration weakness
    THREAT_INTEL = "THREAT_INTEL"          # IOC, feed, or threat actor profile
    RESPONSE_ACTION = "RESPONSE_ACTION"    # Playbook mitigation (e.g. ISOLATE, KILL)


class SecurityEdgeType(str, enum.Enum):
    REQUIRES = "requires"                  # Technique requires Implementation/Asset
    OBSERVED_BY = "observed_by"            # Telemetry observed_by Sensor
    DETECTED_BY = "detected_by"            # Technique detected_by Rule / Model
    AFFECTS = "affects"                    # Technique affects Asset
    MITIGATED_BY = "mitigated_by"          # Technique mitigated_by ResponseAction
    DEPENDS_ON = "depends_on"              # Rule/Model depends_on Telemetry
    VALIDATED_BY = "validated_by"          # Detection validated_by Evidence
    BLOCKED_BY = "blocked_by"              # Attack blocked_by ResponseAction
    EXPOSED_BY = "exposed_by"              # Asset exposed_by Vulnerability


@dataclass
class SecurityNode:
    node_id: str
    node_type: SecurityNodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "properties": self.properties,
            "is_active": self.is_active,
        }


@dataclass
class SecurityEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: SecurityEdgeType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "properties": self.properties,
        }


class SecurityKnowledgeGraph:
    """
    Multi-directed knowledge graph of cybersecurity capabilities, dependencies,
    and attack relationships. Thread-safe via RLock.
    """

    def __init__(self, populate_defaults: bool = True) -> None:
        self._lock = threading.RLock()
        self._nodes: Dict[str, SecurityNode] = {}
        # Adjacency maps: source -> list of out-edges, target -> list of in-edges
        self._out_edges: Dict[str, List[SecurityEdge]] = collections.defaultdict(list)
        self._in_edges: Dict[str, List[SecurityEdge]] = collections.defaultdict(list)

        if populate_defaults:
            self._bootstrap_default_knowledge()

    def add_node(self, node: SecurityNode) -> None:
        with self._lock:
            self._nodes[node.node_id] = node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: SecurityEdgeType,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> SecurityEdge:
        with self._lock:
            edge_id = f"e-{source_id}-{edge_type.value}-{target_id}"
            edge = SecurityEdge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                properties=properties or {},
            )
            self._out_edges[source_id].append(edge)
            self._in_edges[target_id].append(edge)
            return edge

    def get_node(self, node_id: str) -> Optional[SecurityNode]:
        with self._lock:
            return self._nodes.get(node_id)

    def set_sensor_health(self, sensor_id: str, is_healthy: bool) -> bool:
        """Updates health status of an ingestion sensor."""
        with self._lock:
            node = self._nodes.get(sensor_id)
            if node and node.node_type == SecurityNodeType.SENSOR:
                node.is_active = is_healthy
                return True
            return False

    # ── Operational Query APIs ───────────────────────────────────────────────

    def what_enables_detection(self, technique_id: str) -> Dict[str, Any]:
        """
        Answers: 'What currently enables detection of this behavior?'
        Traverses: TECHNIQUE -> detected_by -> RULES / MODELS -> depends_on -> TELEMETRY -> observed_by -> SENSORS.
        """
        with self._lock:
            rules = []
            models = []
            telemetries = set()
            sensors = set()

            for edge in self._out_edges.get(technique_id, []):
                if edge.edge_type == SecurityEdgeType.DETECTED_BY:
                    target = self._nodes.get(edge.target_id)
                    if not target or not target.is_active:
                        continue
                    if target.node_type == SecurityNodeType.DETECTION_RULE:
                        rules.append(target.to_dict())
                    elif target.node_type == SecurityNodeType.ML_MODEL:
                        models.append(target.to_dict())

                    # Check dependencies of this detector
                    for det_edge in self._out_edges.get(target.node_id, []):
                        if det_edge.edge_type == SecurityEdgeType.DEPENDS_ON:
                            tel_node = self._nodes.get(det_edge.target_id)
                            if tel_node:
                                telemetries.add(tel_node.name)
                                # Check sensors observing this telemetry
                                for tel_edge in self._out_edges.get(tel_node.node_id, []):
                                    if tel_edge.edge_type == SecurityEdgeType.OBSERVED_BY:
                                        s_node = self._nodes.get(tel_edge.target_id)
                                        if s_node:
                                            sensors.add(s_node.name)

            return {
                "technique_id": technique_id,
                "is_detected": len(rules) > 0 or len(models) > 0,
                "detection_rules": rules,
                "ml_models": models,
                "required_telemetry": list(telemetries),
                "active_sensors": list(sensors),
            }

    def find_missing_sensors(self, technique_id: str) -> List[Dict[str, Any]]:
        """
        Answers: 'What sensor is missing to observe this technique?'
        Identifies sensors that observe required telemetry but are currently inactive or absent.
        """
        with self._lock:
            missing = []
            for edge in self._out_edges.get(technique_id, []):
                if edge.edge_type == SecurityEdgeType.DETECTED_BY:
                    det = self._nodes.get(edge.target_id)
                    if not det:
                        continue
                    for det_edge in self._out_edges.get(det.node_id, []):
                        if det_edge.edge_type == SecurityEdgeType.DEPENDS_ON:
                            tel = self._nodes.get(det_edge.target_id)
                            if not tel:
                                continue
                            for tel_edge in self._out_edges.get(tel.node_id, []):
                                if tel_edge.edge_type == SecurityEdgeType.OBSERVED_BY:
                                    sensor = self._nodes.get(tel_edge.target_id)
                                    if sensor and not sensor.is_active:
                                        missing.append({
                                            "sensor_id": sensor.node_id,
                                            "sensor_name": sensor.name,
                                            "telemetry": tel.name,
                                            "reason": "Sensor is marked unhealthy or inactive",
                                        })
            return missing

    def detections_affected_by_sensor(self, sensor_id: str) -> List[Dict[str, Any]]:
        """
        Answers: 'Which detections depend on an unhealthy sensor?'
        Finds all detection rules and ML models whose telemetry pipeline passes through this sensor.
        """
        with self._lock:
            affected = []
            # Find telemetries observed by this sensor
            telemetries_observed = set()
            for edge in self._in_edges.get(sensor_id, []):
                if edge.edge_type == SecurityEdgeType.OBSERVED_BY:
                    telemetries_observed.add(edge.source_id)

            # Find detectors that depend on these telemetries
            for tel_id in telemetries_observed:
                for edge in self._in_edges.get(tel_id, []):
                    if edge.edge_type == SecurityEdgeType.DEPENDS_ON:
                        det = self._nodes.get(edge.source_id)
                        if det:
                            affected.append({
                                "detector_id": det.node_id,
                                "detector_type": det.node_type.value,
                                "detector_name": det.name,
                                "telemetry_lost": self._nodes[tel_id].name,
                            })
            return affected

    def mitigating_responses_for_path(self, technique_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Answers: 'Which response actions mitigate this multi-stage attack path?'
        Identifies chokepoint response actions mapped to techniques in the sequence.
        """
        with self._lock:
            actions_by_technique = []
            for tid in technique_ids:
                mitigations = []
                for edge in self._out_edges.get(tid, []):
                    if edge.edge_type in (SecurityEdgeType.MITIGATED_BY, SecurityEdgeType.BLOCKED_BY):
                        resp = self._nodes.get(edge.target_id)
                        if resp:
                            mitigations.append({
                                "response_id": resp.node_id,
                                "action_name": resp.name,
                                "efficacy_weight": edge.weight,
                            })
                actions_by_technique.append({
                    "technique_id": tid,
                    "mitigations": mitigations,
                })
            return actions_by_technique

    # ── Bootstrap Default Architecture Knowledge ─────────────────────────────

    def _bootstrap_default_knowledge(self) -> None:
        """Seeds graph with core AHRAS sensors, detection engines, techniques, and playbooks."""
        # 1. Sensors
        sensors = [
            SecurityNode("sensor-ebpf", SecurityNodeType.SENSOR, "Host eBPF Tetragon Agent"),
            SecurityNode("sensor-zeek", SecurityNodeType.SENSOR, "Zeek Network Sensor"),
            SecurityNode("sensor-cloudtrail", SecurityNodeType.SENSOR, "AWS CloudTrail Adapter"),
            SecurityNode("sensor-win-etw", SecurityNodeType.SENSOR, "Windows ETW Collector", is_active=False),
        ]
        for s in sensors:
            self.add_node(s)

        # 2. Telemetry Types
        telemetries = [
            SecurityNode("tel-process-exec", SecurityNodeType.TELEMETRY, "Process Execution & CmdLine"),
            SecurityNode("tel-network-flow", SecurityNodeType.TELEMETRY, "Network Flow & Packet Statistics"),
            SecurityNode("tel-cloud-audit", SecurityNodeType.TELEMETRY, "Cloud IAM & Control Plane API"),
            SecurityNode("tel-file-io", SecurityNodeType.TELEMETRY, "File System IO & Entropy"),
        ]
        for t in telemetries:
            self.add_node(t)

        # Telemetry -> Sensor links
        self.add_edge("tel-process-exec", "sensor-ebpf", SecurityEdgeType.OBSERVED_BY)
        self.add_edge("tel-file-io", "sensor-ebpf", SecurityEdgeType.OBSERVED_BY)
        self.add_edge("tel-network-flow", "sensor-zeek", SecurityEdgeType.OBSERVED_BY)
        self.add_edge("tel-cloud-audit", "sensor-cloudtrail", SecurityEdgeType.OBSERVED_BY)
        self.add_edge("tel-process-exec", "sensor-win-etw", SecurityEdgeType.OBSERVED_BY)

        # 3. Detection Rules & ML Models
        detectors = [
            SecurityNode("rule-net-005", SecurityNodeType.DETECTION_RULE, "NET-005: SSH Brute Force"),
            SecurityNode("rule-proc-002", SecurityNodeType.DETECTION_RULE, "PROC-002: Shell Exec in CmdLine"),
            SecurityNode("rule-file-001", SecurityNodeType.DETECTION_RULE, "FILE-001: Ransomware High Entropy"),
            SecurityNode("model-iso-forest", SecurityNodeType.ML_MODEL, "Isolation Forest Anomaly Ensemble"),
            SecurityNode("model-multimodal", SecurityNodeType.ML_MODEL, "Multimodal Cross-Attention Encoder"),
        ]
        for d in detectors:
            self.add_node(d)

        # Detector -> Telemetry dependencies
        self.add_edge("rule-net-005", "tel-network-flow", SecurityEdgeType.DEPENDS_ON)
        self.add_edge("rule-proc-002", "tel-process-exec", SecurityEdgeType.DEPENDS_ON)
        self.add_edge("rule-file-001", "tel-file-io", SecurityEdgeType.DEPENDS_ON)
        self.add_edge("model-iso-forest", "tel-network-flow", SecurityEdgeType.DEPENDS_ON)
        self.add_edge("model-multimodal", "tel-process-exec", SecurityEdgeType.DEPENDS_ON)
        self.add_edge("model-multimodal", "tel-network-flow", SecurityEdgeType.DEPENDS_ON)

        # 4. MITRE ATT&CK Techniques
        techniques = [
            SecurityNode("tech-t1110", SecurityNodeType.TECHNIQUE, "T1110: Brute Force"),
            SecurityNode("tech-t1059", SecurityNodeType.TECHNIQUE, "T1059: Command & Scripting Interpreter"),
            SecurityNode("tech-t1486", SecurityNodeType.TECHNIQUE, "T1486: Data Encrypted for Impact (Ransomware)"),
            SecurityNode("tech-t1021", SecurityNodeType.TECHNIQUE, "T1021: Remote Services (Lateral Movement)"),
        ]
        for tech in techniques:
            self.add_node(tech)

        # Technique -> Detector mappings
        self.add_edge("tech-t1110", "rule-net-005", SecurityEdgeType.DETECTED_BY)
        self.add_edge("tech-t1059", "rule-proc-002", SecurityEdgeType.DETECTED_BY)
        self.add_edge("tech-t1486", "rule-file-001", SecurityEdgeType.DETECTED_BY)
        self.add_edge("tech-t1021", "model-iso-forest", SecurityEdgeType.DETECTED_BY)
        self.add_edge("tech-t1021", "model-multimodal", SecurityEdgeType.DETECTED_BY)

        # 5. Response Actions
        responses = [
            SecurityNode("resp-isolate", SecurityNodeType.RESPONSE_ACTION, "Network Host Isolation"),
            SecurityNode("resp-kill", SecurityNodeType.RESPONSE_ACTION, "Terminate Malicious Process"),
            SecurityNode("resp-block-ip", SecurityNodeType.RESPONSE_ACTION, "Firewall Drop IP Rule"),
            SecurityNode("resp-revoke-token", SecurityNodeType.RESPONSE_ACTION, "Revoke Session Token / Credential"),
        ]
        for r in responses:
            self.add_node(r)

        # Technique -> Response mitigations
        self.add_edge("tech-t1110", "resp-block-ip", SecurityEdgeType.MITIGATED_BY, weight=0.95)
        self.add_edge("tech-t1059", "resp-kill", SecurityEdgeType.MITIGATED_BY, weight=0.90)
        self.add_edge("tech-t1486", "resp-isolate", SecurityEdgeType.MITIGATED_BY, weight=0.99)
        self.add_edge("tech-t1021", "resp-isolate", SecurityEdgeType.MITIGATED_BY, weight=0.88)
