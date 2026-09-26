from __future__ import annotations
"""
AHRAS Module / Provenance — Data Models & Schema Contracts
----------------------------------------------------------
Defines heterogeneous provenance nodes, causal graph edges, reconstructed attack
scenarios, and structural graph comparison quality metrics.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ProvenanceNodeType(str, Enum):
    IP = "IP"
    HOST = "HOST"
    USER = "USER"
    ACCOUNT = "ACCOUNT"
    PROCESS = "PROCESS"
    FILE = "FILE"
    DOMAIN = "DOMAIN"
    SOCKET = "SOCKET"
    ASSET = "ASSET"
    ALERT = "ALERT"
    TECHNIQUE = "TECHNIQUE"
    INCIDENT = "INCIDENT"


class ProvenanceEdgeType(str, Enum):
    CONNECTS_TO = "CONNECTS_TO"
    EXECUTES = "EXECUTES"
    READS = "READS"
    WRITES = "WRITES"
    CREATES = "CREATES"
    AUTHENTICATES = "AUTHENTICATES"
    RESOLVES = "RESOLVES"
    COMMUNICATES = "COMMUNICATES"
    TARGETS = "TARGETS"
    USES_TECHNIQUE = "USES_TECHNIQUE"
    TRIGGERS = "TRIGGERS"
    PART_OF = "PART_OF"


@dataclass
class ProvenanceNode:
    """Represents a discrete entity in the heterogeneous forensic provenance graph."""
    node_id: str
    node_type: ProvenanceNodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float = 0.0
    risk_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["node_type"] = self.node_type.value if isinstance(self.node_type, ProvenanceNodeType) else str(self.node_type)
        return res


@dataclass
class ProvenanceEdge:
    """
    Represents a directional causal or structural relationship between two nodes.
    Mandatory fields: timestamp, source, target, confidence, event_id.
    """
    edge_id: str
    source: str                         # source node_id
    target: str                         # target node_id
    relation: ProvenanceEdgeType        # edge relation type
    timestamp: float                    # epoch timestamp of relationship
    confidence: float = 1.0             # causal confidence in [0.0, 1.0]
    event_id: str = ""                  # provenance event identifier
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["relation"] = self.relation.value if isinstance(self.relation, ProvenanceEdgeType) else str(self.relation)
        return res


@dataclass
class ProvenanceAttackScenario:
    """
    Unified multi-stage attack scenario reconstructed from fragmented events and alerts.
    """
    incident_id: str
    stages: List[str]                   # Ordered kill-chain stages (e.g. RECON, INITIAL_ACCESS, LATERAL_MOVEMENT)
    entities: List[str]                 # Participating node IDs
    edges: List[ProvenanceEdge]         # Causal provenance edges forming the scenario DAG
    confidence: float                   # Aggregate scenario confidence [0.0, 1.0]
    path_risk: float                    # Cumulative path risk [0.0, 1.0]
    missing_steps: List[str]            # Hypothesized stealthy / unobserved intermediate steps
    summary: str = ""                   # High-level incident explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "stages": self.stages,
            "entities": self.entities,
            "edges": [e.to_dict() for e in self.edges],
            "confidence": round(self.confidence, 4),
            "path_risk": round(self.path_risk, 4),
            "missing_steps": self.missing_steps,
            "summary": self.summary,
        }


@dataclass
class GraphQualityMetrics:
    """
    Structure/depth-preserving graph comparison metrics against ground-truth scenario graphs.
    """
    node_precision: float
    node_recall: float
    node_f1: float
    edge_precision: float
    edge_recall: float
    edge_f1: float
    path_completeness: float            # Fraction of critical ground-truth paths preserved
    depth_similarity: float             # Length and topology similarity of causal paths
    structural_distance: float          # Normalized graph edit / degree distribution distance [0.0, 1.0]
    overall_graph_f1: float             # Harmonic mean of node F1, edge F1, and path completeness

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) for k, v in asdict(self).items()}
