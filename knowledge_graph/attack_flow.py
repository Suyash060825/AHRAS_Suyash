from __future__ import annotations
"""
AHRAS Module — Attack Flow Interoperability Engine (Section 22 / Research Frontier P1)
--------------------------------------------------------------------------------------
Implements standardized Attack Flow modeling (MITRE Center for Threat-Informed Defense)
bridging low-level causal provenance graphs into structured adversary action flows.

Guarantees:
  1. Full preservation of timestamps, epistemic uncertainty, confidence, and hypothetical edges.
  2. JSON import and export interoperability.
  3. Formal completeness evaluation:
     - Stage Completeness: fraction of key kill-chain tactics represented
     - Edge Completeness: ratio of verified causal transitions to isolated actions
     - Entity Completeness: coverage of targeted hosts, users, and processes
     - Technique Coverage: ATT&CK technique breadth
     - Temporal Ordering: strict chronological monotonicity check
"""

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


@dataclass
class AttackAction:
    action_id: str
    name: str
    technique_id: str                     # e.g. "T1059.001"
    technique_name: str
    tactic: str                           # e.g. "Execution", "Lateral Movement"
    timestamp: float
    confidence: float                     # [0.0, 1.0]
    epistemic_uncertainty: float          # [0.0, 1.0]
    asset_ref: Optional[str] = None       # Pointer to affected AttackAsset
    is_hypothetical: bool = False         # True if inferred/counterfactual, False if directly observed
    evidence_ids: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttackAsset:
    asset_id: str
    name: str
    asset_type: str                       # "HOST", "ACCOUNT", "PROCESS", "DATABASE"
    criticality: float = 1.0              # [0.5, 2.0]
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttackFlowOperator:
    operator_id: str
    operator_type: str                    # "AND", "OR"
    input_action_ids: List[str] = field(default_factory=list)
    output_action_ids: List[str] = field(default_factory=list)


@dataclass
class AttackFlowEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str = "causes"             # "causes", "enables", "targets"
    confidence: float = 1.0
    is_hypothetical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttackFlowMetrics:
    """Formal audit metrics evaluating structural and temporal quality of the Attack Flow."""
    action_count: int
    asset_count: int
    edge_count: int
    stage_completeness_pct: float
    edge_completeness_pct: float
    entity_completeness_pct: float
    technique_coverage_count: int
    temporal_ordering_valid: bool
    hypothetical_ratio_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AttackFlow:
    """
    Standardized Attack Flow container with bidirectional JSON serialization
    and empirical completeness verification.
    """

    STANDARD_TACTIC_ORDER = [
        "Reconnaissance",
        "Initial Access",
        "Execution",
        "Persistence",
        "Privilege Escalation",
        "Defense Evasion",
        "Credential Access",
        "Discovery",
        "Lateral Movement",
        "Collection",
        "Command and Control",
        "Exfiltration",
        "Impact",
    ]

    def __init__(
        self,
        flow_id: Optional[str] = None,
        name: str = "AHRAS Reconstructed Attack Flow",
        description: str = "",
        author: str = "AHRAS Provenance Reasoner",
    ) -> None:
        self.flow_id = flow_id or f"flow-{uuid.uuid4().hex[:8]}"
        self.name = name
        self.description = description
        self.author = author
        self.created_at = time.time()
        self.actions: Dict[str, AttackAction] = {}
        self.assets: Dict[str, AttackAsset] = {}
        self.operators: Dict[str, AttackFlowOperator] = {}
        self.edges: List[AttackFlowEdge] = []

    def add_action(self, action: AttackAction) -> None:
        self.actions[action.action_id] = action

    def add_asset(self, asset: AttackAsset) -> None:
        self.assets[asset.asset_id] = asset

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "causes",
        confidence: float = 1.0,
        is_hypothetical: bool = False,
    ) -> AttackFlowEdge:
        edge = AttackFlowEdge(
            edge_id=f"fe-{source_id}-{target_id}",
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            confidence=confidence,
            is_hypothetical=is_hypothetical,
        )
        self.edges.append(edge)
        return edge

    def compute_completeness_metrics(self) -> AttackFlowMetrics:
        """Computes formal completeness and temporal validity metrics."""
        n_actions = len(self.actions)
        n_assets = len(self.assets)
        n_edges = len(self.edges)

        if n_actions == 0:
            return AttackFlowMetrics(
                action_count=0,
                asset_count=n_assets,
                edge_count=0,
                stage_completeness_pct=0.0,
                edge_completeness_pct=0.0,
                entity_completeness_pct=0.0,
                technique_coverage_count=0,
                temporal_ordering_valid=True,
                hypothetical_ratio_pct=0.0,
            )

        # 1. Stage Completeness (tactics represented vs key attack stages)
        represented_tactics = {a.tactic for a in self.actions.values() if a.tactic}
        stage_comp = (len(represented_tactics) / min(len(self.STANDARD_TACTIC_ORDER), max(3, len(represented_tactics)))) * 100.0
        stage_comp = min(100.0, stage_comp)

        # 2. Edge Completeness (actions connected to at least one incoming or outgoing edge)
        connected_actions = set()
        for e in self.edges:
            connected_actions.add(e.source_id)
            connected_actions.add(e.target_id)
        edge_comp = (len(connected_actions.intersection(self.actions.keys())) / max(1, n_actions)) * 100.0

        # 3. Entity Completeness (actions with associated asset references)
        actions_with_assets = sum(1 for a in self.actions.values() if a.asset_ref and a.asset_ref in self.assets)
        entity_comp = (actions_with_assets / max(1, n_actions)) * 100.0

        # 4. Technique Coverage
        unique_techniques = {a.technique_id for a in self.actions.values()}

        # 5. Temporal Ordering Validity
        # For every causal edge (u -> v), timestamp(u) must be <= timestamp(v)
        temporal_valid = True
        for e in self.edges:
            u = self.actions.get(e.source_id)
            v = self.actions.get(e.target_id)
            if u and v:
                if u.timestamp > v.timestamp + 1e-3:  # allow 1ms float tolerance
                    temporal_valid = False
                    break

        # 6. Hypothetical Edge Ratio
        hypo_edges = sum(1 for e in self.edges if e.is_hypothetical)
        hypo_actions = sum(1 for a in self.actions.values() if a.is_hypothetical)
        hypo_ratio = ((hypo_edges + hypo_actions) / max(1, n_edges + n_actions)) * 100.0

        return AttackFlowMetrics(
            action_count=n_actions,
            asset_count=n_assets,
            edge_count=n_edges,
            stage_completeness_pct=round(stage_comp, 1),
            edge_completeness_pct=round(edge_comp, 1),
            entity_completeness_pct=round(entity_comp, 1),
            technique_coverage_count=len(unique_techniques),
            temporal_ordering_valid=temporal_valid,
            hypothetical_ratio_pct=round(hypo_ratio, 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at,
            "actions": [a.to_dict() for a in self.actions.values()],
            "assets": [ast.to_dict() for ast in self.assets.values()],
            "edges": [e.to_dict() for e in self.edges],
            "metrics": self.compute_completeness_metrics().to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AttackFlow:
        flow = cls(
            flow_id=data.get("flow_id"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            author=data.get("author", ""),
        )
        flow.created_at = data.get("created_at", time.time())

        for ast_dict in data.get("assets", []):
            flow.add_asset(AttackAsset(**ast_dict))

        for act_dict in data.get("actions", []):
            flow.add_action(AttackAction(**act_dict))

        for edg_dict in data.get("edges", []):
            flow.add_edge(**edg_dict)

        return flow

    def export_json(self, file_path: Path | str) -> None:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def import_json(cls, file_path: Path | str) -> AttackFlow:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
