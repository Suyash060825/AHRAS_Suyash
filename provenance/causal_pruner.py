"""
AHRAS Causal Graph Pruner
-------------------------
Removes dependency explosion noise, high-fanout benign administrative chatter,
and non-causal background cycles from provenance DAGs while strictly preserving
all attack-correlated pathways and forensic anchors.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Set, Tuple

from provenance.models import (
    ProvenanceNodeType,
    ProvenanceEdgeType,
    ProvenanceNode,
    ProvenanceEdge,
)
from provenance.graph import ProvenanceGraph


class CausalGraphPruner:
    """
    Dependency explosion mitigation and noise pruning for forensic provenance DAGs.
    """
    def __init__(
        self,
        min_risk_preserve: float = 0.35,
        max_benign_fanout: int = 10,
    ) -> None:
        self.min_risk_preserve = min_risk_preserve
        self.max_benign_fanout = max_benign_fanout

    def prune(
        self,
        graph: ProvenanceGraph,
        pruned_graph_id: str = "pruned-provenance-graph",
    ) -> Tuple[ProvenanceGraph, Dict[str, float]]:
        """
        Prunes redundant high-fanout benign edges and unanchored leaf nodes.
        Preserves all nodes with risk >= min_risk_preserve, alert nodes, and their causal paths.
        """
        raw_node_count = len(graph.nodes)
        raw_edge_count = len(graph.edges)

        # 1. Identify critical anchor nodes (alerts, high risk, known IOCs)
        anchor_nodes: Set[str] = set()
        for nid, node in graph.nodes.items():
            if (
                node.node_type in {ProvenanceNodeType.ALERT, ProvenanceNodeType.TECHNIQUE, ProvenanceNodeType.INCIDENT}
                or node.risk_score >= self.min_risk_preserve
            ):
                anchor_nodes.add(nid)

        # 2. Determine all nodes that lie on paths to/from any anchor node
        nodes_on_anchor_paths: Set[str] = set(anchor_nodes)
        for anchor in anchor_nodes:
            # Backward ancestors
            for path in graph.backward_causal_path(anchor, max_depth=6):
                for e in path:
                    nodes_on_anchor_paths.add(e.source)
                    nodes_on_anchor_paths.add(e.target)
            # Forward reachability
            downstream = graph.forward_reachability(anchor, max_depth=4)
            nodes_on_anchor_paths.update(downstream)

        # 3. Filter edges
        pruned_graph = ProvenanceGraph(graph_id=pruned_graph_id)
        fanout_counters: Dict[str, int] = {}

        # Add nodes on anchor paths or with high risk
        for nid in nodes_on_anchor_paths:
            if nid in graph.nodes:
                pruned_graph.add_node(graph.nodes[nid])

        for eid, edge in graph.edges.items():
            # If both endpoints are on anchor paths, definitely keep
            if edge.source in nodes_on_anchor_paths and edge.target in nodes_on_anchor_paths:
                pruned_graph.add_edge(edge)
                continue

            # For background edges, enforce fanout limit
            fanout_counters[edge.source] = fanout_counters.get(edge.source, 0) + 1
            src_node = graph.get_node(edge.source)
            is_high_risk = src_node and src_node.risk_score >= self.min_risk_preserve

            if not is_high_risk and fanout_counters[edge.source] > self.max_benign_fanout:
                # Prune excess benign fanout
                continue

            # Otherwise include edge if source or target has minimal risk
            if (
                (src_node and src_node.risk_score >= 0.15)
                or (graph.get_node(edge.target) and graph.get_node(edge.target).risk_score >= 0.15)
            ):
                pruned_graph.add_edge(edge)

        pruned_node_count = len(pruned_graph.nodes)
        pruned_edge_count = len(pruned_graph.edges)

        raw_total = raw_node_count + raw_edge_count
        pruned_total = pruned_node_count + pruned_edge_count
        compression_ratio = round(1.0 - (pruned_total / max(1, raw_total)), 4)

        metrics = {
            "raw_nodes": float(raw_node_count),
            "pruned_nodes": float(pruned_node_count),
            "raw_edges": float(raw_edge_count),
            "pruned_edges": float(pruned_edge_count),
            "compression_ratio": max(0.0, compression_ratio),
            "anchor_count": float(len(anchor_nodes)),
        }

        return pruned_graph, metrics
