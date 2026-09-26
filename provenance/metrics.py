from __future__ import annotations
"""
AHRAS Module / Provenance — Graph Reconstruction Quality & Topology Comparison
-------------------------------------------------------------------------------
Evaluates reconstructed attack scenario graphs against ground-truth benchmarks:
  - Node & Edge Precision, Recall, and F1.
  - Path Completeness across critical entry-to-impact trajectories.
  - Depth Similarity preserving causal lineage length.
  - Normalized Structural Distance (Topological Graph Edit Distance).
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from provenance.models import (
    ProvenanceAttackScenario, GraphQualityMetrics, ProvenanceEdge
)
from provenance.graph import ProvenanceGraph


def compute_node_precision_recall(
    pred_nodes: Set[str],
    gt_nodes: Set[str],
) -> Tuple[float, float, float]:
    """Computes Node Precision, Recall, and F1."""
    if not pred_nodes and not gt_nodes:
        return 1.0, 1.0, 1.0
    if not pred_nodes:
        return 0.0, 0.0, 0.0
    if not gt_nodes:
        return 0.0, 0.0, 0.0

    intersection = len(pred_nodes.intersection(gt_nodes))
    precision = intersection / float(len(pred_nodes))
    recall = intersection / float(len(gt_nodes))
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_edge_precision_recall(
    pred_edges: List[ProvenanceEdge],
    gt_edges: List[ProvenanceEdge],
    typed: bool = True,
) -> Tuple[float, float, float]:
    """
    Computes Edge Precision, Recall, and F1.
    If typed=True, matches on (source, target, relation).
    """
    if not pred_edges and not gt_edges:
        return 1.0, 1.0, 1.0
    if not pred_edges:
        return 0.0, 0.0, 0.0
    if not gt_edges:
        return 0.0, 0.0, 0.0

    def edge_key(e: ProvenanceEdge) -> Tuple[str, ...]:
        rel = e.relation.value if hasattr(e.relation, "value") else str(e.relation)
        return (e.source, e.target, rel) if typed else (e.source, e.target)

    pred_keys = {edge_key(e) for e in pred_edges}
    gt_keys = {edge_key(e) for e in gt_edges}

    intersection = len(pred_keys.intersection(gt_keys))
    precision = intersection / float(len(pred_keys))
    recall = intersection / float(len(gt_keys))
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_path_completeness(
    pred_edges: List[ProvenanceEdge],
    gt_critical_paths: List[List[str]],
) -> float:
    """
    Measures what fraction of sequential hops in ground-truth critical paths
    are preserved intact in the reconstructed scenario graph.
    """
    if not gt_critical_paths:
        return 1.0

    pred_pairs = {(e.source, e.target) for e in pred_edges}
    path_scores: List[float] = []

    for path in gt_critical_paths:
        if len(path) <= 1:
            path_scores.append(1.0)
            continue

        total_hops = len(path) - 1
        preserved_hops = 0
        for i in range(total_hops):
            pair = (path[i], path[i + 1])
            if pair in pred_pairs:
                preserved_hops += 1

        path_scores.append(preserved_hops / float(total_hops))

    return float(sum(path_scores) / len(path_scores)) if path_scores else 0.0


def compute_depth_similarity(
    pred_graph: ProvenanceGraph,
    gt_graph: ProvenanceGraph,
) -> float:
    """
    Computes depth similarity by evaluating the preservation of causal DAG hierarchy length:
      depth_sim = 1.0 - |max_depth_gt - max_depth_pred| / max(max_depth_gt, max_depth_pred, 1)
    """
    def get_max_depth(graph: ProvenanceGraph) -> int:
        if not graph.edges:
            return 0
        # Find root nodes
        roots = {e.source for e in graph.edges.values()} - {e.target for e in graph.edges.values()}
        if not roots:
            roots = {next(iter(graph.nodes.keys()))} if graph.nodes else set()

        max_d = 0
        for r in roots:
            visited = set()
            stack = [(r, 1)]
            while stack:
                curr, d = stack.pop()
                max_d = max(max_d, d)
                if curr not in visited and d < 20:
                    visited.add(curr)
                    for e in graph.get_outgoing_edges(curr):
                        stack.append((e.target, d + 1))
        return max_d

    d_gt = get_max_depth(gt_graph)
    d_pred = get_max_depth(pred_graph)

    denom = max(d_gt, d_pred, 1)
    depth_diff = abs(d_gt - d_pred)
    sim = max(0.0, 1.0 - (depth_diff / float(denom)))
    return float(sim)


def compute_structural_distance(
    pred_graph: ProvenanceGraph,
    gt_graph: ProvenanceGraph,
) -> float:
    """
    Computes normalized topological structural distance (bounded in [0.0, 1.0])
    based on symmetric graph difference over vertices and edges.
    """
    v_pred = set(pred_graph.nodes.keys())
    v_gt = set(gt_graph.nodes.keys())

    e_pred = {(e.source, e.target) for e in pred_graph.edges.values()}
    e_gt = {(e.source, e.target) for e in gt_graph.edges.values()}

    node_diff = len(v_pred.symmetric_difference(v_gt))
    edge_diff = len(e_pred.symmetric_difference(e_gt))

    total_elements = len(v_pred) + len(v_gt) + len(e_pred) + len(e_gt)
    if total_elements == 0:
        return 0.0

    dist = (node_diff + edge_diff) / float(total_elements)
    return min(1.0, max(0.0, float(dist)))


def compute_comprehensive_graph_quality(
    pred_scenario: ProvenanceAttackScenario,
    gt_scenario: ProvenanceAttackScenario,
    pred_graph: ProvenanceGraph,
    gt_graph: ProvenanceGraph,
    gt_critical_paths: Optional[List[List[str]]] = None,
) -> GraphQualityMetrics:
    """
    Calculates the full suite of structure/depth-preserving graph reconstruction quality metrics.
    """
    # 1. Node Precision / Recall / F1
    v_pred = set(pred_scenario.entities)
    v_gt = set(gt_scenario.entities)
    node_p, node_r, node_f1 = compute_node_precision_recall(v_pred, v_gt)

    # 2. Edge Precision / Recall / F1
    edge_p, edge_r, edge_f1 = compute_edge_precision_recall(pred_scenario.edges, gt_scenario.edges)

    # 3. Path Completeness
    if not gt_critical_paths and gt_scenario.edges:
        # Synthesize linear critical path from ground-truth edges
        sorted_gt = sorted(gt_scenario.edges, key=lambda e: e.timestamp)
        nodes_in_order = []
        for e in sorted_gt:
            if not nodes_in_order or nodes_in_order[-1] != e.source:
                nodes_in_order.append(e.source)
            if not nodes_in_order or nodes_in_order[-1] != e.target:
                nodes_in_order.append(e.target)
        gt_critical_paths = [nodes_in_order]

    path_comp = compute_path_completeness(pred_scenario.edges, gt_critical_paths or [])

    # 4. Depth Similarity
    depth_sim = compute_depth_similarity(pred_graph, gt_graph)

    # 5. Structural Distance
    struct_dist = compute_structural_distance(pred_graph, gt_graph)

    # 6. Overall Harmonic Graph F1
    components = [node_f1, edge_f1, path_comp]
    valid_comps = [c for c in components if c > 1e-4]
    if len(valid_comps) == 3:
        overall_f1 = 3.0 / sum(1.0 / c for c in components)
    elif valid_comps:
        overall_f1 = float(sum(valid_comps) / len(valid_comps))
    else:
        overall_f1 = 0.0

    return GraphQualityMetrics(
        node_precision=node_p,
        node_recall=node_r,
        node_f1=node_f1,
        edge_precision=edge_p,
        edge_recall=edge_r,
        edge_f1=edge_f1,
        path_completeness=path_comp,
        depth_similarity=depth_sim,
        structural_distance=struct_dist,
        overall_graph_f1=overall_f1,
    )
