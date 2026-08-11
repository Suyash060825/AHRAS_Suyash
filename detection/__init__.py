from __future__ import annotations
"""
AHRAS Detection Engine Package
------------------------------
Exporting unified public API for hybrid signature/ML combiner, multi-modal
statistical engine, adaptive risk engine, and entity graph lateral movement.
"""

from detection.hybrid_engine import (
    DetectionResult,
    HybridCombiner,
    get_combiner,
)

from detection.risk_engine import (
    RiskResult,
    AdaptiveRiskEngine,
    get_risk_engine,
    run_risk_engine,
)

from detection.statistical_engine import (
    StatResult,
    StatisticalEngine,
    get_statistical_engine,
    run_statistical_engine,
    PeerGroupEngine,
    get_peer_group_engine,
    TrendEngine,
    get_trend_engine,
    EntityReportGenerator,
    get_entity_report_generator,
)

from detection.gnn_engine import (
    GraphPathAnomaly,
    EntityGraphEngine,
    get_entity_graph_engine,
)

__all__ = [
    "DetectionResult",
    "HybridCombiner",
    "get_combiner",
    "RiskResult",
    "AdaptiveRiskEngine",
    "get_risk_engine",
    "run_risk_engine",
    "StatResult",
    "StatisticalEngine",
    "get_statistical_engine",
    "run_statistical_engine",
    "PeerGroupEngine",
    "get_peer_group_engine",
    "TrendEngine",
    "get_trend_engine",
    "EntityReportGenerator",
    "get_entity_report_generator",
    "GraphPathAnomaly",
    "EntityGraphEngine",
    "get_entity_graph_engine",
]
